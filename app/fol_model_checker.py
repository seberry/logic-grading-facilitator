from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import app.syntax as syntax


class FolModelParseError(ValueError):
    pass


@dataclass
class PredicateInterpretation:
    name: str
    arity: Optional[int]
    tuples: Set[Tuple[int, ...]]


@dataclass
class FolModel:
    domain: List[int]
    constants: Dict[str, int]
    predicates: Dict[str, PredicateInterpretation]


@dataclass
class SentenceCheckResult:
    index: int
    sentence: str
    parsed_ok: bool
    result: Optional[bool]
    trace: str
    trace_tree: Optional["TraceNode"] = None
    error: Optional[str] = None


@dataclass
class FolCheckResult:
    model_summary: str
    sentence_results: List[SentenceCheckResult]
    all_parsed_ok: bool


@dataclass
class FolTaskCheckResult:
    task_kind: str
    passed: bool
    summary: str
    model_summary: str
    trace_blocks: List[Dict[str, str]]
    draft_comment: str
    sentence_results: List[SentenceCheckResult]


@dataclass
class TraceNode:
    kind: str
    display: str
    result: Optional[bool] = None
    intro: List[str] = field(default_factory=list)
    reductions: List[str] = field(default_factory=list)
    children: List["TraceNode"] = field(default_factory=list)
    context: Dict[str, str] = field(default_factory=dict)
    evaluated_display: str = ""
    truth_display: str = ""


def parse_model_text(text: str) -> FolModel:
    sections = _parse_sections(text)

    domain_text = sections.get("DOMAIN")
    if not domain_text:
        raise FolModelParseError("Missing DOMAIN block.")
    domain = _parse_number_set(domain_text[0], "DOMAIN")
    if not domain:
        raise FolModelParseError("DOMAIN must not be empty.")

    constants: Dict[str, int] = {}
    for raw in sections.get("CONSTANTS", []):
        if not raw.strip() or _is_empty_section_marker(raw):
            continue
        name, value_text = _split_mapping(raw, "constant")
        if not re.fullmatch(r"[a-z]", name):
            raise FolModelParseError(f"Constant '{name}' must be a single lowercase letter.")
        value = _parse_int(value_text, f"constant {name}")
        if value not in domain:
            raise FolModelParseError(f"Constant '{name}' maps to {value}, which is not in the domain.")
        constants[name] = value

    predicates: Dict[str, PredicateInterpretation] = {}
    for raw in sections.get("PREDICATES", []):
        if not raw.strip() or _is_empty_section_marker(raw):
            continue
        name, value_text = _split_mapping(raw, "predicate")
        if not re.fullmatch(r"[A-Z]", name):
            raise FolModelParseError(f"Predicate '{name}' must be a single uppercase letter.")
        predicates[name] = _parse_predicate_value(name, value_text, domain)

    return FolModel(domain=domain, constants=constants, predicates=predicates)


def parse_sentences_text(text: str) -> List[Tuple[int, str]]:
    results: List[Tuple[int, str]] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        match = re.match(r"^(\d+)\s*[\.\):]\s*(.+)$", line)
        if match:
            results.append((int(match.group(1)), match.group(2).strip()))
        else:
            results.append((len(results) + 1, line))
    return results


def check_sentences(model_text: str, sentences_text: str, trace_mode: str = "full") -> FolCheckResult:
    model = parse_model_text(model_text)
    sentences = parse_sentences_text(sentences_text)
    if not sentences:
        raise FolModelParseError("Provide at least one target sentence.")

    sentence_results: List[SentenceCheckResult] = []
    all_parsed_ok = True
    for index, sentence in sentences:
        sentence_results.append(_evaluate_sentence(index, sentence, model, trace_mode=trace_mode))
        all_parsed_ok = all_parsed_ok and sentence_results[-1].parsed_ok

    return FolCheckResult(
        model_summary=_build_model_summary(model),
        sentence_results=sentence_results,
        all_parsed_ok=all_parsed_ok,
    )


def build_trace_blocks(result: FolCheckResult) -> List[Dict[str, str]]:
    blocks: List[Dict[str, str]] = []
    for item in result.sentence_results:
        label = "Parse error" if not item.parsed_ok else ("True" if item.result else "False")
        blocks.append(
            {
                "title": f"Sentence {item.index}: {item.sentence}",
                "result": label,
                "body": item.trace,
            }
        )
    return blocks


def build_draft_comment(result: FolCheckResult) -> str:
    lines = [
        "Question: FOL model",
        "Checker: FOL model checker v0",
        "",
        "Model summary:",
        result.model_summary,
        "",
        "Sentence results:",
    ]
    for item in result.sentence_results:
        if not item.parsed_ok:
            lines.append(f"- {item.index}. {item.sentence}: parse error ({item.error})")
        else:
            verdict = "true" if item.result else "false"
            lines.append(f"- {item.index}. {item.sentence}: {verdict}")
    lines.extend(
        [
            "",
            "Instructor note:",
            "No default scoring rule is configured yet. Edit this comment and the grade manually.",
        ]
    )
    return "\n".join(lines)


def check_fol_task(model_text: str, autograde: Dict[str, Any], trace_mode: str = "full") -> FolTaskCheckResult:
    task_kind = str(autograde.get("task_kind") or "truth_value").strip() or "truth_value"
    prompt_data = autograde.get("prompt_data") or {}
    if not isinstance(prompt_data, dict):
        raise FolModelParseError("autograde.prompt_data must be an object/map.")

    if task_kind in {"truth_value", "truth_in_given_model", "true_false_in_given_model", "make_true", "make_false"}:
        formulas = _require_formula_list(prompt_data.get("formulas"))
        base_result = check_sentences(model_text, _format_sentences(formulas), trace_mode=trace_mode)
        trace_blocks = build_trace_blocks(base_result)
        passed = _basic_task_passed(task_kind, base_result)
        summary = _basic_task_summary(task_kind, base_result)
        return FolTaskCheckResult(
            task_kind=task_kind,
            passed=passed,
            summary=summary,
            model_summary=base_result.model_summary,
            trace_blocks=trace_blocks,
            draft_comment=build_task_draft_comment(task_kind, summary, base_result),
            sentence_results=base_result.sentence_results,
        )

    if task_kind == "joint_consistency":
        formulas = _require_formula_list(prompt_data.get("formulas"))
        base_result = check_sentences(model_text, _format_sentences(formulas), trace_mode=trace_mode)
        passed = base_result.all_parsed_ok and all(bool(item.result) for item in base_result.sentence_results)
        summary = (
            "All target sentences are true in this model, so the set is jointly consistent."
            if passed
            else "Not all target sentences are true in this model, so this is not a successful consistency witness."
        )
        trace_blocks = build_trace_blocks(base_result)
        trace_blocks.append(
            {
                "title": "Task summary: joint consistency",
                "result": "Pass" if passed else "Fail",
                "body": summary,
            }
        )
        return FolTaskCheckResult(
            task_kind=task_kind,
            passed=passed,
            summary=summary,
            model_summary=base_result.model_summary,
            trace_blocks=trace_blocks,
            draft_comment=build_task_draft_comment(task_kind, summary, base_result),
            sentence_results=base_result.sentence_results,
        )

    if task_kind == "non_equivalence":
        formulas = _require_formula_list(prompt_data.get("formulas"))
        if len(formulas) != 2:
            raise FolModelParseError("non_equivalence requires exactly two formulas.")
        base_result = check_sentences(model_text, _format_sentences(formulas), trace_mode=trace_mode)
        left_result, right_result = base_result.sentence_results
        passed = (
            base_result.all_parsed_ok
            and left_result.result is not None
            and right_result.result is not None
            and left_result.result != right_result.result
        )
        summary = (
            "The two formulas differ in truth value in this model, so they are not logically equivalent."
            if passed
            else "The two formulas do not come apart in truth value here, so this is not yet a successful non-equivalence witness."
        )
        trace_blocks = build_trace_blocks(base_result)
        trace_blocks.append(
            {
                "title": "Task summary: non-equivalence",
                "result": "Pass" if passed else "Fail",
                "body": summary,
            }
        )
        return FolTaskCheckResult(
            task_kind=task_kind,
            passed=passed,
            summary=summary,
            model_summary=base_result.model_summary,
            trace_blocks=trace_blocks,
            draft_comment=build_task_draft_comment(task_kind, summary, base_result),
            sentence_results=base_result.sentence_results,
        )

    if task_kind == "non_validity":
        premises = _require_formula_list(prompt_data.get("premises"))
        conclusion = prompt_data.get("conclusion")
        if not isinstance(conclusion, str) or not conclusion.strip():
            raise FolModelParseError("non_validity requires prompt_data.conclusion.")
        base_result = check_sentences(
            model_text,
            _format_sentences([*premises, conclusion.strip()]),
            trace_mode=trace_mode,
        )
        premise_results = base_result.sentence_results[:-1]
        conclusion_result = base_result.sentence_results[-1]
        premises_true = base_result.all_parsed_ok and all(bool(item.result) for item in premise_results)
        conclusion_false = conclusion_result.parsed_ok and (conclusion_result.result is False)
        passed = premises_true and conclusion_false
        summary = (
            "All premises are true and the conclusion is false in this model, so it is a countermodel and the argument is not valid."
            if passed
            else "This model does not make all premises true and the conclusion false, so it is not yet a successful countermodel."
        )
        trace_blocks = build_trace_blocks(base_result)
        trace_blocks.append(
            {
                "title": "Task summary: non-validity",
                "result": "Pass" if passed else "Fail",
                "body": summary,
            }
        )
        return FolTaskCheckResult(
            task_kind=task_kind,
            passed=passed,
            summary=summary,
            model_summary=base_result.model_summary,
            trace_blocks=trace_blocks,
            draft_comment=build_task_draft_comment(task_kind, summary, base_result),
            sentence_results=base_result.sentence_results,
        )

    if task_kind == "non_logical_truth":
        formulas = _require_formula_list(prompt_data.get("formulas"))
        base_result = check_sentences(model_text, _format_sentences(formulas), trace_mode=trace_mode)
        passed = base_result.all_parsed_ok and all(bool(item.result) for item in base_result.sentence_results)
        summary = (
            "The target sentence or sentences are true in this model, so this gives a witness of non-logical truth."
            if passed
            else "At least one target sentence is not true in this model, so this is not yet a successful witness of non-logical truth."
        )
        trace_blocks = build_trace_blocks(base_result)
        trace_blocks.append(
            {
                "title": "Task summary: non-logical truth",
                "result": "Pass" if passed else "Fail",
                "body": summary,
            }
        )
        return FolTaskCheckResult(
            task_kind=task_kind,
            passed=passed,
            summary=summary,
            model_summary=base_result.model_summary,
            trace_blocks=trace_blocks,
            draft_comment=build_task_draft_comment(task_kind, summary, base_result),
            sentence_results=base_result.sentence_results,
        )

    if task_kind == "non_logical_falsehood":
        formulas = _require_formula_list(prompt_data.get("formulas"))
        base_result = check_sentences(model_text, _format_sentences(formulas), trace_mode=trace_mode)
        passed = base_result.all_parsed_ok and all(item.result is False for item in base_result.sentence_results)
        summary = (
            "The target sentence or sentences are false in this model, so this gives a witness of non-logical falsehood."
            if passed
            else "At least one target sentence is not false in this model, so this is not yet a successful witness of non-logical falsehood."
        )
        trace_blocks = build_trace_blocks(base_result)
        trace_blocks.append(
            {
                "title": "Task summary: non-logical falsehood",
                "result": "Pass" if passed else "Fail",
                "body": summary,
            }
        )
        return FolTaskCheckResult(
            task_kind=task_kind,
            passed=passed,
            summary=summary,
            model_summary=base_result.model_summary,
            trace_blocks=trace_blocks,
            draft_comment=build_task_draft_comment(task_kind, summary, base_result),
            sentence_results=base_result.sentence_results,
        )

    raise FolModelParseError(f"Unsupported fol_model task_kind: {task_kind}")


def _evaluate_sentence(index: int, sentence: str, model: FolModel, trace_mode: str = "full") -> SentenceCheckResult:
    original_setting = syntax.predicate_settings
    try:
        syntax.predicate_settings = True
        wff = syntax.parse_formula(sentence)
    finally:
        syntax.predicate_settings = original_setting

    if not wff.is_well_formed:
        return SentenceCheckResult(
            index=index,
            sentence=sentence,
            parsed_ok=False,
            result=None,
            trace=f"Could not parse sentence {index}: {sentence}\nError: {wff.err_msg}",
            trace_tree=TraceNode(
                kind="parse_error",
                display=f"Could not parse sentence {index}: {sentence}",
                result=None,
                intro=[f"Error: {wff.err_msg}"],
            ),
            error=wff.err_msg,
        )

    try:
        trace_tree = _eval_wff(wff, model, {})
        value = bool(trace_tree.result)
        return SentenceCheckResult(
            index=index,
            sentence=sentence,
            parsed_ok=True,
            result=value,
            trace=render_trace_tree(trace_tree, trace_mode=trace_mode),
            trace_tree=trace_tree,
        )
    except FolModelParseError as exc:
        return SentenceCheckResult(
            index=index,
            sentence=sentence,
            parsed_ok=False,
            result=None,
            trace=f"Could not evaluate sentence {index}: {sentence}\nError: {exc}",
            trace_tree=TraceNode(
                kind="evaluation_error",
                display=f"Could not evaluate sentence {index}: {sentence}",
                result=None,
                intro=[f"Error: {exc}"],
            ),
            error=str(exc),
        )


def _eval_wff(
    wff: syntax.Wff,
    model: FolModel,
    env: Dict[str, int],
) -> TraceNode:
    display = _display_formula(wff)
    node = TraceNode(kind="formula", display=display)
    node.evaluated_display = display

    if wff.wff_type == "identity":
        left_val = _resolve_term(wff.my_terms[0], model, env)
        right_val = _resolve_term(wff.my_terms[1], model, env)
        node.kind = "identity"
        node.intro.extend(_describe_term(wff.my_terms[0], left_val, env))
        node.intro.extend(_describe_term(wff.my_terms[1], right_val, env))
        node.result = left_val == right_val
        node.evaluated_display = f"{left_val}={right_val}"
        node.truth_display = _tv(bool(node.result))
        node.reductions.append(f"{left_val} {'=' if node.result else '≠'} {right_val}")
        return node

    if wff.wff_type == "atomic":
        args = [_resolve_term(term, model, env) for term in wff.my_terms]
        node.kind = "atomic"
        for term, value in zip(wff.my_terms, args):
            node.intro.extend(_describe_term(term, value, env))
        predicate = model.predicates.get(wff.my_letter)
        arity = len(args)
        if predicate is None:
            raise FolModelParseError(f"Predicate '{wff.my_letter}' is not defined in the model.")
        if predicate.arity is not None and predicate.arity != arity:
            raise FolModelParseError(
                f"Predicate '{wff.my_letter}' is used with arity {arity}, but the model gives arity {predicate.arity}."
            )
        tuple_value = tuple(args)
        pretty_tuple = _format_tuple(tuple_value)
        node.intro.append(f"{wff.my_letter} = {_format_predicate_set(predicate)}")
        node.intro.append(f"So this is {wff.my_letter}{pretty_tuple}")
        node.result = tuple_value in predicate.tuples
        node.evaluated_display = _format_atomic_display(wff.my_letter, tuple_value)
        node.truth_display = _tv(bool(node.result))
        node.reductions.append(
            f"{pretty_tuple} {'is' if node.result else 'is not'} in {wff.my_letter}"
        )
        return node

    if wff.main_op == "¬" and wff.right_side is not None:
        node.kind = "negation"
        child = _eval_wff(wff.right_side, model, env.copy())
        node.children.append(child)
        node.result = not bool(child.result)
        node.evaluated_display = f"¬({child.evaluated_display})"
        node.truth_display = f"¬{child.truth_display}"
        node.reductions.append(f"not { _tv(bool(child.result)) } = {_tv(node.result)}")
        return node

    if wff.main_op in {"∧", "∨", "→", "↔"} and wff.left_side is not None and wff.right_side is not None:
        node.kind = "binary"
        node.context["operator"] = wff.main_op
        left_node = _eval_wff(wff.left_side, model, env.copy())
        right_node = _eval_wff(wff.right_side, model, env.copy())
        node.children.extend([left_node, right_node])
        left_value = bool(left_node.result)
        right_value = bool(right_node.result)
        if wff.main_op == "∧":
            node.result = left_value and right_value
        elif wff.main_op == "∨":
            node.result = left_value or right_value
        elif wff.main_op == "→":
            node.result = (not left_value) or right_value
        else:
            node.result = left_value == right_value
        node.evaluated_display = f"({left_node.evaluated_display} {wff.main_op} {right_node.evaluated_display})"
        node.truth_display = f"({_tv(left_value)} {wff.main_op} {_tv(right_value)})"
        node.reductions.append(
            f"{_tv(left_value)} {wff.main_op} {_tv(right_value)} = {_tv(bool(node.result))}"
        )
        return node

    if wff.main_op in {"∀", "∃"} and wff.right_side is not None:
        var_name = wff.my_letter
        if not var_name:
            raise FolModelParseError("Quantifier missing bound variable.")
        node.kind = "quantifier"
        node.context["quantifier"] = wff.main_op
        node.context["variable"] = var_name
        node.truth_display = ""
        node.intro.append(f"Domain: {_format_brace_set(model.domain)}")
        if wff.main_op == "∀":
            for obj in model.domain:
                branch = TraceNode(
                    kind="quantifier_branch",
                    display=f"Try object {obj}",
                    context={var_name: str(obj)},
                )
                next_env = env.copy()
                next_env[var_name] = obj
                child = _eval_wff(wff.right_side, model, next_env)
                branch.children.append(child)
                branch.result = child.result
                branch.evaluated_display = child.evaluated_display
                branch.truth_display = child.truth_display
                node.children.append(branch)
                if not child.result:
                    node.result = False
                    node.truth_display = _tv(False)
                    node.reductions.append(f"counterexample: {var_name} = {obj}")
                    return node
            node.result = True
            node.truth_display = _tv(True)
            node.reductions.append("Every object in the domain satisfies the formula")
            return node

        for obj in model.domain:
            branch = TraceNode(
                kind="quantifier_branch",
                display=f"Try witness {var_name} = {obj}",
                context={var_name: str(obj)},
            )
            next_env = env.copy()
            next_env[var_name] = obj
            child = _eval_wff(wff.right_side, model, next_env)
            branch.children.append(child)
            branch.result = child.result
            branch.evaluated_display = child.evaluated_display
            branch.truth_display = child.truth_display
            node.children.append(branch)
            if child.result:
                node.result = True
                node.truth_display = _tv(True)
                node.reductions.append(f"witness found: {var_name} = {obj}")
                return node
        node.result = False
        node.truth_display = _tv(False)
        node.reductions.append("No witness in the domain makes the formula true")
        return node

    raise FolModelParseError(f"Unsupported formula shape: {syntax.wff_to_string(wff)}")


def _describe_term(term: str, value: int, env: Dict[str, int]) -> List[str]:
    if term in env:
        return [f"{term} is assigned {value}"]
    return [f"{term} names {value}"]


def _resolve_term(term: str, model: FolModel, env: Dict[str, int]) -> int:
    if term in env:
        return env[term]
    if term in model.constants:
        return model.constants[term]
    raise FolModelParseError(f"Term '{term}' has no assignment and is not a defined constant.")


def _build_model_summary(model: FolModel) -> str:
    parts = [
        f"DOMAIN: {_format_brace_set(model.domain)}",
        "CONSTANTS:",
    ]
    if model.constants:
        parts.extend([f"  {name}: {value}" for name, value in sorted(model.constants.items())])
    else:
        parts.append("  (none)")
    parts.append("PREDICATES:")
    if model.predicates:
        for name, interp in sorted(model.predicates.items()):
            parts.append(f"  {name}: {_format_predicate_set(interp)}")
    else:
        parts.append("  (none)")
    return "\n".join(parts)


def _parse_sections(text: str) -> Dict[str, List[str]]:
    sections: Dict[str, List[str]] = {}
    current_name: Optional[str] = None

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        records = _extract_labeled_records(line)
        if not records:
            if current_name is None:
                raise FolModelParseError(f"Line is outside any section: {raw_line}")
            sections.setdefault(current_name, []).append(line)
            continue

        for label, value in records:
            normalized_label = _normalize_section_label(label)
            if normalized_label in {"DOMAIN", "CONSTANTS", "PREDICATES", "SENTENCES"}:
                current_name = normalized_label
                sections.setdefault(current_name, [])
                if value:
                    sections[current_name].append(value)
                continue

            if re.fullmatch(r"[a-z]", label):
                target_section = "CONSTANTS"
            elif re.fullmatch(r"[A-Z]", label):
                target_section = "PREDICATES"
            elif current_name in {"CONSTANTS", "PREDICATES"}:
                target_section = current_name
            else:
                raise FolModelParseError(f"Unknown model label '{label}'.")

            current_name = target_section
            sections.setdefault(target_section, []).append(f"{label}: {value}")

    return sections


def _extract_labeled_records(line: str) -> List[Tuple[str, str]]:
    matches = list(
        re.finditer(
            r"(?<![A-Za-z0-9_])"
            r"(DOMAIN|UNIVERSE|CONSTANTS|PREDICATES|SENTENCES|[A-Za-z])"
            r"\s*:",
            line,
            flags=re.IGNORECASE,
        )
    )
    records: List[Tuple[str, str]] = []
    for idx, match in enumerate(matches):
        label = match.group(1).strip()
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(line)
        value = line[start:end].strip()
        value = re.sub(r"^[,;]\s*", "", value)
        value = re.sub(r"\s*[,;]\s*$", "", value)
        records.append((label, value))
    return records


def _normalize_section_label(label: str) -> str:
    upper = label.strip().upper()
    if upper == "UNIVERSE":
        return "DOMAIN"
    return upper


def _split_mapping(raw: str, kind: str) -> Tuple[str, str]:
    if ":" not in raw:
        raise FolModelParseError(f"Bad {kind} line '{raw}'. Expected '<symbol>: <value>'.")
    left, right = raw.split(":", 1)
    return left.strip(), right.strip()


def _is_empty_section_marker(raw: str) -> bool:
    return raw.strip().lower() in {"(none)", "none", "-", "—"}


def _parse_number_set(raw: str, label: str) -> List[int]:
    inner = _unwrap_optional_braces(raw)
    if not inner.strip():
        return []
    values = [_parse_int(part.strip(), label) for part in re.split(r"[,;]", inner) if part.strip()]
    if len(set(values)) != len(values):
        raise FolModelParseError(f"{label} contains duplicates.")
    return values


def _parse_predicate_value(name: str, raw: str, domain: List[int]) -> PredicateInterpretation:
    inner = _unwrap_optional_braces(raw)
    if not inner.strip():
        return PredicateInterpretation(name=name, arity=None, tuples=set())

    if "(" in inner or "<" in inner:
        tuples: Set[Tuple[int, ...]] = set()
        arity = None
        for pair_text in re.findall(r"\([^()]+\)|<[^<>]+>", inner):
            members = [item.strip() for item in pair_text[1:-1].split(",")]
            if arity is None:
                arity = len(members)
            if len(members) != arity:
                raise FolModelParseError(f"Predicate '{name}' has tuples with inconsistent arity.")
            pair = tuple(_parse_int(item, f"predicate {name}") for item in members)
            if any(value not in domain for value in pair):
                raise FolModelParseError(f"Predicate '{name}' contains {pair}, which is outside the domain.")
            tuples.add(pair)
        if not tuples and inner.strip():
            raise FolModelParseError(f"Could not parse predicate '{name}' as a set of ordered pairs.")
        return PredicateInterpretation(name=name, arity=arity, tuples=tuples)

    unary_values = _parse_number_set(raw, f"predicate {name}")
    tuples = {(value,) for value in unary_values}
    for value in unary_values:
        if value not in domain:
            raise FolModelParseError(f"Predicate '{name}' contains {value}, which is outside the domain.")
    return PredicateInterpretation(name=name, arity=1, tuples=tuples)


def _unwrap_braces(raw: str, label: str) -> str:
    text = raw.strip()
    if not (text.startswith("{") and text.endswith("}")):
        raise FolModelParseError(f"{label} must use set notation with braces.")
    return text[1:-1].strip()


def _unwrap_optional_braces(raw: str) -> str:
    text = raw.strip()
    if text.startswith("{") and text.endswith("}"):
        return text[1:-1].strip()
    return text


def _parse_int(raw: str, label: str) -> int:
    if not re.fullmatch(r"-?\d+", raw):
        raise FolModelParseError(f"{label} contains '{raw}', which is not an integer.")
    return int(raw)


def _format_brace_set(values: List[int]) -> str:
    return "{" + ",".join(str(value) for value in values) + "}"


def _format_tuple(value: Tuple[int, ...]) -> str:
    return "(" + ",".join(str(item) for item in value) + ")"


def _format_predicate_set(predicate: PredicateInterpretation) -> str:
    if not predicate.tuples:
        return "{}"
    ordered = sorted(predicate.tuples)
    if predicate.arity == 1 or (predicate.arity is None and all(len(item) == 1 for item in ordered)):
        return "{" + ",".join(str(item[0]) for item in ordered) + "}"
    return "{" + ",".join(_format_tuple(item) for item in ordered) + "}"


def _format_atomic_display(name: str, args: Tuple[int, ...]) -> str:
    if len(args) == 1:
        return f"{name}{args[0]}"
    return f"{name}{_format_tuple(args)}"


def render_trace_tree(node: TraceNode, trace_mode: str = "full") -> str:
    if trace_mode == "terse":
        return _render_trace_tree_terse(node)
    return _render_trace_tree_full(node)


def _render_trace_tree_full(node: TraceNode, indent: int = 0) -> str:
    prefix = "  " * indent
    heading = node.display
    if node.kind in {"formula", "binary", "negation", "quantifier", "identity", "atomic"}:
        heading = f"Evaluating {node.display}:"
    lines = [f"{prefix}{heading}"]
    for detail in node.intro:
        lines.append(f"{prefix}  {detail}")
    for child in node.children:
        lines.append(_render_trace_tree_full(child, indent + 1))
    for reduction in node.reductions:
        lines.append(f"{prefix}  {reduction}")
    if node.result is not None:
        lines.append(f"{prefix}Result: {'True' if node.result else 'False'}")
    return "\n".join(lines)


def _render_trace_tree_terse(node: TraceNode, indent: int = 0) -> str:
    prefix = "  " * indent
    if node.kind == "quantifier":
        lines = [f"{prefix}{node.display}"]
        for child in node.children:
            assignment = ", ".join(f"{var}={value}" for var, value in child.context.items())
            target = child.children[0] if child.children else child
            if _contains_quantifier(target):
                lines.append(f"{prefix}{assignment}")
                lines.append(_render_trace_tree_terse(target, indent + 1))
            else:
                lines.append(f"{prefix}{assignment}: {_collapse_quantifier_free(target)}")
        if node.reductions:
            lines.extend(f"{prefix}{reduction}" for reduction in node.reductions)
        if node.result is not None:
            lines.append(f"{prefix}Therefore {node.display} is {_tv(bool(node.result))}")
        return "\n".join(lines)
    return f"{prefix}{_collapse_quantifier_free(node)}"


def _contains_quantifier(node: TraceNode) -> bool:
    if node.kind == "quantifier":
        return True
    return any(_contains_quantifier(child) for child in node.children)


def _collapse_quantifier_free(node: TraceNode) -> str:
    if node.kind == "negation" and node.children:
        return f"{node.evaluated_display} = {node.truth_display} = {_tv(bool(node.result))}"
    if node.kind == "binary" and len(node.children) == 2:
        return f"{node.evaluated_display} = {node.truth_display} = {_tv(bool(node.result))}"
    if node.kind in {"atomic", "identity"}:
        return f"{node.evaluated_display} = {node.truth_display}"
    return f"{node.evaluated_display or node.display} = {_tv(bool(node.result))}"


def _tv(value: bool) -> str:
    return "TRUE" if value else "FALSE"


def _display_formula(wff: syntax.Wff) -> str:
    original_setting = syntax.predicate_settings
    try:
        syntax.predicate_settings = True
        return syntax.wff_to_string(wff)
    finally:
        syntax.predicate_settings = original_setting


def _require_formula_list(value: Any) -> List[str]:
    if not isinstance(value, list) or not value:
        raise FolModelParseError("Expected a non-empty list of formulas.")
    formulas: List[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise FolModelParseError("Every formula must be a non-empty string.")
        formulas.append(item.strip())
    return formulas


def _format_sentences(formulas: List[str]) -> str:
    return "\n".join(f"{idx}. {formula}" for idx, formula in enumerate(formulas, 1))


def _basic_task_passed(task_kind: str, result: FolCheckResult) -> bool:
    if not result.all_parsed_ok:
        return False
    if task_kind in {"make_false"}:
        return all(item.result is False for item in result.sentence_results)
    return all(item.result is True for item in result.sentence_results)


def _basic_task_summary(task_kind: str, result: FolCheckResult) -> str:
    passed = _basic_task_passed(task_kind, result)
    if task_kind in {"make_false"}:
        return (
            "All target sentences are false in this model."
            if passed
            else "At least one target sentence is not false in this model."
        )
    return (
        "All target sentences have the required truth value in this model."
        if passed
        else "At least one target sentence does not have the required truth value in this model."
    )


def build_task_draft_comment(task_kind: str, summary: str, result: FolCheckResult) -> str:
    lines = [
        "Question: FOL model",
        f"Task kind: {task_kind}",
        "Checker: FOL model checker v0",
        "",
        "Model summary:",
        result.model_summary,
        "",
        "Sentence results:",
    ]
    for item in result.sentence_results:
        if not item.parsed_ok:
            lines.append(f"- {item.index}. {item.sentence}: parse error ({item.error})")
        else:
            verdict = "true" if item.result else "false"
            lines.append(f"- {item.index}. {item.sentence}: {verdict}")
    lines.extend(["", "Task summary:", summary])
    return "\n".join(lines)
