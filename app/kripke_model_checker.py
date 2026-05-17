from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple


class KripkeModelParseError(ValueError):
    pass


@dataclass
class KripkeModel:
    worlds: List[int]
    relation: Set[Tuple[int, int]]
    valuation: Dict[str, Set[int]]


@dataclass
class FormulaNode:
    kind: str
    value: str = ""
    left: Optional["FormulaNode"] = None
    right: Optional["FormulaNode"] = None
    child: Optional["FormulaNode"] = None


@dataclass
class ConstraintResult:
    ok: bool
    line: str
    trace: str = ""


@dataclass
class KripkeTaskCheckResult:
    task_kind: str
    passed: bool
    summary: str
    model_summary: str
    trace_blocks: List[Dict[str, str]]
    draft_comment: str
    constraint_results: List[ConstraintResult]


def parse_model_text(text: str) -> KripkeModel:
    worlds: Optional[List[int]] = None
    relation: Set[Tuple[int, int]] = set()
    valuation: Dict[str, Set[int]] = {}
    current = ""

    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue

        header = re.fullmatch(r"([A-Za-z]+)\s*:?\s*", line)
        if header and header.group(1).upper() in {"WORLDS", "ACCESSIBILITY", "RELATION", "VALUATION"}:
            current = header.group(1).upper()
            continue

        key_value = re.match(r"^([A-Za-zν][A-Za-z0-9_]*)\s*:\s*(.*)$", line)
        if key_value:
            raw_key = key_value.group(1)
            key = raw_key.upper()
            value = key_value.group(2).strip()
            if raw_key == "ν":
                for world, prop, truth_value in _parse_nu_items(value):
                    if truth_value:
                        valuation.setdefault(prop, set()).add(world)
                    else:
                        valuation.setdefault(prop, set())
                current = "VALUATION"
                continue
            if key in {"W", "WORLD", "WORLDS", "DOMAIN"}:
                worlds = _parse_int_list(value, "worlds")
                current = "WORLDS"
                continue
            if key in {"R", "RELATION", "ACCESSIBILITY"}:
                relation.update(_parse_relation_pairs(value))
                current = "ACCESSIBILITY"
                continue
            if re.fullmatch(r"[A-Z]", key_value.group(1)):
                valuation[key_value.group(1).upper()] = set(_parse_int_list(value, f"valuation {key_value.group(1)}"))
                current = "VALUATION"
                continue

        nu_items = _parse_nu_items(line)
        if nu_items:
            for world, prop, value in nu_items:
                if value:
                    valuation.setdefault(prop, set()).add(world)
                else:
                    valuation.setdefault(prop, set())
            continue

        if current == "WORLDS":
            worlds = _parse_int_list(line, "worlds")
        elif current in {"ACCESSIBILITY", "RELATION"}:
            source_map = re.match(r"^(\d+)\s*:\s*(.*)$", line)
            if source_map:
                source = int(source_map.group(1))
                for target in _parse_int_list(source_map.group(2), f"successors of {source}"):
                    relation.add((source, target))
            else:
                relation.update(_parse_relation_pairs(line))
        elif current == "VALUATION":
            prop_map = re.match(r"^([A-Z])\s*:\s*(.*)$", line)
            if not prop_map:
                raise KripkeModelParseError(f"Could not parse valuation line: {line}")
            valuation[prop_map.group(1)] = set(_parse_int_list(prop_map.group(2), f"valuation {prop_map.group(1)}"))
        else:
            raise KripkeModelParseError(f"Could not parse model line: {line}")

    if worlds is None or not worlds:
        raise KripkeModelParseError("Missing non-empty W/WORLDS block.")

    world_set = set(worlds)
    for source, target in relation:
        if source not in world_set or target not in world_set:
            raise KripkeModelParseError(f"Relation pair ({source},{target}) uses a world outside W.")
    for prop, true_worlds in valuation.items():
        extra = true_worlds - world_set
        if extra:
            raise KripkeModelParseError(f"Valuation for {prop} uses worlds outside W: {sorted(extra)}.")

    return KripkeModel(worlds=sorted(worlds), relation=relation, valuation=valuation)


def check_kripke_task(model_text: str, autograde: Dict[str, Any], trace_mode: str = "full") -> KripkeTaskCheckResult:
    model = parse_model_text(model_text)
    task_kind = str(autograde.get("task_kind") or "truth").strip() or "truth"
    prompt_data = autograde.get("prompt_data") or {}
    if not isinstance(prompt_data, dict):
        raise KripkeModelParseError("autograde.prompt_data must be an object/map.")

    constraints = _constraints_for_task(task_kind, prompt_data)
    results = [_evaluate_constraint(model, constraint, trace_mode=trace_mode) for constraint in constraints]
    passed = all(item.ok for item in results)
    summary = _task_summary(task_kind, passed)
    trace_blocks = [
        {
            "title": f"Constraint {idx}",
            "result": "Pass" if result.ok else "Fail",
            "body": result.line + (("\n\n" + result.trace) if result.trace else ""),
        }
        for idx, result in enumerate(results, start=1)
    ]
    trace_blocks.append(
        {
            "title": f"Task summary: {task_kind}",
            "result": "Pass" if passed else "Fail",
            "body": summary,
        }
    )

    return KripkeTaskCheckResult(
        task_kind=task_kind,
        passed=passed,
        summary=summary,
        model_summary=_build_model_summary(model),
        trace_blocks=trace_blocks,
        draft_comment=_build_draft_comment(task_kind, summary, model, results),
        constraint_results=results,
    )


def _constraints_for_task(task_kind: str, prompt_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw_constraints = prompt_data.get("constraints")
    if isinstance(raw_constraints, list) and raw_constraints:
        return [item for item in raw_constraints if isinstance(item, dict)]

    world = int(prompt_data.get("world", 0))
    if task_kind in {"truth", "truth_value", "make_true"}:
        return [{"type": "truth", "world": world, "formula": formula, "expected": True} for formula in _require_formula_list(prompt_data)]
    if task_kind == "make_false":
        return [{"type": "truth", "world": world, "formula": formula, "expected": False} for formula in _require_formula_list(prompt_data)]
    if task_kind in {"different", "non_equivalence"}:
        formulas = _require_formula_list(prompt_data)
        if len(formulas) != 2:
            raise KripkeModelParseError(f"{task_kind} requires exactly two formulas.")
        return [{"type": "different", "world": world, "formulaA": formulas[0], "formulaB": formulas[1]}]
    if task_kind in {"non_entailment", "non_validity"}:
        premises = prompt_data.get("premises")
        conclusion = prompt_data.get("conclusion")
        if not isinstance(premises, list) or not isinstance(conclusion, str) or not conclusion.strip():
            raise KripkeModelParseError(f"{task_kind} requires prompt_data.premises and prompt_data.conclusion.")
        return [
            *[{"type": "truth", "world": world, "formula": str(formula), "expected": True} for formula in premises],
            {"type": "truth", "world": world, "formula": conclusion, "expected": False},
        ]
    raise KripkeModelParseError(f"Unsupported kripke_model task_kind: {task_kind}")


def _evaluate_constraint(model: KripkeModel, constraint: Dict[str, Any], trace_mode: str = "full") -> ConstraintResult:
    kind = str(constraint.get("type") or "").strip()

    if kind == "truth":
        world = _constraint_world(constraint, model)
        formula = str(constraint.get("formula") or "").strip()
        expected = bool(constraint.get("expected"))
        value, trace = _truth_with_trace(model, world, _parse_formula(formula))
        return ConstraintResult(
            ok=value == expected,
            line=f"At w{world}: {formula} is {'T' if value else 'F'} (expected {'T' if expected else 'F'})",
            trace=trace if trace_mode != "terse" else "",
        )

    if kind == "different":
        world = _constraint_world(constraint, model)
        formula_a = str(constraint.get("formulaA") or constraint.get("formula_a") or "").strip()
        formula_b = str(constraint.get("formulaB") or constraint.get("formula_b") or "").strip()
        value_a, trace_a = _truth_with_trace(model, world, _parse_formula(formula_a))
        value_b, trace_b = _truth_with_trace(model, world, _parse_formula(formula_b))
        return ConstraintResult(
            ok=value_a != value_b,
            line=(
                f"At w{world}: {formula_a} = {'T' if value_a else 'F'}, "
                f"{formula_b} = {'T' if value_b else 'F'} (need them to differ)"
            ),
            trace=(trace_a + "\n\n" + trace_b) if trace_mode != "terse" else "",
        )

    if kind == "relation":
        prop = str(constraint.get("property") or "").strip().lower()
        expected = bool(constraint.get("expected"))
        value = _relation_property(model, prop)
        descriptor = prop.capitalize() if prop else "Unknown relation property"
        return ConstraintResult(
            ok=value == expected,
            line=f"Accessibility: {descriptor} {'holds' if value else 'fails'} (expected {'to hold' if expected else 'to fail'})",
        )

    raise KripkeModelParseError(f"Unsupported constraint type: {kind}")


def _constraint_world(constraint: Dict[str, Any], model: KripkeModel) -> int:
    world = int(constraint.get("world", 0))
    if world not in set(model.worlds):
        raise KripkeModelParseError(f"Constraint refers to world {world}, which is not in W.")
    return world


def _truth_with_trace(model: KripkeModel, world: int, formula: FormulaNode, indent: int = 0) -> Tuple[bool, str]:
    prefix = "  " * indent
    display = _formula_display(formula)
    if formula.kind == "prop":
        value = world in model.valuation.get(formula.value, set())
        return value, f"{prefix}At w{world}, {formula.value} is {'true' if value else 'false'} by the valuation."
    if formula.kind == "not":
        child_value, child_trace = _truth_with_trace(model, world, _require_child(formula), indent + 1)
        value = not child_value
        return value, f"{prefix}At w{world}, {display} is {'true' if value else 'false'} because its negand is {'false' if value else 'true'}.\n{child_trace}"
    if formula.kind in {"and", "or", "implies"}:
        left = _require_left(formula)
        right = _require_right(formula)
        left_value, left_trace = _truth_with_trace(model, world, left, indent + 1)
        right_value, right_trace = _truth_with_trace(model, world, right, indent + 1)
        if formula.kind == "and":
            value = left_value and right_value
            reason = "both conjuncts are true" if value else "at least one conjunct is false"
        elif formula.kind == "or":
            value = left_value or right_value
            reason = "at least one disjunct is true" if value else "both disjuncts are false"
        else:
            value = (not left_value) or right_value
            reason = "the antecedent is false or the consequent is true" if value else "the antecedent is true and the consequent is false"
        return value, f"{prefix}At w{world}, {display} is {'true' if value else 'false'} because {reason}.\n{left_trace}\n{right_trace}"
    if formula.kind == "box":
        successors = sorted(target for source, target in model.relation if source == world)
        child = _require_child(formula)
        child_results = [_truth_with_trace(model, target, child, indent + 1) for target in successors]
        value = all(result for result, _ in child_results)
        if not successors:
            reason = "there are no accessible worlds"
        else:
            reason = "the embedded formula is true at every accessible world" if value else "the embedded formula fails at an accessible world"
        traces = "\n".join(trace for _, trace in child_results)
        return value, f"{prefix}At w{world}, {display} is {'true' if value else 'false'} because {reason}." + (f"\n{traces}" if traces else "")
    if formula.kind == "diamond":
        successors = sorted(target for source, target in model.relation if source == world)
        child = _require_child(formula)
        child_results = [_truth_with_trace(model, target, child, indent + 1) for target in successors]
        value = any(result for result, _ in child_results)
        reason = "the embedded formula is true at some accessible world" if value else "no accessible world makes the embedded formula true"
        traces = "\n".join(trace for _, trace in child_results)
        return value, f"{prefix}At w{world}, {display} is {'true' if value else 'false'} because {reason}." + (f"\n{traces}" if traces else "")
    raise KripkeModelParseError(f"Unknown formula node kind: {formula.kind}")


class _FormulaParser:
    def __init__(self, text: str):
        self.tokens = _tokenize_formula(text)
        self.pos = 0

    def parse(self) -> FormulaNode:
        node = self._parse_implies()
        if self._peek() is not None:
            raise KripkeModelParseError(f"Unexpected token in formula: {self._peek()}")
        return node

    def _parse_implies(self) -> FormulaNode:
        left = self._parse_or()
        if self._accept("->"):
            right = self._parse_implies()
            return FormulaNode("implies", left=left, right=right)
        return left

    def _parse_or(self) -> FormulaNode:
        node = self._parse_and()
        while self._accept("|"):
            node = FormulaNode("or", left=node, right=self._parse_and())
        return node

    def _parse_and(self) -> FormulaNode:
        node = self._parse_unary()
        while self._accept("&"):
            node = FormulaNode("and", left=node, right=self._parse_unary())
        return node

    def _parse_unary(self) -> FormulaNode:
        if self._accept("~"):
            return FormulaNode("not", child=self._parse_unary())
        if self._accept("[]"):
            return FormulaNode("box", child=self._parse_unary())
        if self._accept("<>"):
            return FormulaNode("diamond", child=self._parse_unary())
        if self._accept("("):
            node = self._parse_implies()
            self._expect(")")
            return node
        token = self._peek()
        if token and re.fullmatch(r"[A-Z]", token):
            self.pos += 1
            return FormulaNode("prop", value=token)
        raise KripkeModelParseError(f"Expected formula at token: {token or 'end of input'}")

    def _peek(self) -> Optional[str]:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def _accept(self, token: str) -> bool:
        if self._peek() == token:
            self.pos += 1
            return True
        return False

    def _expect(self, token: str) -> None:
        if not self._accept(token):
            raise KripkeModelParseError(f"Expected {token} in formula.")


def _parse_formula(text: str) -> FormulaNode:
    if not text.strip():
        raise KripkeModelParseError("Missing modal formula.")
    return _FormulaParser(text).parse()


def _tokenize_formula(text: str) -> List[str]:
    normalized = (
        text.replace("□", "[]")
        .replace("◇", "<>")
        .replace("¬", "~")
        .replace("∧", "&")
        .replace("∨", "|")
        .replace("v", "|")
        .replace("→", "->")
    )
    tokens: List[str] = []
    index = 0
    while index < len(normalized):
        ch = normalized[index]
        if ch.isspace():
            index += 1
            continue
        two = normalized[index : index + 2]
        if two in {"[]", "<>", "->"}:
            tokens.append(two)
            index += 2
            continue
        if ch in {"~", "&", "|", "(", ")"}:
            tokens.append(ch)
            index += 1
            continue
        if re.fullmatch(r"[A-Z]", ch):
            tokens.append(ch)
            index += 1
            continue
        raise KripkeModelParseError(f"Unexpected character in formula: {ch}")
    return tokens


def _relation_property(model: KripkeModel, prop: str) -> bool:
    worlds = set(model.worlds)
    if prop == "reflexive":
        return all((world, world) in model.relation for world in worlds)
    if prop == "symmetric":
        return all((target, source) in model.relation for source, target in model.relation)
    if prop == "transitive":
        return all(
            (source, end) in model.relation
            for source, mid in model.relation
            for mid2, end in model.relation
            if mid == mid2
        )
    raise KripkeModelParseError(f"Unknown relation property: {prop}")


def _build_model_summary(model: KripkeModel) -> str:
    worlds = ", ".join(f"w{world}" for world in model.worlds)
    pairs = ", ".join(f"<w{source},w{target}>" for source, target in sorted(model.relation)) or "-"
    props = sorted(model.valuation)
    valuation_lines = []
    for prop in props:
        true_worlds = ", ".join(f"w{world}" for world in sorted(model.valuation[prop])) or "-"
        valuation_lines.append(f"{prop}: {true_worlds}")
    return "\n".join(["W: " + worlds, "R: " + pairs, "Valuation: " + ("; ".join(valuation_lines) if valuation_lines else "-")])


def _build_draft_comment(task_kind: str, summary: str, model: KripkeModel, results: List[ConstraintResult]) -> str:
    lines = [
        "Question: Kripke model",
        "Checker: Kripke model checker v0",
        "",
        "Model summary:",
        _build_model_summary(model),
        "",
        "Constraint results:",
    ]
    lines.extend(f"- {result.line}" for result in results)
    lines.extend(["", "Task verdict:", summary])
    return "\n".join(lines)


def _task_summary(task_kind: str, passed: bool) -> str:
    if task_kind in {"non_entailment", "non_validity"}:
        return (
            "All premises are true and the conclusion is false at the target world, so this is a countermodel."
            if passed
            else "This model does not make all premises true and the conclusion false at the target world."
        )
    if task_kind in {"different", "non_equivalence"}:
        return (
            "The target formulas differ in truth value at the target world."
            if passed
            else "The target formulas do not differ in truth value at the target world."
        )
    return "The model meets all Kripke constraints." if passed else "The model does not meet all Kripke constraints."


def _require_formula_list(prompt_data: Dict[str, Any]) -> List[str]:
    formulas = prompt_data.get("formulas")
    if not isinstance(formulas, list) or not formulas:
        raise KripkeModelParseError("prompt_data.formulas must be a non-empty list.")
    return [str(item).strip() for item in formulas if str(item).strip()]


def _parse_int_list(text: str, label: str) -> List[int]:
    cleaned = text.strip().strip("{}[]()")
    if not cleaned or cleaned in {"-", "—"}:
        return []
    values: List[int] = []
    for part in re.split(r"[,;\s]+", cleaned):
        if not part:
            continue
        normalized = part.lower()
        if normalized.startswith("w"):
            normalized = normalized[1:]
        if not re.fullmatch(r"\d+", normalized):
            raise KripkeModelParseError(f"Could not parse {label} value: {part}")
        values.append(int(normalized))
    return sorted(dict.fromkeys(values))


def _parse_relation_pairs(text: str) -> Set[Tuple[int, int]]:
    cleaned = text.strip()
    if not cleaned or cleaned in {"-", "—", "{}"}:
        return set()
    pairs: Set[Tuple[int, int]] = set()
    for left, right in re.findall(r"[<⟨(]\s*w?(\d+)\s*,\s*w?(\d+)\s*[>⟩)]", cleaned, flags=re.IGNORECASE):
        pairs.add((int(left), int(right)))
    if not pairs and re.search(r"\d", cleaned):
        raise KripkeModelParseError(f"Could not parse relation pairs: {text}")
    return pairs


def _parse_nu_items(line: str) -> List[Tuple[int, str, bool]]:
    items: List[Tuple[int, str, bool]] = []
    for world, prop, value in re.findall(r"ν\s*(\d+)\s*\(\s*([A-Z])\s*\)\s*=\s*([TF])", line):
        items.append((int(world), prop, value == "T"))
    return items


def _formula_display(node: FormulaNode) -> str:
    if node.kind == "prop":
        return node.value
    if node.kind == "not":
        return "~" + _formula_display(_require_child(node))
    if node.kind == "box":
        return "[]" + _formula_display(_require_child(node))
    if node.kind == "diamond":
        return "<>" + _formula_display(_require_child(node))
    if node.kind in {"and", "or", "implies"}:
        symbol = {"and": "&", "or": "v", "implies": "->"}[node.kind]
        return f"({_formula_display(_require_left(node))} {symbol} {_formula_display(_require_right(node))})"
    return node.kind


def _require_child(node: FormulaNode) -> FormulaNode:
    if node.child is None:
        raise KripkeModelParseError("Internal formula error: missing child.")
    return node.child


def _require_left(node: FormulaNode) -> FormulaNode:
    if node.left is None:
        raise KripkeModelParseError("Internal formula error: missing left child.")
    return node.left


def _require_right(node: FormulaNode) -> FormulaNode:
    if node.right is None:
        raise KripkeModelParseError("Internal formula error: missing right child.")
    return node.right
