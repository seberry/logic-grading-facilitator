"""Rubric-driven truth-in-a-model assistance, with legacy Quiz 9 support.

The browser's approved explanation text comes exclusively from the vendored
JavaScript module. This module supplies the printed quiz facts and checks them
with the existing Python model checker before offering assistance.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import tempfile
from pathlib import Path
from typing import Any

from PIL import Image

from app.fol_model_checker import check_sentences
from app.fol_truth_config import checker_model, normalize_formula, object_token, validate_model
from app.quiz_variants import available_quiz_versions, problem_variant


ASSESSMENT = "quiz9_monadic_fol_truth"
SCHEMA = 1
HALF_FOR_TRUTH = "half_for_truth"
ANSWER_RESPONSE_FORMAT = {
    'type': 'json_schema', 'json_schema': {'name': 'fol_answer_reading', 'strict': True,
        'schema': {'type': 'object', 'properties': {
            'truth': {'type': 'string', 'enum': ['true', 'false', 'blank', 'unclear']},
            'written': {'type': 'string'}}, 'required': ['truth', 'written'], 'additionalProperties': False}}}


class VisionReadingFormatError(ValueError):
    """A malformed transcription reply, not a grade or a transport failure."""

MODELS = {
    "A": {"domain": [1, 2, 3], "constants": {"c": 2}, "predicates": {"P": [1, 2], "Q": [1, 3], "R": [1]}},
    "B": {"domain": [1, 2, 3], "constants": {"c": 3}, "predicates": {"P": [1, 2, 3], "Q": [], "R": [1, 2]}},
    "C": {"domain": [1, 2, 3], "constants": {"c": 1}, "predicates": {"P": [2], "Q": [3], "R": [1, 2]}},
}
# The Python checker uses compact predicates and fully grouped binary formulas.
# Each spelling below was checked against the printed PDF and the upstream JS.
QUESTIONS = {
    "Q1": ("¬P(c) ∨ Q(c)", "(¬Pc ∨ Qc)", False),
    "Q2": ("∃x (P(x) ∧ Q(x) ∧ R(x))", "∃x((Px ∧ Qx) ∧ Rx)", True),
    "Q3": ("∃x (P(x) ∧ ¬Q(x))", "∃x(Px ∧ ¬Qx)", True),
    "Q4": ("∀x (P(x) → (Q(x) ∨ R(x)))", "∀x(Px → (Qx ∨ Rx))", True),
    "Q5": ("∀x (P(x) ∨ Q(x))", "∀x(Px ∨ Qx)", True),
    "Q6": ("(∃x P(x)) ∧ (∃x Q(x))", "(∃xPx ∧ ∃xQx)", False),
    "Q7": ("(∀x P(x)) → (∃x Q(x))", "(∀xPx → ∃xQx)", False),
}


def location_prompt(problem_id: str, description: str = "") -> str:
    """Locate an assisted question, including its unanswered response area."""
    return (
        f"Locate question {problem_id}: {description}. Identify the entire answer area: "
        "the printed number and sentence, both True and False choices, any circle or correction, "
        "and the witness/counterexample line when printed. A blank student answer still counts "
        "as a PRESENT printed question; include its blank answer area. Return present:false only "
        "when the printed question itself is absent from this page. Return ONLY one JSON object "
        "with target_problem_id, present, bbox, comment, and confidence. bbox is normalized "
        "[x_min,y_min,x_max,y_max], each value from 0 to 1; for example [0.1,0.2,0.8,0.3]. "
        "Do not answer the logic problem or transcribe any student name."
    )


def enabled(grading_key: Any) -> bool:
    config = getattr(grading_key, "fol_truth_assistance", None)
    legacy = isinstance(config, dict) and config.get("schema") == SCHEMA and config.get("assessment") == ASSESSMENT
    return legacy or any(question_enabled(grading_key, p.id, legacy=False)
                         for p in getattr(grading_key, "problems", []))


def question_enabled(grading_key: Any, problem_id: str, *, legacy: bool = True) -> bool:
    spec = next((p for p in getattr(grading_key, "problems", []) if p.id == problem_id), None)
    if spec is not None and (spec.fol_truth is not None or
                            any(v.fol_truth is not None for v in spec.variants.values())):
        return True
    config = getattr(grading_key, "fol_truth_assistance", None)
    return (legacy and isinstance(config, dict) and config.get("schema") == SCHEMA
            and config.get("assessment") == ASSESSMENT and problem_id in QUESTIONS)


def assistance_schema(grading_key: Any) -> int:
    versions = available_quiz_versions(grading_key) or [None]
    schema = 1
    for spec in grading_key.problems:
        for version in versions:
            config = _rubric_config(spec, version)
            if config is None:
                continue
            schema = 2
            if config.get("manual_evidence") or (config.get("scoring") or {}).get("mode") == HALF_FOR_TRUTH:
                return 3
    return schema


def assert_supported(grading_key: Any) -> None:
    config = getattr(grading_key, "fol_truth_assistance", None)
    if config is not None and not (isinstance(config, dict) and config.get("schema") == SCHEMA
                                   and config.get("assessment") == ASSESSMENT):
        raise ValueError(
            "This session requests unsupported FOL truth assistance. "
            "Use grader code with Quiz 9 FOL assistance schema 1, or remove the opt-in configuration."
        )
    versions = available_quiz_versions(grading_key) or [None]
    for spec in grading_key.problems:
        if not question_enabled(grading_key, spec.id, legacy=False):
            continue
        for version in versions:
            question_context(grading_key, spec.id, version, None)


def model_text(model: dict) -> str:
    lines = ["DOMAIN: " + ",".join(map(str, model["domain"])), "CONSTANTS:"]
    lines += [f"{key}: {value}" for key, value in model["constants"].items()]
    lines.append("PREDICATES:")
    for key, values in model["predicates"].items():
        members = ["(" + ",".join(map(str, value)) + ")" if isinstance(value, list) else str(value)
                   for value in values]
        lines.append(f"{key}: " + ("{" + ",".join(members) + "}" if members else "{}"))
    return "\n".join(lines)


def model_display(model: dict) -> str:
    def label(value):
        return "(" + ", ".join(map(str, value)) + ")" if isinstance(value, list) else str(value)
    parts = ["D = {" + ", ".join(map(str, model["domain"])) + "}"]
    parts += [f"{key} = {value}" for key, value in model["constants"].items()]
    parts += [f"{key} = {{" + ", ".join(label(x) for x in values) + "}"
              for key, values in model["predicates"].items()]
    return "; ".join(parts)


def _rubric_config(spec, version):
    override = problem_variant(spec, version)
    variant_config = override.fol_truth if override else None
    if spec.fol_truth is None and variant_config is None:
        return None
    return {**(spec.fol_truth or {}), **(variant_config or {})}


def question_context(grading_key: Any, problem_id: str, version: str | None,
                     segment: dict | None, source_path: Path | None = None) -> dict | None:
    spec = next((p for p in grading_key.problems if p.id == problem_id), None)
    if spec is not None and question_enabled(grading_key, problem_id, legacy=False):
        config = _rubric_config(spec, version)
        if config is None:
            return None
        if set(config) - {"model", "formula", "asks_object", "scoring", "manual_evidence"}:
            raise ValueError(f"Unknown FOL truth configuration field for {problem_id}.")
        model = validate_model(config.get("model"))
        formula = normalize_formula(config.get("formula"), len(model["domain"]))
        asks_object = config.get("asks_object", False)
        if not isinstance(asks_object, bool):
            raise ValueError(f"asks_object must be true or false for {problem_id}.")
        if asks_object and not formula.startswith(("∀", "∃")):
            raise ValueError(f"{problem_id} requests an object but its outermost operator is not a quantifier.")
        rule = config.get("scoring")
        manual_evidence = config.get("manual_evidence", False)
        if not isinstance(manual_evidence, bool) or (manual_evidence and (asks_object or rule is not None)):
            raise ValueError(f"{problem_id} with manual evidence review can suggest truth-value feedback only.")
        _validate_scoring(rule, spec.points, asks_object, problem_id)
        result = check_sentences(model_text(checker_model(model)), formula).sentence_results[0]
        if not result.parsed_ok or result.result is None:
            raise ValueError(f"Invalid FOL sentence for {problem_id} {version or ''}: {result.error}")
        evidence = {}
        if manual_evidence:
            evidence["manual_evidence"] = True
        if rule and rule.get("mode") == HALF_FOR_TRUTH:
            evidence["asks_object_when_applicable"] = asks_object
            asks_object = asks_object and formula.startswith("∃" if result.result else "∀")
        revision = None
        if source_path is not None and source_path.is_file():
            stat = source_path.stat()
            revision = [stat.st_size, stat.st_mtime_ns]
        identity = {"schema": 2, "config": config, "points": spec.points, "version": version,
                    "problem": problem_id, "segment": segment, "source_revision": revision}
        fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        return {"formula": formula, "model": model, "version": version, "truth": result.result,
                "source_filename": segment.get("page_filename") if segment else None,
                "score_rule": rule, "asks_object": asks_object, "fingerprint": fingerprint,
                "object_choices": [{"value": object_token(x), "label": str(x)} for x in model["domain"]],
                "model_display": model_display(model), **evidence}
    if not enabled(grading_key) or problem_id not in QUESTIONS or version not in MODELS:
        return None
    # A mixed assessment must not acquire Quiz 9 facts merely because it has Q1.
    legacy = getattr(grading_key, "fol_truth_assistance", None)
    if not isinstance(legacy, dict) or legacy.get("assessment") != ASSESSMENT:
        return None
    formula, python_formula, asks_object = QUESTIONS[problem_id]
    model = MODELS[version]
    scoring = grading_key.fol_truth_assistance.get("scoring") or {}
    score_rule = scoring.get(problem_id) if isinstance(scoring, dict) else None
    if score_rule is not None:
        spec = next((p for p in grading_key.problems if p.id == problem_id), None)
        if not isinstance(score_rule, dict) or spec is None:
            raise ValueError(f"Invalid Quiz 9 scoring rule for {problem_id}.")
        truth_points, object_points = score_rule.get("truth"), score_rule.get("object", 0)
        if (not isinstance(truth_points, (int, float)) or isinstance(truth_points, bool) or
            not isinstance(object_points, (int, float)) or isinstance(object_points, bool) or
            not math.isfinite(truth_points) or not math.isfinite(object_points) or
            truth_points < 0 or object_points < 0 or
            abs(truth_points + object_points - spec.points) > 1e-9 or
            bool(object_points) != asks_object or
            ("blank_na_credit" in score_rule and not isinstance(score_rule["blank_na_credit"], bool))):
            raise ValueError(f"Invalid Quiz 9 scoring rule for {problem_id}.")
    result = check_sentences(model_text(model), python_formula).sentence_results[0]
    if not result.parsed_ok or result.result is None:
        raise ValueError(f"Quiz 9 configuration cannot be checked for {version} {problem_id}: {result.error}")
    # A crop/page correction, version correction, or configuration change makes
    # an old proposal stale. Final human comments remain ordinary grade records.
    source_revision = None
    if source_path is not None and source_path.is_file():
        stat = source_path.stat()
        source_revision = [stat.st_size, stat.st_mtime_ns]
    identity = {"schema": SCHEMA, "assessment": ASSESSMENT, "config": grading_key.fol_truth_assistance,
                "version": version, "problem": problem_id, "segment": segment,
                "model": model, "formula": formula, "python_formula": python_formula,
                "source_revision": source_revision}
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    return {"formula": formula, "model": model, "version": version,
            "source_filename": segment.get("page_filename") if segment else None,
            "truth": result.result, "score_rule": score_rule,
            "asks_object": asks_object, "fingerprint": fingerprint,
            "object_choices": [{"value": object_token(x), "label": str(x)} for x in model["domain"]],
            "model_display": model_display(model)}


def _validate_scoring(rule: Any, points: float, asks_object: bool, problem_id: str) -> None:
    if not math.isfinite(points) or points < 0:
        raise ValueError(f"Invalid points for assisted question {problem_id}.")
    # Omitting scoring gives feedback assistance with manual score entry.
    if rule is None:
        return
    if not isinstance(rule, dict):
        raise ValueError(f"Invalid FOL scoring rule for {problem_id}.")
    mode = rule.get("mode", "components")
    if not isinstance(mode, str):
        raise ValueError(f"Invalid FOL scoring mode for {problem_id}.")
    fields = {"mode", "truth", "object", "blank_na_credit"} if mode == "components" else {"mode", "points", "blank_na_credit"}
    if mode == HALF_FOR_TRUTH:
        fields = {"mode", "points"}
    values = [rule.get("truth"), rule.get("object", 0)] if mode == "components" else [rule.get("points")]
    if (mode not in {"components", "all_or_nothing", HALF_FOR_TRUTH} or set(rule) - fields or
            any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in values) or
            abs(sum(values) - points) > 1e-9 or
            (mode == "components" and not asks_object and rule.get("object", 0) != 0) or
            ("blank_na_credit" in rule and not isinstance(rule["blank_na_credit"], bool))):
        raise ValueError(f"Invalid FOL scoring rule for {problem_id}; its maximum must match the question points.")


def prepare_vision_reading(client: Any, image_path: Path, bbox: list, asks_object: bool,
                          domain: list | None = None) -> dict:
    """Read a single saved crop, yielding only a proposal for TF confirmation."""
    if len(bbox) != 4 or any(not isinstance(v, (int, float)) or not 0 <= v <= 1 for v in bbox):
        raise ValueError("Saved question crop is invalid.")
    with Image.open(image_path) as source:
        width, height = source.size
        x0, y0, x1, y1 = bbox
        if x0 >= x1 or y0 >= y1:
            raise ValueError("Saved question crop is invalid.")
        margin_x = .04 * width
        margin_y = .02 * height
        box = (max(0, int(x0 * width - margin_x)), max(0, int(y0 * height - margin_y)),
               min(width, int(x1 * width + margin_x)), min(height, int(y1 * height + margin_y)))
        crop = source.crop(box).convert("RGB")
        with tempfile.TemporaryDirectory(prefix="fol_truth_read_") as temp_dir:
            path = Path(temp_dir) / "question.jpg"
            crop.save(path, quality=90)
            prompt = (
                "Read the student's marks in this one quiz question. The printed choices are True and False; "
                "a student may circle a word, circle a small bubble, cross out a mark, or leave it blank. "
                "Transcribe the final unambiguous choice only. If both choices appear marked, a correction is "
                "uncertain, or the choice is hard to see, use unclear. Do not solve the logic problem or infer "
                'an answer from the printed sentence. Return exactly one JSON object with two fields: '
                '"truth" is one of the strings "true", "false", "blank", "unclear"; '
                '"written" is the student\'s exact short writing or an empty string. '
                'For example: {"truth":"false","written":""}. No prose or Markdown.'
            )
            if not asks_object:
                prompt += " This question has no written response field; set written to an empty string."
            structured = getattr(client, 'chat_with_image_json', None)
            raw = structured(path, prompt, ANSWER_RESPONSE_FORMAT, max_tokens=512) if callable(structured) else client.chat_with_image_prompt(path, prompt)
    cleaned = raw.strip().lstrip('\ufeff') if isinstance(raw, str) else raw
    if isinstance(cleaned, str) and cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            cleaned = "\n".join(lines[1:-1]).strip()
    try:
        parsed = json.loads(cleaned)
    except (ValueError, TypeError):
        raise VisionReadingFormatError("Vision reading was not valid JSON; no proposal was saved.") from None
    if not isinstance(parsed, dict):
        raise VisionReadingFormatError("Vision reading was not a JSON object; no proposal was saved.")
    if set(parsed) != {'truth', 'written'} or not isinstance(parsed['written'], str):
        raise VisionReadingFormatError("Vision reading had invalid answer fields; no proposal was saved.")
    # JSON false is a legitimate reading; never discard it as a falsy value.
    truth = str(parsed['truth']).strip().lower()
    if truth not in {"true", "false", "blank", "unclear"}:
        truth = "unclear"
    written = parsed.get("written")
    if not isinstance(written, str):
        written = ""
    written = written.strip()[:200] if asks_object else ""
    if not written:
        obj = "blank"
    elif domain is not None and written in [str(x) for x in domain]:
        matches = [x for x in domain if str(x) == written]
        obj = object_token(matches[0]) if len(matches) == 1 else "unclear"
    elif re.fullmatch(r"-?(?:0|[1-9]\d*)", written) and abs(int(written)) <= 2**53 - 1:
        obj = written
    elif re.fullmatch(r"n\s*/?\s*a", written, flags=re.I):
        obj = "na"
    else:
        obj = "unclear"
    return {"proposed_truth": truth, "proposed_object": obj if asks_object else None,
            "proposed_raw_object": written if asks_object else None}
