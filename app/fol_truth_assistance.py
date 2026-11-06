"""Opt-in Quiz 9 semantic configuration and independent Python checks.

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


ASSESSMENT = "quiz9_monadic_fol_truth"
SCHEMA = 1
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


def location_prompt(problem_id: str) -> str:
    """Ask for the printed Quiz 9 answer area, including blank responses."""
    number = int(problem_id[1:])
    return (
        f"Locate printed question {number} on this Quiz 9 page. Identify the entire answer area: "
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
    return isinstance(config, dict) and config.get("schema") == SCHEMA and config.get("assessment") == ASSESSMENT


def assert_supported(grading_key: Any) -> None:
    config = grading_key.fol_truth_assistance
    if config is not None and not enabled(grading_key):
        raise ValueError(
            "This session requests unsupported FOL truth assistance. "
            "Use grader code with Quiz 9 FOL assistance schema 1, or remove the opt-in configuration."
        )


def model_text(model: dict) -> str:
    lines = ["DOMAIN: " + ",".join(map(str, model["domain"])), "CONSTANTS:"]
    lines += [f"{key}: {value}" for key, value in model["constants"].items()]
    lines.append("PREDICATES:")
    lines += [f"{key}: " + (",".join(map(str, values)) if values else "{}") for key, values in model["predicates"].items()]
    return "\n".join(lines)


def question_context(grading_key: Any, problem_id: str, version: str | None,
                     segment: dict | None, source_path: Path | None = None) -> dict | None:
    if not enabled(grading_key) or problem_id not in QUESTIONS or version not in MODELS:
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
            "asks_object": asks_object, "fingerprint": fingerprint}


def prepare_vision_reading(client: Any, image_path: Path, bbox: list, asks_object: bool) -> dict:
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
                "an answer from the printed sentence. Return only JSON with truth as true, false, blank, or "
                "unclear, and written as the student's exact short writing or an empty string."
            )
            if not asks_object:
                prompt += " This question has no written response field; set written to an empty string."
            raw = client.chat_with_image_prompt(path, prompt)
    cleaned = raw.strip() if isinstance(raw, str) else raw
    if isinstance(cleaned, str) and cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            cleaned = "\n".join(lines[1:-1]).strip()
    try:
        parsed = json.loads(cleaned)
    except (ValueError, TypeError):
        raise ValueError("Vision reading was not valid JSON; no proposal was saved.") from None
    if not isinstance(parsed, dict):
        raise ValueError("Vision reading was not a JSON object; no proposal was saved.")
    truth = str(parsed.get("truth") or "").strip().lower()
    if truth not in {"true", "false", "blank", "unclear"}:
        truth = "unclear"
    written = parsed.get("written")
    if not isinstance(written, str):
        written = ""
    written = written.strip()[:200] if asks_object else ""
    if not written:
        obj = "blank"
    elif re.fullmatch(r"[123]", written):
        obj = written
    elif re.fullmatch(r"n\s*/?\s*a", written, flags=re.I):
        obj = "na"
    else:
        obj = "unclear"
    return {"proposed_truth": truth, "proposed_object": obj if asks_object else None,
            "proposed_raw_object": written if asks_object else None}
