"""Focused unary-verifier probe harness for isolated modal line crops."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import cv2

from modal_ocr.verifier import (
    UNARY_SENSITIVE_RULE_HEADS,
    build_forced_choice_unary_verifier_prompt,
    build_structured_unary_verifier_prompt,
    leading_unary_prefix,
)
from proofs_ocr.domains import modal
from proofs_ocr.pipelines.modal_v2_runtime import build_vlm_client


def normalize_probe_formula_text(value: Any, scope_role: str = "derived_line") -> str:
    """Normalize manual probe formulas, including ASCII operator fallbacks."""

    if str(scope_role or "").strip() == "strict_open":
        return "□"

    normalized = str(value or "").strip()
    replacements = (
        ("<->", "↔"),
        ("->", "→"),
        ("=>", "→"),
        ("[]", "□"),
        ("<>", "⋄"),
        ("◇", "⋄"),
        ("◊", "⋄"),
        ("~", "¬"),
        ("&", "∧"),
        ("^", "∧"),
    )
    for source, target in replacements:
        normalized = normalized.replace(source, target)
    return modal.normalize_formula_text(normalized)


def normalize_probe_justification_text(value: Any) -> str:
    """Normalize justifications for unary-verifier prompts."""

    normalized = str(value or "").strip().replace(",", " ")
    normalized = normalized.replace("–", "-").replace("—", "-")
    normalized = " ".join(normalized.split())
    return normalized


def unary_formula_skeleton(formula_text: str) -> str:
    """Return a skeleton that ignores unary-prefix multiplicity but keeps the body."""

    normalized = normalize_probe_formula_text(formula_text)
    prefix = leading_unary_prefix(normalized)
    body = normalized[len(prefix) :].strip()
    modal_class = "□" if "□" in prefix else "⋄" if "⋄" in prefix else "atom"
    return f"{modal_class}:{body}"


def build_candidate_prefixes(*, formula_text: str, justification_text: str) -> list[str]:
    """Generate nearby unary prefixes to probe with forced-choice verification."""

    prefix = leading_unary_prefix(formula_text)
    if not prefix:
        return []

    rule_head = justification_text.split(" ", 1)[0] if justification_text else ""
    contains_box = "□" in prefix
    contains_diamond = "⋄" in prefix
    negation_count = prefix.count("¬")

    candidates: list[str] = [prefix]
    if contains_diamond:
        pool = ("⋄", "⋄¬", "⋄¬¬", "¬⋄", "¬⋄¬", "¬⋄¬¬")
    elif contains_box or negation_count >= 2 or rule_head in UNARY_SENSITIVE_RULE_HEADS:
        pool = ("□", "□¬", "□¬¬", "¬□", "¬□¬", "¬□¬¬")
    else:
        pool = ("¬", "¬¬", "¬¬¬")

    for candidate in pool:
        if candidate not in candidates:
            candidates.append(candidate)
    if not contains_box and not contains_diamond and prefix.endswith("¬") and prefix.count("¬") < 4:
        extended = f"{prefix}¬"
        if extended not in candidates:
            candidates.append(extended)
    return candidates


def build_unary_formula_candidates(
    *,
    formula_text: str,
    justification_text: str = "",
) -> list[str]:
    """Generate nearby full-formula candidates for forced-choice verification."""

    normalized_formula = normalize_probe_formula_text(formula_text)
    normalized_justification = normalize_probe_justification_text(justification_text)
    prefix = leading_unary_prefix(normalized_formula)
    if not prefix:
        return [normalized_formula] if normalized_formula else []

    body = normalized_formula[len(prefix) :].strip()
    if not body:
        return [normalized_formula]

    skeleton = unary_formula_skeleton(normalized_formula)
    candidates: list[str] = []
    for candidate_prefix in build_candidate_prefixes(
        formula_text=normalized_formula,
        justification_text=normalized_justification,
    ):
        candidate = f"{candidate_prefix}{body}"
        if unary_formula_skeleton(candidate) != skeleton:
            continue
        if candidate not in candidates:
            candidates.append(candidate)
    return candidates or [normalized_formula]


def parse_probe_response(raw_response: str) -> dict[str, Any]:
    """Parse one JSON object returned by the verifier model."""

    parsed = modal.parse_json_object_response(raw_response)
    return dict(parsed)


def _upscale_for_probe(crop_img, *, scale: int = 4):
    if crop_img.size == 0:
        return crop_img
    return cv2.resize(crop_img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)


def int_field(value: Any, default: int = 0, minimum: int = 0, maximum: int = 3) -> int:
    """Clamp an integer-like verifier field into the expected unary-count range."""

    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, parsed))


def formula_components(formula_text: str) -> tuple[str, str]:
    """Split a formula into its unary prefix and remaining body."""

    prefix = leading_unary_prefix(normalize_probe_formula_text(formula_text))
    body = normalize_probe_formula_text(formula_text)[len(prefix) :].strip()
    return prefix, body


def derive_formula_from_structured_counts(
    *,
    formula_text: str,
    structured_parsed: dict[str, Any],
    scope_role: str = "derived_line",
) -> str | None:
    """Reconstruct a unary formula directly from the structured negation counts."""

    normalized_formula = normalize_probe_formula_text(formula_text, scope_role)
    _, body = formula_components(normalized_formula)
    if not body:
        return None

    modal_value = str(structured_parsed.get("main_modal_operator") or "").strip().lower()
    if modal_value == "box":
        modal_symbol = "□"
    elif modal_value == "diamond":
        modal_symbol = "⋄"
    else:
        modal_symbol = ""

    before_modal = int_field(structured_parsed.get("negations_before_modal"), 0, 0, 3)
    before_atom = int_field(structured_parsed.get("negations_before_final_atom"), 0, 0, 3)

    if not modal_symbol:
        return f"{'¬' * before_atom}{body}"
    return f"{'¬' * before_modal}{modal_symbol}{'¬' * before_atom}{body}"


def run_unary_probe(
    *,
    image_path: Path,
    formula_text: str,
    justification_text: str = "",
    scope_role: str = "derived_line",
    output_dir: Path,
    verifier_base_url: str | None = None,
    verifier_model: str | None = None,
    candidates: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Run the old unary verifier prompts on one isolated crop."""

    output_dir.mkdir(parents=True, exist_ok=True)
    crop_img = cv2.imread(str(image_path))
    if crop_img is None:
        raise FileNotFoundError(f"Could not read verifier crop at {image_path}")

    normalized_formula = normalize_probe_formula_text(formula_text, scope_role)
    normalized_justification = normalize_probe_justification_text(justification_text)
    normalized_candidates = [
        normalize_probe_formula_text(candidate, scope_role)
        for candidate in (candidates or build_unary_formula_candidates(
            formula_text=normalized_formula,
            justification_text=normalized_justification,
        ))
        if str(candidate or "").strip()
    ]
    if normalized_formula and normalized_formula not in normalized_candidates:
        normalized_candidates.insert(0, normalized_formula)

    entry = {
        "formula_text": normalized_formula,
        "justification_text": normalized_justification,
        "scope_role": scope_role,
    }

    client = build_vlm_client(
        base_url=verifier_base_url,
        model=verifier_model,
        timeout=90,
    )

    upscaled_img = _upscale_for_probe(crop_img)
    probe_crop_path = output_dir / "probe_crop.jpg"
    cv2.imwrite(str(probe_crop_path), upscaled_img)

    structured_prompt = build_structured_unary_verifier_prompt(
        entry,
        normalize_formula_text=lambda value, current_scope_role: normalize_probe_formula_text(value, current_scope_role),
        normalize_justification_text=normalize_probe_justification_text,
    )
    structured_raw = client.chat_with_image_prompt(probe_crop_path, structured_prompt)
    structured_parsed = parse_probe_response(structured_raw)
    structured_formula = derive_formula_from_structured_counts(
        formula_text=normalized_formula,
        structured_parsed=structured_parsed,
        scope_role=scope_role,
    )

    forced_prompt = build_forced_choice_unary_verifier_prompt(
        entry,
        normalized_candidates,
        normalize_formula_text=lambda value, current_scope_role: normalize_probe_formula_text(value, current_scope_role),
        normalize_justification_text=normalize_probe_justification_text,
    )
    forced_raw = client.chat_with_image_prompt(probe_crop_path, forced_prompt)
    forced_parsed = parse_probe_response(forced_raw)

    payload = {
        "image_path": str(image_path),
        "probe_crop_path": str(probe_crop_path),
        "entry": entry,
        "candidates": normalized_candidates,
        "structured_prompt": structured_prompt,
        "structured_raw_response": structured_raw,
        "structured_parsed": structured_parsed,
        "structured_formula": structured_formula,
        "forced_choice_prompt": forced_prompt,
        "forced_choice_raw_response": forced_raw,
        "forced_choice_parsed": forced_parsed,
        "verifier_model": getattr(client, "model", verifier_model),
        "verifier_base_url": getattr(client, "base_url", verifier_base_url),
    }
    (output_dir / "unary_probe_result.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return payload


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the legacy unary verifier prompts on one isolated crop.")
    parser.add_argument("--image", required=True, help="Path to the isolated line crop.")
    parser.add_argument("--formula", required=True, help="Current OCR formula for the crop.")
    parser.add_argument("--justification", default="", help="Current OCR justification for the crop.")
    parser.add_argument("--scope-role", default="derived_line", help="Scope role for prompt context.")
    parser.add_argument("--output-dir", required=True, help="Directory where probe artifacts should be written.")
    parser.add_argument("--candidate", action="append", default=None, help="Explicit forced-choice candidate formula. Repeatable.")
    parser.add_argument("--verifier-base-url", default=None, help="LM Studio base URL override.")
    parser.add_argument("--verifier-model", default=None, help="LM Studio model override.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    payload = run_unary_probe(
        image_path=Path(args.image),
        formula_text=args.formula,
        justification_text=args.justification,
        scope_role=args.scope_role,
        output_dir=Path(args.output_dir),
        verifier_base_url=args.verifier_base_url,
        verifier_model=args.verifier_model,
        candidates=args.candidate,
    )

    forced_choice = str(payload.get("forced_choice_parsed", {}).get("selected_formula") or "").strip()
    confidence = payload.get("forced_choice_parsed", {}).get("confidence")
    print(f"Probe crop: {payload['probe_crop_path']}")
    print(f"Current OCR formula: {payload['entry']['formula_text']}")
    print(f"Candidates: {', '.join(payload['candidates'])}")
    print(f"Structured parsed: {json.dumps(payload['structured_parsed'], ensure_ascii=False)}")
    print(f"Forced-choice parsed: {json.dumps(payload['forced_choice_parsed'], ensure_ascii=False)}")
    if forced_choice:
        print(f"Selected formula: {forced_choice} (confidence={confidence})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
