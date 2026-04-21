"""Formula-focused reread helpers for modal_v2 review flows."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from modal_ocr.verifier import (
    UNARY_VERIFIER_RULE_HEADS,
    build_forced_choice_unary_verifier_prompt,
    build_structured_unary_verifier_prompt,
    leading_unary_prefix,
)
from proofs_ocr.common import artifacts
from proofs_ocr.domains import modal
from proofs_ocr.pipelines.modal_v2_line_zones import extract_line_zone_crops, extract_relaxed_formula_crop

_LEAKED_JUSTIFICATION_TOKEN_RE = re.compile(r"(¬E|~E|→E|□E|□I|DN|MC|PR)")


def build_formula_crop(line_crop_img: np.ndarray) -> np.ndarray | None:
    """Return the formula-side crop for one saved line image."""

    if line_crop_img.size == 0:
        return None
    formula_crop, _, _ = extract_line_zone_crops(line_crop_img)
    if formula_crop.size == 0:
        return None
    return formula_crop


def build_boxed_formula_context_crop(line_crop_img: np.ndarray) -> np.ndarray | None:
    """Return the full line with the formula region boxed for context-aware rereads."""

    if line_crop_img.size == 0:
        return None
    _, _, geometry = extract_line_zone_crops(line_crop_img)
    formula_x1 = int(geometry.get("formula_x1") or 0)
    formula_x2 = int(geometry.get("formula_x2") or 0)
    if formula_x2 <= formula_x1:
        return None

    context = line_crop_img.copy()
    img_h, img_w = context.shape[:2]
    x1 = max(0, min(img_w - 1, formula_x1))
    x2 = max(x1 + 1, min(img_w, formula_x2))
    cv2.rectangle(context, (x1, 0), (x2 - 1, max(0, img_h - 1)), (0, 0, 255), 2)
    return context


def upscale_for_reread(crop_img: np.ndarray, *, scale: int = 4) -> np.ndarray:
    """Upscale a formula reread crop to give the local VLM more visual detail."""

    if crop_img.size == 0:
        return crop_img
    return cv2.resize(crop_img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)


def maybe_reread_formula_text(
    *,
    output_dir: Path,
    client: Any | None,
    node_id: str,
    row_order: int,
    line_crop_img: np.ndarray,
    parsed_entry: Mapping[str, Any],
    modal_config: Mapping[str, Any] | None = None,
    precomputed_attempts: Sequence[Mapping[str, Any]] | None = None,
    formula_zone_crop: np.ndarray | None = None,
    formula_zone_geometry: Mapping[str, Any] | None = None,
) -> tuple[dict[str, str], dict[str, Any] | None, str]:
    """Reread a line's formula using formula-only crops and accept only on consensus."""

    resolved = _resolved_line_entry(parsed_entry)
    if client is None or line_crop_img.size == 0:
        return resolved, None, ""

    prompt_formula = modal.build_targeted_formula_ocr_prompt(modal_config, focus_mode="formula_crop")
    prompt_boxed = modal.build_targeted_formula_ocr_prompt(modal_config, focus_mode="boxed_formula")
    attempt_specs = [
        (
            "formula_crop",
            build_formula_crop(line_crop_img),
            prompt_formula,
            output_dir / artifacts.NODE_LINE_FORMULA_REREAD_CROP_TEMPLATE.format(node_id=node_id, row_order=row_order),
        ),
        (
            "boxed_formula",
            build_boxed_formula_context_crop(line_crop_img),
            prompt_boxed,
            output_dir / artifacts.NODE_LINE_FORMULA_BOXED_REREAD_CROP_TEMPLATE.format(node_id=node_id, row_order=row_order),
        ),
    ]

    attempts: list[dict[str, Any]] = []
    candidate_formulas: list[str] = []
    for precomputed in precomputed_attempts or ():
        attempt = dict(precomputed)
        attempts.append(attempt)
        structured_formula = modal.normalize_formula_text(attempt.get("structured_formula") or "")
        if structured_formula:
            candidate_formulas.append(structured_formula)
    for attempt_kind, crop_img, prompt, crop_path in attempt_specs:
        if crop_img is None or crop_img.size == 0:
            continue
        scaled = upscale_for_reread(crop_img)
        cv2.imwrite(str(crop_path), scaled)
        raw_response = client.chat_with_image_prompt(crop_path, prompt)
        parse_error = ""
        parsed: dict[str, Any] = {}
        structured_formula = ""
        confidence = ""
        try:
            parsed = modal.parse_json_object_response(raw_response)
            structured_formula = modal.normalize_formula_text(parsed.get("formula_text") or "")
            confidence = str(parsed.get("confidence") or "").strip()
        except (ValueError, json.JSONDecodeError) as exc:
            parse_error = str(exc)
        attempts.append(
            {
                "attempt_kind": attempt_kind,
                "crop_filename": crop_path.name,
                "prompt": prompt,
                "raw_response": raw_response,
                "parse_error": parse_error,
                "parsed": parsed,
                "structured_formula": structured_formula,
                "confidence": confidence,
            }
        )
        if structured_formula:
            candidate_formulas.append(structured_formula)

    if _should_try_relaxed_formula_fallback(
        baseline_formula=resolved.get("formula_text") or "",
        attempt_formulas=candidate_formulas,
        formula_zone_crop=formula_zone_crop,
        formula_zone_geometry=formula_zone_geometry,
    ):
        relaxed_crop, relaxed_geometry = extract_relaxed_formula_crop(
            formula_zone_crop,
            formula_zone_geometry=formula_zone_geometry,
        )
        if relaxed_crop.size != 0:
            prompt_relaxed = modal.build_targeted_formula_ocr_prompt(modal_config, focus_mode="formula_crop")
            relaxed_path = output_dir / artifacts.NODE_LINE_FORMULA_THIN_REREAD_CROP_TEMPLATE.format(
                node_id=node_id,
                row_order=row_order,
            )
            scaled_relaxed = upscale_for_reread(relaxed_crop)
            cv2.imwrite(str(relaxed_path), scaled_relaxed)
            raw_response = client.chat_with_image_prompt(relaxed_path, prompt_relaxed)
            parse_error = ""
            parsed: dict[str, Any] = {}
            structured_formula = ""
            confidence = ""
            try:
                parsed = modal.parse_json_object_response(raw_response)
                structured_formula = modal.normalize_formula_text(parsed.get("formula_text") or "")
                confidence = str(parsed.get("confidence") or "").strip()
            except (ValueError, json.JSONDecodeError) as exc:
                parse_error = str(exc)
            attempts.append(
                {
                    "attempt_kind": "thin_formula_fallback",
                    "crop_filename": relaxed_path.name,
                    "prompt": prompt_relaxed,
                    "raw_response": raw_response,
                    "parse_error": parse_error,
                    "parsed": parsed,
                    "structured_formula": structured_formula,
                    "confidence": confidence,
                    "geometry": relaxed_geometry,
                }
            )
            if structured_formula:
                candidate_formulas.append(structured_formula)

    if not attempts:
        return resolved, None, ""

    proposed_formula = _select_formula_proposal_with_fallback(
        baseline_formula=resolved.get("formula_text") or "",
        attempt_formulas=candidate_formulas,
    )
    unary_entry = dict(resolved)
    if proposed_formula:
        unary_entry["formula_text"] = proposed_formula
    unary_artifact = None
    unary_recovery = ""
    if attempts:
        unary_candidate, unary_artifact, unary_recovery = maybe_reread_unary_formula_text(
            output_dir=output_dir,
            client=client,
            node_id=node_id,
            row_order=row_order,
            line_crop_img=line_crop_img,
            formula_crop_img=formula_zone_crop,
            parsed_entry=unary_entry,
        )
        if unary_recovery:
            proposed_formula = unary_candidate.get("formula_text") or proposed_formula
    applied = bool(proposed_formula)
    candidate = dict(resolved)
    if applied:
        candidate["formula_text"] = proposed_formula
    artifact = {
        "attempt_kind": "formula_consensus_reread",
        "baseline_formula": resolved.get("formula_text") or "",
        "proposed_formula": proposed_formula,
        "candidate_attempts": attempts,
        "applied": applied,
    }
    if unary_artifact is not None:
        artifact["unary_formula_reread"] = unary_artifact
    if applied:
        return candidate, artifact, "formula_consensus_reread"
    return resolved, artifact, ""


def maybe_reread_unary_formula_text(
    *,
    output_dir: Path,
    client: Any | None,
    node_id: str,
    row_order: int,
    line_crop_img: np.ndarray,
    formula_crop_img: np.ndarray | None,
    parsed_entry: Mapping[str, Any],
) -> tuple[dict[str, str], dict[str, Any] | None, str]:
    """Run a narrow unary-count verifier on suspicious unary/modal formulas."""

    resolved = _resolved_line_entry(parsed_entry)
    if client is None or not _should_try_unary_formula_reread(resolved):
        return resolved, None, ""

    probe_source = formula_crop_img if formula_crop_img is not None and formula_crop_img.size != 0 else build_formula_crop(line_crop_img)
    if probe_source is None or probe_source.size == 0:
        return resolved, None, ""

    candidates = _build_unary_formula_candidates(
        formula_text=resolved.get("formula_text") or "",
        justification_text=resolved.get("justification_text") or "",
    )
    if len(candidates) < 2:
        return resolved, None, ""

    probe_path = output_dir / artifacts.NODE_LINE_FORMULA_UNARY_REREAD_CROP_TEMPLATE.format(
        node_id=node_id,
        row_order=row_order,
    )
    cv2.imwrite(str(probe_path), upscale_for_reread(probe_source))

    entry_payload = {
        "formula_text": resolved.get("formula_text") or "",
        "justification_text": resolved.get("justification_text") or "",
        "scope_role": "derived_line",
    }
    structured_prompt = build_structured_unary_verifier_prompt(
        entry_payload,
        normalize_formula_text=lambda value, _scope_role: modal.normalize_formula_text(value),
        normalize_justification_text=modal.normalize_justification_text,
    )
    structured_raw = client.chat_with_image_prompt(probe_path, structured_prompt)
    structured_parse_error = ""
    structured_parsed: dict[str, Any] = {}
    structured_formula = ""
    try:
        structured_parsed = modal.parse_json_object_response(structured_raw)
        structured_formula = _derive_unary_formula_from_structured_counts(
            formula_text=resolved.get("formula_text") or "",
            structured_parsed=structured_parsed,
        ) or ""
    except (ValueError, json.JSONDecodeError) as exc:
        structured_parse_error = str(exc)

    forced_prompt = build_forced_choice_unary_verifier_prompt(
        entry_payload,
        candidates,
        normalize_formula_text=lambda value, _scope_role: modal.normalize_formula_text(value),
        normalize_justification_text=modal.normalize_justification_text,
    )
    forced_raw = client.chat_with_image_prompt(probe_path, forced_prompt)
    forced_parse_error = ""
    forced_parsed: dict[str, Any] = {}
    forced_formula = ""
    try:
        forced_parsed = modal.parse_json_object_response(forced_raw)
        forced_formula = modal.normalize_formula_text(forced_parsed.get("selected_formula") or "")
    except (ValueError, json.JSONDecodeError) as exc:
        forced_parse_error = str(exc)

    proposed_formula = _accept_unary_formula_reread(
        original_formula=resolved.get("formula_text") or "",
        structured_formula=structured_formula,
        forced_formula=forced_formula,
        candidates=candidates,
    )
    applied = bool(proposed_formula)
    artifact = {
        "crop_filename": probe_path.name,
        "candidates": candidates,
        "structured_prompt": structured_prompt,
        "structured_raw_response": structured_raw,
        "structured_parse_error": structured_parse_error,
        "structured_parsed": structured_parsed,
        "structured_formula": structured_formula,
        "forced_choice_prompt": forced_prompt,
        "forced_choice_raw_response": forced_raw,
        "forced_choice_parse_error": forced_parse_error,
        "forced_choice_parsed": forced_parsed,
        "forced_formula": forced_formula,
        "proposed_formula": proposed_formula,
        "applied": applied,
    }
    if applied:
        candidate = dict(resolved)
        candidate["formula_text"] = proposed_formula
        return candidate, artifact, "unary_formula_reread"
    return resolved, artifact, ""


def _select_consensus_formula(*, baseline_formula: str, attempt_formulas: list[str]) -> str:
    if len(attempt_formulas) < 2:
        return ""
    normalized_baseline = modal.canonicalize_formula_for_comparison(baseline_formula)
    canonical_attempts = [
        (modal.canonicalize_formula_for_comparison(value), modal.normalize_formula_text(value))
        for value in attempt_formulas
        if modal.canonicalize_formula_for_comparison(value)
    ]
    if len(canonical_attempts) < 2:
        return ""
    counts = Counter(candidate for candidate, _display in canonical_attempts if candidate)
    if not counts:
        return ""
    best_canonical, best_count = counts.most_common(1)[0]
    if best_count < 2:
        return ""
    tied = [candidate for candidate, count in counts.items() if count == best_count]
    if len(tied) != 1 or best_canonical == normalized_baseline:
        return ""
    for canonical_value, display_value in canonical_attempts:
        if canonical_value == best_canonical:
            return display_value
    return ""


def _select_formula_proposal_with_fallback(
    *,
    baseline_formula: str,
    attempt_formulas: list[str],
) -> str:
    consensus = _select_consensus_formula(
        baseline_formula=baseline_formula,
        attempt_formulas=attempt_formulas,
    )
    if consensus:
        return consensus

    non_empty = [modal.normalize_formula_text(value) for value in attempt_formulas if modal.normalize_formula_text(value)]
    if len(non_empty) != 1:
        return ""
    candidate = non_empty[0]
    baseline_normalized = modal.normalize_formula_text(baseline_formula)
    if not baseline_normalized:
        return ""
    if not _looks_like_justification_leak(baseline_normalized):
        return ""
    if _looks_like_justification_leak(candidate):
        return ""
    if len(modal.canonicalize_formula_for_comparison(candidate)) >= len(modal.canonicalize_formula_for_comparison(baseline_normalized)):
        return ""
    return candidate


def _looks_like_justification_leak(formula_text: str) -> bool:
    normalized = modal.normalize_formula_text(formula_text)
    if not normalized:
        return False
    return bool(_LEAKED_JUSTIFICATION_TOKEN_RE.search(normalized))


def _should_try_relaxed_formula_fallback(
    *,
    baseline_formula: str,
    attempt_formulas: list[str],
    formula_zone_crop: np.ndarray | None,
    formula_zone_geometry: Mapping[str, Any] | None,
) -> bool:
    if formula_zone_crop is None or formula_zone_crop.size == 0 or not formula_zone_geometry:
        return False
    baseline_normalized = modal.normalize_formula_text(baseline_formula)
    if not _looks_suspiciously_short_modal_formula(baseline_normalized):
        return False
    crop_height = int(formula_zone_crop.shape[0])
    y1 = int(formula_zone_geometry.get("y1", 0) or 0)
    y2 = int(formula_zone_geometry.get("y2", crop_height) or crop_height)
    trimmed_height = max(0, y2 - y1)
    if trimmed_height <= 0 or crop_height <= 0:
        return False
    trim_ratio = float(trimmed_height) / float(crop_height)
    if trim_ratio > 0.65 and trimmed_height > 18:
        return False
    longer_attempts = [
        modal.normalize_formula_text(value)
        for value in attempt_formulas
        if len(modal.canonicalize_formula_for_comparison(value))
        > len(modal.canonicalize_formula_for_comparison(baseline_normalized))
    ]
    return len(longer_attempts) >= 2


def _looks_suspiciously_short_modal_formula(formula_text: str) -> bool:
    canonical = modal.canonicalize_formula_for_comparison(formula_text)
    return canonical in {"□", "⋄", "¬"}


def _should_try_unary_formula_reread(entry: Mapping[str, Any]) -> bool:
    formula_text = modal.normalize_formula_text(entry.get("formula_text") or "")
    if not formula_text:
        return False
    prefix = leading_unary_prefix(formula_text)
    if not prefix:
        return False
    body = formula_text[len(prefix) :].strip()
    if not body or any(char in body for char in "()∧∨→↔"):
        return False
    if not (body.isalpha() or body == "⊥"):
        return False
    rule_head = modal.normalize_justification_head(entry.get("justification_text") or "")
    return rule_head in UNARY_VERIFIER_RULE_HEADS or ("□" in prefix or "⋄" in prefix)


def _build_candidate_prefixes(*, formula_text: str, justification_text: str) -> list[str]:
    prefix = leading_unary_prefix(formula_text)
    if not prefix:
        return []
    rule_head = modal.normalize_justification_head(justification_text)
    contains_box = "□" in prefix
    contains_diamond = "⋄" in prefix
    negation_count = prefix.count("¬")
    candidates: list[str] = [prefix]
    if contains_diamond:
        pool = ("⋄", "⋄¬", "⋄¬¬", "¬⋄", "¬⋄¬", "¬⋄¬¬")
    elif contains_box or negation_count >= 2 or rule_head in UNARY_VERIFIER_RULE_HEADS:
        pool = ("□", "□¬", "□¬¬", "¬□", "¬□¬", "¬□¬¬")
    else:
        pool = ("¬", "¬¬", "¬¬¬")
    for candidate in pool:
        if candidate not in candidates:
            candidates.append(candidate)
    return candidates


def _build_unary_formula_candidates(*, formula_text: str, justification_text: str) -> list[str]:
    normalized_formula = modal.normalize_formula_text(formula_text)
    prefix = leading_unary_prefix(normalized_formula)
    if not prefix:
        return [normalized_formula] if normalized_formula else []
    body = normalized_formula[len(prefix) :].strip()
    if not body:
        return [normalized_formula]
    skeleton = _unary_formula_skeleton(normalized_formula)
    candidates: list[str] = []
    for candidate_prefix in _build_candidate_prefixes(
        formula_text=normalized_formula,
        justification_text=modal.normalize_justification_text(justification_text),
    ):
        candidate = f"{candidate_prefix}{body}"
        if _unary_formula_skeleton(candidate) != skeleton:
            continue
        if candidate not in candidates:
            candidates.append(candidate)
    return candidates or [normalized_formula]


def _unary_formula_skeleton(formula_text: str) -> str:
    normalized = modal.normalize_formula_text(formula_text)
    prefix = leading_unary_prefix(normalized)
    body = normalized[len(prefix) :].strip()
    modal_class = "□" if "□" in prefix else "⋄" if "⋄" in prefix else "atom"
    return f"{modal_class}:{body}"


def _derive_unary_formula_from_structured_counts(
    *,
    formula_text: str,
    structured_parsed: Mapping[str, Any],
) -> str | None:
    normalized_formula = modal.normalize_formula_text(formula_text)
    prefix = leading_unary_prefix(normalized_formula)
    body = normalized_formula[len(prefix) :].strip()
    if not body:
        return None
    modal_value = str(structured_parsed.get("main_modal_operator") or "").strip().lower()
    if modal_value == "box":
        modal_symbol = "□"
    elif modal_value == "diamond":
        modal_symbol = "⋄"
    else:
        modal_symbol = ""
    before_modal = _int_field(structured_parsed.get("negations_before_modal"), 0, 0, 3)
    before_atom = _int_field(structured_parsed.get("negations_before_final_atom"), 0, 0, 3)
    if not modal_symbol:
        return f"{'¬' * before_atom}{body}"
    return f"{'¬' * before_modal}{modal_symbol}{'¬' * before_atom}{body}"


def _int_field(value: Any, default: int = 0, minimum: int = 0, maximum: int = 3) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, parsed))


def _accept_unary_formula_reread(
    *,
    original_formula: str,
    structured_formula: str,
    forced_formula: str,
    candidates: Sequence[str],
) -> str:
    original_normalized = modal.normalize_formula_text(original_formula)
    structured_normalized = modal.normalize_formula_text(structured_formula)
    forced_normalized = modal.normalize_formula_text(forced_formula)
    if not structured_normalized or structured_normalized != forced_normalized:
        return ""
    if structured_normalized == original_normalized:
        return ""
    if structured_normalized not in candidates:
        return ""
    if _unary_formula_skeleton(structured_normalized) != _unary_formula_skeleton(original_normalized):
        return ""
    if _unary_prefix_distance(original_normalized, structured_normalized) != 1:
        return ""
    return structured_normalized


def _unary_prefix_distance(original: str, candidate: str) -> int:
    original_prefix = leading_unary_prefix(original)
    candidate_prefix = leading_unary_prefix(candidate)
    original_modal = "□" if "□" in original_prefix else "⋄" if "⋄" in original_prefix else ""
    candidate_modal = "□" if "□" in candidate_prefix else "⋄" if "⋄" in candidate_prefix else ""
    if original_modal != candidate_modal:
        return 10
    if original_modal:
        original_before_modal, original_after_modal = original_prefix.split(original_modal, 1)
        candidate_before_modal, candidate_after_modal = candidate_prefix.split(candidate_modal, 1)
        return abs(original_before_modal.count("¬") - candidate_before_modal.count("¬")) + abs(
            original_after_modal.count("¬") - candidate_after_modal.count("¬")
        )
    return abs(original_prefix.count("¬") - candidate_prefix.count("¬"))


def _resolved_line_entry(entry: Mapping[str, Any]) -> dict[str, str]:
    return {
        "line_label": str(entry.get("line_label") or "").strip(),
        "formula_text": modal.normalize_formula_text(entry.get("formula_text") or ""),
        "justification_text": modal.normalize_justification_text(entry.get("justification_text") or ""),
        "scope_role": modal.normalize_scope_role(entry.get("scope_role") or "derived_line"),
        "confidence": str(entry.get("confidence") or "").strip(),
    }
