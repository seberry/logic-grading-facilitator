"""Formula-focused reread helpers for modal_v2 review flows."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np

from proofs_ocr.common import artifacts
from proofs_ocr.domains import modal
from proofs_ocr.pipelines.modal_v2_line_zones import extract_line_zone_crops

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

    if not attempts:
        return resolved, None, ""

    proposed_formula = _select_formula_proposal_with_fallback(
        baseline_formula=resolved.get("formula_text") or "",
        attempt_formulas=candidate_formulas,
    )
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
    if applied:
        return candidate, artifact, "formula_consensus_reread"
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
    first_canonical = canonical_attempts[0][0]
    if not first_canonical or any(candidate != first_canonical for candidate, _display in canonical_attempts[1:]):
        return ""
    if first_canonical == normalized_baseline:
        return ""
    return canonical_attempts[0][1]


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


def _resolved_line_entry(entry: Mapping[str, Any]) -> dict[str, str]:
    return {
        "line_label": str(entry.get("line_label") or "").strip(),
        "formula_text": modal.normalize_formula_text(entry.get("formula_text") or ""),
        "justification_text": modal.normalize_justification_text(entry.get("justification_text") or ""),
        "scope_role": modal.normalize_scope_role(entry.get("scope_role") or "derived_line"),
        "confidence": str(entry.get("confidence") or "").strip(),
    }
