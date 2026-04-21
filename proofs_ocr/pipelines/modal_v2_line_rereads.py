"""Targeted line-level rereads for narrow modal_v2 OCR failure modes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np

from proofs_ocr.common import artifacts
from proofs_ocr.domains import modal


def maybe_reread_contradiction_line(
    *,
    output_dir: Path,
    client: Any | None,
    node_id: str,
    row_order: int,
    node_crop_img: np.ndarray,
    row_record: Mapping[str, Any],
    line_crop_img: np.ndarray,
    parsed_entry: Mapping[str, Any],
) -> tuple[dict[str, str], dict[str, Any] | None, str]:
    """Reread a suspicious negation-elimination line as a possible contradiction."""

    resolved = _resolved_line_entry(parsed_entry)
    if client is None or not _should_reread_contradiction_line(resolved):
        return resolved, None, ""

    reread_source = _extract_tight_reread_crop(node_crop_img=node_crop_img, row_record=row_record, fallback=line_crop_img)
    reread_img = _upscale_for_reread(reread_source)
    reread_crop_path = output_dir / artifacts.NODE_LINE_CONTRADICTION_REREAD_CROP_TEMPLATE.format(
        node_id=node_id,
        row_order=row_order,
    )
    cv2.imwrite(str(reread_crop_path), reread_img)

    prompt = build_contradiction_reread_prompt(resolved)
    raw_response = client.chat_with_image_prompt(reread_crop_path, prompt)
    parse_error = ""
    candidate = _resolved_line_entry({})
    try:
        candidate = _resolved_line_entry(modal.parse_line_ocr_response(raw_response))
    except (ValueError, json.JSONDecodeError) as exc:
        parse_error = str(exc)

    applied = _should_accept_contradiction_reread(original=resolved, candidate=candidate)
    artifact = {
        "crop_filename": reread_crop_path.name,
        "prompt": prompt,
        "raw_response": raw_response,
        "parse_error": parse_error,
        "resolved": candidate,
        "applied": applied,
    }
    if applied:
        return candidate, artifact, "contradiction_reread"
    return resolved, artifact, ""


def build_contradiction_reread_prompt(current_entry: Mapping[str, Any]) -> str:
    """Build a narrow reread prompt for contradiction lines."""

    current_formula = modal.normalize_formula_text(current_entry.get("formula_text") or "")
    current_justification = modal.normalize_justification_text(current_entry.get("justification_text") or "")
    lines = [
        "You are rereading one visible line from a modal Fitch-style proof.",
        "This line is suspicious because negation-elimination text may have been placed into the formula field.",
        "Return only JSON with keys: line_label, formula_text, justification_text, scope_role, confidence.",
        "If the formula is a contradiction symbol, return formula_text as ⊥.",
        "A contradiction line may be written as ⊥, _|_, _ | _, or a short vertical stroke above a short horizontal bar.",
        "Do not place justification text such as ¬E 4,8 or ~E 4,8 into formula_text.",
        "Keep the rule and citations in justification_text.",
        "If the crop shows no contradiction symbol, return the visible formula honestly.",
    ]
    if current_formula or current_justification:
        lines.append(
            "Current OCR hint: "
            f'formula_text="{current_formula}", justification_text="{current_justification}". '
            "Use this only as a weak hint and prefer the image."
        )
    return "\n".join(lines)


def _should_reread_contradiction_line(entry: Mapping[str, Any]) -> bool:
    formula_text = modal.normalize_formula_text(entry.get("formula_text") or "")
    justification_text = modal.normalize_justification_text(entry.get("justification_text") or "")
    if formula_text == "⊥":
        return False
    formula_head = modal.normalize_justification_head(formula_text)
    justification_head = modal.normalize_justification_head(justification_text)
    return formula_head in {"¬E", "~E"} or justification_head in {"¬E", "~E"}


def _should_accept_contradiction_reread(
    *,
    original: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> bool:
    candidate_formula = modal.normalize_formula_text(candidate.get("formula_text") or "")
    if candidate_formula != "⊥":
        return False
    candidate_head = modal.normalize_justification_head(candidate.get("justification_text") or "")
    original_head = modal.normalize_justification_head(original.get("formula_text") or "")
    original_justification_head = modal.normalize_justification_head(original.get("justification_text") or "")
    return candidate_head in {"¬E", "~E"} or original_head in {"¬E", "~E"} or original_justification_head in {"¬E", "~E"}


def _resolved_line_entry(entry: Mapping[str, Any]) -> dict[str, str]:
    return {
        "line_label": str(entry.get("line_label") or "").strip(),
        "formula_text": modal.normalize_formula_text(entry.get("formula_text") or ""),
        "justification_text": modal.normalize_justification_text(entry.get("justification_text") or ""),
        "scope_role": modal.normalize_scope_role(entry.get("scope_role") or "derived_line"),
        "confidence": str(entry.get("confidence") or "").strip(),
    }


def _upscale_for_reread(crop_img: np.ndarray, *, scale: int = 4) -> np.ndarray:
    if crop_img.size == 0:
        return crop_img
    return cv2.resize(crop_img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)


def _extract_tight_reread_crop(
    *,
    node_crop_img: np.ndarray,
    row_record: Mapping[str, Any],
    fallback: np.ndarray,
) -> np.ndarray:
    if node_crop_img.size == 0:
        return fallback
    img_h, img_w = node_crop_img.shape[:2]
    y1 = int(row_record.get("y1", 0))
    y2 = int(row_record.get("y2", 0))
    if y2 <= y1:
        return fallback
    band_height = max(1, y2 - y1)
    pad_y = max(2, min(4, band_height // 5))
    top = max(0, y1 - pad_y)
    bottom = min(img_h, y2 + pad_y)
    crop = node_crop_img[top:bottom, :img_w]
    return crop if crop.size else fallback
