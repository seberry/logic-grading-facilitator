"""Targeted line-level rereads for narrow modal_v2 OCR failure modes."""

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

_JUSTIFICATION_TOKEN_RE = re.compile(r"(¬E|~E|→E|□E|□I|DN|MC|PR)")
_DIGITS_ONLY_JUSTIFICATION_RE = re.compile(r"[\d,\- ]+")


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


def maybe_reread_justification_head(
    *,
    output_dir: Path,
    client: Any | None,
    node_id: str,
    row_order: int,
    line_crop_img: np.ndarray,
    parsed_entry: Mapping[str, Any],
) -> tuple[dict[str, str], dict[str, Any] | None, str]:
    """Reread a suspicious justification head from a boxed region on the line crop."""

    resolved = _resolved_line_entry(parsed_entry)
    if client is None or not _should_reread_justification_head(resolved, node_id=node_id, row_order=row_order):
        return resolved, None, ""

    reread_img = _build_boxed_justification_context_crop(line_crop_img)
    if reread_img is None or reread_img.size == 0:
        return resolved, None, ""
    reread_img = _upscale_for_reread(reread_img)
    reread_crop_path = output_dir / artifacts.NODE_LINE_JUSTIFICATION_REREAD_CROP_TEMPLATE.format(
        node_id=node_id,
        row_order=row_order,
    )
    cv2.imwrite(str(reread_crop_path), reread_img)

    prompt = modal.build_targeted_justification_ocr_prompt()
    raw_response = client.chat_with_image_prompt(reread_crop_path, prompt)
    parse_error = ""
    parsed: dict[str, Any] = {}
    candidate = dict(resolved)
    try:
        parsed = modal.parse_json_object_response(raw_response)
        candidate["justification_text"] = modal.normalize_justification_text(
            parsed.get("justification_text") or parsed.get("jstr") or ""
        )
        candidate["confidence"] = str(parsed.get("confidence") or "").strip()
    except (ValueError, json.JSONDecodeError) as exc:
        parse_error = str(exc)

    applied = _should_accept_justification_head_reread(
        original=resolved,
        candidate=candidate,
        node_id=node_id,
        row_order=row_order,
    )
    artifact = {
        "crop_filename": reread_crop_path.name,
        "prompt": prompt,
        "raw_response": raw_response,
        "parse_error": parse_error,
        "parsed": parsed,
        "resolved": candidate,
        "applied": applied,
    }
    if applied:
        return candidate, artifact, "justification_head_reread"
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


def _should_reread_justification_head(
    entry: Mapping[str, Any],
    *,
    node_id: str,
    row_order: int,
) -> bool:
    scope_role = modal.normalize_scope_role(entry.get("scope_role") or "derived_line")
    formula_text = modal.normalize_formula_text(entry.get("formula_text") or "")
    justification_text = modal.normalize_justification_text(entry.get("justification_text") or "")
    if not formula_text:
        return False
    later_nonroot_row = str(node_id or "").strip() != "ROOT" and int(row_order) > 1
    malformed_justification = _looks_malformed_justification(justification_text)
    if scope_role == "derived_line" and justification_text in {"", "PR"}:
        return True
    if scope_role == "derived_line" and _DIGITS_ONLY_JUSTIFICATION_RE.fullmatch(justification_text):
        return True
    if scope_role == "derived_line" and malformed_justification:
        return True
    if bool(_JUSTIFICATION_TOKEN_RE.search(formula_text)):
        return True
    if later_nonroot_row and justification_text in {"", "PR"}:
        return True
    if later_nonroot_row and malformed_justification:
        return True
    return False


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


def _should_accept_justification_head_reread(
    *,
    original: Mapping[str, Any],
    candidate: Mapping[str, Any],
    node_id: str,
    row_order: int,
) -> bool:
    original_formula = modal.normalize_formula_text(original.get("formula_text") or "")
    original_scope_role = modal.normalize_scope_role(original.get("scope_role") or "derived_line")
    original_justification = modal.normalize_justification_text(original.get("justification_text") or "")
    candidate_justification = modal.normalize_justification_text(candidate.get("justification_text") or "")
    if not candidate_justification or candidate_justification == original_justification:
        return False
    candidate_head = modal.normalize_justification_head(candidate_justification)
    if candidate_head in {"", "other"}:
        return False
    later_nonroot_row = str(node_id or "").strip() != "ROOT" and int(row_order) > 1
    malformed_justification = _looks_malformed_justification(original_justification)
    if malformed_justification:
        visible_heads = _extract_visible_justification_heads(original_justification)
        if visible_heads and candidate_head not in visible_heads:
            return False
    if original_scope_role == "derived_line" and original_justification in {"", "PR"}:
        return True
    if _DIGITS_ONLY_JUSTIFICATION_RE.fullmatch(original_justification):
        return True
    if malformed_justification:
        return True
    if later_nonroot_row and original_justification in {"", "PR"}:
        return True
    return bool(_JUSTIFICATION_TOKEN_RE.search(original_formula))


def _looks_malformed_justification(justification_text: str) -> bool:
    text = modal.normalize_justification_text(justification_text or "")
    if not text:
        return False
    if modal.normalize_justification_head(text) != "other":
        return False
    return bool(_JUSTIFICATION_TOKEN_RE.search(text))


def _extract_visible_justification_heads(justification_text: str) -> set[str]:
    heads = {
        modal.normalize_justification_head(fragment)
        for fragment in _JUSTIFICATION_TOKEN_RE.findall(justification_text or "")
    }
    return {head for head in heads if head not in {"", "other"}}


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


def _build_boxed_justification_context_crop(line_crop_img: np.ndarray) -> np.ndarray | None:
    if line_crop_img.size == 0:
        return None
    _, justification_crop, zone_geometry = extract_line_zone_crops(line_crop_img)
    box = _detect_justification_focus_box(justification_crop)
    if box is None:
        return None

    context = line_crop_img.copy()
    justification_x1 = int(zone_geometry.get("justification_x1") or 0)
    x, y, w, h = box
    abs_x1 = max(0, min(context.shape[1] - 1, justification_x1 + x))
    abs_y1 = max(0, min(context.shape[0] - 1, y))
    abs_x2 = max(abs_x1 + 1, min(context.shape[1], justification_x1 + x + w))
    abs_y2 = max(abs_y1 + 1, min(context.shape[0], y + h))
    cv2.rectangle(context, (abs_x1, abs_y1), (abs_x2 - 1, abs_y2 - 1), (0, 0, 255), 2)
    return context


def _detect_justification_focus_box(justification_crop: np.ndarray) -> tuple[int, int, int, int] | None:
    if justification_crop.size == 0:
        return None

    gray = cv2.cvtColor(justification_crop, cv2.COLOR_BGR2GRAY)
    dark = (gray < 185).astype(np.uint8)
    if int(dark.sum()) <= 0:
        return None

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(dark, 8)
    components: list[tuple[int, int, int, int]] = []
    for label_index in range(1, num_labels):
        area = int(stats[label_index, cv2.CC_STAT_AREA])
        left = int(stats[label_index, cv2.CC_STAT_LEFT])
        top = int(stats[label_index, cv2.CC_STAT_TOP])
        width = int(stats[label_index, cv2.CC_STAT_WIDTH])
        height = int(stats[label_index, cv2.CC_STAT_HEIGHT])
        if area < 12 or width <= 2 or height <= 2:
            continue
        components.append((left, top, width, height))

    if not components:
        return None

    components.sort(key=lambda item: (item[0], item[1]))
    head_components: list[tuple[int, int, int, int]] = [components[0]]
    right_edge = components[0][0] + components[0][2]
    head_top = components[0][1]
    head_bottom = components[0][1] + components[0][3]
    for component in components[1:]:
        left, top, width, height = component
        gap = left - right_edge
        comp_bottom = top + height
        vertical_overlap = min(head_bottom, comp_bottom) - max(head_top, top)
        min_height = max(1, min(head_bottom - head_top, height))
        same_row = vertical_overlap >= int(0.35 * min_height)
        if gap > 30 or not same_row:
            break
        head_components.append(component)
        right_edge = max(right_edge, left + width)
        head_top = min(head_top, top)
        head_bottom = max(head_bottom, comp_bottom)
        if len(head_components) >= 5:
            break

    x1 = min(component[0] for component in head_components)
    y1 = min(component[1] for component in head_components)
    x2 = max(component[0] + component[2] for component in head_components)
    y2 = max(component[1] + component[3] for component in head_components)
    pad_x = 6
    pad_y = 4
    img_h, img_w = justification_crop.shape[:2]
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(img_w, x2 + pad_x)
    y2 = min(img_h, y2 + pad_y)
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2 - x1, y2 - y1)
