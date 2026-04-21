"""Alternate-model reread helpers for mismatched baseline OCR lines."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from proofs_ocr.common import stitch
from proofs_ocr.domains import modal
from proofs_ocr.pipelines.modal_v2_line_zones import extract_line_zone_crops

FAILED_LINE_REREADS_FILENAME = "failed_line_rereads.json"
FAILED_LINE_REREAD_CROP_TEMPLATE = "failed_line_reread_{order_index}_{mode}.jpg"


def load_source_aware_structure(output_dir: Path) -> list[Any]:
    """Rebuild the final structure while preserving source crop metadata."""

    library = _load_node_entry_library(output_dir)
    root_entries = list(library.get("ROOT", []))
    if not any(modal.has_visible_formula(entry) for entry in root_entries) and "1" in library:
        root_entries = list(library["1"])

    final_structure = stitch.recursive_stitch(
        root_entries,
        library,
        resolve_item=_resolve_modal_ocr_entry_with_source,
    )
    while isinstance(final_structure, list) and len(final_structure) == 1 and isinstance(final_structure[0], list):
        final_structure = final_structure[0]
    stitch.renumber_lines(final_structure)
    return final_structure


def run_failed_line_rereads(
    *,
    output_dir: Path,
    line_records: Sequence[Mapping[str, Any]],
    client: Any,
    modal_config: Mapping[str, Any] | None = None,
    modes: Sequence[str] = ("whole_line", "boxed_context"),
    model_name: str = "",
    scale: int = 4,
) -> list[dict[str, Any]]:
    """Reread mismatched lines using saved crops and an alternate local model."""

    requested_modes = _normalize_modes(modes)
    prompt_cache = {
        mode: (
            modal.build_targeted_justification_ocr_prompt(modal_config)
            if mode == "boxed_justification"
            else modal.build_targeted_line_ocr_prompt(modal_config, focus_mode=mode)
        )
        for mode in requested_modes
    }
    records: list[dict[str, Any]] = []

    for line_record in line_records:
        actual = line_record.get("actual")
        if not isinstance(actual, Mapping):
            continue
        if _record_is_fully_correct(line_record):
            continue

        source_node_id = str(actual.get("source_node_id") or "").strip()
        source_row_order = int(actual.get("source_row_order") or 0)
        if not source_node_id or source_row_order <= 0:
            continue

        expected = line_record.get("expected")
        order_index = int(line_record.get("order_index") or 0)
        attempts: list[dict[str, Any]] = []
        for mode in requested_modes:
            probe_crop_path = _build_probe_crop(
                output_dir=output_dir,
                order_index=order_index,
                node_id=source_node_id,
                row_order=source_row_order,
                mode=mode,
                scale=scale,
            )
            if probe_crop_path is None:
                continue

            raw_response = client.chat_with_image_prompt(probe_crop_path, prompt_cache[mode])
            parse_error = ""
            resolved = _default_line_entry(actual)
            try:
                if mode == "boxed_justification":
                    resolved = _merge_boxed_justification_result(
                        actual=actual,
                        parsed=modal.parse_json_object_response(raw_response),
                    )
                else:
                    resolved = modal.parse_line_ocr_response(raw_response)
            except (ValueError, json.JSONDecodeError) as exc:
                parse_error = str(exc)

            attempts.append(
                {
                    "mode": mode,
                    "prompt_filename": "",
                    "probe_crop_filename": probe_crop_path.name,
                    "raw_response": raw_response,
                    "parse_error": parse_error,
                    "resolved": resolved,
                }
            )

        if not attempts:
            continue

        records.append(
            {
                "order_index": order_index,
                "structure_matches": bool(line_record.get("structure_matches")),
                "formula_matches": bool(line_record.get("formula_matches")),
                "justification_matches": bool(line_record.get("justification_matches")),
                "actual": dict(actual),
                "expected": dict(expected) if isinstance(expected, Mapping) else None,
                "probe_model": model_name,
                "attempts": attempts,
            }
        )

    (output_dir / FAILED_LINE_REREADS_FILENAME).write_text(
        json.dumps(records, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return records


def _record_is_fully_correct(line_record: Mapping[str, Any]) -> bool:
    return bool(line_record.get("formula_matches")) and bool(line_record.get("justification_matches")) and bool(
        line_record.get("structure_matches")
    )


def _normalize_modes(modes: Sequence[str]) -> list[str]:
    normalized: list[str] = []
    for mode in modes:
        text = str(mode or "").strip().lower().replace("-", "_")
        if text not in {"whole_line", "boxed_context", "boxed_justification"}:
            continue
        if text not in normalized:
            normalized.append(text)
    return normalized or ["whole_line", "boxed_context"]


def _default_line_entry(actual: Mapping[str, Any]) -> dict[str, str]:
    return {
        "line_label": str(actual.get("source_line_label") or actual.get("line_label") or "").strip(),
        "formula_text": modal.normalize_formula_text(actual.get("wffstr") or actual.get("formula_text") or ""),
        "justification_text": modal.normalize_justification_text(actual.get("jstr") or actual.get("justification_text") or ""),
        "scope_role": modal.normalize_scope_role(actual.get("scope_role") or "derived_line"),
        "confidence": "",
    }


def _merge_boxed_justification_result(
    *,
    actual: Mapping[str, Any],
    parsed: Mapping[str, Any],
) -> dict[str, str]:
    resolved = _default_line_entry(actual)
    justification_text = modal.normalize_justification_text(parsed.get("justification_text") or parsed.get("jstr") or "")
    resolved["justification_text"] = justification_text
    resolved["confidence"] = str(parsed.get("confidence") or "").strip()
    return resolved


def _load_node_entry_library(output_dir: Path) -> dict[str, list[dict[str, Any]]]:
    library: dict[str, list[dict[str, Any]]] = {}
    for path in sorted(output_dir.glob("node_entries_*.json")):
        node_id = path.stem.removeprefix("node_entries_")
        parsed = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(parsed, list):
            library[node_id] = [dict(item) for item in parsed if isinstance(item, Mapping)]
    return library


def _resolve_modal_ocr_entry_with_source(entry: Any, index: int) -> dict[str, Any] | None:
    resolved = modal.resolve_modal_ocr_entry(entry, index)
    if resolved is None or not isinstance(entry, Mapping):
        return resolved
    source_node_id = str(entry.get("_ocr_node_id") or "").strip()
    source_row_order = int(entry.get("_ocr_row_order") or 0)
    if source_node_id:
        resolved["source_node_id"] = source_node_id
    if source_row_order > 0:
        resolved["source_row_order"] = source_row_order
    line_label = str(entry.get("line_label") or "").strip()
    if line_label:
        resolved["source_line_label"] = line_label
    return resolved


def _build_probe_crop(
    *,
    output_dir: Path,
    order_index: int,
    node_id: str,
    row_order: int,
    mode: str,
    scale: int,
) -> Path | None:
    if mode == "whole_line":
        source_path = output_dir / f"node_line_{node_id}_{row_order}.jpg"
        crop_img = cv2.imread(str(source_path))
    elif mode == "boxed_context":
        crop_img = _build_boxed_context_crop(output_dir=output_dir, node_id=node_id, row_order=row_order)
    elif mode == "boxed_justification":
        crop_img = _build_boxed_justification_crop(output_dir=output_dir, node_id=node_id, row_order=row_order)
    else:
        return None

    if crop_img is None or crop_img.size == 0:
        return None
    scaled = _upscale(crop_img, scale=scale)
    target_path = output_dir / FAILED_LINE_REREAD_CROP_TEMPLATE.format(order_index=order_index, mode=mode)
    cv2.imwrite(str(target_path), scaled)
    return target_path


def _build_boxed_context_crop(*, output_dir: Path, node_id: str, row_order: int) -> Any:
    node_crop_path = output_dir / f"crop_{node_id}.jpg"
    response_path = output_dir / f"node_line_{node_id}_{row_order}.json"
    node_crop = cv2.imread(str(node_crop_path))
    if node_crop is None or node_crop.size == 0 or not response_path.exists():
        return None

    payload = json.loads(response_path.read_text(encoding="utf-8"))
    y1 = int(payload.get("y1") or 0)
    y2 = int(payload.get("y2") or 0)
    if y2 <= y1:
        return None

    img_h, img_w = node_crop.shape[:2]
    band_height = max(1, y2 - y1)
    padding = max(24, band_height * 2)
    context_y1 = max(0, y1 - padding)
    context_y2 = min(img_h, y2 + padding)
    context = node_crop[context_y1:context_y2, 0:img_w].copy()
    if context.size == 0:
        return None

    box_top = max(0, y1 - context_y1)
    box_bottom = min(context.shape[0] - 1, max(box_top + 1, y2 - context_y1 - 1))
    cv2.rectangle(context, (0, box_top), (max(1, context.shape[1] - 1), box_bottom), (0, 0, 255), 2)
    return context


def _build_boxed_justification_crop(*, output_dir: Path, node_id: str, row_order: int) -> Any:
    line_crop_path = output_dir / f"node_line_{node_id}_{row_order}.jpg"
    line_crop = cv2.imread(str(line_crop_path))
    if line_crop is None or line_crop.size == 0:
        return None

    _, justification_crop, zone_geometry = extract_line_zone_crops(line_crop)
    box = _detect_justification_focus_box(justification_crop)
    if box is None:
        return None

    context = line_crop.copy()
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
        if area < 12:
            continue
        if width <= 2 or height <= 2:
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


def _upscale(crop_img: Any, *, scale: int) -> Any:
    if crop_img is None or getattr(crop_img, "size", 0) == 0 or scale <= 1:
        return crop_img
    return cv2.resize(crop_img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
