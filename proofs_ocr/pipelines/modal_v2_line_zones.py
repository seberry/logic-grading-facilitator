"""Line-zone OCR helpers for the modal_v2 pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Tuple

import cv2
import numpy as np

from proofs_ocr.common import artifacts
from proofs_ocr.domains import modal, modal_line_zones


def extract_line_zone_crops(crop_img: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Split one line crop into formula-side and justification-side subcrops."""

    if crop_img.size == 0:
        empty = crop_img.copy()
        return empty, empty, {
            "split_x": 0,
            "formula_x1": 0,
            "formula_x2": 0,
            "justification_x1": 0,
            "justification_x2": 0,
            "width": 0,
            "height": 0,
        }

    img_h, img_w = crop_img.shape[:2]
    split_x = _choose_line_zone_split_x(crop_img)
    overlap = max(10, int(img_w * 0.02))

    formula_x1 = 0
    formula_x2 = max(1, min(img_w, split_x + overlap))
    justification_x1 = max(0, min(img_w - 1, split_x - overlap))
    justification_x2 = img_w

    formula_crop = crop_img[:, formula_x1:formula_x2]
    justification_crop = crop_img[:, justification_x1:justification_x2]
    if formula_crop.size == 0:
        formula_crop = crop_img.copy()
    if justification_crop.size == 0:
        justification_crop = crop_img.copy()

    return formula_crop, justification_crop, {
        "split_x": int(split_x),
        "formula_x1": int(formula_x1),
        "formula_x2": int(formula_x2),
        "justification_x1": int(justification_x1),
        "justification_x2": int(justification_x2),
        "width": int(img_w),
        "height": int(img_h),
    }


def read_line_with_zone_separation(
    *,
    output_dir: Path,
    client: Any,
    verifier_client: Any | None,
    formula_prompt: str,
    justification_prompt: str,
    node: Any,
    line_crop_img: np.ndarray,
    row_record: Mapping[str, Any],
) -> Mapping[str, str]:
    """Run formula/justification OCR on separate crops and merge the result."""

    row_order = int(row_record.get("row_order", 0))
    crop_name = artifacts.NODE_LINE_CROP_TEMPLATE.format(node_id=node.id, row_order=row_order)
    crop_path = output_dir / crop_name
    cv2.imwrite(str(crop_path), line_crop_img)

    skip_reason = modal.detect_label_only_crop(line_crop_img)
    formula_raw = ""
    formula_parse_error = ""
    formula_resolved = {"line_label": "", "formula_text": "", "confidence": ""}
    justification_raw = ""
    justification_parse_error = ""
    justification_resolved = {"justification_text": "", "confidence": ""}
    merged = {
        "line_label": "",
        "formula_text": "",
        "justification_text": "",
        "scope_role": "derived_line",
        "confidence": "",
    }
    recovery_applied = ""
    zone_geometry = {
        "split_x": 0,
        "formula_x1": 0,
        "formula_x2": 0,
        "justification_x1": 0,
        "justification_x2": 0,
        "width": int(line_crop_img.shape[1]) if line_crop_img.size else 0,
        "height": int(line_crop_img.shape[0]) if line_crop_img.size else 0,
    }
    formula_crop_name = ""
    justification_crop_name = ""
    tight_justification_crop_name = ""
    tight_justification_geometry: dict[str, int | bool] = {}
    tight_justification_reread: dict[str, Any] | None = None

    if not skip_reason:
        formula_crop, justification_crop, zone_geometry = extract_line_zone_crops(line_crop_img)
        formula_crop_name = artifacts.NODE_LINE_FORMULA_CROP_TEMPLATE.format(node_id=node.id, row_order=row_order)
        justification_crop_name = artifacts.NODE_LINE_JUSTIFICATION_CROP_TEMPLATE.format(node_id=node.id, row_order=row_order)
        formula_crop_path = output_dir / formula_crop_name
        justification_crop_path = output_dir / justification_crop_name
        cv2.imwrite(str(formula_crop_path), formula_crop)
        cv2.imwrite(str(justification_crop_path), justification_crop)

        formula_raw = client.chat_with_image_prompt(formula_crop_path, formula_prompt)
        try:
            formula_parsed = modal.parse_json_object_response(formula_raw)
            formula_resolved = modal_line_zones.normalize_formula_zone_result(formula_parsed)
        except (ValueError, json.JSONDecodeError) as exc:
            formula_parse_error = str(exc)

        justification_raw = client.chat_with_image_prompt(justification_crop_path, justification_prompt)
        try:
            justification_parsed = modal.parse_json_object_response(justification_raw)
            justification_resolved = modal_line_zones.normalize_justification_zone_result(justification_parsed)
        except (ValueError, json.JSONDecodeError) as exc:
            justification_parse_error = str(exc)

        merged = modal_line_zones.merge_line_zone_results(
            formula_result=formula_resolved,
            justification_result=justification_resolved,
            is_root_node=str(node.id) == "ROOT",
            is_first_visible_row_of_nonroot=(str(node.id) != "ROOT" and row_order == 1),
        )
        if verifier_client is not None and _should_tight_reread_justification(
            merged_entry=merged,
            justification_result=justification_resolved,
            justification_crop=justification_crop,
        ):
            tight_justification_crop, tight_justification_geometry = extract_tight_justification_crop(justification_crop)
            if bool(tight_justification_geometry.get("applied")) and tight_justification_crop.size != 0:
                tight_justification_crop_name = artifacts.NODE_LINE_TIGHT_JUSTIFICATION_CROP_TEMPLATE.format(
                    node_id=node.id,
                    row_order=row_order,
                )
                tight_justification_crop_path = output_dir / tight_justification_crop_name
                cv2.imwrite(str(tight_justification_crop_path), tight_justification_crop)
                reread_client = verifier_client or client
                tight_raw = reread_client.chat_with_image_prompt(tight_justification_crop_path, justification_prompt)
                tight_parse_error = ""
                tight_resolved = {"justification_text": "", "confidence": ""}
                try:
                    tight_parsed = modal.parse_json_object_response(tight_raw)
                    tight_resolved = modal_line_zones.normalize_justification_zone_result(tight_parsed)
                except (ValueError, json.JSONDecodeError) as exc:
                    tight_parse_error = str(exc)
                    tight_parsed = {}
                tight_candidate_text = modal.normalize_justification_text(tight_resolved.get("justification_text") or "")
                applied = bool(tight_candidate_text)
                if applied:
                    justification_resolved = tight_resolved
                    merged = modal_line_zones.merge_line_zone_results(
                        formula_result=formula_resolved,
                        justification_result=justification_resolved,
                        is_root_node=str(node.id) == "ROOT",
                        is_first_visible_row_of_nonroot=(str(node.id) != "ROOT" and row_order == 1),
                    )
                tight_justification_reread = {
                    "crop_filename": tight_justification_crop_name,
                    "geometry": tight_justification_geometry,
                    "raw": tight_raw,
                    "parsed": tight_parsed,
                    "resolved": tight_resolved,
                    "parse_error": tight_parse_error,
                    "applied": applied,
                }
        merged, recovery_applied = modal.recover_line_entry_from_crop(
            line_crop_img=line_crop_img,
            parsed_entry=merged,
        )
        if str(node.id) != "ROOT" and row_order == 1:
            strict_open_resolved, strict_open_recovery = modal.recover_first_nonroot_strict_open_from_crop(
                line_crop_img=line_crop_img,
                parsed_entry=merged,
            )
            if strict_open_recovery:
                merged = strict_open_resolved
                recovery_applied = strict_open_recovery

    payload = {
        "node_id": node.id,
        "row_order": row_order,
        "band_index": row_record.get("band_index"),
        "y1": row_record.get("y1"),
        "y2": row_record.get("y2"),
        "mode": "formula_justification_separation",
        "skip_reason": skip_reason,
        "zone_geometry": zone_geometry,
        "formula_crop_filename": formula_crop_name,
        "justification_crop_filename": justification_crop_name,
        "tight_justification_crop_filename": tight_justification_crop_name,
        "tight_justification_geometry": tight_justification_geometry,
        "formula_raw_response": formula_raw,
        "formula_parse_error": formula_parse_error,
        "formula_resolved": formula_resolved,
        "justification_raw_response": justification_raw,
        "justification_parse_error": justification_parse_error,
        "justification_resolved": justification_resolved,
        "resolved": merged,
        "recovery_applied": recovery_applied,
    }
    if tight_justification_reread is not None:
        payload["tight_justification_reread"] = tight_justification_reread
    (output_dir / artifacts.NODE_LINE_RESPONSE_TEMPLATE.format(node_id=node.id, row_order=row_order)).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return merged


def _choose_line_zone_split_x(crop_img: np.ndarray) -> int:
    if crop_img.size == 0:
        return 0

    gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    dark = (gray < 185).astype(np.uint8)
    if dark.size == 0:
        return max(0, int(crop_img.shape[1] * 0.6))

    col_counts = dark.sum(axis=0).astype(np.float32)
    if col_counts.size == 0:
        return max(0, int(crop_img.shape[1] * 0.6))

    kernel_width = max(9, min(31, int(crop_img.shape[1] * 0.06)))
    if kernel_width % 2 == 0:
        kernel_width += 1
    kernel = np.ones(kernel_width, dtype=np.float32) / float(kernel_width)
    smoothed = np.convolve(col_counts, kernel, mode="same")

    total_ink = float(col_counts.sum())
    img_w = int(crop_img.shape[1])
    search_start = max(12, int(img_w * 0.18))
    search_end = min(img_w - 12, int(img_w * 0.72))
    preferred = float(img_w * 0.42)

    prefix = np.cumsum(col_counts)
    best_x = max(search_start, min(search_end, int(preferred)))
    best_score = float("inf")

    for split_x in range(search_start, max(search_start + 1, search_end)):
        left_ink = float(prefix[split_x - 1]) if split_x > 0 else 0.0
        right_ink = total_ink - left_ink
        if left_ink < 10.0 or right_ink < 10.0:
            continue
        score = float(smoothed[split_x]) + (abs(float(split_x) - preferred) * 0.03)
        if score < best_score:
            best_score = score
            best_x = split_x

    min_split = max(18, int(img_w * 0.22))
    max_split = max(min_split + 1, int(img_w * 0.62))
    return max(min_split, min(max_split, best_x))


def extract_tight_justification_crop(justification_crop: np.ndarray) -> tuple[np.ndarray, dict[str, int | bool]]:
    """Trim obvious spill above a justification band while preserving the full citation text."""

    if justification_crop.size == 0:
        empty = justification_crop.copy()
        return empty, {
            "x1": 0,
            "y1": 0,
            "x2": 0,
            "y2": 0,
            "width": 0,
            "height": 0,
            "applied": False,
        }

    img_h, img_w = justification_crop.shape[:2]
    gray = cv2.cvtColor(justification_crop, cv2.COLOR_BGR2GRAY)
    dark = (gray < 185).astype(np.uint8)
    if int(dark.sum()) <= 0:
        return justification_crop.copy(), {
            "x1": 0,
            "y1": 0,
            "x2": int(img_w),
            "y2": int(img_h),
            "width": int(img_w),
            "height": int(img_h),
            "applied": False,
        }

    row_counts = dark.sum(axis=1).astype(np.int32)
    max_row_count = int(row_counts.max())
    row_threshold = max(2, int(max_row_count * 0.3))
    primary_run = _choose_primary_ink_run(row_counts, row_threshold)
    if primary_run is None:
        ys, xs = np.where(dark > 0)
        y1 = int(ys.min())
        y2 = int(ys.max()) + 1
        x1 = int(xs.min())
        x2 = int(xs.max()) + 1
    else:
        run_y1, run_y2 = primary_run
        band = dark[run_y1:run_y2, :]
        ys, xs = np.where(band > 0)
        if ys.size == 0 or xs.size == 0:
            return justification_crop.copy(), {
                "x1": 0,
                "y1": 0,
                "x2": int(img_w),
                "y2": int(img_h),
                "width": int(img_w),
                "height": int(img_h),
                "applied": False,
            }
        y1 = int(run_y1 + ys.min())
        y2 = int(run_y1 + ys.max()) + 1
        x1 = int(xs.min())
        x2 = int(xs.max()) + 1

    pad_left = 3
    pad_right = 10
    pad_y = 3
    x1 = max(0, x1 - pad_left)
    x2 = min(img_w, x2 + pad_right)
    y1 = max(0, y1 - pad_y)
    y2 = min(img_h, y2 + pad_y)
    if x2 <= x1 or y2 <= y1:
        return justification_crop.copy(), {
            "x1": 0,
            "y1": 0,
            "x2": int(img_w),
            "y2": int(img_h),
            "width": int(img_w),
            "height": int(img_h),
            "applied": False,
        }

    tight_crop = justification_crop[y1:y2, x1:x2]
    applied = x1 > 0 or y1 > 0 or x2 < img_w or y2 < img_h
    return tight_crop, {
        "x1": int(x1),
        "y1": int(y1),
        "x2": int(x2),
        "y2": int(y2),
        "width": int(img_w),
        "height": int(img_h),
        "applied": bool(applied),
    }


def _should_tight_reread_justification(
    *,
    merged_entry: Mapping[str, Any],
    justification_result: Mapping[str, Any],
    justification_crop: np.ndarray,
) -> bool:
    text = modal.normalize_justification_text(merged_entry.get("justification_text") or "")
    if text:
        return False
    del justification_result
    return _crop_has_visible_ink(justification_crop)


def _crop_has_visible_ink(crop_img: np.ndarray) -> bool:
    if crop_img.size == 0:
        return False
    gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    return int((gray < 185).sum()) > 12


def _choose_primary_ink_run(row_counts: np.ndarray, threshold: int) -> tuple[int, int] | None:
    active_rows = row_counts >= threshold
    base_runs: list[tuple[int, int, int]] = []
    start = -1
    for index, is_active in enumerate(active_rows.tolist()):
        if is_active and start < 0:
            start = index
        elif not is_active and start >= 0:
            base_runs.append((start, index, int(row_counts[start:index].sum())))
            start = -1
    if start >= 0:
        base_runs.append((start, int(len(row_counts)), int(row_counts[start:].sum())))
    if not base_runs:
        return None

    runs: list[tuple[int, int, int]] = []
    merge_gap = 3
    for run_start, run_end, run_sum in base_runs:
        if not runs:
            runs.append((run_start, run_end, run_sum))
            continue
        prev_start, prev_end, prev_sum = runs[-1]
        if run_start - prev_end <= merge_gap:
            runs[-1] = (prev_start, run_end, prev_sum + run_sum)
        else:
            runs.append((run_start, run_end, run_sum))

    start, end, _ = max(runs, key=lambda item: (item[2], item[0]))
    return (int(start), int(end))
