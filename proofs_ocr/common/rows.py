"""Shared row-band detection and row-record generation helpers.

These helpers cover the geometry-heavy part of line discovery inside one proof
scope crop. Domain-specific prompt wording and OCR interpretation stay outside
this module.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple

import cv2
import numpy as np

Band = Tuple[int, int]
Rect = Tuple[int, int, int, int]
RowRecord = Dict[str, Any]


def band_height(band: Band) -> int:
    return max(0, int(band[1]) - int(band[0]) + 1)


def adaptive_band_padding(*, band: Band, img_h: int) -> int:
    height = band_height(band)
    if height <= 10:
        return min(24, max(16, img_h // 12))
    if height <= 14:
        return min(20, max(14, img_h // 14))
    if height <= 20:
        return min(16, max(12, img_h // 16))
    return 12


def band_gap(upper: Band, lower: Band) -> int:
    return int(lower[0]) - int(upper[1]) - 1


def merge_fragment_bands(bands: List[Band]) -> List[Band]:
    if not bands:
        return []

    merged = [tuple(map(int, band)) for band in sorted(bands)]
    changed = True
    while changed and len(merged) > 1:
        changed = False
        for idx, band in enumerate(list(merged)):
            height = band_height(band)
            if height > 12:
                continue

            prev_gap = band_gap(merged[idx - 1], band) if idx > 0 else None
            next_gap = band_gap(band, merged[idx + 1]) if idx + 1 < len(merged) else None
            candidate_idx: int | None = None
            candidate_gap: int | None = None

            if prev_gap is not None and prev_gap <= 18:
                candidate_idx = idx - 1
                candidate_gap = prev_gap
            if next_gap is not None and next_gap <= 18 and (candidate_gap is None or next_gap < candidate_gap):
                candidate_idx = idx + 1
                candidate_gap = next_gap
            if candidate_idx is None:
                continue

            other = merged[candidate_idx]
            new_band = (min(band[0], other[0]), max(band[1], other[1]))
            low_idx = min(idx, candidate_idx)
            high_idx = max(idx, candidate_idx)
            merged[low_idx] = new_band
            del merged[high_idx]
            changed = True
            break
    return merged


def detect_text_row_bands(crop_img: np.ndarray) -> List[Band]:
    gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    img_h, img_w = gray.shape[:2]
    formula_w = min(img_w, max(80, int(img_w * 0.50)))
    focused = gray[:, :formula_w]

    dark_mask = (focused < 185).astype(np.uint8)
    if dark_mask.size == 0:
        return []

    # Remove long proof bars and scope underlines so they do not merge adjacent rows.
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(20, int(formula_w * 0.18)), 1))
    horizontal_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, horizontal_kernel)
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(20, int(img_h * 0.35))))
    vertical_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, vertical_kernel)
    text_mask = dark_mask.copy()
    text_mask[horizontal_mask > 0] = 0
    text_mask[vertical_mask > 0] = 0

    row_counts = text_mask.sum(axis=1).astype(np.float32)
    if row_counts.size == 0:
        return []

    kernel = np.ones(9, dtype=np.float32) / 9.0
    smoothed = np.convolve(row_counts, kernel, mode="same")
    threshold = max(6.0, float(smoothed.max()) * 0.16)
    active_rows = np.where(smoothed >= threshold)[0]
    if active_rows.size == 0:
        return []

    bands: List[Band] = []
    start = int(active_rows[0])
    prev = int(active_rows[0])
    for idx in active_rows[1:]:
        idx = int(idx)
        if idx <= prev + 4:
            prev = idx
            continue
        bands.append((start, prev))
        start = idx
        prev = idx
    bands.append((start, prev))

    merged: List[Band] = []
    for band_start, band_end in bands:
        if merged and band_start <= merged[-1][1] + 10:
            merged[-1] = (merged[-1][0], band_end)
        else:
            merged.append((band_start, band_end))

    split_bands: List[Band] = []
    valley_threshold = max(2.0, threshold * 0.45)
    min_band_height = 6
    min_gap_height = 4
    for band_start, band_end in merged:
        if band_end - band_start < 22:
            split_bands.append((band_start, band_end))
            continue

        current_start = band_start
        gap_start: int | None = None
        gap_end: int | None = None
        for row in range(band_start, band_end + 1):
            if smoothed[row] < valley_threshold:
                if gap_start is None:
                    gap_start = row
                gap_end = row
                continue

            if gap_start is not None and gap_end is not None:
                gap_height = gap_end - gap_start + 1
                left_height = gap_start - current_start
                right_height = band_end - gap_end
                if gap_height >= min_gap_height and left_height >= min_band_height and right_height >= min_band_height:
                    split_bands.append((current_start, gap_start - 1))
                    current_start = gap_end + 1
                gap_start = None
                gap_end = None

        if band_end - current_start + 1 >= min_band_height:
            split_bands.append((current_start, band_end))

    normalized = [(start, end) for start, end in split_bands if end - start >= min_band_height]
    if not normalized:
        normalized = [(start, end) for start, end in merged if end - start >= 6]

    component_bands: List[Band] = []
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(text_mask, 8)
    raw_component_bands: List[Band] = []
    for label_index in range(1, num_labels):
        area = int(stats[label_index, cv2.CC_STAT_AREA])
        if area < 10:
            continue
        top = int(stats[label_index, cv2.CC_STAT_TOP])
        height = int(stats[label_index, cv2.CC_STAT_HEIGHT])
        raw_component_bands.append((top, top + height - 1))

    for band_start, band_end in sorted(raw_component_bands):
        if component_bands and band_start <= component_bands[-1][1] + 8:
            component_bands[-1] = (component_bands[-1][0], max(component_bands[-1][1], band_end))
        else:
            component_bands.append((band_start, band_end))
    component_bands = [(start, end) for start, end in component_bands if end - start >= 4]

    if len(component_bands) > len(normalized):
        return component_bands
    return normalized


def extract_line_row_crop(
    *,
    node_crop_img: np.ndarray,
    y1: int,
    y2: int,
    suppress_left_scope_bar: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    if node_crop_img.size == 0:
        return node_crop_img
    img_h, img_w = node_crop_img.shape[:2]
    pad_y = adaptive_band_padding(band=(int(y1), int(y2)), img_h=img_h)
    top = max(0, int(y1) - pad_y)
    bottom = min(img_h, int(y2) + pad_y)
    crop = node_crop_img[top:bottom, :img_w]
    if crop.size == 0:
        return node_crop_img
    return suppress_left_scope_bar(crop)


def band_has_text_outside_vertical_window(
    node_crop_img: np.ndarray,
    band: Band,
    *,
    window_y1: int,
    window_y2: int,
    suppress_left_scope_bar: Callable[[np.ndarray], np.ndarray],
) -> bool:
    if node_crop_img.size == 0:
        return False

    img_h, img_w = node_crop_img.shape[:2]
    pad_y = adaptive_band_padding(band=band, img_h=img_h)
    top = max(0, int(band[0]) - pad_y)
    bottom = min(img_h, int(band[1]) + pad_y)
    if bottom <= top:
        return False

    crop = suppress_left_scope_bar(node_crop_img[top:bottom, :img_w])
    if crop.size == 0:
        return False

    local_window_y1 = max(0, min(crop.shape[0], int(window_y1) - top))
    local_window_y2 = max(0, min(crop.shape[0], int(window_y2) - top))
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    left_ignore = min(20, max(12, gray.shape[1] // 8))
    right_limit = min(gray.shape[1], max(80, int(gray.shape[1] * 0.68)))
    if right_limit <= left_ignore:
        return False

    focused = gray[:, left_ignore:right_limit]
    mask = np.ones(focused.shape[:2], dtype=np.uint8)
    if local_window_y2 > local_window_y1:
        mask[local_window_y1:local_window_y2, :] = 0

    dark = ((focused < 185) & (mask > 0)).astype(np.uint8)
    return int(dark.sum()) >= 14


def band_matches_child_token_artifact(
    *,
    band: Band,
    node: Any,
    node_crop_img: np.ndarray,
    token_rect_for_child: Callable[..., Rect],
    band_has_text_outside_vertical_window_fn: Callable[..., bool],
    center_tolerance: int = 2,
    overlap_ratio_threshold: float = 0.45,
) -> bool:
    current_band_height = band_height(band)
    if current_band_height <= 0 or not getattr(node, "children", None):
        return False

    band_center = (int(band[0]) + int(band[1])) / 2.0
    for child in node.children:
        _, token_top, _, token_bottom = token_rect_for_child(node=node, child=child, img_shape=node_crop_img.shape)
        overlap_top = max(int(band[0]), int(token_top))
        overlap_bottom = min(int(band[1]), int(token_bottom))
        overlap = overlap_bottom - overlap_top + 1
        overlap_ratio = (overlap / current_band_height) if overlap > 0 else 0.0
        center_inside = (int(token_top) - center_tolerance) <= band_center <= (int(token_bottom) + center_tolerance)
        if center_inside:
            return True
        if overlap_ratio < overlap_ratio_threshold:
            continue
        if not band_has_text_outside_vertical_window_fn(
            node_crop_img,
            band,
            window_y1=int(token_top),
            window_y2=int(token_bottom) + 1,
        ):
            return True
    return False


def filter_bands_near_child_tokens(
    *,
    bands: List[Band],
    node: Any,
    node_crop_img: np.ndarray,
    band_matches_child_token_artifact_fn: Callable[..., bool],
) -> List[Band]:
    if not bands or not getattr(node, "children", None):
        return bands

    filtered: List[Band] = []
    for band in bands:
        if not band_matches_child_token_artifact_fn(
            band=band,
            node=node,
            node_crop_img=node_crop_img,
        ):
            filtered.append(band)
    return filtered


def row_overlaps_child_token(
    *,
    node: Any,
    row_record: RowRecord,
    node_crop_img: np.ndarray,
    band_matches_child_token_artifact_fn: Callable[..., bool],
) -> bool:
    band = (int(row_record.get("y1", 0)), int(row_record.get("y2", 0)))
    return band_matches_child_token_artifact_fn(
        band=band,
        node=node,
        node_crop_img=node_crop_img,
    )


def band_has_textlike_ink(
    node_crop_img: np.ndarray,
    band: Band,
    *,
    extract_band_crop_fn: Callable[..., np.ndarray],
) -> bool:
    band_crop = extract_band_crop_fn(node_crop_img=node_crop_img, bands=[band], band_index=0)
    if band_crop.size == 0:
        return False
    gray = cv2.cvtColor(band_crop, cv2.COLOR_BGR2GRAY)
    left_ignore = min(20, max(12, gray.shape[1] // 8))
    focused = gray[:, left_ignore : min(gray.shape[1], max(80, int(gray.shape[1] * 0.60)))]
    dark = (focused < 185).astype(np.uint8)
    if dark.size == 0:
        return False
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(18, int(focused.shape[1] * 0.2)), 1))
    horizontal_mask = cv2.morphologyEx(dark, cv2.MORPH_OPEN, horizontal_kernel)
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(16, int(focused.shape[0] * 0.75))))
    vertical_mask = cv2.morphologyEx(dark, cv2.MORPH_OPEN, vertical_kernel)
    text_mask = dark.copy()
    text_mask[horizontal_mask > 0] = 0
    text_mask[vertical_mask > 0] = 0
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(text_mask, 8)
    tall_components = 0
    total_area = 0
    for label_index in range(1, num_labels):
        area = int(stats[label_index, cv2.CC_STAT_AREA])
        width = int(stats[label_index, cv2.CC_STAT_WIDTH])
        height = int(stats[label_index, cv2.CC_STAT_HEIGHT])
        if area < 8:
            continue
        total_area += area
        if height >= 6 and width >= 2:
            tall_components += 1
    current_band_height = band_height(band)
    if current_band_height < 12 and total_area < 36:
        return False
    return tall_components >= 1 and total_area >= 18


def segment_node_rows(
    *,
    node: Any,
    node_crop_img: np.ndarray,
    suppress_left_scope_bar: Callable[[np.ndarray], np.ndarray],
    detect_text_row_bands_fn: Callable[[np.ndarray], List[Band]],
    band_has_textlike_ink_fn: Callable[[np.ndarray, Band], bool],
    merge_fragment_bands_fn: Callable[[List[Band]], List[Band]],
    filter_bands_near_child_tokens_fn: Callable[..., List[Band]],
    token_rect_for_child_fn: Callable[..., Rect],
    write_row_artifacts: Callable[[List[RowRecord]], None] | None = None,
    write_band_debug_artifacts: Callable[[Dict[str, Any]], None] | None = None,
) -> List[RowRecord]:
    cleaned = suppress_left_scope_bar(node_crop_img)
    detected_bands = detect_text_row_bands_fn(cleaned)
    textlike_bands: List[Band] = []
    raw_detected_bands: List[Dict[str, Any]] = []
    for band in detected_bands:
        keep = bool(band_has_textlike_ink_fn(cleaned, band))
        raw_detected_bands.append(
            {
                "y1": int(band[0]),
                "y2": int(band[1]),
                "height": int(band_height(band)),
                "kept_after_textlike_filter": keep,
                "rejection_reason": "" if keep else "band_has_textlike_ink=false",
            }
        )
        if keep:
            textlike_bands.append(band)

    merged_bands = merge_fragment_bands_fn(textlike_bands)
    bands = filter_bands_near_child_tokens_fn(bands=merged_bands, node=node, node_crop_img=cleaned)

    if write_band_debug_artifacts is not None:
        merged_records = [{"y1": int(band[0]), "y2": int(band[1]), "height": int(band_height(band))} for band in merged_bands]
        final_records = [{"y1": int(band[0]), "y2": int(band[1]), "height": int(band_height(band))} for band in bands]
        token_filtered_records = [
            {
                "y1": int(band[0]),
                "y2": int(band[1]),
                "height": int(band_height(band)),
                "rejection_reason": "child_token_overlap",
            }
            for band in merged_bands
            if band not in bands
        ]
        write_band_debug_artifacts(
            {
                "raw_detected_bands": raw_detected_bands,
                "merged_bands": merged_records,
                "final_bands": final_records,
                "rejected_after_merge": token_filtered_records,
            }
        )

    row_records: List[RowRecord] = []
    for band_index, (y1, y2) in enumerate(bands):
        row_records.append(
            {
                "kind": "line",
                "band_index": band_index,
                "y1": int(y1),
                "y2": int(y2),
                "center_y": float((y1 + y2) / 2.0),
            }
        )
    img_h = node_crop_img.shape[0]
    for child in sorted(getattr(node, "children", []), key=lambda c: c.box[1]):
        _, token_top, _, token_bottom = token_rect_for_child_fn(node=node, child=child, img_shape=node_crop_img.shape)
        center_y = float((token_top + token_bottom) / 2.0)
        row_records.append(
            {
                "kind": "subproof_token",
                "child_id": child.id,
                "y1": int(max(0, token_top)),
                "y2": int(min(img_h - 1, token_bottom)),
                "center_y": center_y,
            }
        )

    row_records.sort(
        key=lambda record: (
            float(record.get("center_y", 0.0)),
            0 if str(record.get("kind") or "line") == "line" else 1,
            int(record.get("band_index", 0)),
        )
    )
    for row_order, record in enumerate(row_records, start=1):
        record["row_order"] = row_order

    if write_row_artifacts is not None:
        write_row_artifacts(row_records)
    return row_records
