"""Shared row-band detection and row-record generation helpers.

These helpers cover the geometry-heavy part of line discovery inside one proof
scope crop. Domain-specific prompt wording and OCR interpretation stay outside
this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple

import cv2
import numpy as np

Band = Tuple[int, int]
Rect = Tuple[int, int, int, int]
RowRecord = Dict[str, Any]


@dataclass(frozen=True)
class _CompactComponent:
    top: int
    bottom: int
    left: int
    right: int
    width: int
    height: int
    area: int


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
            if height > 11:
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


def _projection_bands_from_dark_mask(dark_mask: np.ndarray) -> List[Band]:
    if dark_mask.size == 0:
        return []

    img_h, img_w = dark_mask.shape[:2]
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(20, int(img_w * 0.18)), 1))
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

    return component_bands if len(component_bands) > len(normalized) else normalized


def _split_first_band_on_opener_underline(gray: np.ndarray, bands: List[Band]) -> List[Band]:
    if not bands:
        return bands

    first_start, first_end = bands[0]
    if first_start > 30 or band_height((first_start, first_end)) < 26:
        return bands

    img_h, img_w = gray.shape[:2]
    left_w = min(img_w, max(120, int(img_w * 0.38)))
    left_window = gray[:, :left_w]
    dark = (left_window < 185).astype(np.uint8)
    kernel_w = min(100, max(20, int(left_w * 0.20)))
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, 1))
    horizontal_mask = cv2.morphologyEx(dark, cv2.MORPH_OPEN, horizontal_kernel)
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(horizontal_mask, 8)

    underline: tuple[int, int, int, int] | None = None
    for label_index in range(1, num_labels):
        area = int(stats[label_index, cv2.CC_STAT_AREA])
        x = int(stats[label_index, cv2.CC_STAT_LEFT])
        y = int(stats[label_index, cv2.CC_STAT_TOP])
        w = int(stats[label_index, cv2.CC_STAT_WIDTH])
        h = int(stats[label_index, cv2.CC_STAT_HEIGHT])
        if area < 40 or h > 5:
            continue
        if y < first_start + 4 or y > first_end - 8:
            continue
        if w < 35 or w > int(left_w * 0.85):
            continue
        if underline is None or w > underline[2]:
            underline = (x, y, w, h)

    if underline is None:
        return bands

    _, line_y, _, line_h = underline
    top_end = line_y - 2
    bottom_start = line_y + line_h + 4
    if band_height((first_start, top_end)) < 6 or band_height((bottom_start, first_end)) < 6:
        return bands

    below_dark = int(dark[bottom_start : first_end + 1, :].sum())
    above_dark = int(dark[first_start : top_end + 1, :].sum())
    if above_dark < 20 or below_dark < 20:
        return bands

    return [(first_start, top_end), (bottom_start, first_end), *bands[1:]]


def _recover_sparse_lower_compact_bands(
    *,
    dark_mask: np.ndarray,
    candidate_bands: List[Band],
) -> List[Band]:
    if dark_mask.size == 0:
        return []

    img_h, img_w = dark_mask.shape[:2]
    top_limit = min(img_h - 1, max(48, img_h // 2))
    left_limit = min(img_w - 1, max(90, int(img_w * 0.28)))
    min_left = max(8, img_w // 40)
    max_component_span = max(18, int(img_h * 0.16))
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(dark_mask, 8)
    components: List[_CompactComponent] = []
    for label_index in range(1, num_labels):
        area = int(stats[label_index, cv2.CC_STAT_AREA])
        left = int(stats[label_index, cv2.CC_STAT_LEFT])
        top = int(stats[label_index, cv2.CC_STAT_TOP])
        width = int(stats[label_index, cv2.CC_STAT_WIDTH])
        height = int(stats[label_index, cv2.CC_STAT_HEIGHT])
        if area < 3 or area > 24:
            continue
        if top <= top_limit:
            continue
        if left < min_left or left > left_limit:
            continue
        if width > max_component_span or height > max_component_span:
            continue
        components.append(
            _CompactComponent(
                top=top,
                bottom=top + height - 1,
                left=left,
                right=left + width - 1,
                width=width,
                height=height,
                area=area,
            )
        )

    if len(components) < 2:
        return []

    components.sort(key=lambda item: (item.top, item.left))
    grouped: List[List[_CompactComponent]] = []
    current_group = [components[0]]
    for component in components[1:]:
        current_bottom = max(item.bottom for item in current_group)
        if component.top <= current_bottom + 6:
            current_group.append(component)
        else:
            grouped.append(current_group)
            current_group = [component]
    grouped.append(current_group)

    rescued: List[Band] = []
    for group in grouped:
        if len(group) < 2:
            continue
        band_top = min(item.top for item in group)
        band_bottom = max(item.bottom for item in group)
        band_left = min(item.left for item in group)
        band_right = max(item.right for item in group)
        band_width = band_right - band_left + 1
        band_height_value = band_bottom - band_top + 1
        total_area = sum(item.area for item in group)
        has_horizontal = any(item.width >= 4 and item.height <= 3 for item in group)
        has_vertical = any(item.height >= 4 and item.width <= 3 for item in group)
        if total_area < 10:
            continue
        if band_height_value < 5 or band_height_value > 18:
            continue
        if band_width > 28:
            continue
        if not (has_horizontal and has_vertical):
            continue
        if any(not (band_bottom < existing_top - 3 or band_top > existing_bottom + 3) for existing_top, existing_bottom in candidate_bands):
            continue
        rescued.append((band_top, band_bottom))
    return rescued


def _has_lower_half_split_advantage(
    *,
    base_bands: List[Band],
    alt_bands: List[Band],
    img_h: int,
) -> bool:
    if len(alt_bands) < 2 or not base_bands:
        return False

    lower_half_y = int(img_h * 0.55)
    max_split_gap = max(14, img_h // 10)
    tolerance = 4
    for upper_alt, lower_alt in zip(alt_bands, alt_bands[1:]):
        if upper_alt[0] < lower_half_y:
            continue
        if band_gap(upper_alt, lower_alt) > max_split_gap:
            continue

        spanning_base: List[Band] = []
        for base_band in base_bands:
            overlaps_upper = not (
                base_band[1] < upper_alt[0] - tolerance or base_band[0] > upper_alt[1] + tolerance
            )
            overlaps_lower = not (
                base_band[1] < lower_alt[0] - tolerance or base_band[0] > lower_alt[1] + tolerance
            )
            if overlaps_upper and overlaps_lower:
                spanning_base.append(base_band)
        if len(spanning_base) != 1:
            continue

        base_band = spanning_base[0]
        split_span_height = lower_alt[1] - upper_alt[0] + 1
        if (
            base_band[0] <= upper_alt[0] + tolerance
            and base_band[1] >= lower_alt[0] - tolerance
            and split_span_height >= band_height(base_band) + max(6, max_split_gap // 2)
        ):
            return True
    return False


def _recover_split_post_child_bands_from_right_window(
    *,
    bands: List[Band],
    node: Any,
    node_crop_img: np.ndarray,
    token_rect_for_child_fn: Callable[..., Rect],
) -> List[Band]:
    if not bands or not getattr(node, "children", None):
        return bands

    img_h, img_w = node_crop_img.shape[:2]
    if img_w < 220:
        return bands

    token_bottoms = [
        int(token_rect_for_child_fn(node=node, child=child, img_shape=node_crop_img.shape)[3])
        for child in sorted(getattr(node, "children", []), key=lambda c: c.box[1])
    ]
    if not token_bottoms:
        return bands

    last_token_bottom = max(token_bottoms)
    post_child_bands = [band for band in bands if band[0] > last_token_bottom + 4]
    if len(post_child_bands) != 1:
        return bands

    gray = cv2.cvtColor(node_crop_img, cv2.COLOR_BGR2GRAY)
    formula_w = min(img_w, max(80, int(img_w * 0.50)))
    right_start = min(img_w - 1, max(formula_w, int(img_w * 0.42)))
    right_end = max(right_start + 1, min(img_w, int(img_w * 0.92)))
    right_window = gray[:, right_start:right_end]
    right_dark_mask = (right_window < 185).astype(np.uint8)
    post_child_right_bands = [
        band for band in _projection_bands_from_dark_mask(right_dark_mask) if band[0] > last_token_bottom + 4
    ]
    if len(post_child_right_bands) != 2:
        return bands

    target_band = post_child_bands[0]
    if not _has_lower_half_split_advantage(
        base_bands=[target_band],
        alt_bands=post_child_right_bands,
        img_h=img_h,
    ):
        return bands

    refined: List[Band] = []
    for band in bands:
        if band == target_band:
            refined.extend(post_child_right_bands)
        else:
            refined.append(band)
    return refined


def detect_text_row_bands(
    crop_img: np.ndarray,
    *,
    enable_sparse_lower_compact_band_recovery: bool = False,
) -> List[Band]:
    gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    img_h, img_w = gray.shape[:2]
    formula_w = min(img_w, max(80, int(img_w * 0.50)))
    focused = gray[:, :formula_w]
    dark_mask = (focused < 185).astype(np.uint8)
    candidate_bands = _projection_bands_from_dark_mask(dark_mask)

    if img_w >= 220:
        right_start = min(img_w - 1, max(formula_w, int(img_w * 0.42)))
        right_end = max(right_start + 1, min(img_w, int(img_w * 0.92)))
        right_window = gray[:, right_start:right_end]
        right_dark_mask = (right_window < 185).astype(np.uint8)
        right_candidate_bands = _projection_bands_from_dark_mask(right_dark_mask)
        if len(right_candidate_bands) > len(candidate_bands):
            candidate_bands = right_candidate_bands

    candidate_bands = _split_first_band_on_opener_underline(gray, candidate_bands)
    if enable_sparse_lower_compact_band_recovery:
        candidate_bands = sorted(
            candidate_bands + _recover_sparse_lower_compact_bands(dark_mask=dark_mask, candidate_bands=candidate_bands)
        )

    recovered_compact_bands: List[Band] = []
    num_raw_labels, _, raw_stats, _ = cv2.connectedComponentsWithStats(dark_mask, 8)
    top_limit = min(img_h - 1, max(48, img_h // 2))
    left_limit = min(focused.shape[1], max(90, int(formula_w * 0.22)))
    max_component_span = max(28, int(img_h * 0.28))
    for label_index in range(1, num_raw_labels):
        area = int(raw_stats[label_index, cv2.CC_STAT_AREA])
        if area < 36:
            continue
        left = int(raw_stats[label_index, cv2.CC_STAT_LEFT])
        top = int(raw_stats[label_index, cv2.CC_STAT_TOP])
        width = int(raw_stats[label_index, cv2.CC_STAT_WIDTH])
        height = int(raw_stats[label_index, cv2.CC_STAT_HEIGHT])
        if top > top_limit or left > left_limit:
            continue
        if width < 8 or height < 8:
            continue
        if width > max_component_span or height > max_component_span:
            continue
        bottom = top + height - 1
        if any(not (bottom < band_start - 3 or top > band_end + 3) for band_start, band_end in candidate_bands):
            continue
        recovered_compact_bands.append((top, bottom))

    combined = sorted(candidate_bands + recovered_compact_bands)
    if not combined:
        return []

    merged_combined: List[Band] = []
    for band_start, band_end in combined:
        if merged_combined and band_start <= merged_combined[-1][1] + 6:
            merged_combined[-1] = (merged_combined[-1][0], max(merged_combined[-1][1], band_end))
        else:
            merged_combined.append((band_start, band_end))
    return merged_combined


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
    bands = _recover_split_post_child_bands_from_right_window(
        bands=bands,
        node=node,
        node_crop_img=cleaned,
        token_rect_for_child_fn=token_rect_for_child_fn,
    )

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
