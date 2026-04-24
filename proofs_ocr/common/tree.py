"""Shared subproof bar detection and proof-tree construction helpers.

These functions are copied from the current stable modal OCR implementation to
seed a cleaner shared library. They are not yet wired into a new pipeline.
"""

from __future__ import annotations

from typing import List

import cv2
import numpy as np

from .types import Box, SubproofNode


def _split_bars_on_gaps(
    img: np.ndarray, boxes: List[Box]
) -> List[Box]:
    """Post-process merged bars: split any bar with a significant vertical gap
    in its actual pixel data, indicating two separate subproofs that were
    erroneously merged because they sit at the same X position.

    Uses the same adaptive-threshold + morphological pipeline as bar detection
    so that the gap analysis is done on the same thick, smeared representation
    that caused the merge in the first place.
    """

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_h, img_w = gray.shape[:2]

    # Reproduce the same binary pipeline used by detect_vertical_bars.
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 91, 15,
    )
    dilate_kernel = np.ones((1, 6), np.uint8)
    dilated = cv2.dilate(binary, dilate_kernel, iterations=1)
    vert_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 80))
    vert_mask = cv2.morphologyEx(dilated, cv2.MORPH_OPEN, vert_kern)

    result: List[Box] = []
    for x, y, w, h in boxes:
        # A bar must be tall enough to even consider splitting.
        if h < 60:
            result.append((x, y, w, h))
            continue

        # Extract the morphological mask within the bar's bounding box,
        # with a small horizontal margin.
        x1 = max(0, x - 4)
        x2 = min(img_w, x + w + 4)
        y1 = max(0, y)
        y2 = min(img_h, y + h)
        roi_mask = vert_mask[y1:y2, x1:x2]
        if roi_mask.size == 0:
            result.append((x, y, w, h))
            continue

        row_counts = (roi_mask > 0).sum(axis=1)
        min_row_count = max(1, int(0.15 * (x2 - x1)))

        active_rows = [idx for idx, count in enumerate(row_counts) if count >= min_row_count]
        if len(active_rows) < 2:
            result.append((x, y, w, h))
            continue

        # Find contiguous segments of active rows.
        segments: List[tuple[int, int]] = []
        seg_start = active_rows[0]
        prev = active_rows[0]
        for idx in active_rows[1:]:
            if idx == prev + 1:
                prev = idx
                continue
            segments.append((seg_start, prev))
            seg_start = idx
            prev = idx
        segments.append((seg_start, prev))

        if len(segments) <= 1:
            result.append((x, y, w, h))
            continue

        # Only split on gaps that are proportionally significant.
        gap_threshold = max(15, int(h * 0.08))

        split_boxes: List[Box] = []
        for i, (seg_s, seg_e) in enumerate(segments):
            seg_h = seg_e - seg_s + 1
            if seg_h < 20:
                continue
            if i == 0:
                split_boxes.append((x, y1 + seg_s, w, seg_h))
            else:
                prev_seg_end = segments[i - 1][1]
                gap = seg_s - prev_seg_end
                if gap >= gap_threshold:
                    split_boxes.append((x, y1 + seg_s, w, seg_h))
                else:
                    # Gap too small — merge back with the previous split box.
                    if split_boxes:
                        last = split_boxes[-1]
                        new_y = last[1]
                        new_h = (y1 + seg_e + 1) - new_y
                        split_boxes[-1] = (last[0], new_y, last[2], new_h)
                    else:
                        split_boxes.append((x, y1 + seg_s, w, seg_h))

        if len(split_boxes) <= 1:
            result.append((x, y, w, h))
        else:
            result.extend(split_boxes)

    return result


def refine_vertical_bar_box(img: np.ndarray, box: Box) -> Box:
    x, y, w, h = box
    img_h, img_w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(img_w, x + w)
    y2 = min(img_h, y + h)
    roi = gray[y1:y2, x1:x2]
    if roi.size == 0:
        return box

    dark_mask = roi < 140
    col_counts = dark_mask.sum(axis=0)
    min_count = max(20, int(0.2 * roi.shape[0]))
    active_cols = [idx for idx, count in enumerate(col_counts) if count >= min_count]
    if not active_cols:
        return box

    groups: List[tuple[int, int]] = []
    start = active_cols[0]
    prev = active_cols[0]
    for idx in active_cols[1:]:
        if idx == prev + 1:
            prev = idx
            continue
        groups.append((start, prev))
        start = idx
        prev = idx
    groups.append((start, prev))

    best_left, best_right = max(
        groups,
        key=lambda group: (group[1], int(col_counts[group[0] : group[1] + 1].sum())),
    )
    group_mask = dark_mask[:, best_left : best_right + 1]
    active_rows = np.where(group_mask.any(axis=1))[0]
    if len(active_rows) == 0:
        return box

    refined_x = x1 + best_left
    refined_y = y1 + int(active_rows[0])
    refined_w = best_right - best_left + 1
    refined_h = int(active_rows[-1] - active_rows[0] + 1)
    return (refined_x, refined_y, refined_w, refined_h)


def detect_vertical_bars(img: np.ndarray) -> List[Box]:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        91,
        15,
    )
    dilate_kernel = np.ones((1, 6), np.uint8)
    dilated = cv2.dilate(binary, dilate_kernel, iterations=1)
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 80))
    vertical_mask = cv2.morphologyEx(dilated, cv2.MORPH_OPEN, vertical_kernel)
    contours, _ = cv2.findContours(vertical_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [cv2.boundingRect(c) for c in contours]
    boxes.sort(key=lambda box: (box[0], box[1]))

    merged: List[Box] = []
    if not boxes:
        return merged

    curr_x, curr_y, curr_w, curr_h = boxes[0]
    for nx, ny, nw, nh in boxes[1:]:
        if abs(curr_x - nx) < 20 and (ny - (curr_y + curr_h) < 100):
            new_x = min(curr_x, nx)
            new_y = min(curr_y, ny)
            new_w = max(curr_x + curr_w, nx + nw) - new_x
            new_h = max(curr_y + curr_h, ny + nh) - new_y
            curr_x, curr_y, curr_w, curr_h = new_x, new_y, new_w, new_h
        else:
            merged.append((curr_x, curr_y, curr_w, curr_h))
            curr_x, curr_y, curr_w, curr_h = nx, ny, nw, nh
    merged.append((curr_x, curr_y, curr_w, curr_h))

    # Post-merge: split bars that have significant vertical gaps (e.g. two
    # adjacent subproofs at the same X position that were erroneously merged).
    split = _split_bars_on_gaps(img, merged)

    refined = [refine_vertical_bar_box(img, box) for box in split]
    return [box for box in refined if box[3] > 50]


def build_proof_tree(boxes: List[Box], img_w: int, img_h: int) -> SubproofNode:
    root = SubproofNode("ROOT", (0, 0, img_w, img_h))
    nodes = [SubproofNode(str(i + 1), box) for i, box in enumerate(sorted(boxes, key=lambda b: b[0]))]
    for node in nodes:
        best_parent = root
        nx, ny, _, nh = node.box
        center_y = ny + nh / 2
        for candidate in reversed(nodes):
            if candidate is node:
                continue
            if candidate.box[0] >= nx:
                continue
            _, cy, _, ch = candidate.box
            if cy <= center_y <= (cy + ch):
                best_parent = candidate
                break
        node.parent = best_parent
        best_parent.children.append(node)
    return root


def crop_region(
    img: np.ndarray,
    box: Box,
    *,
    x_padding: int = 10,
    top_padding: int = 10,
    bottom_padding: int = 80,
) -> np.ndarray:
    x, y, w, h = box
    img_h, img_w = img.shape[:2]
    x1 = max(0, x - x_padding)
    y1 = max(0, y - top_padding)
    x2 = img_w
    y2 = min(img_h, y + h + bottom_padding)
    return img[y1:y2, x1:x2]
