"""Early bar-filtering experiments for modal_v2 tree construction."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from proofs_ocr.common.types import Box


def filter_false_left_separator_bars(
    *,
    image: np.ndarray,
    bars: Sequence[Box],
) -> tuple[list[Box], dict[str, Any]]:
    """Optionally drop one suspicious full-height left separator before tree build."""

    img_h, img_w = image.shape[:2]
    raw_bars = [tuple(int(value) for value in box) for box in bars]
    payload: dict[str, Any] = {
        "raw_bars": [list(box) for box in raw_bars],
        "filtered_bars": [list(box) for box in raw_bars],
        "rejected_bars": [],
        "applied": False,
        "reason": "",
    }
    if len(raw_bars) < 2:
        payload["reason"] = "too_few_bars"
        return list(raw_bars), payload

    leftmost = min(raw_bars, key=lambda box: (box[0], box[1], -box[3]))
    x, y, _, h = leftmost
    if x > max(24, int(img_w * 0.12)):
        payload["reason"] = "leftmost_not_near_edge"
        return list(raw_bars), payload
    if y > max(8, int(img_h * 0.05)):
        payload["reason"] = "leftmost_not_top_anchored"
        return list(raw_bars), payload
    if h < max(80, int(img_h * 0.78)):
        payload["reason"] = "leftmost_not_tall_enough"
        return list(raw_bars), payload

    right_bars = [
        box
        for box in raw_bars
        if box != leftmost and box[0] >= x + max(12, leftmost[2] + 4)
    ]
    if not right_bars:
        payload["reason"] = "no_right_bar"
        return list(raw_bars), payload

    lower_right_bars = [
        box
        for box in right_bars
        if box[1] >= y + max(28, int(img_h * 0.12)) and box[3] >= max(40, int(img_h * 0.20))
    ]
    if not lower_right_bars:
        payload["reason"] = "no_lower_right_bar"
        return list(raw_bars), payload

    filtered = [box for box in raw_bars if box != leftmost]
    payload["filtered_bars"] = [list(box) for box in filtered]
    payload["rejected_bars"] = [list(leftmost)]
    payload["applied"] = True
    payload["reason"] = "removed_false_left_separator_candidate"
    return filtered, payload
