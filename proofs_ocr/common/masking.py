"""Shared child-masking and placeholder-token geometry helpers.

These functions are copied from the current stable modal OCR implementation to
seed a cleaner shared library. They are not yet wired into a new pipeline.
"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

from .types import SubproofNode


def fit_token_layout(
    token: str,
    *,
    start_x: int,
    center_y: int,
    max_width: int,
) -> Tuple[float, int, Tuple[int, int, int, int]]:
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2
    pad = 8
    scale = 1.0
    min_scale = 0.4

    while scale >= min_scale:
        (tw, th), _baseline = cv2.getTextSize(token, font, scale, thickness)
        total_width = tw + 2 * pad
        if total_width <= max_width:
            left = start_x
            top = max(0, center_y - th - pad)
            right = left + total_width
            bottom = center_y + pad
            return scale, thickness, (left, top, right, bottom)
        scale -= 0.1

    scale = min_scale
    (tw, th), _baseline = cv2.getTextSize(token, font, scale, thickness)
    total_width = min(max_width, tw + 2 * pad)
    left = start_x
    top = max(0, center_y - th - pad)
    right = left + total_width
    bottom = center_y + pad
    return scale, thickness, (left, top, right, bottom)


def mask_subproof_in_parent(parent_img: np.ndarray, child_node: SubproofNode) -> np.ndarray:
    masked = parent_img.copy()
    x, y, _, h = child_node.box
    img_h, img_w = masked.shape[:2]
    cv2.rectangle(masked, (x, y), (img_w, y + h), (255, 255, 255), -1)

    token = f"<SUBPROOF_ID_{child_node.id}>"
    font = cv2.FONT_HERSHEY_SIMPLEX
    start_x = min(max(8, x + 20), max(8, img_w - 40))
    center_y = min(max(24, y + max(18, h // 2)), img_h - 12)
    max_width = max(60, img_w - start_x - 8)
    scale, thickness, (left, top, right, bottom) = fit_token_layout(
        token,
        start_x=start_x,
        center_y=center_y,
        max_width=max_width,
    )
    cv2.rectangle(masked, (left, top), (right, bottom), (0, 0, 255), 2)
    cv2.putText(masked, token, (left + 8, center_y), font, scale, (0, 0, 255), thickness)
    return masked


def relative_child_box(node: SubproofNode, child: SubproofNode) -> Tuple[int, int, int, int]:
    if node.id == "ROOT":
        return child.box
    px, py, _, _ = node.box
    return (child.box[0] - px + 10, child.box[1] - py + 10, child.box[2], child.box[3])


def token_center_y_for_child(node: SubproofNode, child: SubproofNode, img_h: int) -> int:
    _, y, _, h = relative_child_box(node, child)
    return int(min(max(24, y + max(18, h // 2)), max(24, img_h - 12)))


def token_rect_for_child(
    *,
    node: SubproofNode,
    child: SubproofNode,
    img_shape: Tuple[int, int, int] | Tuple[int, int],
) -> Tuple[int, int, int, int]:
    img_h, img_w = img_shape[:2]
    x, y, _, h = relative_child_box(node, child)
    token = f"<SUBPROOF_ID_{child.id}>"
    start_x = min(max(8, x + 20), max(8, img_w - 40))
    center_y = int(min(max(24, y + max(18, h // 2)), img_h - 12))
    max_width = max(60, img_w - start_x - 8)
    _, _, rect = fit_token_layout(
        token,
        start_x=start_x,
        center_y=center_y,
        max_width=max_width,
    )
    return rect
