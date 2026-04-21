"""Formula-focused reread crop helpers for modal_v2 review flows."""

from __future__ import annotations

import cv2
import numpy as np

from proofs_ocr.pipelines.modal_v2_line_zones import extract_line_zone_crops


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
