"""Crop and row geometry helpers for verifier-guided TFL rereads."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np

from proofs_ocr.common.rows import band_height, detect_text_row_bands


def fullwidth_sibling_candidates(crop_path: Path, node_id: Any) -> list[Path]:
    node = str(node_id or "").strip()
    candidates: list[Path] = []
    if node:
        candidates.extend(
            [
                crop_path.with_name(f"crop_{node}_fullwidth.jpg"),
                crop_path.with_name(f"crop_{node}_4b_fullwidth.jpg"),
            ]
        )
    if crop_path.name.endswith(".jpg") and "fullwidth" not in crop_path.stem:
        candidates.extend(
            [
                crop_path.with_name(f"{crop_path.stem}_fullwidth.jpg"),
                crop_path.with_name(f"{crop_path.stem}_4b_fullwidth.jpg"),
            ]
        )
    return candidates


def locate_or_derive_full_width_crop(
    *,
    ocr: Mapping[str, Any],
    source_image: np.ndarray | None,
    output_dir: Path,
) -> tuple[Path | None, str]:
    """Find or create a full-width crop for a provenance record."""
    crop_path = Path(str(ocr.get("crop_path") or ""))
    if crop_path.exists() and "fullwidth" in crop_path.stem:
        return crop_path, "provenance_fullwidth"

    if crop_path.exists():
        for candidate in fullwidth_sibling_candidates(crop_path, ocr.get("node_id")):
            if candidate.exists():
                return candidate, "sibling_fullwidth"

    if source_image is None:
        if crop_path.exists():
            return crop_path, "provenance_fallback_not_fullwidth"
        return None, "source image unavailable for full-width derivation"

    node_id = str(ocr.get("node_id") or "node")
    if node_id == "ROOT":
        full = source_image
    else:
        box = ocr.get("node_box")
        if not (
            isinstance(box, list)
            and len(box) == 4
            and all(isinstance(value, (int, float)) for value in box)
        ):
            return None, "missing node box for full-width derivation"
        _x, y, _w, h = [int(value) for value in box]
        h_img = source_image.shape[0]
        y1 = max(0, y - 10)
        masked_crop = cv2.imread(str(crop_path)) if crop_path.exists() else None
        if str(ocr.get("parent_id") or "") == "ROOT":
            y2 = h_img
        elif masked_crop is not None:
            y2 = min(h_img, y1 + masked_crop.shape[0])
        else:
            y2 = min(h_img, y + h)
        full = source_image[y1:y2, :].copy()
        if crop_path.exists():
            x1 = max(0, int(box[0]) - 10)
            if (
                masked_crop is not None
                and masked_crop.shape[0] == full.shape[0]
                and x1 < full.shape[1]
            ):
                paste_w = min(masked_crop.shape[1], full.shape[1] - x1)
                full[:, x1 : x1 + paste_w] = masked_crop[:, :paste_w]

    derived_dir = output_dir / "verifier_line_rereads_27b"
    derived_dir.mkdir(parents=True, exist_ok=True)
    full_path = derived_dir / f"node_{node_id}_fullwidth_source.jpg"
    cv2.imwrite(str(full_path), full)
    return full_path, "derived_from_source"


def detect_ocr_row_bounds(crop_path: Path, *, row_count: int, padding: int = 18) -> list[tuple[int, int]]:
    """Detect row y-bounds in the exact image sent to OCR."""
    img = cv2.imread(str(crop_path))
    if img is None or row_count < 1:
        return []

    bands = [band for band in detect_text_row_bands(img) if band_height(band) >= 18]
    if len(bands) != row_count:
        h_img = img.shape[0]
        step = h_img / float(row_count)
        bands = []
        for idx in range(row_count):
            center = (idx + 0.5) * step
            half_height = max(24.0, step * 0.42)
            bands.append((int(max(0, center - half_height)), int(min(h_img - 1, center + half_height))))

    h_img = img.shape[0]
    return [
        (max(0, int(y1) - padding), min(h_img, int(y2) + padding))
        for y1, y2 in bands
    ]


def detect_visible_gutter_line_bounds(
    crop_path: Path,
    *,
    target_line: int,
    first_line: int,
    last_line: int,
    node_box: list[Any] | None = None,
    padding: int = 18,
) -> tuple[int, int] | None:
    """Estimate a row y-window from visible left-gutter line-number bands."""
    if target_line < first_line or target_line > last_line:
        return None
    img = cv2.imread(str(crop_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    h_img, w_img = img.shape[:2]
    if node_box and len(node_box) >= 1 and isinstance(node_box[0], (int, float)):
        x_end = max(20, min(int(node_box[0]) - 45, w_img // 3))
    else:
        x_end = max(20, w_img // 6)
    x_start = max(0, x_end - 60)
    gutter = img[:, x_start:x_end]
    if gutter.size == 0:
        return None

    mask = gutter < 160
    projection = mask.sum(axis=1)
    raw_bands: list[tuple[int, int]] = []
    in_band = False
    start = 0
    for y, value in enumerate(projection):
        if value > 1 and not in_band:
            start = y
            in_band = True
        if (value <= 1 or y == len(projection) - 1) and in_band:
            end = y
            in_band = False
            if end - start >= 4:
                raw_bands.append((start, end))

    merged: list[tuple[int, int]] = []
    for band in raw_bands:
        if merged and band[0] - merged[-1][1] <= 8:
            merged[-1] = (merged[-1][0], band[1])
        else:
            merged.append(band)
    bands = [band for band in merged if band[1] - band[0] >= 8]

    expected_count = last_line - first_line + 1
    if len(bands) < expected_count:
        return None
    index = target_line - first_line
    if index >= len(bands):
        return None
    y1, y2 = bands[index]
    return max(0, int(y1) - padding), min(h_img, int(y2) + padding)


def extract_single_line_crop(
    crop_path: Path,
    *,
    local_index: int,
    expected_lines: int,
    output_path: Path,
    padding: int = 18,
    row_bounds: tuple[int, int] | None = None,
) -> Path:
    """Extract a generous full-width row crop from a node crop."""
    img = cv2.imread(str(crop_path))
    if img is None:
        raise FileNotFoundError(f"Could not read crop image: {crop_path}")
    if local_index < 1 or expected_lines < 1 or local_index > expected_lines:
        raise ValueError(f"Invalid row {local_index} of {expected_lines} for {crop_path}")

    if row_bounds is not None:
        y1, y2 = row_bounds
        y1 = max(0, int(y1))
        y2 = min(img.shape[0], int(y2))
    else:
        bands = [band for band in detect_text_row_bands(img) if band_height(band) >= 18]
        if len(bands) != expected_lines:
            h_img = img.shape[0]
            step = h_img / float(expected_lines)
            bands = []
            for idx in range(expected_lines):
                center = (idx + 0.5) * step
                half_height = max(24.0, step * 0.42)
                bands.append((int(max(0, center - half_height)), int(min(h_img - 1, center + half_height))))

        y1, y2 = bands[local_index - 1]
        y1 = max(0, int(y1) - padding)
        y2 = min(img.shape[0], int(y2) + padding)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), img[y1:y2, :])
    return output_path
