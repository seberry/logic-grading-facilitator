"""Debug artifact writing for proof OCR experiments."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Mapping, Sequence

import cv2
import numpy as np

from .types import Box

STATE_FILENAME = "state.json"
PROMPT_FILENAME = "ocr_prompt.txt"
FINAL_STRUCTURE_FILENAME = "final_structure.json"
BASELINE_FINAL_STRUCTURE_FILENAME = "final_structure_baseline.json"
DEBUG_BARS_FILENAME = "debug_bars.jpg"
SOURCE_IMAGE_STEM = "source_image"
INTERMEDIATE_TEMPLATE = "intermediate_{node_id}.json"
RAW_RESPONSE_TEMPLATE = "response_{node_id}.json"
NODE_ROWS_TEMPLATE = "node_rows_{node_id}.json"
NODE_ROW_BANDS_TEMPLATE = "node_row_bands_{node_id}.json"
NODE_ROWS_OVERLAY_TEMPLATE = "node_rows_overlay_{node_id}.jpg"
NODE_ENTRIES_TEMPLATE = "node_entries_{node_id}.json"
NODE_CROP_TEMPLATE = "crop_{node_id}.jpg"
NODE_LINE_CROP_TEMPLATE = "node_line_{node_id}_{row_order}.jpg"
NODE_LINE_FORMULA_CROP_TEMPLATE = "node_line_formula_{node_id}_{row_order}.jpg"
NODE_LINE_JUSTIFICATION_CROP_TEMPLATE = "node_line_justification_{node_id}_{row_order}.jpg"
NODE_LINE_RESPONSE_TEMPLATE = "node_line_{node_id}_{row_order}.json"
BAR_FILTER_DEBUG_FILENAME = "bar_filter_debug.json"
FALSE_LEFT_SEPARATOR_VERIFIER_CROP_FILENAME = "false_left_separator_candidate.jpg"
UNARY_STRUCTURED_PROBE_FILENAME = "unary_structured_probe.json"
UNARY_STRUCTURED_LINE_CROP_TEMPLATE = "unary_structured_line_{node_id}_{row_order}.jpg"
UNARY_STRUCTURED_FORMULA_PROBE_FILENAME = "unary_structured_formula_probe.json"
UNARY_STRUCTURED_FORMULA_LINE_CROP_TEMPLATE = "unary_structured_formula_{node_id}_{row_order}.jpg"
UNARY_BLOB_PROBE_FILENAME = "unary_blob_probe.json"
UNARY_BLOB_LINE_WINDOW_TEMPLATE = "unary_blob_line_{node_id}_{row_order}.jpg"


def clear_debug_dir(output_dir: Path, *, preserve_paths: Sequence[Path] | None = None) -> None:
    """Remove stale debug artifacts from one output directory."""

    output_dir.mkdir(parents=True, exist_ok=True)
    preserved = {path.resolve() for path in (preserve_paths or [])}
    for child in output_dir.iterdir():
        if child.resolve() in preserved:
            continue
        if child.is_dir():
            shutil.rmtree(child)
            continue
        child.unlink()


def write_bar_overlay(*, image: np.ndarray, bars: Sequence[Box], output_path: Path) -> None:
    """Write an image showing detected subproof bars."""

    overlay = image.copy()
    for index, (x, y, w, h) in enumerate(bars, start=1):
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            overlay,
            str(index),
            (x, max(18, y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
    cv2.imwrite(str(output_path), overlay)


def write_row_overlay(
    *,
    node_crop_img: np.ndarray,
    row_records: Sequence[Mapping[str, object]],
    output_path: Path,
) -> None:
    """Write a crop with detected rows or child-token slots highlighted."""

    overlay = node_crop_img.copy()
    img_h, img_w = overlay.shape[:2]
    for record in row_records:
        kind = str(record.get("kind") or "line")
        y1 = max(0, min(img_h - 1, int(record.get("y1", 0))))
        y2 = max(0, min(img_h - 1, int(record.get("y2", 0))))
        row_order = int(record.get("row_order", 0))
        if kind == "subproof_token":
            color = (0, 0, 255)
            label = f"T{row_order}:{record.get('child_id', '')}"
        else:
            color = (0, 180, 0)
            label = f"L{row_order}"
        cv2.rectangle(overlay, (0, y1), (img_w - 1, y2), color, 2)
        cv2.putText(
            overlay,
            label,
            (6, max(18, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
        )
    cv2.imwrite(str(output_path), overlay)
