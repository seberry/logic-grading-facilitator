"""Real-crop blob-count experiments for unary formula lines."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

from . import unary_probe


@dataclass(frozen=True)
class BlobProbeVariant:
    name: str
    dark_threshold: int
    min_area: int
    close_kernel: int = 0


DEFAULT_BLOB_VARIANTS = (
    BlobProbeVariant(name="strict170", dark_threshold=170, min_area=15, close_kernel=0),
    BlobProbeVariant(name="balanced185", dark_threshold=185, min_area=15, close_kernel=0),
    BlobProbeVariant(name="closed185", dark_threshold=185, min_area=15, close_kernel=2),
)


def count_unary_formula_symbol_slots(formula_text: str) -> int:
    """Count visible unary/modal prefix symbols plus one body slot."""

    normalized = unary_probe.normalize_probe_formula_text(formula_text)
    prefix = unary_probe.leading_unary_prefix(normalized)
    body = normalized[len(prefix) :].strip()
    return len(prefix) + (1 if body else 0)


def extract_formula_window(
    line_crop_img: np.ndarray,
    *,
    left_fraction: float = 0.09,
    right_fraction: float = 0.36,
) -> tuple[np.ndarray, dict[str, int]]:
    """Extract a crude formula-zone window from a whole line crop."""

    if line_crop_img.size == 0:
        return line_crop_img, {"x1": 0, "x2": 0, "width": 0, "height": 0}

    img_h, img_w = line_crop_img.shape[:2]
    x1 = max(0, min(img_w - 1, int(round(img_w * left_fraction))))
    x2 = max(x1 + 1, min(img_w, int(round(img_w * right_fraction))))
    window = line_crop_img[:, x1:x2].copy()
    return window, {"x1": x1, "x2": x2, "width": int(window.shape[1]), "height": int(window.shape[0])}


def count_blob_components(
    formula_window_img: np.ndarray,
    *,
    dark_threshold: int,
    min_area: int,
    close_kernel: int = 0,
) -> dict[str, Any]:
    """Count glyph-like connected components inside a formula-only window."""

    if formula_window_img.size == 0:
        return {
            "component_count": 0,
            "components": [],
        }

    gray = cv2.cvtColor(formula_window_img, cv2.COLOR_BGR2GRAY)
    dark = (gray < dark_threshold).astype(np.uint8)
    if close_kernel > 1:
        dark = cv2.morphologyEx(
            dark,
            cv2.MORPH_CLOSE,
            np.ones((close_kernel, close_kernel), dtype=np.uint8),
        )

    component_count, _, stats, _ = cv2.connectedComponentsWithStats(dark, 8)
    components: list[dict[str, int]] = []
    for label_index in range(1, component_count):
        area = int(stats[label_index, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        components.append(
            {
                "area": area,
                "x": int(stats[label_index, cv2.CC_STAT_LEFT]),
                "y": int(stats[label_index, cv2.CC_STAT_TOP]),
                "w": int(stats[label_index, cv2.CC_STAT_WIDTH]),
                "h": int(stats[label_index, cv2.CC_STAT_HEIGHT]),
            }
        )

    components.sort(key=lambda item: (item["x"], item["y"]))
    return {
        "component_count": len(components),
        "components": components,
    }


def run_blob_probe(
    *,
    image_path: Path,
    current_ocr_formula: str,
    output_dir: Path,
    left_fraction: float = 0.09,
    right_fraction: float = 0.36,
    variants: Sequence[BlobProbeVariant] | None = None,
) -> dict[str, Any]:
    """Run the crude formula-window blob-count experiment on one real line crop."""

    output_dir.mkdir(parents=True, exist_ok=True)
    line_crop_img = cv2.imread(str(image_path))
    if line_crop_img is None:
        raise FileNotFoundError(f"Could not read line crop at {image_path}")

    formula_window_img, window_meta = extract_formula_window(
        line_crop_img,
        left_fraction=left_fraction,
        right_fraction=right_fraction,
    )
    window_path = output_dir / "formula_window.jpg"
    cv2.imwrite(str(window_path), formula_window_img)

    normalized_formula = unary_probe.normalize_probe_formula_text(current_ocr_formula)
    ocr_symbol_count = count_unary_formula_symbol_slots(normalized_formula)
    variant_records: list[dict[str, Any]] = []
    for variant in (variants or DEFAULT_BLOB_VARIANTS):
        counts = count_blob_components(
            formula_window_img,
            dark_threshold=variant.dark_threshold,
            min_area=variant.min_area,
            close_kernel=variant.close_kernel,
        )
        variant_records.append(
            {
                "name": variant.name,
                "dark_threshold": variant.dark_threshold,
                "min_area": variant.min_area,
                "close_kernel": variant.close_kernel,
                "component_count": counts["component_count"],
                "components": counts["components"],
                "blob_count_exceeds_ocr": counts["component_count"] > ocr_symbol_count,
            }
        )

    payload = {
        "image_path": str(image_path),
        "current_ocr_formula": normalized_formula,
        "ocr_symbol_count": ocr_symbol_count,
        "window": window_meta,
        "formula_window_path": str(window_path),
        "variants": variant_records,
        "any_blob_flagged": any(record["blob_count_exceeds_ocr"] for record in variant_records),
        "max_component_count": max((record["component_count"] for record in variant_records), default=0),
    }
    (output_dir / "unary_blob_probe.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return payload


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a formula-window blob-count experiment on one OCR line crop.")
    parser.add_argument("--image", required=True, help="Path to the full line crop image.")
    parser.add_argument("--formula", required=True, help="Current OCR formula for the line.")
    parser.add_argument("--output-dir", required=True, help="Directory where probe artifacts should be written.")
    parser.add_argument("--left-fraction", type=float, default=0.09, help="Left crop boundary as a width fraction.")
    parser.add_argument("--right-fraction", type=float, default=0.36, help="Right crop boundary as a width fraction.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    payload = run_blob_probe(
        image_path=Path(args.image),
        current_ocr_formula=args.formula,
        output_dir=Path(args.output_dir),
        left_fraction=float(args.left_fraction),
        right_fraction=float(args.right_fraction),
    )
    print(
        json.dumps(
            {
                "ocr_symbol_count": payload["ocr_symbol_count"],
                "max_component_count": payload["max_component_count"],
                "any_blob_flagged": payload["any_blob_flagged"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
