"""Line-level connected-component count experiments for OCR suspicion flags."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

from proofs_ocr.pipelines.modal_v2_line_zones import extract_line_zone_crops


@dataclass(frozen=True)
class LineCharCountVariant:
    name: str
    dark_threshold: int
    min_area: int
    close_kernel: int = 0


DEFAULT_LINE_COUNT_VARIANTS = (
    LineCharCountVariant(name="strict170", dark_threshold=170, min_area=15, close_kernel=0),
    LineCharCountVariant(name="balanced185", dark_threshold=185, min_area=15, close_kernel=0),
    LineCharCountVariant(name="closed185", dark_threshold=185, min_area=15, close_kernel=2),
)


def count_visible_char_slots(value: Any) -> int:
    """Count non-space visible characters in an OCR string."""

    return sum(1 for char in str(value or "") if not char.isspace())


def extract_label_window(
    line_crop_img: np.ndarray,
    *,
    right_fraction: float = 0.10,
) -> tuple[np.ndarray, dict[str, int]]:
    """Extract a crude far-left label window from one line crop."""

    if line_crop_img.size == 0:
        return line_crop_img, {"x1": 0, "x2": 0, "width": 0, "height": 0}

    img_h, img_w = line_crop_img.shape[:2]
    x1 = 0
    x2 = max(1, min(img_w, int(round(img_w * right_fraction))))
    window = line_crop_img[:, x1:x2].copy()
    return window, {"x1": x1, "x2": x2, "width": int(window.shape[1]), "height": int(img_h)}


def count_blob_components(
    zone_img: np.ndarray,
    *,
    dark_threshold: int,
    min_area: int,
    close_kernel: int = 0,
) -> dict[str, Any]:
    """Count glyph-like connected components inside one line zone."""

    if zone_img.size == 0:
        return {"component_count": 0, "components": []}

    gray = cv2.cvtColor(zone_img, cv2.COLOR_BGR2GRAY)
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
    return {"component_count": len(components), "components": components}


def build_zone_summary(
    *,
    name: str,
    expected_text: str,
    variants: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Build one zone summary with excess/deficit suspicion flags."""

    expected_count = count_visible_char_slots(expected_text)
    component_counts = [int(variant.get("component_count") or 0) for variant in variants]
    min_count = min(component_counts) if component_counts else 0
    max_count = max(component_counts) if component_counts else 0
    return {
        "zone_name": name,
        "expected_text": str(expected_text or ""),
        "expected_char_count": expected_count,
        "min_component_count": min_count,
        "max_component_count": max_count,
        "component_count_gap_min": int(min_count - expected_count),
        "component_count_gap_max": int(max_count - expected_count),
        "any_excess_flagged": any(count >= expected_count + 1 for count in component_counts),
        "any_large_excess_flagged": any(count >= expected_count + 2 for count in component_counts),
        "all_below_expected": bool(component_counts) and all(count < expected_count for count in component_counts),
        "variants": list(variants),
    }


def run_line_char_count_probe(
    *,
    image_path: Path,
    output_dir: Path,
    line_label: str = "",
    formula_text: str = "",
    justification_text: str = "",
    variants: Sequence[LineCharCountVariant] | None = None,
) -> dict[str, Any]:
    """Run a component-count probe on a whole line plus formula/justification zones."""

    output_dir.mkdir(parents=True, exist_ok=True)
    line_crop_img = cv2.imread(str(image_path))
    if line_crop_img is None:
        raise FileNotFoundError(f"Could not read line crop at {image_path}")

    formula_crop_img, justification_crop_img, zone_geometry = extract_line_zone_crops(line_crop_img)
    label_crop_img, label_geometry = extract_label_window(line_crop_img)
    cv2.imwrite(str(output_dir / "line_crop.jpg"), line_crop_img)
    cv2.imwrite(str(output_dir / "formula_zone.jpg"), formula_crop_img)
    cv2.imwrite(str(output_dir / "justification_zone.jpg"), justification_crop_img)
    cv2.imwrite(str(output_dir / "label_zone.jpg"), label_crop_img)

    active_variants = variants or DEFAULT_LINE_COUNT_VARIANTS
    zone_images = {
        "whole_line": line_crop_img,
        "label": label_crop_img,
        "formula": formula_crop_img,
        "justification": justification_crop_img,
    }
    zone_expected = {
        "whole_line": f"{line_label}{formula_text}{justification_text}",
        "label": line_label,
        "formula": formula_text,
        "justification": justification_text,
    }
    zone_variant_records: dict[str, list[dict[str, Any]]] = {}
    for zone_name, zone_img in zone_images.items():
        records: list[dict[str, Any]] = []
        for variant in active_variants:
            counts = count_blob_components(
                zone_img,
                dark_threshold=variant.dark_threshold,
                min_area=variant.min_area,
                close_kernel=variant.close_kernel,
            )
            records.append(
                {
                    "name": variant.name,
                    "dark_threshold": variant.dark_threshold,
                    "min_area": variant.min_area,
                    "close_kernel": variant.close_kernel,
                    "component_count": counts["component_count"],
                    "components": counts["components"],
                }
            )
        zone_variant_records[zone_name] = records

    zone_summaries = {
        zone_name: build_zone_summary(
            name=zone_name,
            expected_text=zone_expected[zone_name],
            variants=records,
        )
        for zone_name, records in zone_variant_records.items()
    }
    payload = {
        "image_path": str(image_path),
        "line_label": str(line_label or ""),
        "formula_text": str(formula_text or ""),
        "justification_text": str(justification_text or ""),
        "zone_geometry": dict(zone_geometry),
        "label_geometry": dict(label_geometry),
        "zones": zone_summaries,
        "any_zone_excess_flagged": any(bool(summary.get("any_excess_flagged")) for summary in zone_summaries.values()),
        "any_zone_large_excess_flagged": any(
            bool(summary.get("any_large_excess_flagged")) for summary in zone_summaries.values()
        ),
    }
    (output_dir / "line_char_count_probe.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return payload


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a line-level connected-component count probe on one OCR crop.")
    parser.add_argument("--image", required=True, help="Path to the full line crop image.")
    parser.add_argument("--output-dir", required=True, help="Directory where probe artifacts should be written.")
    parser.add_argument("--line-label", default="", help="Current OCR line label for the crop.")
    parser.add_argument("--formula", default="", help="Current OCR formula for the crop.")
    parser.add_argument("--justification", default="", help="Current OCR justification for the crop.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    payload = run_line_char_count_probe(
        image_path=Path(args.image),
        output_dir=Path(args.output_dir),
        line_label=args.line_label,
        formula_text=args.formula,
        justification_text=args.justification,
    )
    summary = {
        zone_name: {
            "expected_char_count": zone["expected_char_count"],
            "min_component_count": zone["min_component_count"],
            "max_component_count": zone["max_component_count"],
            "any_excess_flagged": zone["any_excess_flagged"],
            "any_large_excess_flagged": zone["any_large_excess_flagged"],
        }
        for zone_name, zone in payload["zones"].items()
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
