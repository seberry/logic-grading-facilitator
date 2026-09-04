"""Extract version-aware question images from a blank multi-version quiz PDF.

The output follows the app's conventional layout:
    <output>/<quiz_id>/<problem_id>_<version>.png

This script is for blank assessment PDFs only. Do not use it on student work.
"""

from __future__ import annotations

import argparse
import io
import re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import fitz  # type: ignore
from PIL import Image

Box = Tuple[float, float, float, float]


def safe_asset_component(value: object, fallback: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "").strip()).strip("._")
    return cleaned or fallback


# Normalized boxes for the Fall 2026 P250 Quiz 1 three-page PDF. Q2 and Q3
# combine the shared symbolization key with the individual question, avoiding
# the large blank answer area between them.
P250_QUIZ1_REGIONS: Dict[str, List[Box]] = {
    "Q1": [(0.065, 0.125, 0.935, 0.270)],
    "Q2": [(0.065, 0.278, 0.935, 0.360), (0.065, 0.365, 0.935, 0.495)],
    "Q3": [(0.065, 0.278, 0.935, 0.360), (0.065, 0.505, 0.935, 0.610)],
    "EC1": [(0.065, 0.670, 0.495, 0.815)],
    "EC2": [(0.495, 0.670, 0.935, 0.815)],
}

PRESETS = {"p250_quiz1": P250_QUIZ1_REGIONS}


def _crop_normalized(image: Image.Image, box: Box) -> Image.Image:
    width, height = image.size
    left, top, right, bottom = box
    pixel_box = (
        round(left * width),
        round(top * height),
        round(right * width),
        round(bottom * height),
    )
    return image.crop(pixel_box)


def _stack(crops: Sequence[Image.Image], gap: int = 12) -> Image.Image:
    if len(crops) == 1:
        return crops[0]
    width = max(crop.width for crop in crops)
    height = sum(crop.height for crop in crops) + gap * (len(crops) - 1)
    result = Image.new("RGB", (width, height), "white")
    y = 0
    for crop in crops:
        result.paste(crop, (0, y))
        y += crop.height + gap
    return result


def extract_reference_images(
    source_pdf: Path,
    output_dir: Path,
    quiz_id: str,
    versions: Sequence[str],
    regions: Dict[str, List[Box]],
    scale: float = 2.5,
) -> List[Path]:
    """Render and crop one blank quiz page per version."""
    if scale <= 0:
        raise ValueError("scale must be positive")
    document = fitz.open(str(source_pdf))
    try:
        if len(document) < len(versions):
            raise ValueError(
                f"PDF has {len(document)} page(s), but {len(versions)} versions were requested."
            )
        quiz_dir = output_dir / safe_asset_component(quiz_id, "quiz")
        quiz_dir.mkdir(parents=True, exist_ok=True)
        written: List[Path] = []
        matrix = fitz.Matrix(scale, scale)
        for page_index, version in enumerate(versions):
            pixmap = document[page_index].get_pixmap(matrix=matrix, alpha=False)
            page_image = Image.open(io.BytesIO(pixmap.tobytes("png"))).convert("RGB")
            for problem_id, boxes in regions.items():
                crops = [_crop_normalized(page_image, box) for box in boxes]
                question_image = _stack(crops, gap=max(8, round(5 * scale)))
                filename = (
                    f"{safe_asset_component(problem_id, 'problem')}_"
                    f"{safe_asset_component(version, 'UNKNOWN')}.png"
                )
                target = quiz_dir / filename
                question_image.save(target, format="PNG", optimize=True)
                written.append(target)
        return written
    finally:
        document.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract blank question images for the version-aware grading header."
    )
    parser.add_argument("source_pdf", type=Path)
    parser.add_argument("--quiz-id", required=True)
    parser.add_argument("--versions", nargs="+", default=["A", "B", "C"])
    parser.add_argument("--preset", choices=sorted(PRESETS), default="p250_quiz1")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("working") / "rubric_assets",
        help="Parent rubric-assets directory (default: working/rubric_assets).",
    )
    parser.add_argument("--scale", type=float, default=2.5)
    args = parser.parse_args()

    if not args.source_pdf.is_file():
        parser.error(f"PDF not found: {args.source_pdf}")

    written = extract_reference_images(
        source_pdf=args.source_pdf,
        output_dir=args.output_dir,
        quiz_id=args.quiz_id,
        versions=[str(value).strip().upper() for value in args.versions],
        regions=PRESETS[args.preset],
        scale=args.scale,
    )
    print(f"Wrote {len(written)} blank-question image(s) below {args.output_dir.resolve()}")
    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
