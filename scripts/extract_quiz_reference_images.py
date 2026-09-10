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

# Fall 2026 P250 truth-table quiz: two pages per version. These crops contain
# only the printed question, not the large blank student-answer rectangle.
P250_TRUTH_TABLE_QUIZ_REGIONS: Dict[str, List[Box]] = {
    "Q1": [(0.065, 0.115, 0.935, 0.165)],
    "Q2": [(0.065, 0.365, 0.935, 0.430)],
    "Q3": [(0.065, 0.115, 0.935, 0.210)],
    "EC1": [(0.065, 0.515, 0.935, 0.615)],
}

PRESETS = {
    "p250_quiz1": P250_QUIZ1_REGIONS,
    "p250_truth_table_quiz": P250_TRUTH_TABLE_QUIZ_REGIONS,
}
PRESET_PAGES_PER_VERSION = {
    "p250_quiz1": 1,
    "p250_truth_table_quiz": 2,
}
PRESET_PAGE_OFFSETS = {
    "p250_quiz1": {},
    "p250_truth_table_quiz": {"Q1": 0, "Q2": 0, "Q3": 1, "EC1": 1},
}


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
    pages_per_version: int = 1,
    page_offsets: Dict[str, int] | None = None,
) -> List[Path]:
    """Render and crop blank quiz pages for each version.

    ``page_offsets`` maps each problem id to its zero-based page within a
    version. The defaults preserve the original one-page-per-version behavior.
    """
    if scale <= 0:
        raise ValueError("scale must be positive")
    if pages_per_version <= 0:
        raise ValueError("pages_per_version must be positive")
    offsets = page_offsets or {}
    document = fitz.open(str(source_pdf))
    try:
        required_pages = len(versions) * pages_per_version
        if len(document) < required_pages:
            raise ValueError(
                f"PDF has {len(document)} page(s), but {required_pages} are needed "
                f"for {len(versions)} version(s) with {pages_per_version} page(s) each."
            )
        quiz_dir = output_dir / safe_asset_component(quiz_id, "quiz")
        quiz_dir.mkdir(parents=True, exist_ok=True)
        written: List[Path] = []
        matrix = fitz.Matrix(scale, scale)
        for version_index, version in enumerate(versions):
            rendered_pages: Dict[int, Image.Image] = {}
            for problem_id, boxes in regions.items():
                offset = offsets.get(problem_id, 0)
                if offset < 0 or offset >= pages_per_version:
                    raise ValueError(
                        f"Page offset {offset} for {problem_id} is outside a "
                        f"{pages_per_version}-page version."
                    )
                page_index = version_index * pages_per_version + offset
                if offset not in rendered_pages:
                    pixmap = document[page_index].get_pixmap(matrix=matrix, alpha=False)
                    rendered_pages[offset] = Image.open(
                        io.BytesIO(pixmap.tobytes("png"))
                    ).convert("RGB")
                page_image = rendered_pages[offset]
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
        pages_per_version=PRESET_PAGES_PER_VERSION[args.preset],
        page_offsets=PRESET_PAGE_OFFSETS[args.preset],
    )
    print(f"Wrote {len(written)} blank-question image(s) below {args.output_dir.resolve()}")
    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
