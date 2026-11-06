"""Prepare the approved Quiz 9 rubric and blank question images from its PDF."""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import fitz
import yaml


def prepare(pdf_path: Path, output_dir: Path) -> Path:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {output_dir}")
    source = Path(__file__).resolve().parents[1] / "docs/quiz9_fol_truth_rubric.yaml"
    rubric = yaml.safe_load(source.read_text(encoding="utf-8"))
    quiz_id = rubric["quiz_id"]
    doc = fitz.open(pdf_path)
    if len(doc) != 3:
        raise ValueError("Expected the three-page A/B/C blank Quiz 9 PDF.")
    assets = output_dir / "rubric_assets" / quiz_id
    rubric_dir = output_dir / "rubric"
    assets.mkdir(parents=True)
    rubric_dir.mkdir(parents=True)
    for version, page in zip("ABC", doc):
        text = page.get_text()
        if f"Version {version}" not in text:
            raise ValueError(f"Expected Version {version} on page {version}.")
        starts: dict[int, float] = {}
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                content = "".join(span["text"] for span in line["spans"]).strip()
                match = re.match(r"([1-7])\.\s", content)
                if match:
                    starts[int(match.group(1))] = line["bbox"][1]
        if set(starts) != set(range(1, 8)):
            raise ValueError(f"Could not locate all seven questions on Version {version}.")
        for number, problem in enumerate(rubric["problems"], 1):
            bottom = starts.get(number + 1, starts[number] + 55) - 8
            crop = fitz.Rect(50, starts[number] - 11, 565, bottom)
            if crop.y1 <= crop.y0:
                raise ValueError(f"Invalid crop for Version {version} question {number}.")
            filename = f"Q{number}_{version}.png"
            page.get_pixmap(matrix=fitz.Matrix(2, 2), clip=crop, alpha=False).save(assets / filename)
            problem["variants"][version]["question_image"] = f"{quiz_id}/{filename}"
    target = rubric_dir / f"{quiz_id}.yaml"
    target.write_text(yaml.safe_dump(rubric, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("working/quiz9_prepared_rubric"))
    args = parser.parse_args()
    print(prepare(args.pdf, args.output_dir))
