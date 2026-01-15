"""
Generate a small fake dataset under DATA_DIR for safe local testing.

Creates:
- roster.csv
- rubric/quiz1.yaml
- raw_scans/*.png (simple pages with student name + Q1/Q2 text)

Run:
  python scripts/generate_fake_data.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

from PIL import Image, ImageDraw, ImageFont

# Ensure repo root on sys.path for direct script execution.
import os
import sys

project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from app.storage import DATA_DIR, RAW_SCANS_DIR, RUBRIC_DIR, DEFAULT_ROSTER_PATH


DEFAULT_STUDENTS: List[Tuple[str, str]] = [
    ("S001", "Alice Example"),
    ("S002", "Ben Sample"),
    ("S003", "Cora Test"),
    ("S004", "Devon Demo"),
    ("S005", "Evan Placeholder"),
    ("S006", "Frank Mock"),
]


def _load_font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", size=size)
    except Exception:
        return ImageFont.load_default()


def _write_roster(path: Path, students: List[Tuple[str, str]]) -> None:
    lines = ["student_id,name"]
    for sid, name in students:
        lines.append(f"{sid},{name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_rubric(path: Path, quiz_id: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = (
        f"quiz_id: {quiz_id}\n"
        "max_points: 10\n"
        "problems:\n"
        "  - id: Q1\n"
        "    label: Problem 1\n"
        "    points: 5\n"
        "  - id: Q2\n"
        "    label: Problem 2\n"
        "    points: 5\n"
    )
    path.write_text(content, encoding="utf-8")


def _clear_dir(path: Path) -> None:
    for child in path.iterdir():
        if child.is_file():
            child.unlink()
        elif child.is_dir():
            for sub in sorted(child.rglob("*"), reverse=True):
                if sub.is_file():
                    sub.unlink()
                elif sub.is_dir():
                    sub.rmdir()
            child.rmdir()


def _draw_page(
    out_path: Path,
    quiz_id: str,
    student_id: str,
    student_name: str,
    page_index: int,
) -> None:
    width, height = 1700, 2200
    img = Image.new("RGB", (width, height), color="white")
    draw = ImageDraw.Draw(img)

    title_font = _load_font(48)
    body_font = _load_font(32)
    mono_font = _load_font(28)

    y = 60
    draw.text((80, y), f"Quiz: {quiz_id}", fill="black", font=title_font)
    y += 80
    draw.text((80, y), f"Student: {student_name}", fill="black", font=title_font)
    y += 60
    draw.text((80, y), f"ID: {student_id}", fill="black", font=body_font)
    y += 80
    draw.text((80, y), f"Page {page_index}", fill="black", font=body_font)

    y += 120
    draw.text((80, y), "Q1.", fill="black", font=title_font)
    y += 60
    draw.text((140, y), "Answer: A -> B", fill="black", font=body_font)
    y += 120

    draw.text((80, y), "Q2.", fill="black", font=title_font)
    y += 60
    draw.text((140, y), "Answer: (A v B) -> C", fill="black", font=body_font)
    y += 120

    for i in range(6):
        draw.text((140, y + i * 50), f"Work line {i+1}: ...", fill="black", font=mono_font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, format="PNG")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate fake roster, rubric, and scans.")
    parser.add_argument("--quiz-id", default="quiz1", help="Quiz id used in rubric and page headers.")
    parser.add_argument("--pages-per-student", type=int, default=2, help="Pages to generate per student.")
    parser.add_argument("--force", action="store_true", help="Clear raw_scans before generating.")
    args = parser.parse_args()

    quiz_id = args.quiz_id
    pages_per_student = max(1, args.pages_per_student)

    RAW_SCANS_DIR.mkdir(parents=True, exist_ok=True)
    RUBRIC_DIR.mkdir(parents=True, exist_ok=True)

    if args.force:
        _clear_dir(RAW_SCANS_DIR)

    _write_roster(DEFAULT_ROSTER_PATH, DEFAULT_STUDENTS)
    _write_rubric(RUBRIC_DIR / f"{quiz_id}.yaml", quiz_id)

    counter = 1
    for sid, name in DEFAULT_STUDENTS:
        for page_idx in range(1, pages_per_student + 1):
            filename = f"scan_{counter:02d}.png"
            counter += 1
            _draw_page(
                RAW_SCANS_DIR / filename,
                quiz_id=quiz_id,
                student_id=sid,
                student_name=name,
                page_index=page_idx,
            )

    print(f"DATA_DIR: {DATA_DIR}")
    print(f"Roster: {DEFAULT_ROSTER_PATH}")
    print(f"Rubric: {RUBRIC_DIR / f'{quiz_id}.yaml'}")
    print(f"Raw scans: {RAW_SCANS_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()
