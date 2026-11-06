"""Create an isolated fake Quiz 9 session for reviewing the FOL truth UI."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from app.fol_truth_assistance import MODELS
from app.models import RosterEntry, Session, WorkUnit
from app.storage import load_grading_key


def build(output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    scans = output / "raw_scans"
    grades = output / "grades"
    scans.mkdir(parents=True)
    grades.mkdir(parents=True)
    rubric = load_grading_key(Path(__file__).resolve().parents[1] / "docs/quiz9_fol_truth_rubric.yaml")
    fake_answers = {
        "A": [("True", ""), ("False", "1"), ("True", "2"), ("False", "2"), ("True", "N/A"), ("True", ""), ("True", "")],
        "B": [("False", ""), ("False", "N/A"), ("True", "x = 2"), ("True", "3"), ("True", "N/A"), ("False", ""), ("False", "")],
        "C": [("True", ""), ("False", "N/A"), ("True", "2"), ("True", "N/A"), ("False", "1"), ("True", ""), ("True", "")],
    }
    units = []
    roster = []
    locations = {f"Q{i}": [] for i in range(1, 8)}
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 27)
        title_font = ImageFont.truetype("DejaVuSans.ttf", 34)
    except OSError:
        font = ImageFont.load_default()
        title_font = font
    for unit_id, version in enumerate(MODELS, 1):
        image = Image.new("RGB", (1600, 2200), "white")
        draw = ImageDraw.Draw(image)
        model = MODELS[version]
        draw.text((90, 70), f"FAKE QUIZ 9 - VERSION {version} - DEMO ONLY", font=title_font, fill="black")
        draw.text((90, 125), f"D = {{1,2,3}}, c = {model['constants']['c']}", font=font, fill="black")
        draw.text((90, 170), ", ".join(f"{p} = {{{','.join(map(str, values))}}}" for p, values in model["predicates"].items()), font=font, fill="black")
        for number, (answer, written) in enumerate(fake_answers[version], 1):
            top = 255 + (number - 1) * 255
            draw.text((90, top), f"{number}. {rubric.problems[number-1].variants[version].prompt}", font=font, fill="black")
            draw.text((150, top + 55), "True       False", font=font, fill="black")
            # Color distinguishes these synthetic marks from real student work.
            center = 175 if answer == "True" else 315
            draw.ellipse((center - 14, top + 43, center + 92, top + 95), outline="#a33a25", width=5)
            if 2 <= number <= 5:
                draw.text((150, top + 115), f"Written response: {written}", font=font, fill="#a33a25")
            locations[f"Q{number}"].append({
                "unit_id": unit_id, "page_filename": f"fake_version_{version}.jpg", "page_index": 0,
                "bbox": [0.04, max(0, (top - 20) / 2200), .96, min(1, (top + 190) / 2200)],
                "present": True, "confidence": 1.0, "comment": "Synthetic demo crop",
            })
        filename = f"fake_version_{version}.jpg"
        image.save(scans / filename, quality=90)
        roster.append(RosterEntry(student_id=f"fake{unit_id}", name=f"Demo Student {version}"))
        units.append(WorkUnit(unit_id=unit_id, page_filenames=[filename], student_id=f"fake{unit_id}",
                              quiz_version=version, per_problem={f"Q{i}": None for i in range(1, 8)}, comment=None))
    session = Session(roster=roster, grading_key=rubric, work_units=units, unclaimed_pages=[])
    (grades / "session.json").write_text(session.model_dump_json(indent=2), encoding="utf-8")
    for problem, rows in locations.items():
        (grades / f"segments_{problem}.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (output / "README.txt").write_text(
        "Synthetic Quiz 9 UI demo. All names, marks, and scans are fake.\n"
        "Run with DATA_DIR set to this directory and open /problem/Q1/strips.\n"
        "Choose a truth value; for Q2-Q5 also choose an object or N/A.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("working/quiz9_fol_truth_demo_scored"))
    args = parser.parse_args()
    build(args.output_dir)
    print("Created fake demo under", args.output_dir.resolve())
