"""
Generate a FERPA-safe printed FOL model exam demo dataset.

The output is a complete DATA_DIR fixture with:
- roster.csv
- rubric/fol_model_exam_demo.yaml
- raw_scans/*.png
- grades/session.json
- grades/segments_Q*.json

Run:
  python scripts/generate_fol_model_exam_demo.py --force

Then start the app with:
  $env:DATA_DIR='data/samples/fol_model_exam_demo'
  python -m uvicorn app.server:app --reload --port 5000
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import yaml
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "data" / "samples" / "fol_model_exam_demo"

PAGE_WIDTH = 1600
PAGE_HEIGHT = 2100
QUESTION_BOXES = {
    "Q1": [0.06, 0.18, 0.94, 0.39],
    "Q2": [0.06, 0.43, 0.94, 0.64],
    "Q3": [0.06, 0.68, 0.94, 0.89],
}


@dataclass(frozen=True)
class DemoStudent:
    student_id: str
    name: str
    filename: str
    answers: Dict[str, str]


STUDENTS: List[DemoStudent] = [
    DemoStudent(
        student_id="FOL001",
        name="Ada Example",
        filename="fol_model_demo_ada_example_quant_p01.png",
        answers={
            "Q1": "Domain: 1,2\nP: 1",
            "Q2": "Domain: 1,2\nP: 1\nQ: 1",
            "Q3": "Domain: 1,2\nP: 1\nQ: 2",
        },
    ),
    DemoStudent(
        student_id="FOL002",
        name="Bert Sample",
        filename="fol_model_demo_bert_sample_quant_p01.png",
        answers={
            "Q1": "Domain: 1\nP: 1",
            "Q2": "Domain: 1,2\nP: 1\nQ: 1,2",
            "Q3": "Domain: 1,2\nP: 1\nQ: 1",
        },
    ),
    DemoStudent(
        student_id="FOL003",
        name="Cleo Demo",
        filename="fol_model_demo_cleo_demo_quant_p01.png",
        answers={
            "Q1": "Domain: 1,2,3\nP: 1,3",
            "Q2": "Domain: 1,2,3\nP: 2\nQ: 2",
            "Q3": "Domain: 1,2,3\nP: 1,3\nQ: 2",
        },
    ),
    DemoStudent(
        student_id="FOL004",
        name="Dev Test",
        filename="fol_model_demo_dev_test_quant_p01.png",
        answers={
            "Q1": "Domain: 1,2\nP: 1,2",
            "Q2": "Domain: 1,2\nP: 1\nQ: 1",
            "Q3": "Domain: 1,2\nP: 1\nQ: 2",
        },
    ),
]


def _load_font(size: int, *, mono: bool = False) -> ImageFont.ImageFont:
    if mono:
        candidates = [
            "DejaVuSansMono.ttf",
            str(Path("C:/Windows/Fonts/DejaVuSansMono.ttf")),
            "consola.ttf",
            "cour.ttf",
        ]
    else:
        candidates = [
            "DejaVuSans.ttf",
            str(Path("C:/Windows/Fonts/DejaVuSans.ttf")),
            "seguisym.ttf",
            str(Path("C:/Windows/Fonts/seguisym.ttf")),
            "arial.ttf",
            "calibri.ttf",
        ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def _write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_roster(root: Path) -> None:
    lines = ["student_id,name"]
    for student in STUDENTS:
        lines.append(f"{student.student_id},{student.name}")
    (root / "roster.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_rubric(root: Path) -> None:
    rubric = {
        "quiz_id": "fol_model_exam_demo",
        "max_points": 9,
        "problems": [
            {
                "id": "Q1",
                "label": "Give a model where the two sentences have different truth values.",
                "points": 3,
                "problem_type": "fol_model",
                "section": "FOL Models",
                "problem_number": "1",
                "prompt": "Give a model showing that ∀x Px and ∃x Px are not equivalent.",
                "segmentation_hint": "Look for the printed Q1 model answer box.",
                "autograde": {
                    "task_kind": "non_equivalence",
                    "default_score_correct": 3,
                    "default_score_incorrect": 0,
                    "prompt_data": {
                        "formulas": ["∀x Px", "∃x Px"],
                    },
                    "ocr": {"expected_answer_shape": "model_only"},
                },
            },
            {
                "id": "Q2",
                "label": "Give a countermodel to the argument.",
                "points": 3,
                "problem_type": "fol_model",
                "section": "FOL Models",
                "problem_number": "2",
                "prompt": "Give a model making both premises true and the conclusion false.",
                "segmentation_hint": "Look for the printed Q2 model answer box.",
                "autograde": {
                    "task_kind": "non_validity",
                    "default_score_correct": 3,
                    "default_score_incorrect": 0,
                    "prompt_data": {
                        "premises": ["∀x (Px → Qx)", "∃x Px"],
                        "conclusion": "∀x Qx",
                    },
                    "ocr": {"expected_answer_shape": "model_only"},
                },
            },
            {
                "id": "Q3",
                "label": "Give one model making all target sentences true.",
                "points": 3,
                "problem_type": "fol_model",
                "section": "FOL Models",
                "problem_number": "3",
                "prompt": "Give a model witnessing joint consistency.",
                "segmentation_hint": "Look for the printed Q3 model answer box.",
                "autograde": {
                    "task_kind": "joint_consistency",
                    "default_score_correct": 3,
                    "default_score_incorrect": 0,
                    "prompt_data": {
                        "formulas": ["∃x Px", "∃x Qx", "∀x (Px → ¬Qx)"],
                    },
                    "ocr": {"expected_answer_shape": "model_only"},
                },
            },
        ],
    }
    _write_yaml(root / "rubric" / "fol_model_exam_demo.yaml", rubric)


def _draw_wrapped_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    *,
    font: ImageFont.ImageFont,
    fill: str = "black",
    line_gap: int = 8,
) -> None:
    x, y = xy
    for line in text.splitlines():
        draw.text((x, y), line, fill=fill, font=font)
        bbox = draw.textbbox((x, y), line or " ", font=font)
        y += (bbox[3] - bbox[1]) + line_gap


def _draw_student_page(root: Path, student: DemoStudent) -> None:
    img = Image.new("RGB", (PAGE_WIDTH, PAGE_HEIGHT), "white")
    draw = ImageDraw.Draw(img)
    title_font = _load_font(48)
    body_font = _load_font(30)
    small_font = _load_font(24)
    mono_font = _load_font(34, mono=True)

    draw.text((80, 55), "FOL Model Exam Demo", fill="black", font=title_font)
    draw.text((80, 125), f"Student: {student.name}", fill="black", font=body_font)
    draw.text((80, 170), f"ID: {student.student_id}", fill="black", font=body_font)
    draw.text(
        (80, 220),
        "Printed fake data for local autograde workflow testing.",
        fill="#555555",
        font=small_font,
    )

    prompts = {
        "Q1": "Q1. Make ∀x Px and ∃x Px have different truth values.",
        "Q2": "Q2. Make ∀x(Px → Qx), ∃x Px true and ∀x Qx false.",
        "Q3": "Q3. Make ∃x Px, ∃x Qx, and ∀x(Px → ¬Qx) all true.",
    }
    for problem_id, bbox in QUESTION_BOXES.items():
        x1 = int(bbox[0] * PAGE_WIDTH)
        y1 = int(bbox[1] * PAGE_HEIGHT)
        x2 = int(bbox[2] * PAGE_WIDTH)
        y2 = int(bbox[3] * PAGE_HEIGHT)
        draw.rectangle((x1, y1, x2, y2), outline="#111111", width=3)
        draw.text((x1 + 24, y1 + 18), prompts[problem_id], fill="black", font=body_font)
        _draw_wrapped_text(
            draw,
            (x1 + 54, y1 + 80),
            student.answers[problem_id],
            font=mono_font,
        )

    out_path = root / "raw_scans" / student.filename
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, format="PNG")


def _write_session(root: Path) -> None:
    work_units = []
    for index, student in enumerate(STUDENTS, 1):
        work_units.append(
            {
                "unit_id": index,
                "page_filenames": [student.filename],
                "student_id": student.student_id,
                "per_problem": {"Q1": None, "Q2": None, "Q3": None},
                "per_problem_comments": {"Q1": None, "Q2": None, "Q3": None},
                "comment": None,
            }
        )

    session = {
        "roster": [{"student_id": s.student_id, "name": s.name} for s in STUDENTS],
        "grading_key": yaml.safe_load((root / "rubric" / "fol_model_exam_demo.yaml").read_text(encoding="utf-8")),
        "work_units": work_units,
        "unclaimed_pages": [],
        "pinned_note": "Fake printed FOL model demo. Human grades remain authoritative.",
        "problem_notes": {},
        "problem_types": {},
    }
    _write_json(root / "grades" / "session.json", session)


def _write_segments(root: Path) -> None:
    for problem_id, bbox in QUESTION_BOXES.items():
        rows = []
        for index, student in enumerate(STUDENTS, 1):
            rows.append(
                {
                    "unit_id": index,
                    "page_filename": student.filename,
                    "page_index": 0,
                    "bbox": bbox,
                    "present": True,
                    "comment": f"Precomputed demo crop for {problem_id}.",
                    "confidence": 1.0,
                }
            )
        _write_json(root / "grades" / f"segments_{problem_id}.json", rows)


def _write_readme(root: Path) -> None:
    text = """# FOL Model Exam Demo

FERPA-safe printed fake assignment for testing the inline `fol_model` strip workflow.

Run from the repo root:

```powershell
$env:DATA_DIR='data/samples/fol_model_exam_demo'
python -m uvicorn app.server:app --reload --port 5000
```

Then open:

- `http://127.0.0.1:5000/problem/Q1/strips`
- `http://127.0.0.1:5000/problem/Q2/strips`
- `http://127.0.0.1:5000/problem/Q3/strips`

Each problem is tagged `problem_type: fol_model` in the rubric. The dataset
includes precomputed `grades/segments_Q*.json` files so strip view can open
without asking the local vision model to locate answer boxes first.

Expected behavior:

- `Start OCR` sends each printed answer crop through the FOL model OCR prompt.
- `Save + Check` works even without OCR; edit/type loose model text directly.
- `Apply Suggestion` copies the suggested score/comment into the normal strip
  score/comment fields, but it does not save the whole strip form by itself.
"""
    (root / "README.md").write_text(text, encoding="utf-8")


def generate(output_dir: Path, *, force: bool = False) -> None:
    if output_dir.exists() and force:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for child in ("raw_scans", "rubric", "grades", "debug_fol_model"):
        (output_dir / child).mkdir(parents=True, exist_ok=True)

    _write_roster(output_dir)
    _write_rubric(output_dir)
    for student in STUDENTS:
        _draw_student_page(output_dir, student)
    _write_session(output_dir)
    _write_segments(output_dir)
    _write_readme(output_dir)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate the printed FOL model exam demo dataset.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="DATA_DIR fixture to create.")
    parser.add_argument("--force", action="store_true", help="Delete and recreate the output directory first.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = (ROOT / output_dir).resolve()
    generate(output_dir, force=args.force)
    print(f"Wrote FOL model exam demo to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
