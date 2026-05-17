"""
Generate FERPA-safe synthetic logic exam fixtures for OCR/autograde debugging.

First supported family:
- fol_model

The output is a complete DATA_DIR fixture:
- roster.csv
- rubric/synthetic_fol_models.yaml
- raw_scans/*.png
- grades/session.json
- grades/segments_Q*.json
- gold/fol_model_gold.json
- README.md

Run:
  python scripts/generate_synthetic_logic_exam.py --family fol_model --style phone_scan_light --count 8 --force

Then start the app with:
  $env:DATA_DIR='data/samples/synthetic_fol_models'
  python -m uvicorn app.server:app --reload --port 5000
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping

import yaml
from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "data" / "samples" / "synthetic_fol_models"

PAGE_WIDTH = 1600
PAGE_HEIGHT = 2100
QUESTION_BOXES: Dict[str, List[float]] = {
    "Q1": [0.06, 0.18, 0.94, 0.39],
    "Q2": [0.06, 0.43, 0.94, 0.64],
    "Q3": [0.06, 0.68, 0.94, 0.89],
}


@dataclass(frozen=True)
class ProblemTemplate:
    problem_id: str
    label: str
    points: int
    prompt: str
    autograde: Dict[str, Any]


@dataclass(frozen=True)
class AnswerVariant:
    model_text: str
    expected: bool
    note: str


PROBLEMS: List[ProblemTemplate] = [
    ProblemTemplate(
        problem_id="Q1",
        label="Give a model where the two sentences have different truth values.",
        points=3,
        prompt="Give a model showing that ∀x Px and ∃x Px are not equivalent.",
        autograde={
            "task_kind": "non_equivalence",
            "default_score_correct": 3,
            "default_score_incorrect": 0,
            "prompt_data": {"formulas": ["∀x Px", "∃x Px"]},
            "ocr": {"expected_answer_shape": "model_only"},
        },
    ),
    ProblemTemplate(
        problem_id="Q2",
        label="Give a countermodel to the argument.",
        points=3,
        prompt="Make ∀x(Px → Qx), ∃x Px true and ∀x Qx false.",
        autograde={
            "task_kind": "non_validity",
            "default_score_correct": 3,
            "default_score_incorrect": 0,
            "prompt_data": {
                "premises": ["∀x (Px → Qx)", "∃x Px"],
                "conclusion": "∀x Qx",
            },
            "ocr": {"expected_answer_shape": "model_only"},
        },
    ),
    ProblemTemplate(
        problem_id="Q3",
        label="Give one model making all target sentences true.",
        points=3,
        prompt="Make ∃x Px, ∃x Qx, and ∀x(Px → ¬Qx) all true.",
        autograde={
            "task_kind": "joint_consistency",
            "default_score_correct": 3,
            "default_score_incorrect": 0,
            "prompt_data": {"formulas": ["∃x Px", "∃x Qx", "∀x (Px → ¬Qx)"]},
            "ocr": {"expected_answer_shape": "model_only"},
        },
    ),
]


ANSWER_BANK: Dict[str, List[AnswerVariant]] = {
    "Q1": [
        AnswerVariant("Domain: 1,2\nP: 1", True, "standard two-object witness"),
        AnswerVariant("DOMAIN: {1}\n\nCONSTANTS:\n\nPREDICATES:\nP: {1}", False, "one-object all-P model"),
        AnswerVariant("Domain: 1,2,3\nPredicates:\nP: 1,3", True, "larger witness"),
        AnswerVariant("Domain: 1,2\nP: 1,2", False, "universal and existential both true"),
    ],
    "Q2": [
        AnswerVariant("Domain: 1,2\nP: 1\nQ: 1", True, "countermodel: second object not Q"),
        AnswerVariant("Domain: 1,2\nP: 1\nQ: 1,2", False, "conclusion true"),
        AnswerVariant("Domain: 1,2,3\nP: 2\nQ: 2", True, "larger countermodel"),
        AnswerVariant("DOMAIN: {1,2}\nPREDICATES:\nP: {1,2}\nQ: {1}", False, "premise false"),
    ],
    "Q3": [
        AnswerVariant("Domain: 1,2\nP: 1\nQ: 2", True, "separates P and Q"),
        AnswerVariant("Domain: 1,2\nP: 1\nQ: 1", False, "violates disjointness target"),
        AnswerVariant("Domain: 1,2,3\nP: 1,3\nQ: 2", True, "three-object witness"),
        AnswerVariant("DOMAIN: {1}\nPREDICATES:\nP: {1}\nQ: {}", False, "no Q witness"),
    ],
}


STUDENT_NAMES = [
    "Ada Example",
    "Bert Sample",
    "Cleo Demo",
    "Dev Test",
    "Eli Fixture",
    "Fran Synthetic",
    "Gus Placeholder",
    "Hana Trial",
    "Ivy Mock",
    "Jules Sandbox",
    "Kai Local",
    "Lena Probe",
]


def generate(
    output_dir: Path,
    *,
    family: str = "fol_model",
    style: str = "printed",
    count: int = 8,
    seed: int = 1337,
    force: bool = False,
) -> None:
    if family != "fol_model":
        raise ValueError("Only family='fol_model' is supported in the first synthetic generator pass.")
    if style not in {"printed", "handwriting_font", "phone_scan_light"}:
        raise ValueError("style must be one of: printed, handwriting_font, phone_scan_light")
    if count < 1:
        raise ValueError("count must be at least 1")

    rng = random.Random(seed)
    if output_dir.exists() and force:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for child in ("raw_scans", "rubric", "grades", "gold", "debug_fol_model"):
        (output_dir / child).mkdir(parents=True, exist_ok=True)

    students = _student_records(count)
    assignments = _assign_answers(students, rng)

    _write_roster(output_dir, students)
    _write_rubric(output_dir)
    for student in students:
        _draw_student_page(output_dir, student, assignments[student["student_id"]], style=style, rng=rng)
    _write_session(output_dir, students, style=style)
    _write_segments(output_dir, students)
    _write_gold(output_dir, students, assignments, style=style, seed=seed)
    _write_readme(output_dir, style=style, count=count, seed=seed)


def _student_records(count: int) -> List[Dict[str, str]]:
    records: List[Dict[str, str]] = []
    for index in range(count):
        name = STUDENT_NAMES[index % len(STUDENT_NAMES)]
        sid = f"SYN{index + 1:03d}"
        slug = "_".join(name.lower().split())
        records.append(
            {
                "student_id": sid,
                "name": name,
                "filename": f"synthetic_fol_{index + 1:03d}_{slug}_p01.png",
            }
        )
    return records


def _assign_answers(students: List[Dict[str, str]], rng: random.Random) -> Dict[str, Dict[str, AnswerVariant]]:
    assignments: Dict[str, Dict[str, AnswerVariant]] = {}
    for student_index, student in enumerate(students):
        per_problem: Dict[str, AnswerVariant] = {}
        for problem in PROBLEMS:
            bank = ANSWER_BANK[problem.problem_id]
            base_index = student_index % len(bank)
            if student_index >= len(bank) and rng.random() < 0.35:
                base_index = rng.randrange(len(bank))
            per_problem[problem.problem_id] = bank[base_index]
        assignments[student["student_id"]] = per_problem
    return assignments


def _write_roster(root: Path, students: List[Dict[str, str]]) -> None:
    lines = ["student_id,name"]
    lines.extend(f"{student['student_id']},{student['name']}" for student in students)
    (root / "roster.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_rubric(root: Path) -> None:
    rubric = {
        "quiz_id": "synthetic_fol_models",
        "max_points": sum(problem.points for problem in PROBLEMS),
        "problems": [
            {
                "id": problem.problem_id,
                "label": problem.label,
                "points": problem.points,
                "problem_type": "fol_model",
                "section": "Synthetic FOL Models",
                "problem_number": str(index),
                "prompt": problem.prompt,
                "segmentation_hint": f"Look for the {problem.problem_id} model answer box.",
                "autograde": problem.autograde,
            }
            for index, problem in enumerate(PROBLEMS, start=1)
        ],
    }
    _write_yaml(root / "rubric" / "synthetic_fol_models.yaml", rubric)


def _write_session(root: Path, students: List[Dict[str, str]], *, style: str) -> None:
    problem_ids = [problem.problem_id for problem in PROBLEMS]
    session = {
        "roster": [{"student_id": s["student_id"], "name": s["name"]} for s in students],
        "grading_key": yaml.safe_load((root / "rubric" / "synthetic_fol_models.yaml").read_text(encoding="utf-8")),
        "work_units": [
            {
                "unit_id": index,
                "page_filenames": [student["filename"]],
                "student_id": student["student_id"],
                "per_problem": {problem_id: None for problem_id in problem_ids},
                "per_problem_comments": {problem_id: None for problem_id in problem_ids},
                "comment": None,
            }
            for index, student in enumerate(students, start=1)
        ],
        "unclaimed_pages": [],
        "pinned_note": f"Synthetic FOL model fixture generated in {style!r} style. Human grades remain authoritative.",
        "problem_notes": {},
        "problem_types": {},
    }
    _write_json(root / "grades" / "session.json", session)


def _write_segments(root: Path, students: List[Dict[str, str]]) -> None:
    for problem in PROBLEMS:
        rows = [
            {
                "unit_id": index,
                "page_filename": student["filename"],
                "page_index": 0,
                "bbox": QUESTION_BOXES[problem.problem_id],
                "present": True,
                "comment": f"Precomputed synthetic crop for {problem.problem_id}.",
                "confidence": 1.0,
            }
            for index, student in enumerate(students, start=1)
        ]
        _write_json(root / "grades" / f"segments_{problem.problem_id}.json", rows)


def _write_gold(
    root: Path,
    students: List[Dict[str, str]],
    assignments: Dict[str, Dict[str, AnswerVariant]],
    *,
    style: str,
    seed: int,
) -> None:
    payload: Dict[str, Any] = {
        "fixture_kind": "synthetic_logic_exam",
        "family": "fol_model",
        "style": style,
        "seed": seed,
        "problems": {problem.problem_id: problem.autograde for problem in PROBLEMS},
        "answers": {},
    }
    answers: Dict[str, Any] = {}
    for student in students:
        sid = student["student_id"]
        answers[sid] = {
            problem_id: {
                "model_text": variant.model_text,
                "expected_passed": variant.expected,
                "note": variant.note,
            }
            for problem_id, variant in assignments[sid].items()
        }
    payload["answers"] = answers
    _write_json(root / "gold" / "fol_model_gold.json", payload)


def _draw_student_page(
    root: Path,
    student: Mapping[str, str],
    answers: Mapping[str, AnswerVariant],
    *,
    style: str,
    rng: random.Random,
) -> None:
    paper = _paper_color(style)
    img = Image.new("RGB", (PAGE_WIDTH, PAGE_HEIGHT), paper)
    draw = ImageDraw.Draw(img)
    if style in {"handwriting_font", "phone_scan_light"}:
        _draw_lined_paper(draw, rng)

    title_font = _load_font(48, role="body")
    body_font = _load_font(29, role="body")
    small_font = _load_font(23, role="body")
    answer_font = _load_font(37 if style == "printed" else 41, role="answer" if style != "printed" else "mono")

    ink = _ink_color(style)
    draw.text((80, 55), "Synthetic FOL Model Exam", fill="#111111", font=title_font)
    draw.text((80, 125), f"Student: {student['name']}", fill="#111111", font=body_font)
    draw.text((80, 170), f"ID: {student['student_id']}", fill="#111111", font=body_font)
    draw.text((80, 220), f"Style: {style}", fill="#555555", font=small_font)

    for problem in PROBLEMS:
        bbox = QUESTION_BOXES[problem.problem_id]
        x1 = int(bbox[0] * PAGE_WIDTH)
        y1 = int(bbox[1] * PAGE_HEIGHT)
        x2 = int(bbox[2] * PAGE_WIDTH)
        y2 = int(bbox[3] * PAGE_HEIGHT)
        outline = "#222222" if style == "printed" else "#7c8da0"
        draw.rounded_rectangle((x1, y1, x2, y2), radius=8, outline=outline, width=3)
        draw.text((x1 + 24, y1 + 18), f"{problem.problem_id}. {problem.prompt}", fill="#111111", font=body_font)
        _draw_answer_text(
            draw,
            (x1 + 54, y1 + 84),
            answers[problem.problem_id].model_text,
            font=answer_font,
            fill=ink,
            style=style,
            rng=rng,
        )

    if style == "phone_scan_light":
        img = _apply_phone_scan_effects(img, rng)
    elif style == "handwriting_font":
        img = _apply_light_handwriting_effects(img, rng)

    out_path = root / "raw_scans" / student["filename"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, format="PNG")


def _draw_answer_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    *,
    font: ImageFont.ImageFont,
    fill: str,
    style: str,
    rng: random.Random,
) -> None:
    base_x, y = xy
    for line in text.splitlines():
        x = base_x
        if style in {"handwriting_font", "phone_scan_light"}:
            x += rng.randint(-6, 10)
            y += rng.randint(-2, 3)
        draw.text((x, y), line, fill=fill, font=font)
        bbox = draw.textbbox((x, y), line or " ", font=font)
        line_height = bbox[3] - bbox[1]
        y += line_height + (11 if style == "printed" else rng.randint(10, 18))


def _draw_lined_paper(draw: ImageDraw.ImageDraw, rng: random.Random) -> None:
    spacing = 58
    line_color = "#d8e7f1"
    margin_color = "#f2b5b5"
    for y in range(300 + rng.randint(-4, 4), PAGE_HEIGHT - 80, spacing):
        draw.line((70, y, PAGE_WIDTH - 70, y + rng.randint(-1, 1)), fill=line_color, width=2)
    draw.line((132, 290, 132, PAGE_HEIGHT - 80), fill=margin_color, width=2)


def _apply_light_handwriting_effects(img: Image.Image, rng: random.Random) -> Image.Image:
    img = ImageEnhance.Contrast(img).enhance(0.94)
    img = ImageEnhance.Color(img).enhance(0.96)
    if rng.random() < 0.5:
        img = img.filter(ImageFilter.GaussianBlur(radius=0.25))
    return img


def _apply_phone_scan_effects(img: Image.Image, rng: random.Random) -> Image.Image:
    img = _affine_jitter(img, rng)
    img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.82, 0.95))
    img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.94, 1.04))
    img = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.25, 0.55)))
    img = _add_shadow_gradient(img, rng)
    img = _add_noise(img, rng)
    return img


def _affine_jitter(img: Image.Image, rng: random.Random) -> Image.Image:
    shear = rng.uniform(-0.018, 0.018)
    shifted = img.transform(
        img.size,
        Image.Transform.AFFINE,
        (1, shear, -shear * img.height / 2, rng.uniform(-0.01, 0.01), 1, 0),
        resample=Image.Resampling.BICUBIC,
        fillcolor=_paper_color("phone_scan_light"),
    )
    angle = rng.uniform(-1.1, 1.1)
    return shifted.rotate(angle, resample=Image.Resampling.BICUBIC, fillcolor=_paper_color("phone_scan_light"))


def _add_shadow_gradient(img: Image.Image, rng: random.Random) -> Image.Image:
    width, height = img.size
    overlay = Image.new("L", img.size, 0)
    draw = ImageDraw.Draw(overlay)
    max_alpha = rng.randint(18, 36)
    for x in range(width):
        alpha = int(max_alpha * (x / width) ** 1.7)
        draw.line((x, 0, x, height), fill=alpha)
    shadow = Image.new("RGB", img.size, (70, 65, 55))
    return Image.composite(shadow, img, overlay)


def _add_noise(img: Image.Image, rng: random.Random) -> Image.Image:
    noise = Image.effect_noise(img.size, rng.uniform(7.0, 14.0)).convert("L")
    noise_rgb = Image.merge("RGB", (noise, noise, noise))
    blended = ImageChops.add(img, ImageEnhance.Contrast(noise_rgb).enhance(0.18), scale=2.0, offset=0)
    return Image.blend(img, blended, 0.18)


def _paper_color(style: str) -> str:
    if style == "phone_scan_light":
        return "#fbf8ef"
    if style == "handwriting_font":
        return "#fffdf7"
    return "white"


def _ink_color(style: str) -> str:
    if style == "printed":
        return "#111111"
    if style == "phone_scan_light":
        return "#253c78"
    return "#1f3e85"


def _load_font(size: int, *, role: str) -> ImageFont.ImageFont:
    candidates: List[str]
    if role == "answer":
        candidates = [
            "segoepr.ttf",
            "segoeprb.ttf",
            "Inkfree.ttf",
            "comic.ttf",
            "BRADHITC.TTF",
            "Gabriola.ttf",
            "arial.ttf",
        ]
    elif role == "mono":
        candidates = ["DejaVuSansMono.ttf", "consola.ttf", "cour.ttf", "arial.ttf"]
    else:
        candidates = ["DejaVuSans.ttf", "seguisym.ttf", "arial.ttf", "calibri.ttf"]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def _write_yaml(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_readme(root: Path, *, style: str, count: int, seed: int) -> None:
    text = f"""# Synthetic FOL Models

FERPA-safe synthetic fixture for FOL model OCR/autograde debugging.

- family: `fol_model`
- style: `{style}`
- count: `{count}`
- seed: `{seed}`

Run from the repo root:

```powershell
$env:DATA_DIR='{root.as_posix()}'
python -m uvicorn app.server:app --reload --port 5000
```

Then open:

- `http://127.0.0.1:5000/problem/Q1/strips`
- `http://127.0.0.1:5000/problem/Q2/strips`
- `http://127.0.0.1:5000/problem/Q3/strips`

Gold metadata lives at:

- `gold/fol_model_gold.json`

This fixture intentionally separates semantic variation from visual variation:
the gold file records the intended model text and checker pass/fail verdict for
each student/problem, while the PNG pages provide increasingly noisy renderings
for OCR and strip-view debugging.
"""
    (root / "README.md").write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate synthetic logic exam fixtures.")
    parser.add_argument("--family", default="fol_model", choices=["fol_model"], help="Problem family to generate.")
    parser.add_argument(
        "--style",
        default="printed",
        choices=["printed", "handwriting_font", "phone_scan_light"],
        help="Visual rendering style.",
    )
    parser.add_argument("--count", type=int, default=8, help="Number of synthetic students/pages.")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed for answer assignment and visual jitter.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="DATA_DIR fixture to create.")
    parser.add_argument("--force", action="store_true", help="Delete and recreate the output directory first.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = (ROOT / output_dir).resolve()
    generate(output_dir, family=args.family, style=args.style, count=args.count, seed=args.seed, force=args.force)
    print(f"Wrote synthetic {args.family} fixture to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
