from __future__ import annotations

import argparse
import html
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional

import yaml
from PIL import Image

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from app.autograde import autograde_submission
from scripts.debug_fol_model_ocr import transcribe_image_to_canonical_text


TranscribeFn = Callable[..., tuple[str, str]]


@dataclass
class SyntheticFolOcrCaseResult:
    case_id: str
    unit_id: int
    student_id: str
    problem_id: str
    image_path: str
    crop_path: str
    artifacts_dir: str
    ocr_status: str
    checker_status: str
    expected_passed: Optional[bool]
    ocr_passed: Optional[bool]
    verdict_match: Optional[bool]
    summary: str
    ocr_model_text: str = ""
    gold_model_text: str = ""
    error: str = ""


def run_evaluation(
    *,
    fixture_root: Path,
    output_dir: Path,
    problem_ids: Optional[Iterable[str]] = None,
    student_ids: Optional[Iterable[str]] = None,
    limit: Optional[int] = None,
    transcribe_fn: TranscribeFn = transcribe_image_to_canonical_text,
) -> List[SyntheticFolOcrCaseResult]:
    session = _read_json(fixture_root / "grades" / "session.json")
    gold = _read_json(fixture_root / "gold" / "fol_model_gold.json")
    problems = {str(problem["id"]): problem for problem in session["grading_key"]["problems"]}
    selected_problem_ids = set(problem_ids or problems.keys())
    selected_student_ids = set(student_ids or [])

    output_dir.mkdir(parents=True, exist_ok=True)
    results: List[SyntheticFolOcrCaseResult] = []
    for work_unit in session["work_units"]:
        unit_id = int(work_unit["unit_id"])
        student_id = str(work_unit.get("student_id") or "")
        if selected_student_ids and student_id not in selected_student_ids:
            continue
        page_filenames = list(work_unit.get("page_filenames") or [])
        if not page_filenames:
            continue
        image_path = fixture_root / "raw_scans" / page_filenames[0]

        for problem_id in sorted(selected_problem_ids):
            if problem_id not in problems:
                continue
            if limit is not None and len(results) >= limit:
                _write_summaries(results, output_dir)
                return results

            case_id = f"u{unit_id:03d}_{student_id}_{problem_id}".lower()
            artifacts_dir = output_dir / "cases" / case_id
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            crop_path = artifacts_dir / "crop.png"
            expected = (gold.get("answers") or {}).get(student_id, {}).get(problem_id, {})
            gold_model_text = str(expected.get("model_text") or "")
            (artifacts_dir / "gold_model_text.txt").write_text(gold_model_text, encoding="utf-8")

            try:
                segment = _segment_for_case(fixture_root, problem_id, unit_id)
                _write_crop(image_path=image_path, crop_path=crop_path, bbox=segment["bbox"])
                raw, ocr_model_text = transcribe_fn(
                    image_path=crop_path,
                    output_dir=artifacts_dir,
                    autograde_config=problems[problem_id]["autograde"],
                )
                (artifacts_dir / "ocr_raw_response.txt").write_text(raw, encoding="utf-8")
                (artifacts_dir / "ocr_model_text.txt").write_text(ocr_model_text, encoding="utf-8")
                checker = autograde_submission(
                    problem_type="fol_model",
                    answer_text=ocr_model_text,
                    autograde_config=problems[problem_id]["autograde"],
                    runtime_options={"trace_mode": "terse"},
                )
                expected_passed = _optional_bool(expected.get("expected_passed"))
                verdict_match = checker.passed == expected_passed if expected_passed is not None else None
                results.append(
                    SyntheticFolOcrCaseResult(
                        case_id=case_id,
                        unit_id=unit_id,
                        student_id=student_id,
                        problem_id=problem_id,
                        image_path=str(image_path),
                        crop_path=str(crop_path),
                        artifacts_dir=str(artifacts_dir),
                        ocr_status="ok",
                        checker_status=checker.status,
                        expected_passed=expected_passed,
                        ocr_passed=checker.passed,
                        verdict_match=verdict_match,
                        summary=checker.summary,
                        ocr_model_text=ocr_model_text,
                        gold_model_text=gold_model_text,
                        error="; ".join(checker.errors),
                    )
                )
            except Exception as exc:
                results.append(
                    SyntheticFolOcrCaseResult(
                        case_id=case_id,
                        unit_id=unit_id,
                        student_id=student_id,
                        problem_id=problem_id,
                        image_path=str(image_path),
                        crop_path=str(crop_path),
                        artifacts_dir=str(artifacts_dir),
                        ocr_status="error",
                        checker_status="not_run",
                        expected_passed=_optional_bool(expected.get("expected_passed")),
                        ocr_passed=None,
                        verdict_match=None,
                        summary="Synthetic OCR evaluation failed before checker completed.",
                        ocr_model_text="",
                        gold_model_text=gold_model_text,
                        error=str(exc),
                    )
                )

    _write_summaries(results, output_dir)
    return results


def _segment_for_case(fixture_root: Path, problem_id: str, unit_id: int) -> Mapping[str, Any]:
    segments = _read_json(fixture_root / "grades" / f"segments_{problem_id}.json")
    for row in segments:
        if int(row.get("unit_id", -1)) == unit_id:
            return row
    raise FileNotFoundError(f"No segment found for {problem_id} unit {unit_id}.")


def _write_crop(*, image_path: Path, crop_path: Path, bbox: Any) -> None:
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise ValueError(f"Bad bbox for crop: {bbox!r}")
    with Image.open(image_path) as image:
        w, h = image.size
        x1 = max(0, min(w - 1, int(float(bbox[0]) * w)))
        y1 = max(0, min(h - 1, int(float(bbox[1]) * h)))
        x2 = max(x1 + 1, min(w, int(float(bbox[2]) * w)))
        y2 = max(y1 + 1, min(h, int(float(bbox[3]) * h)))
        crop_path.parent.mkdir(parents=True, exist_ok=True)
        image.crop((x1, y1, x2, y2)).save(crop_path)


def _write_summaries(results: List[SyntheticFolOcrCaseResult], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps([asdict(item) for item in results], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_html_summary(results, output_dir / "summary.html", output_dir)


def _write_html_summary(results: List[SyntheticFolOcrCaseResult], output_path: Path, output_dir: Path) -> None:
    total = len(results)
    ocr_ok = sum(1 for item in results if item.ocr_status == "ok")
    checker_ok = sum(1 for item in results if item.checker_status == "scored")
    matches = sum(1 for item in results if item.verdict_match is True)
    mismatches = sum(1 for item in results if item.verdict_match is False)

    rows = []
    for item in results:
        crop_rel = _safe_relpath(Path(item.crop_path), output_dir)
        row_class = "pass" if item.verdict_match is True else ("fail" if item.verdict_match is False else "unknown")
        rows.append(
            f"""
            <tr class="{row_class}">
              <td>{html.escape(item.case_id)}</td>
              <td>{html.escape(item.problem_id)}</td>
              <td>{html.escape(item.student_id)}</td>
              <td><img src="{html.escape(crop_rel)}" alt="crop for {html.escape(item.case_id)}" /></td>
              <td>{html.escape(item.ocr_status)} / {html.escape(item.checker_status)}</td>
              <td>{item.expected_passed}</td>
              <td>{item.ocr_passed}</td>
              <td>{item.verdict_match}</td>
              <td><pre>{html.escape(item.ocr_model_text)}</pre></td>
              <td><pre>{html.escape(item.gold_model_text)}</pre></td>
              <td>{html.escape(item.summary)}<br /><pre>{html.escape(item.error)}</pre></td>
            </tr>
            """
        )

    html_text = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>Synthetic FOL OCR Evaluation</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 2rem; color: #1f2937; background: #f6f8fb; }}
    table {{ width: 100%; border-collapse: collapse; background: white; }}
    th, td {{ border: 1px solid #d1d5db; padding: 0.65rem; vertical-align: top; text-align: left; }}
    th {{ background: #e5eef9; }}
    tr.pass {{ background: #f0fdf4; }}
    tr.fail {{ background: #fef2f2; }}
    img {{ max-width: 260px; border: 1px solid #cbd5e1; }}
    pre {{ margin: 0.25rem 0 0; white-space: pre-wrap; font-family: Consolas, monospace; font-size: 0.85rem; }}
    .metrics {{ display: flex; gap: 1rem; flex-wrap: wrap; margin-bottom: 1rem; }}
    .metric {{ background: white; border: 1px solid #d1d5db; padding: 0.75rem 1rem; border-radius: 6px; }}
  </style>
</head>
<body>
  <h1>Synthetic FOL OCR Evaluation</h1>
  <div class="metrics">
    <div class="metric"><strong>Total</strong><br />{total}</div>
    <div class="metric"><strong>OCR OK</strong><br />{ocr_ok}</div>
    <div class="metric"><strong>Checker scored</strong><br />{checker_ok}</div>
    <div class="metric"><strong>Verdict matches</strong><br />{matches}</div>
    <div class="metric"><strong>Verdict mismatches</strong><br />{mismatches}</div>
  </div>
  <table>
    <thead>
      <tr>
        <th>Case</th>
        <th>Problem</th>
        <th>Student</th>
        <th>Crop</th>
        <th>Status</th>
        <th>Expected</th>
        <th>OCR Verdict</th>
        <th>Match</th>
        <th>OCR Model Text</th>
        <th>Gold Model Text</th>
        <th>Summary / Error</th>
      </tr>
    </thead>
    <tbody>
      {''.join(rows)}
    </tbody>
  </table>
</body>
</html>
"""
    output_path.write_text(html_text, encoding="utf-8")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _optional_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    return bool(value)


def _safe_relpath(path: Path, start: Path) -> str:
    try:
        return path.resolve().relative_to(start.resolve()).as_posix()
    except Exception:
        return path.as_posix()


def _parse_csv(raw: str | None) -> Optional[List[str]]:
    if not raw:
        return None
    return [part.strip() for part in raw.split(",") if part.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run OCR/checker evaluation over a synthetic FOL model fixture.")
    parser.add_argument("--fixture-root", default="data/samples/synthetic_fol_models", help="Synthetic DATA_DIR fixture root.")
    parser.add_argument("--output-dir", default="working/synthetic_fol_ocr_report", help="Directory for report artifacts.")
    parser.add_argument("--problem-id", help="Comma-separated problem ids to run, e.g. Q1,Q2.")
    parser.add_argument("--student-id", help="Comma-separated student ids to run, e.g. SYN001,SYN002.")
    parser.add_argument("--limit", type=int, help="Maximum number of cases to run.")
    args = parser.parse_args()

    results = run_evaluation(
        fixture_root=Path(args.fixture_root),
        output_dir=Path(args.output_dir),
        problem_ids=_parse_csv(args.problem_id),
        student_ids=_parse_csv(args.student_id),
        limit=args.limit,
    )

    matches = sum(1 for item in results if item.verdict_match is True)
    mismatches = sum(1 for item in results if item.verdict_match is False)
    print(f"[SYNTH-OCR] Wrote {Path(args.output_dir) / 'summary.json'}")
    print(f"[SYNTH-OCR] Wrote {Path(args.output_dir) / 'summary.html'}")
    print(f"[SYNTH-OCR] Cases: {len(results)}; matches: {matches}; mismatches: {mismatches}")
    for item in results:
        print(
            f"- {item.case_id}: ocr={item.ocr_status}, checker={item.checker_status}, "
            f"expected={item.expected_passed}, ocr_passed={item.ocr_passed}, match={item.verdict_match}"
        )
    return 0 if mismatches == 0 and all(item.ocr_status == "ok" for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
