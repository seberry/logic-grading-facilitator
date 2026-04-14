from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from app.autograde import autograde_submission
from scripts.debug_fol_model_ocr import _load_autograde_config, transcribe_image_to_canonical_text


@dataclass
class BatchSampleResult:
    scan_slug: str
    image_path: str
    config_source: str
    ocr_status: str
    checker_status: str
    passed: Optional[bool]
    score: Optional[float]
    summary: str
    error: str = ""


def _find_image(sample_dir: Path) -> Path:
    candidates = sorted(
        p for p in sample_dir.iterdir() if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    if not candidates:
        raise FileNotFoundError(f"No image file found in {sample_dir}")
    return candidates[0]


def _load_checker_text(sample_dir: Path, mode: str) -> str:
    state_path = sample_dir / "state.json"
    if not state_path.exists():
        raise FileNotFoundError(f"No state.json found in {sample_dir}")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if mode == "gold":
        return str(state.get("model_text") or "")

    proposed_path = sample_dir / "batch_ocr" / "ocr_model_text.txt"
    if not proposed_path.exists():
        raise FileNotFoundError(f"OCR proposal not found at {proposed_path}")
    return proposed_path.read_text(encoding="utf-8")


def _write_html_summary(results: List[BatchSampleResult], output_path: Path) -> None:
    rows = []
    for item in results:
        rows.append(
            f"""
            <tr>
              <td>{item.scan_slug}</td>
              <td>{item.ocr_status}</td>
              <td>{item.checker_status}</td>
              <td>{item.passed}</td>
              <td>{"" if item.score is None else item.score}</td>
              <td>{item.summary}</td>
              <td><pre>{item.error}</pre></td>
            </tr>
            """
        )
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>FOL OCR Batch Summary</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 2rem; background: #f6f8fb; color: #1f2937; }}
    table {{ width: 100%; border-collapse: collapse; background: #fff; }}
    th, td {{ border: 1px solid #d1d5db; padding: 0.75rem; text-align: left; vertical-align: top; }}
    th {{ background: #e5eef9; }}
    pre {{ margin: 0; white-space: pre-wrap; font-family: Consolas, monospace; }}
  </style>
</head>
<body>
  <h1>FOL OCR Batch Summary</h1>
  <table>
    <thead>
      <tr>
        <th>Sample</th>
        <th>OCR</th>
        <th>Checker</th>
        <th>Passed</th>
        <th>Score</th>
        <th>Summary</th>
        <th>Error</th>
      </tr>
    </thead>
    <tbody>
      {''.join(rows)}
    </tbody>
  </table>
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Batch-run the FOL OCR handoff pipeline.\n"
            "For each sample folder: image -> OCR proposal -> deterministic checker."
        )
    )
    parser.add_argument(
        "--samples-root",
        default="data/samples/debug_fol_model",
        help="Root directory containing one sample folder per scan slug.",
    )
    parser.add_argument(
        "--checker-input",
        choices=("ocr", "gold"),
        default="ocr",
        help="Use OCR proposal text or the sample's known-good model_text as checker input.",
    )
    parser.add_argument(
        "--output-dir",
        default="working/fol_ocr_batch",
        help="Directory for batch summary artifacts.",
    )
    args = parser.parse_args()

    samples_root = Path(args.samples_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: List[BatchSampleResult] = []
    for sample_dir in sorted(p for p in samples_root.iterdir() if p.is_dir()):
        state_path = sample_dir / "state.json"
        if not state_path.exists():
            continue

        scan_slug = sample_dir.name
        image_path = _find_image(sample_dir)
        sample_output_dir = sample_dir / "batch_ocr"
        try:
            autograde_config = _load_autograde_config(state_path)

            # This simulates the "Re-run OCR" button: produce a proposed canonical
            # transcription exactly as the browser flow would, but in batch.
            transcribe_image_to_canonical_text(
                image_path=image_path,
                output_dir=sample_output_dir,
                autograde_config=autograde_config,
            )
            checker_text = _load_checker_text(sample_dir, args.checker_input)
            result = autograde_submission(
                problem_type="fol_model",
                answer_text=checker_text,
                autograde_config=autograde_config,
                runtime_options={"trace_mode": "terse"},
            )
            results.append(
                BatchSampleResult(
                    scan_slug=scan_slug,
                    image_path=str(image_path),
                    config_source=str(state_path),
                    ocr_status="ok",
                    checker_status=result.status,
                    passed=result.passed,
                    score=result.score,
                    summary=result.summary,
                    error="; ".join(result.errors),
                )
            )
        except Exception as exc:
            results.append(
                BatchSampleResult(
                    scan_slug=scan_slug,
                    image_path=str(image_path),
                    config_source=str(state_path),
                    ocr_status="error",
                    checker_status="not_run",
                    passed=None,
                    score=None,
                    summary="Batch run failed before checker completed.",
                    error=str(exc),
                )
            )

    json_path = output_dir / "summary.json"
    html_path = output_dir / "summary.html"
    json_path.write_text(json.dumps([asdict(item) for item in results], indent=2), encoding="utf-8")
    _write_html_summary(results, html_path)

    print(f"[BATCH] Wrote JSON summary to {json_path}")
    print(f"[BATCH] Wrote HTML summary to {html_path}")
    for item in results:
        print(
            f"- {item.scan_slug}: ocr={item.ocr_status}, checker={item.checker_status}, "
            f"passed={item.passed}, score={item.score}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
