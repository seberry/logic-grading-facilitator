#!/usr/bin/env python
"""Probe per-row justification rereads on a saved full-width Fitch crop.

This is an experiment script, not part of the production OCR pipeline.  It
extracts generous full-width row bands from an existing crop image and asks the
default LM Studio vision model to read only the justification for each row.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import cv2

from app.vlm_client import LMStudioQwenClient
from proofs_ocr.common.rows import band_height, detect_text_row_bands


PROMPT = """You are transcribing one row from a handwritten Fitch proof.
The image is a full-width horizontal row crop.  It may include the printed line
number on the left, the formula in the middle, and the justification on the
right.  Return only JSON with keys "line", "wffstr", and "jstr".
Use the absolute line number visible at the left if present.  Put only the
right-side rule and line references in "jstr".  If no justification is printed,
use an empty string.  Do not infer or repair the student's proof."""


def _json_from_text(text: str) -> Any:
    content = text.strip()
    if "```json" in content:
        content = content.split("```json", 1)[1].split("```", 1)[0]
    elif "```" in content:
        content = content.split("```", 1)[1].split("```", 1)[0]
    return json.loads(content)


def _safe_stem(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "crop"


def extract_row_crops(
    crop_path: Path,
    output_dir: Path,
    padding: int,
    *,
    expected_lines: int | None = None,
) -> list[Path]:
    img = cv2.imread(str(crop_path))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {crop_path}")

    bands = detect_text_row_bands(img)
    bands = [band for band in bands if band_height(band) >= 18]
    if not bands:
        raise RuntimeError(f"No text row bands detected in {crop_path}")
    if expected_lines and len(bands) != expected_lines:
        print(
            f"Detected {len(bands)} row band(s), expected {expected_lines}; "
            "using evenly spaced overlapping row windows."
        )
        h_img = img.shape[0]
        step = h_img / float(expected_lines)
        bands = []
        for idx in range(expected_lines):
            center = (idx + 0.5) * step
            half_height = max(24.0, step * 0.42)
            bands.append((int(max(0, center - half_height)), int(min(h_img - 1, center + half_height))))

    output_dir.mkdir(parents=True, exist_ok=True)
    h_img = img.shape[0]
    row_paths: list[Path] = []
    stem = _safe_stem(crop_path.stem)

    for index, band in enumerate(bands, start=1):
        y1 = max(0, int(band[0]) - padding)
        y2 = min(h_img, int(band[1]) + padding)
        row_img = img[y1:y2, :]
        row_path = output_dir / f"{stem}_row_{index:02d}_y{y1}_{y2}.jpg"
        cv2.imwrite(str(row_path), row_img)
        row_paths.append(row_path)

    return row_paths


def reread_rows(row_paths: list[Path], *, model: str | None, timeout: int) -> list[dict[str, Any]]:
    client = LMStudioQwenClient(model=model, timeout=timeout)
    records: list[dict[str, Any]] = []

    for index, row_path in enumerate(row_paths, start=1):
        print(f"[row {index}] Sending {row_path.name}...")
        messages = client.build_messages_for_prompt(row_path, PROMPT)
        started = time.time()
        raw = client.chat(messages, model=model, timeout=timeout, max_tokens=256)
        elapsed = time.time() - started
        try:
            parsed = _json_from_text(raw)
        except Exception as exc:  # noqa: BLE001 - probe should record parse failures.
            parsed = {"parse_error": str(exc), "raw": raw}

        record = {
            "row_index": index,
            "row_crop": str(row_path),
            "elapsed_s": round(elapsed, 3),
            "parsed": parsed,
            "raw": raw,
            "metadata": getattr(client, "last_response_metadata", {}) or {},
        }
        records.append(record)
        print(json.dumps(parsed, ensure_ascii=False))

    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("crop", type=Path, help="Saved full-width crop image to split into rows.")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "working" / "tfl_row_reread")
    parser.add_argument("--padding", type=int, default=18, help="Vertical padding around each detected row band.")
    parser.add_argument("--expected-lines", type=int, default=None, help="Use overlapping equal row windows if detected row count differs.")
    parser.add_argument("--model", default=None, help="LM Studio model override. Defaults to the configured 4b model.")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--no-call", action="store_true", help="Only write row crops; do not call the model.")
    args = parser.parse_args()

    crop_path = args.crop.resolve()
    output_dir = args.output_dir.resolve()
    row_paths = extract_row_crops(
        crop_path,
        output_dir,
        args.padding,
        expected_lines=args.expected_lines,
    )
    print(f"Wrote {len(row_paths)} row crop(s) to {output_dir}")
    for row_path in row_paths:
        print(f"  {row_path}")

    if args.no_call:
        return

    records = reread_rows(row_paths, model=args.model, timeout=args.timeout)
    report_path = output_dir / f"{_safe_stem(crop_path.stem)}_row_reread.json"
    report_path.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
