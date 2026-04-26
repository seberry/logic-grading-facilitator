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


ORDERED_JUSTIFICATIONS_PROMPT = """You are reading the RIGHT-SIDE justifications
in a cropped handwritten Fitch subproof.

Important: the first/top proof line in this child subproof is an assumption or
premise and has NO printed justification.  Ignore that top assumption line.

Return only a JSON array of strings: the printed justifications for the remaining
derived lines, in top-to-bottom order.  Do not include formulas.  Do not infer
corrections.  Preserve rule symbols and citation numbers as written.

Example output shape: ["∧E 7", "→E 2,8", "∧I 8,9"]"""


SEPARATED_LISTS_PROMPT = """You are transcribing a cropped handwritten Fitch
subproof.  Return exactly one JSON object with keys "formulas" and
"justifications".

Rules:
- "formulas" is a top-to-bottom array of every formula visible in the subproof.
- The first/top formula is the child subproof assumption or premise.
- The first/top formula has no printed justification.  Do not include a
  justification for it.
- "justifications" is a top-to-bottom array of the printed right-side
  justifications for only the derived lines after the first formula.
- Preserve citation numbers exactly as written.  They are absolute proof line
  numbers, not local crop numbers.
- Do not infer corrections or validate the proof.

Example output shape:
{"formulas": ["W∧A", "W", "S", "S∧A"], "justifications": ["∧E 7", "→E 2,8", "∧I 8,9"]}"""


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


def reread_ordered_justifications(crop_path: Path, *, model: str | None, timeout: int) -> dict[str, Any]:
    client = LMStudioQwenClient(model=model, timeout=timeout)
    print(f"[ordered] Sending {crop_path.name}...")
    messages = client.build_messages_for_prompt(crop_path, ORDERED_JUSTIFICATIONS_PROMPT)
    started = time.time()
    raw = client.chat(messages, model=model, timeout=timeout, max_tokens=128)
    elapsed = time.time() - started
    try:
        parsed = _json_from_text(raw)
    except Exception as exc:  # noqa: BLE001 - probe should record parse failures.
        parsed = {"parse_error": str(exc), "raw": raw}

    record = {
        "crop": str(crop_path),
        "elapsed_s": round(elapsed, 3),
        "parsed": parsed,
        "raw": raw,
        "metadata": getattr(client, "last_response_metadata", {}) or {},
    }
    print(json.dumps(parsed, ensure_ascii=False))
    return record


def _stitch_separated_lists(parsed: Any) -> list[dict[str, str]] | None:
    if not isinstance(parsed, dict):
        return None
    formulas = parsed.get("formulas")
    justifications = parsed.get("justifications")
    if not isinstance(formulas, list) or not isinstance(justifications, list):
        return None
    if len(formulas) != len(justifications) + 1:
        return None

    stitched: list[dict[str, str]] = []
    for index, formula in enumerate(formulas):
        jstr = "" if index == 0 else str(justifications[index - 1] or "")
        stitched.append({"wffstr": str(formula or ""), "jstr": jstr})
    return stitched


def reread_separated_lists(crop_path: Path, *, model: str | None, timeout: int) -> dict[str, Any]:
    client = LMStudioQwenClient(model=model, timeout=timeout)
    print(f"[separated] Sending {crop_path.name}...")
    messages = client.build_messages_for_prompt(crop_path, SEPARATED_LISTS_PROMPT)
    started = time.time()
    raw = client.chat(messages, model=model, timeout=timeout, max_tokens=256)
    elapsed = time.time() - started
    try:
        parsed = _json_from_text(raw)
    except Exception as exc:  # noqa: BLE001 - probe should record parse failures.
        parsed = {"parse_error": str(exc), "raw": raw}

    stitched = _stitch_separated_lists(parsed)
    record = {
        "crop": str(crop_path),
        "elapsed_s": round(elapsed, 3),
        "parsed": parsed,
        "stitched": stitched,
        "counts_match": stitched is not None,
        "raw": raw,
        "metadata": getattr(client, "last_response_metadata", {}) or {},
    }
    print(json.dumps(parsed, ensure_ascii=False))
    if stitched is not None:
        print("stitched:", json.dumps(stitched, ensure_ascii=False))
    else:
        print("stitched: <count mismatch or invalid shape>")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("crop", type=Path, help="Saved full-width crop image to split into rows.")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "working" / "tfl_row_reread")
    parser.add_argument("--padding", type=int, default=18, help="Vertical padding around each detected row band.")
    parser.add_argument("--expected-lines", type=int, default=None, help="Use overlapping equal row windows if detected row count differs.")
    parser.add_argument("--model", default=None, help="LM Studio model override. Defaults to the configured 4b model.")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--no-call", action="store_true", help="Only write row crops; do not call the model.")
    parser.add_argument(
        "--ordered-justifications",
        action="store_true",
        help="Ask for the derived-line justifications as one ordered list from the whole crop.",
    )
    parser.add_argument(
        "--separated-lists",
        action="store_true",
        help="Ask for formulas and derived-line justifications as separate lists.",
    )
    args = parser.parse_args()

    crop_path = args.crop.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.ordered_justifications:
        if args.no_call:
            print("--ordered-justifications requires a model call; ignoring --no-call.")
        record = reread_ordered_justifications(crop_path, model=args.model, timeout=args.timeout)
        report_path = output_dir / f"{_safe_stem(crop_path.stem)}_ordered_justifications.json"
        report_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Wrote {report_path}")
        return

    if args.separated_lists:
        if args.no_call:
            print("--separated-lists requires a model call; ignoring --no-call.")
        record = reread_separated_lists(crop_path, model=args.model, timeout=args.timeout)
        report_path = output_dir / f"{_safe_stem(crop_path.stem)}_separated_lists.json"
        report_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Wrote {report_path}")
        return

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
