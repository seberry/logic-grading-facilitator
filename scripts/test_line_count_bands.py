#!/usr/bin/env python
"""Validate detect_text_row_bands on TFL leaf crops.

Compares the modal pipeline's morphological band detector against the old
naive horizontal projection counter, using known expected line counts from
the gold regression fixtures.

Also tests a post-filtered counting strategy that removes thin underline
artifacts and estimates line counts for merged tall bands.

Usage
-----
    python scripts/test_line_count_bands.py
    python scripts/test_line_count_bands.py --verbose   # show per-band details
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from proofs_ocr.common.rows import detect_text_row_bands, band_height
from proofs_ocr.pipelines.tfl_fitch import count_text_lines_in_region

WORKING_DIR = REPO_ROOT / "working" / "tfl_gold_regressions"
FIXTURE_DIR = REPO_ROOT / "proofs_ocr" / "regression" / "fixtures" / "tfl" / "handwritten"

Band = Tuple[int, int]

# Known expected line counts for leaf crops, derived from expected_structure.json.
EXPECTED = {
    "tfl_handwritten_q3_P_and_Q_imp_R": {
        "crop_2": 5,
    },
    "tfl_handwritten_q4_N_or_K_and_W": {
        "crop_2": 2,
        "crop_3": 2,
    },
    "tfl_handwritten_q6_S_iff_W": {
        "crop_2": 4,
        "crop_3": 4,
    },
}


def estimate_line_count_from_bands(bands: List[Band], img_h: int) -> int:
    """Estimate number of proof lines from raw bands.

    Post-processing strategy:
    1. Remove bands shorter than MIN_BAND_HEIGHT (underline artifacts).
    2. Compute a reference line height from the remaining bands.
    3. For any band significantly taller than the reference, estimate how
       many merged lines it contains.
    """
    if not bands:
        return 0

    MIN_BAND_HEIGHT = 18  # Real handwritten text lines are at least this tall

    # Step 1: Remove thin artifact bands
    real_bands = [b for b in bands if band_height(b) >= MIN_BAND_HEIGHT]
    if not real_bands:
        # Fall back to raw count if all bands were filtered
        return len(bands)

    # Step 2: Compute reference height (median of all real bands)
    heights = sorted([band_height(b) for b in real_bands])
    ref_height = heights[len(heights) // 2]  # median

    # Step 3: Estimate merged lines in tall bands
    MERGE_FACTOR = 1.6  # A band > 1.6x the reference probably has merged lines
    count = 0
    for b in real_bands:
        h = band_height(b)
        if h > ref_height * MERGE_FACTOR and ref_height > 0:
            # Estimate how many lines are packed in this band
            estimated = max(1, round(h / ref_height))
            count += estimated
        else:
            count += 1

    return count


def main():
    # Force UTF-8 output on Windows (emoji/symbol characters in results)
    if sys.stdout.encoding != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if sys.stderr.encoding != "utf-8":
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    verbose = "--verbose" in sys.argv

    print("=" * 100)
    print("LINE COUNT BAND DIAGNOSTIC")
    print("  Old = count_text_lines_in_region (naive projection)")
    print("  Bands = detect_text_row_bands (raw morphological bands)")
    print("  Filtered = bands with thin artifact removal + tall-band merge estimation")
    print("=" * 100)
    print()

    # Header
    print(
        f"  {'Case':<35s} {'Crop':<10s} {'Exp':>4s} "
        f"{'Old':>5s} {'Bands':>6s} {'Filt':>5s} "
        f"{'Old':>5s} {'Raw':>5s} {'Filt':>5s}"
    )
    print(
        f"  {'':35s} {'':10s} {'':4s} "
        f"{'':5s} {'':6s} {'':5s} "
        f"{'OK?':>5s} {'OK?':>5s} {'OK?':>5s}"
    )
    print("  " + "-" * 95)

    total = 0
    old_correct = 0
    raw_correct = 0
    filt_correct = 0

    for case_id, leaf_crops in sorted(EXPECTED.items()):
        case_dir = WORKING_DIR / case_id
        if not case_dir.is_dir():
            print(f"  {case_id}: WORKING DIR NOT FOUND")
            continue

        for crop_name, expected_lines in sorted(leaf_crops.items()):
            crop_path = case_dir / f"{crop_name}.jpg"
            if not crop_path.exists():
                print(f"  {case_id}/{crop_name}: NOT FOUND")
                continue

            img = cv2.imread(str(crop_path))
            if img is None:
                print(f"  {case_id}/{crop_name}: READ FAILED")
                continue

            h, w = img.shape[:2]

            # Old method
            old_count = count_text_lines_in_region(img)

            # New method (raw bands)
            bands = detect_text_row_bands(img)
            raw_count = len(bands)

            # Filtered method
            filt_count = estimate_line_count_from_bands(bands, h)

            old_ok = old_count == expected_lines
            raw_ok = raw_count == expected_lines
            filt_ok = filt_count == expected_lines

            total += 1
            if old_ok: old_correct += 1
            if raw_ok: raw_correct += 1
            if filt_ok: filt_correct += 1

            def mark(ok, val):
                return "  OK" if ok else f"  X({val})"

            short_case = case_id.replace("tfl_handwritten_", "")
            print(
                f"  {short_case:<35s} {crop_name:<10s} {expected_lines:>4d} "
                f"{old_count:>5d} {raw_count:>6d} {filt_count:>5d} "
                f"{mark(old_ok, old_count):>5s} {mark(raw_ok, raw_count):>5s} {mark(filt_ok, filt_count):>5s}"
            )

            if verbose:
                print(f"    Image size: {w}x{h}")
                real_bands = [b for b in bands if band_height(b) >= 18]
                heights = sorted([band_height(b) for b in real_bands]) if real_bands else []
                ref_h = heights[len(heights) // 2] if heights else 0
                print(f"    Raw bands: {len(bands)}, Real bands (>17px): {len(real_bands)}, Ref height: {ref_h}px")
                for i, (y1, y2) in enumerate(bands):
                    bh = band_height((y1, y2))
                    tag = ""
                    if bh < 18:
                        tag = "  <- ARTIFACT (filtered out)"
                    elif ref_h > 0 and bh > ref_h * 1.6:
                        est = max(1, round(bh / ref_h))
                        tag = f"  <- MERGED (est ~{est} lines)"
                    print(f"    Band {i+1}: rows {y1}-{y2} (height {bh}px){tag}")
                print()

    # Summary
    print()
    print("  " + "=" * 95)
    print(f"  SUMMARY: {total} leaf crops tested")
    print(f"    Old (count_text_lines_in_region):  {old_correct}/{total} correct")
    print(f"    Raw (detect_text_row_bands):        {raw_correct}/{total} correct")
    print(f"    Filtered (post-processed bands):    {filt_correct}/{total} correct")
    print()
    if filt_correct == total:
        print("  ALL FILTERED COUNTS MATCH -- safe to wire into pipeline")
    elif filt_correct > raw_correct:
        print("  FILTERED IS BETTER THAN RAW -- promising, investigate remaining failures")
    else:
        print("  FILTERED IS NOT BETTER -- tune parameters before wiring in")
    print("  " + "=" * 95)


if __name__ == "__main__":
    main()