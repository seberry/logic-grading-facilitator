# Development Workflows & Cheat Sheet

Use these recipes to test individual "cogs" of the machine without running the full web server.

## 1. The "Coloring Book" Script (CV)
**Goal:** Manually verify that the Computer Vision script can find and color the vertical logic bars.
* **Script:** `scripts/bold_fitch_lines.py`
* **Input:** One or more image files.
* **Command:**
    ```bash
    # Single file
    python scripts/bold_fitch_lines.py data/raw_scans/my_scan.jpg
    
    # Multiple files (wildcard)
    python scripts/bold_fitch_lines.py data/raw_scans/*.jpg
    ```
* **Output:** Creates `*_highlighted.jpg` copies in the same folder.
* **Code Reference:** See `highlight_proof_scopes` function in `scripts/bold_fitch_lines.py`.

## 2. The Red Box Segmentation (VLM)
**Goal:** Ask the local model where it thinks "Q1" is on a page and generate a debug HTML report.
* **Script:** `scripts/debug_segment_view.py`
* **Command:**
    ```bash
    python -m scripts.debug_segment_view \
      --raw-scans-dir "data/raw_scans" \
      --problem-id "Q1" \
      --problem-description "True/False question"
    ```
* **Output:** View results at `debug_output_segment_all/index.html`.

## 3. The Logic Parser (Text -> JSON)
**Goal:** Test the parsing logic that converts ASCII proofs into JSON for the verifier.
* **Script:** `app/ascii_to_json.py`
* **Command:**
    ```bash
    python -m app.ascii_to_json
    ```
* **How to edit:** This runs the test block at the bottom of the file. Open `app/ascii_to_json.py` and modify the `test_ascii` string variable to test new proof formats.

## 4. Recursive Fitch OCR (Strip -> Nested JSON)
**Goal:** Run the end-to-end "chop, mask, transcribe, stitch" loop on a cropped Fitch proof scan.
* **Script:** `scripts/debug_fitch_ocr.py`
* **Input:** A single proof scan in `data/raw_scans/` (ideally pre-cropped to the proof grid).
* **Command:**
    ```bash
    python scripts/debug_fitch_ocr.py quiz1_studentA_p1.jpg
    ```
* **What it does:**
  - Detects vertical subproof bars and builds a nesting tree.
  - Recursively masks each child subproof with placeholder tokens, calls the local VLM, and saves per-scope JSON.
  - Stitches the placeholders back together, renumbers lines globally, and writes `data/debug_fitch/<scan_stem>/final_structure.json`.
* **Debug artifacts:** The script drops intermediate crops, masks, and VLM responses into `data/debug_fitch/<scan_stem>/` so you can spot where a layer went wrong.
