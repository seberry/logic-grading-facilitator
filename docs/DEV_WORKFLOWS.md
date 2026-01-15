# Development Workflows & Cheat Sheet

Use these recipes to test individual "cogs" of the machine without running the full web server.
All `data/...` paths below are relative to `DATA_DIR` (defaults to `data/samples`).

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
  - Stitches the placeholders back together, normalizes symbols/justification labels for `app/verifier.py`, infers top-level premises, renumbers lines globally, and writes `data/debug_fitch/<scan_stem>/final_structure.json`.
  - If the verifier imports cleanly, runs a proof check and prints any rule/structure issues alongside the inferred conclusion.
* **Debug artifacts:** The script drops intermediate crops, masks, and VLM responses into `data/debug_fitch/<scan_stem>/` so you can spot where a layer went wrong.

## 5. Fitch proof display & human correction page
**Goal:** Preview the highlighted proof image, hand-edit `final_structure.json`, and run the verifier from the new in-browser review page.

1) **Generate artifacts once** (creates the dropdown options and preview assets):
   ```bash
   # Place a cropped proof scan under data/raw_scans/, e.g. quiz1_studentA_p1.jpg
   python scripts/debug_fitch_ocr.py quiz1_studentA_p1.jpg
   # Artifacts land in data/debug_fitch/quiz1_studentA_p1/
   ```

2) **Start the server** (same as normal UI):
   ```bash
   python -m uvicorn app.server:app --reload --port 5000
   ```

3) **Open the review page:** visit `http://127.0.0.1:5000/debug/fitch` and pick your scan slug from the dropdown (the slug matches the filename stem from `data/debug_fitch/<slug>/`).

4) **Re-run the OCR/parse loop:** click **Re-run OCR/parse** to call `scripts/debug_fitch_ocr.py` on the underlying scan. The textarea refreshes with the regenerated `final_structure.json`, and stdout/stderr appear in the log panel.

5) **Hand-correct structure:** edit the JSON in the **final_structure.json** panel and click **Save edits**. The file is written back to `data/debug_fitch/<slug>/final_structure.json`; keep the top-level value as a list.

6) **Verify the proof:** click **Run grader** to send the saved structure through `app/verifier.py`. The log shows inferred premise count, guessed conclusion, and any rule issues without leaving the browser.
