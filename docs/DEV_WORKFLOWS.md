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

## 6. FOL model OCR + debug page
**Goal:** Use the local vision model to propose a canonical FOL model transcription from a cropped answer image, then hand-correct it and run the deterministic model checker.

1) **Prepare a safe image sample**
   - Put a cropped image under `data/raw_scans/` or `data/debug_fol_model/<scan_slug>/`.
   - Prefer fake/sanitized samples rather than real student data unless you are intentionally working inside a protected local folder.

2) **Optional direct OCR script run**
   ```bash
   python scripts/debug_fol_model_ocr.py \
     --image data/raw_scans/my_fol_crop.png \
     --output-dir data/debug_fol_model/my_fol_crop \
     --autograde-file data/debug_fol_model/my_fol_crop/ocr_request.yaml
   ```
   This script is only the OCR proposal stage:
   `image -> local model -> proposed canonical text`.
   It does not run the deterministic checker.
   `--autograde-file` may point either to a plain autograde YAML file or to a
   saved debug `state.json` file containing `problem_setup_text`.
   The script writes:
   - `ocr_prompt.txt`
   - `ocr_raw_response.txt`
   - `ocr_model_text.txt`

3) **Start the server**
   ```bash
   python -m uvicorn app.server:app --reload --port 5000
   ```

4) **Open the FOL debug page**
   - Visit `http://127.0.0.1:5000/debug/fol-model`
   - Pick the scan slug from the dropdown.
   - Optionally load a rubric-backed `fol_model` problem so the YAML setup box is prefilled from `autograde`.
   - FERPA-safe synthetic samples now live under `data/samples/debug_fol_model/` if you want ready-made test inputs.

5) **Run OCR**
   - Click **Re-run OCR**.
   - The page calls `scripts/debug_fol_model_ocr.py` on the selected image.
   - The proposed canonical transcription is loaded into the **Model** textarea.
   - The OCR log and raw model response are shown below the trace panel.

6) **Hand-correct and check**
   - Edit the model text if needed.
   - Choose `Terse` or `Full` trace mode.
   - Click **Run checker** to evaluate the configured task through the shared `app/autograde/` layer and inspect the recursive trace.

**Current scope / known limits**
- OCR currently aims at canonical model transcription, not full grading.
- The best-supported answer shape right now is `model_only`.
- The page is meant for safe local debugging before wiring the same flow into strip grading.

## 7. FOL OCR batch handoff test
**Goal:** Exercise the full non-browser handoff:
`image -> OCR proposal -> checker input -> deterministic autograde`.

```bash
python scripts/run_fol_ocr_batch.py
```

Useful flags:

```bash
python scripts/run_fol_ocr_batch.py --checker-input ocr
python scripts/run_fol_ocr_batch.py --checker-input gold
```

Interpretation:
- `ocr`: simulate “human clicked approve” with no manual edits; feed the OCR proposal directly into the checker.
- `gold`: bypass OCR text as checker input and use the known-good sample transcription from `state.json`; useful for isolating OCR problems from checker problems.

Artifacts:
- `working/fol_ocr_batch/summary.json`
- `working/fol_ocr_batch/summary.html`

## 8. Running tests (offline, fast)

Unit and integration tests live in `tests/`. Most are fully offline (no VLM needed).

```bash
# Run everything
pytest tests/

# Run only TFL-related tests
pytest tests/ -k tfl

# Run only modal-related tests
pytest tests/ -k modal

# Skip tests that require a live LM Studio server
pytest tests/ -m "not needs_vlm"
```

Key test files:
- `test_tfl_fitch_gold_fixtures.py` — validates fixture `expected_structure.json` files parse correctly (offline, fast)
- `test_tfl_fitch_ocr_regression.py` — runs the regression runner in offline/gold mode
- `test_proofs_ocr_regression_runner.py` — shared regression infrastructure
- `test_debug_fitch_ocr.py` — exercises the debug Fitch OCR script path

## 9. Running OCR gold regressions (live VLM, slow)

The gold regression suite re-runs the full TFL Fitch OCR pipeline against curated handwritten proof images and compares results line-by-line against a hand-curated "expected" structure. **This calls a live VLM endpoint** (LM Studio or similar) and costs real tokens/time.

```bash
# Run all TFL gold regressions
python -m proofs_ocr.regression.tfl_runner run-gold

# Results are written to:
#   working/tfl_gold_regressions/<case_id>/
```

Each fixture directory under `proofs_ocr/regression/fixtures/tfl/handwritten/` contains:
- One or more source images (the handwritten proof scan)
- `expected_structure.json` — the gold-standard proof structure

### Interpreting output

For each case, the runner writes three files to `working/tfl_gold_regressions/<case_id>/`:

| File | Purpose |
|------|---------|
| `gold_regression_result.json` | Top-level pass/fail + error message (if any) |
| `gold_regression_metrics.json` | Line-by-line comparison: formula, justification, and nesting structure matches |
| `gold_regression_diff.txt` | Human-readable diff (generated when there are differences) |

Key metrics in `gold_regression_metrics.json`:
- `formula_correct_line_count` / `expected_line_count` — how many formulas matched
- `justification_correct_line_count` — how many justifications matched
- `both_correct_line_count` — lines where both formula AND justification matched
- `structure_mismatch_count` — lines where nesting depth was wrong
- `extra_line_count` — VLM hallucinated lines beyond the expected proof
- `missing_line_count` — expected lines the VLM failed to produce
- `error` — non-empty if the pipeline crashed (e.g. `"maximum recursion depth exceeded"`)
