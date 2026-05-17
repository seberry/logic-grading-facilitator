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
- OCR currently aims at canonical model transcription; deterministic grading
  happens after the proposed text is accepted or edited.
- The best-supported OCR answer shape right now is `model_only`.
- The same OCR/checker path is now also available from strip grading for
  rubric problems tagged `problem_type: fol_model`.

## 6a. FOL model strip workflow
**Goal:** Review one FOL model problem across all students with the image, OCR
proposal, target requirement, verdict, and editable score/comment visible in
each strip.

1) Tag the rubric problem:
   ```yaml
   problem_type: fol_model
   autograde:
     task_kind: non_equivalence
     default_score_correct: 4
     default_score_incorrect: 0
     prompt_data:
       formulas:
         - "∀x Px"
         - "∃x Px"
     ocr:
       expected_answer_shape: model_only
   ```

2) Open `/problem/{problem_id}/strips`.

3) Click **Start FOL model OCR**. The page crops each visible answer strip,
   writes debug artifacts under `DATA_DIR/debug_fol_model/<slug>/`, and stores
   review state in `DATA_DIR/grades/fol_model_ocr/fol_model_{problem_id}.json`.

4) For each strip, edit the proposed model text if needed and click the
   per-strip checker button. The checker verdict and suggested score/comment
   update in place; the human-entered score remains authoritative.

The OCR prompt infers expected constants and predicates from
`prompt_data.formulas`. Use `ocr.constants` and `ocr.predicates` only when the
rubric needs to pin the OCR vocabulary more explicitly.

## 6b. Synthetic FOL model exam generator
**Goal:** Create repeatable FERPA-safe fake assignments for initial OCR/checker
debugging without waiting on new hand-made examples.

```powershell
python scripts/generate_synthetic_logic_exam.py `
  --family fol_model `
  --style phone_scan_light `
  --count 8 `
  --seed 1337 `
  --output-dir data/samples/synthetic_fol_models `
  --force
```

Supported styles:

- `printed` - clean baseline for testing fixture/session/checker plumbing.
- `handwriting_font` - locally available handwriting-like font, line jitter,
  faint lined paper, and mild degradation.
- `phone_scan_light` - handwriting-font rendering plus mild skew, blur, noise,
  contrast loss, and a shadow gradient.
- `phone_scan_medium` - stronger skew, blur, noise, contrast loss, and shadow
  while keeping the answer inside the precomputed crop boxes.
- `faint_pencil` - low-contrast gray writing with mild blur/noise.
- `crowded_answer` - smaller, tighter handwriting to stress line separation.
- `messy_layout` - display-only student-ish notation such as missing section
  headers or compact predicate lines while preserving semantic gold text.

The generator writes a complete `DATA_DIR` fixture:

- `roster.csv`
- `rubric/synthetic_fol_models.yaml`
- `raw_scans/*.png`
- `grades/session.json`
- `grades/segments_Q*.json`
- `gold/fol_model_gold.json`
- `README.md`

The gold file records the intended model text and expected checker verdict for
each student/problem. The image rendering is deliberately separate from the
semantic gold data, so OCR failures can be distinguished from checker failures.

To run a live OCR/checker report over the generated fixture:

```powershell
python scripts/run_synthetic_fol_ocr_eval.py `
  --fixture-root data/samples/synthetic_fol_models `
  --output-dir working/synthetic_fol_ocr_report `
  --limit 6
```

The report writes:

- `working/synthetic_fol_ocr_report/summary.json`
- `working/synthetic_fol_ocr_report/summary.html`
- per-case crops and OCR artifacts under `working/synthetic_fol_ocr_report/cases/`

Use `--limit`, `--problem-id Q1`, or `--student-id SYN001` for small live-model
probes before running the full fixture.

The report includes first-pass failure labels such as `missing_domain`,
`missing_predicates_section`, `bad_constants_section`, `checker_parse_failure`,
`invented_default_constants`, and `verdict_mismatch`. These are heuristics for
triage; inspect the crop and OCR text before treating a label as diagnostic.

## 6c. Kripke model checker starter
**Goal:** Autograde already-transcribed Kripke model text through the shared
`app/autograde/` layer. OCR/strip review for graph drawings is not wired yet.

Accepted model text can be compact:

```text
W: {0,1}
R: <0,1>, <1,1>
P: {1}
Q: {}
```

The parser also accepts `w0` labels in `W`/`R` lines and worksheet-style
valuation entries such as `ν: ν0(P) = F, ν1(P) = T`.

Example rubric:

```yaml
problem_type: kripke_model
autograde:
  task_kind: non_entailment
  default_score_correct: 4
  default_score_incorrect: 0
  prompt_data:
    world: 0
    premises:
      - "<>P"
    conclusion: "[]Q"
```

The checker also accepts worksheet-style explicit constraints:

```yaml
problem_type: kripke_model
autograde:
  task_kind: different
  prompt_data:
    constraints:
      - type: different
        world: 0
        formulaA: "<>P"
        formulaB: "[]P"
      - type: relation
        world: 0
        property: reflexive
        expected: true
```

Supported formula syntax includes `[]`, `<>`, `~`, `&`, `|`, `->`, and the
Unicode variants `□`, `◇`, `¬`, `∧`, `∨`, `→`.

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

Before running TFL/Fitch live OCR with the 27b reread/fallback model, make sure
LM Studio is running and the 27b model has thinking/reasoning disabled in the
LM Studio model/runtime settings. If you change that setting while the model is
loaded, unload and reload the model before testing. The pipeline also sends
`/no_think`, `chat_template_kwargs.enable_thinking=false`, and low token caps
when the relevant `TFL_FITCH_27B_*` environment variables are enabled, but the
GUI/runtime setting is still important. A healthy no-thinking run reports
`thinking_chars=0` and `reasoning_tokens=0` in `response_metadata_*.json`.

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
