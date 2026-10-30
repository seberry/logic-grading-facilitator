# TFL/Fitch OCR Operating Guide

This is the current source of truth for handwritten TFL/Fitch OCR work. The
long dated history was moved to [FITCH_OCR_EXPERIMENT_LOG.md](FITCH_OCR_EXPERIMENT_LOG.md).
Use that log when you need old experiment evidence; use this file when you need
to change or run the current pipeline.

## Working Boundary

- Main pipeline: `proofs_ocr/pipelines/tfl_fitch.py`
- Thin CLI wrapper: `scripts/debug_fitch_ocr.py`
- Gold regression runner: `proofs_ocr/regression/tfl_runner.py`
- Fixture manifest: `proofs_ocr/regression/manifest.yaml`
- Private real-scan tools:
  - `proofs_ocr/regression/private_tfl_exporter.py`
  - `proofs_ocr/regression/private_tfl_runner.py`
  - `proofs_ocr/regression/private_tfl_narrator.py`

Keep behavior in importable modules under `proofs_ocr/`. Scripts should remain
thin wrappers or local probes.

## Privacy Rules

Real student scans, crops, handwriting, rosters, LMS IDs, timestamps, and local
paths are protected local data.

- Do not add real scans/crops to tracked fixtures or docs.
- Do not paste raw private gold structures, OCR diffs, path names, or image
  contents into agent conversations.
- Use anonymous case IDs and aggregate/redacted summaries.
- For visual inspection of protected OCR artifacts, prefer
  `private_tfl_narrator ask`; it asks a local model and writes a redacted
  observation.
- Keep private eval corpora outside the repo, for example
  `C:\grading-data\ocr-eval\...`.

## Fast Offline Commands

These are CI-safe and do not require a live VLM:

```powershell
$env:PYTHONPATH='.'
pytest tests/test_debug_fitch_ocr.py -q
pytest tests/test_tfl_fitch_gold_fixtures.py -q
pytest tests/test_tfl_fitch_ocr_regression.py -q
pytest tests/test_tfl_private_runner.py tests/test_tfl_private_exporter.py -q
pytest tests/test_tfl_private_narrator.py -q
python -m proofs_ocr.regression.tfl_runner summary
```

Use these before live OCR unless the change is docs-only.

## Live OCR Commands

Live runs now default to IU REALLMS Gemma through `app/vlm_provider.py` and
require `REALLMS_API_KEY`. Set `REALLMS_MODEL=qwen3.8-27b` for an IU Qwen
comparison, or `GRADING_VLM_PROVIDER=lmstudio` for the existing local baseline.
Provider selection changes the transport/model, not the chopping, masking,
stitching, normalization, or regression comparison. Name-recognition success is
not proof-OCR validation; run bounded instructor-authored/fake gold cases first.

```powershell
python -m proofs_ocr.regression.tfl_runner run-gold
python -m proofs_ocr.regression.tfl_runner run-gold --case tfl_handwritten_q4_N_or_K_and_W
python -m proofs_ocr.regression.tfl_runner run-gold --output-root working/my_experiment
```

For one cropped proof:

```powershell
python scripts/debug_fitch_ocr.py path\to\proof_crop.jpg
```

The optional `LMSTUDIO_MODEL_27B` reread route is for explicitly configured local
experiments; leave it unset for IU runs. For local 27B probes, confirm LM Studio has thinking/reasoning
disabled for the loaded model. The pipeline writes `response_metadata_*.json`;
healthy no-thinking probes should show `thinking_chars=0` and
`reasoning_tokens=0`.

## Pipeline Shape

The handwritten TFL/Fitch OCR flow is:

1. Load image.
2. Detect vertical Fitch bars.
3. Build a subproof tree.
4. Recursively crop and mask child subproofs.
5. Ask the VLM to transcribe each crop.
6. Stitch `<SUBPROOF_ID_N>` placeholders into nested proof structure.
7. Normalize formulas, justifications, depth, and blank assumptions/premises.
8. Write `final_structure.json`.
9. Optionally run verifier/reread reporting.

Typical artifacts are written beside the crop under `DATA_DIR/debug_fitch/<slug>/`
or under a regression output root:

| Artifact | Purpose |
|---|---|
| `debug_bars.jpg` | Bar detection overlay. |
| `crop_ROOT.jpg`, `crop_N.jpg` | Images sent to the VLM. |
| `crop_N_fullwidth.jpg` | Full-width small-leaf crop used by the optional 27B route. |
| `response_N.json` | Raw VLM response. |
| `response_metadata_N.json` | Runtime and token metadata. |
| `thinking_N.txt` | Captured Qwen reasoning text when present. |
| `final_structure.json` | Stitched and normalized proof JSON. |
| `gold_regression_*.json/txt` | Regression result, metrics, and diff files. |

## Current OCR Status

- Line-count estimation is active and uses `proofs_ocr/common/rows.py`.
- Letterboxed tight crops are the default for small leaves.
- Context-aware green-box crops are not the default because earlier Qwen runs
  ignored the highlighted region.
- ROOT line-number blotting is active to prevent line numbers from entering
  formula text.
- Outer Fitch proof frame handling treats the ordinary printed spine as page
  scaffold; logical proof depth starts at 0.
- The 27B small-crop route is opt-in through `LMSTUDIO_MODEL_27B`; leave it
  unset for normal fast runs.
- Reread candidates are report-only in strip view. The grader must accept or
  correct by hand.

## Private Real-Scan Evaluation

Use the private runner/exporter for real student proof crops. The manifest,
crops, gold files, and detailed runs must remain outside the repo.

Seed cases from accepted strip OCR/debug state:

```powershell
python -m proofs_ocr.regression.private_tfl_exporter seed-from-strip `
  --eval-root C:\grading-data\ocr-eval\exam1 `
  --assignment-id exam1_spring2026 `
  --problem-id Q6 `
  --grades-dir C:\grading-data\live\grades `
  --debug-fitch-dir C:\grading-data\live\debug_fitch `
  --split dev_regression
```

Run private regressions:

```powershell
python -m proofs_ocr.regression.private_tfl_runner `
  --manifest C:\grading-data\ocr-eval\exam1\manifest.yaml `
  run-gold --split dev_regression
```

Share only `shareable_summary.txt/json` and anonymous case IDs back into agent
workflows.

## Maintenance Notes

- New durable OCR behavior, flags, artifact expectations, and redacted current
  findings belong here.
- Dated experiment narratives belong in
  [FITCH_OCR_EXPERIMENT_LOG.md](FITCH_OCR_EXPERIMENT_LOG.md).
- Broader command recipes belong in [AI_WORKFLOW.md](AI_WORKFLOW.md) or
  [DEV_WORKFLOWS.md](DEV_WORKFLOWS.md).
- Do not create another narrow OCR markdown file unless these owner docs cannot
  reasonably hold the information.

## Initial IU model comparison and next experiment

The provider migration is committed at `8701bce`. A bounded comparison used the
registered instructor-authored Q3, Q4, and Q5 handwritten gold fixtures (20 proof
lines total), including genuine subproofs and an intentionally invalid citation.
No real student proofs were used. The production chop/mask/stitch algorithm was
left unchanged during this comparison.

| Model / strategy | Exact gold structures | Total time for three proofs |
|---|---:|---:|
| Local Qwen3-VL-4B, existing pipeline | 1/3 | 50.8 s |
| IU Gemma, existing pipeline | 1/3 | 9.5 s |
| IU Qwen, existing pipeline | 0/3 | 12.3 s |
| IU Gemma, constrained whole-page experiment | 3/3 | 3.8 s |
| IU Qwen, constrained whole-page experiment | 3/3 | 8.4 s |

The existing pipeline produced an extra line and incorrect nesting on Q3 with
both IU models. Both changed Q5's visibly written invalid citation `1,4` to the
logically appropriate `1,3`. These are unacceptable transcription changes even
when a formula is otherwise read correctly. Gemma read all 20 aligned expected
formulas correctly, but the extra line still makes the complete result wrong.

A first unconstrained whole-page prompt produced inconsistent list nesting,
letter case, and rule formatting. A second version used a literal-transcription
instruction, uppercase TFL atoms, compact Unicode rule heads, a generic nested
array example, and a JSON schema constraining lines versus subproof arrays.
Both IU models then matched formulas, justifications, and nested structure for
all three fixtures, preserving the intentional bad citation. The schema defines
output shape, not the correct proof; gold answers were not supplied to the model.

This is a three-proof exploratory result, with changed prompts/output contracts
and single timing samples. It does not validate general handwriting accuracy,
deep nesting, FOL, or unattended grading. The whole-page strategy is **not yet
integrated into production OCR**; only provider selection has changed there.

The retained local probe is `working/iu-fitch-comparison.py`; detailed outputs
and the combined summary are under `working/iu-fitch-comparison/`. `gemma` and
`qwen` contain the existing-pipeline runs; `gemma_whole_page_v2` and
`qwen_whole_page_v2` contain the constrained whole-page runs. These are fake/
instructor-authored artifacts safe for agent inspection. Do not generalize that
permission to protected student corpora.

Suggested fresh-session prompt:

```text
Continue the IU Fitch whole-page OCR experiment in this repository. Read AGENTS.md,
docs/FITCH_OCR_AGENT.md, and docs/QUIZ_VARIANT_TF_WORKFLOW.md. IU Gemma is now the
production default; Qwen is an explicit alternative. Provider migration commit:
8701bce. Preserve the pre-existing unrelated working-tree changes.

Inspect working/iu-fitch-comparison.py, its summary.json, and the three instructor-
authored gold fixtures. Constrained whole-page Gemma and Qwen each matched 3/3;
the existing chopping pipeline scored local 1/3, Gemma 1/3, and Qwen 0/3.

Evaluate the whole-page contract on the remaining registered fake/instructor-
authored cases, especially nested subproofs and intentional errors. Keep literal
citations and errors; never repair a student's reasoning during transcription.
Check explicit line numbers as well as formulas, rules, nesting, and extra/missing
lines. Gold answers must never be included in model inputs. Keep runs bounded and
record transport/model metadata. Use no protected student scans without an
explicitly authorized private workflow.

If the results support it, implement an importable opt-in whole-page strategy
under proofs_ocr/, with the current pipeline retained for comparison. Add meaningful
offline tests, verify strip-view integration and human acceptance behavior, and
run a small live canary. Report the evidence before changing the production OCR
strategy default. Do not infer correctness from model confidence or verifier
success alone. Commit a concrete, reviewable implementation if successful.
```
