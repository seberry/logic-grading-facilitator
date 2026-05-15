# AI Workflow

This note is for coding agents and slower local models that need a stable map of
the repo before changing OCR or regression code.

## Fast entry points

- Read [`proofs_ocr/regression/manifest.yaml`](../proofs_ocr/regression/manifest.yaml) for fixture ids, domains, and paths.
- Read [`docs/FITCH_OCR_AGENT.md`](FITCH_OCR_AGENT.md) before changing handwritten TFL/Fitch OCR behavior.
- Use [`proofs_ocr/regression/runner.py`](../proofs_ocr/regression/runner.py) for modal gold-regression helpers.
- Use [`proofs_ocr/regression/tfl_runner.py`](../proofs_ocr/regression/tfl_runner.py) for handwritten TFL/Fitch gold-regression helpers.
- Use [`proofs_ocr/pipelines/tfl_fitch.py`](../proofs_ocr/pipelines/tfl_fitch.py) for the importable Fitch OCR pipeline.
- Keep [`scripts/debug_fitch_ocr.py`](../scripts/debug_fitch_ocr.py) as a CLI wrapper, not the primary implementation.

## Safe and fast commands

- `pytest tests/test_debug_fitch_ocr.py -q`
- `pytest tests/test_tfl_fitch_gold_fixtures.py -q`
- `pytest tests/test_tfl_fitch_ocr_regression.py -q`
- `pytest tests/test_tfl_private_runner.py tests/test_tfl_private_exporter.py -q`
- `pytest tests/test_tfl_private_narrator.py -q`
- `pytest tests/test_proofs_ocr_regression_runner.py -q`
- `python -m proofs_ocr.regression summary`
- `python -m proofs_ocr.regression.tfl_runner summary`

These are the preferred commands for CI-safe edits because they do not require a
live vision model. They validate harness logic and fixture plumbing, not the
quality of live handwritten OCR.

## Slow or live-model workflows

- `python -m proofs_ocr.regression run-gold ...`
- `python -m proofs_ocr.regression.tfl_runner run-gold ...`
- `python -m proofs_ocr.regression.private_tfl_runner --manifest <external_manifest.yaml> run-gold ...`
- `RUN_LIVE_TFL_OCR=1 pytest tests/test_tfl_fitch_ocr_regression.py -q -k Live`
- `python scripts/debug_fitch_ocr.py <scan>.jpg`

Treat these as local/manual workflows. They may require LM Studio or another
running local VLM endpoint. When a live OCR run is enabled, expect strict
gold-match failures until the handwritten pipeline actually improves.

For TFL/Fitch runs that exercise the 27b reread/fallback path, confirm LM
Studio has thinking/reasoning disabled for the loaded 27b model before timing
or judging failures. The API payload also sends no-thinking controls when the
`TFL_FITCH_27B_*` flags are set, but the LM Studio runtime setting is the
important control. Check `response_metadata_*.json`: successful no-thinking
probes should show `thinking_chars=0` and `reasoning_tokens=0`.

For OCR behavior changes, assume the user usually wants a bounded slow run
after the fast tests pass. Default to the smallest relevant live case first
using `--case ... --output-root working/<descriptive_run_name>`, then summarize
whether the change helped before proposing broader runs.

Before a live OCR run on new proof geometry, confirm the pre-VLM structure first:
bar count, subproof tree shape, and leaf line counts should be plausible. Small
leaf crops can route to slower verifier/larger-model paths, so bar
over-detection can multiply expensive calls quickly.

For real student proof crops, use the private TFL runner instead of adding
fixtures to the repo manifest. The manifest, crops, gold files, and run outputs
must stay outside the repo, for example under `C:\grading-data\ocr-eval\...`.
The runner writes local detailed reports plus `shareable_summary.txt/json`,
which omit paths, images, line text, diffs, and timestamps:

```powershell
python -m proofs_ocr.regression.private_tfl_exporter seed-from-strip --eval-root C:\grading-data\ocr-eval\exam1 --assignment-id exam1_spring2026 --problem-id Q6 --grades-dir C:\grading-data\live\grades --debug-fitch-dir C:\grading-data\live\debug_fitch --split dev_regression
python -m proofs_ocr.regression.private_tfl_runner --manifest C:\grading-data\ocr-eval\exam1\manifest.yaml summary
python -m proofs_ocr.regression.private_tfl_runner --manifest C:\grading-data\ocr-eval\exam1\manifest.yaml run-gold --split dev_regression
python -m proofs_ocr.regression.private_tfl_narrator suggest-cases --summary C:\grading-data\ocr-eval\exam1\runs\tfl_private_gold_regressions\shareable_summary.json --category structure
python -m proofs_ocr.regression.private_tfl_narrator list-artifacts --manifest C:\grading-data\ocr-eval\exam1\manifest.yaml --run-root C:\grading-data\ocr-eval\exam1\runs\tfl_private_gold_regressions --case exam1_spring2026_q6_case_0007
python -m proofs_ocr.regression.private_tfl_narrator ask --manifest C:\grading-data\ocr-eval\exam1\manifest.yaml --run-root C:\grading-data\ocr-eval\exam1\runs\tfl_private_gold_regressions --case exam1_spring2026_q6_case_0007 --artifact debug_bars --model qwen/qwen3-vl-27b --disable-thinking --question "Why might the bar detector or line grouping be failing here?"
```

Only paste the shareable summary back into agent conversations unless the case is
fake or fully sanitized.

## OCR architecture

Handwritten Fitch/TFL OCR currently flows through:

1. image load
2. vertical-bar detection
3. subproof tree construction
4. recursive crop and mask
5. VLM transcription of each crop
6. token stitch-back into nested proof structure
7. normalization and blank-justification repair
8. final structure write
9. optional verifier pass

The importable implementation lives in
[`proofs_ocr/pipelines/tfl_fitch.py`](../proofs_ocr/pipelines/tfl_fitch.py).

## Fixture source of truth

- Modal and TFL handwritten fixtures are declared in [`proofs_ocr/regression/manifest.yaml`](../proofs_ocr/regression/manifest.yaml).
- Curated expected structures are the source of truth for OCR regressions.
- Semantic caveats belong in the sidecar `semantic_notes.yaml` file next to the fixture.

## Working conventions

- Prefer importable modules under `proofs_ocr/` for new OCR logic.
- Keep shell scripts thin wrappers around importable functions.
- Write long-running debug artifacts under `working/` for regression runs.
- Avoid live-model requirements in default test targets.
