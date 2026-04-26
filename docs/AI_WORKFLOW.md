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
- `pytest tests/test_proofs_ocr_regression_runner.py -q`
- `python -m proofs_ocr.regression summary`
- `python -m proofs_ocr.regression.tfl_runner summary`

These are the preferred commands for CI-safe edits because they do not require a
live vision model. They validate harness logic and fixture plumbing, not the
quality of live handwritten OCR.

## Slow or live-model workflows

- `python -m proofs_ocr.regression run-gold ...`
- `python -m proofs_ocr.regression.tfl_runner run-gold ...`
- `RUN_LIVE_TFL_OCR=1 pytest tests/test_tfl_fitch_ocr_regression.py -q -k Live`
- `python scripts/debug_fitch_ocr.py <scan>.jpg`

Treat these as local/manual workflows. They may require LM Studio or another
running local VLM endpoint. When a live OCR run is enabled, expect strict
gold-match failures until the handwritten pipeline actually improves.

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
