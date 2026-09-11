# Agent Entry Point

This repo has several semi-independent workflows. Start here, then read the
domain-specific guide for the task.

## First Reads

- General repo map: `README.md`
- Documentation index: `docs/README.md`
- Coding/test recipes: `docs/AI_WORKFLOW.md`
- Versioned paper-quiz preparation, TA handoff, and release work:
  `docs/QUIZ_VARIANT_TF_WORKFLOW.md`
- Active TFL/Fitch OCR work: `docs/FITCH_OCR_AGENT.md`

Choose the domain guide that matches the task. Do not read the OCR guide as a
prerequisite for ordinary paper-quiz setup, grading, or release work.

## Current Paper-Quiz Ownership

- Setup, scan intake, name review, and session creation: `app/server.py`
- Name/version VLM classification: `app/vlm_classifier.py`,
  `app/vlm_client.py`
- Problem crops and saved question locations: `app/routes/problem_strips.py`
- Final exports and model-free TA/TF bundle: `app/routes/exam_exports.py`
- Grading and preparation UI: `app/templates/problem_strips.html`,
  `app/templates/finish.html`, `app/templates/name_review.html`
- Blank-question image extraction: `scripts/extract_quiz_reference_images.py`
- Instructor operating/release source of truth:
  `docs/QUIZ_VARIANT_TF_WORKFLOW.md`
- TA/TF instructions: `docs/TA_GRADING_GUIDE.md`

## Current OCR Ownership

- Main handwritten TFL/Fitch OCR pipeline:
  `proofs_ocr/pipelines/tfl_fitch.py`
- TFL gold regression runner:
  `proofs_ocr/regression/tfl_runner.py`
- TFL handwritten fixture registry:
  `proofs_ocr/regression/manifest.yaml`
- Thin debug CLI wrapper:
  `scripts/debug_fitch_ocr.py`

Keep new OCR behavior in importable modules under `proofs_ocr/` when possible.
Scripts should usually be probes or thin wrappers.

## Privacy Boundary

Treat real student scans, handwriting, rosters, LMS metadata, and generated
crops from real submissions as protected local data. Do not move those artifacts
into tracked fixtures or docs. Sanitized/gold JSON structures and aggregate
metrics can be used for code and regression work when they contain no names,
paths, timestamps, handwriting images, or linkable identifiers.

## Working Directory

`working/` is for local experiment outputs and live-model regression artifacts.
It is intentionally ignored except for `working/.gitkeep` and
`working/README.md`. Do not clean or delete experiment outputs unless the user
explicitly asks.

## Useful Commands

```powershell
$env:PYTHONPATH='.'
pytest tests/test_debug_fitch_ocr.py tests/test_tfl_fitch_ocr_regression.py -q
python -m proofs_ocr.regression.tfl_runner summary
python -m proofs_ocr.regression.tfl_runner run-gold --case tfl_handwritten_q6_S_iff_W
```
