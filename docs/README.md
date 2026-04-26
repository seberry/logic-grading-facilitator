# Documentation Index

This folder has accumulated design notes, active agent guides, and historical
experiment logs. Use this index to avoid reading everything.

## Active Entry Points

- `AI_WORKFLOW.md` - short coding-agent map and safe/slow test commands.
- `FITCH_OCR_AGENT.md` - active handwritten TFL/Fitch OCR guide and current
  experiment history. Read this before changing `proofs_ocr/pipelines/tfl_fitch.py`.
- `DEV_WORKFLOWS.md` - practical command recipes for server/debug workflows.
- `ARCHITECTURE.md` - broader app architecture and debug UI overview.
- `SCHEMA.md` - app/session/rubric data shape notes.

## OCR Reference Docs

- `PROOF_OCR_PIPELINE_OVERVIEW.md` - stable modal proof OCR mental model.
- `PROOF_OCR_REFACTOR_STATUS.md` - modal OCR refactor status; useful but not
  the TFL/Fitch source of truth.
- `PROOF_OCR_LLM_README.yaml` - machine-readable modal OCR orientation.
- `HANDWRITTEN_OCR_PROBE_RESULTS.md` - historical handwritten OCR probes.
- `TFL_OCR_FUTURE_IDEAS.md` - older TFL OCR idea backlog.
- `CROSS_PIPELINE_INVESTIGATION.md` - notes comparing modal/TFL approaches.
- `FITCH_DEBUG_PIPELINE_TRACE_PROGRESS.md` - historical trace/debug progress.

## Canvas And Grading App Docs

- `CANVAS_INTEGRATION.md`
- `CANVAS_COMMENT_POSTING.md`
- `CANVAS_USER_ID_MAPPING.md`
- `CANVAS_WORKFLOW_REDESIGN_SPEC.md`
- `AUTOGRADE_ARCHITECTURE.md`
- `FUTURE_AUTOMATION.md`
- `CLASSWORK_BACKEND_MAP.md`
- `CLASSWORK_VARIANT_SPEC.md`
- `COLUMN_MATCHING.md`
- `DESIGN_v2.md`
- `STRATEGY_FITCH_PROOFS.md`

## Status Notes

- `FITCH_OCR_AGENT.md` is the current source of truth for TFL/Fitch OCR.
- The `PROOF_OCR_*` docs mostly describe the modal proof OCR path. They are
  still useful, but they can contradict TFL/Fitch details if read as current
  TFL guidance.
- Older probe/result docs are preserved for history. Prefer updating
  `FITCH_OCR_AGENT.md` with current OCR findings instead of creating another
  one-off result file.

## Cleanup Candidates

These should not be deleted by an agent without explicit user approval, but
they are likely clutter from the perspective of future coding sessions:

- Root-level generated/media artifacts: `Gemini_Generated_*.png`,
  `icon_*.png`, `icon_*.ico`, `temp_root_zoom_strip.jpg`,
  `P251_Exam_1-solutions.pdf`, `deduct-master.zip`, `repo-structure.txt`.
- Root-level prototype tests: `test_column_matching.py`,
  `test_column_matching_simple.py`. If still useful, move them under `tests/`.
- Historical debug folders: `debug_output/`, `debug_output_segment_all/`, and
  most timestamped or one-off subdirectories under `working/`.
- Script clutter: many `scripts/probe_*`, icon-processing scripts, and old
  segmentation utilities are intentionally indexed in `scripts/README.md`, but
  could later be split into `scripts/ocr/`, `scripts/canvas/`, and
  `scripts/legacy/` if the repo needs a stronger cleanup.

## Privacy Note

Do not add real student handwriting, names, LMS IDs, timestamps, or original
file paths to docs. Use sanitized case IDs and aggregate metrics only.
