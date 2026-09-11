# Documentation Index

This folder has accumulated design notes, active agent guides, and historical
experiment logs. Use this index to avoid reading everything.

## Source Of Truth

Use these docs first. If one of these contradicts an older note, prefer the
entry in this table.

| Doc | Current responsibility |
|---|---|
| `AI_WORKFLOW.md` | Coding-agent map, safe test commands, and OCR workflow guardrails. |
| `ARCHITECTURE.md` | Current FastAPI/Jinja app architecture, active routes, data flow, caches, and UI integration points. |
| `SCHEMA.md` | Persisted JSON/YAML data shapes, including session files, problem types, and proof OCR review state. |
| `QUIZ_VARIANT_TF_WORKFLOW.md` | Current versioned paper-quiz workflow, implementation map, product decisions, TA/TF handoff, and release continuation. |
| `FITCH_OCR_AGENT.md` | Current handwritten TFL/Fitch OCR operating guide. Read before changing `proofs_ocr/pipelines/tfl_fitch.py` or strip-view TFL/Fitch OCR integration. |
| `DEV_WORKFLOWS.md` | Practical command recipes for server/debug workflows. |

For TFL/Fitch OCR UI integration, read `ARCHITECTURE.md` for the strip-view
state/routes and `FITCH_OCR_AGENT.md` for the OCR pipeline/review-artifact
context. Do not create another one-off proof OCR UI note unless the current
entry points cannot hold the information cleanly.

## OCR Reference Docs

- `PROOF_OCR_PIPELINE_OVERVIEW.md` - stable modal proof OCR mental model.
- `PROOF_OCR_REFACTOR_STATUS.md` - modal OCR refactor status; useful but not
  the TFL/Fitch source of truth.
- `PROOF_OCR_LLM_README.yaml` - machine-readable modal OCR orientation.
- `HANDWRITTEN_OCR_PROBE_RESULTS.md` - historical handwritten OCR probes.
- `TFL_OCR_FUTURE_IDEAS.md` - older TFL OCR idea backlog.
- `CROSS_PIPELINE_INVESTIGATION.md` - notes comparing modal/TFL approaches.
- `FITCH_DEBUG_PIPELINE_TRACE_PROGRESS.md` - historical trace/debug progress.
- `FITCH_OCR_EXPERIMENT_LOG.md` - archived long-form TFL/Fitch OCR notes and
  dated experiment history.

These are reference or historical context unless explicitly called out by an
active entry point. Do not use them as the source of truth for current TFL/Fitch
behavior without checking `FITCH_OCR_AGENT.md`.

## Canvas And Grading App Docs

- `QUIZ_VARIANT_TF_WORKFLOW.md` - instructor workflow for versioned paper quizzes and model-free TF handoff.
- `TA_GRADING_GUIDE.md` - short operating/troubleshooting guide to include with a TF grading bundle.
- `TA_RELEASE_CHECKLIST.md` - instructor checklist for publishing an audited code release and sending the private bundle separately.
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

These docs are useful for background and roadmap work. For current exam grading
routes and data stores, prefer `ARCHITECTURE.md` and `SCHEMA.md`.

## Status Notes

- `FITCH_OCR_AGENT.md` is the current source of truth for TFL/Fitch OCR. The
  long historical notes are in `FITCH_OCR_EXPERIMENT_LOG.md`.
- The `PROOF_OCR_*` docs mostly describe the modal proof OCR path. They are
  still useful, but they can contradict TFL/Fitch details if read as current
  TFL guidance.
- Older probe/result docs are preserved for history. Prefer updating
  `FITCH_OCR_AGENT.md` with current OCR findings instead of creating another
  one-off result file.

## Documentation Maintenance Rules

- New persisted JSON/YAML shapes belong in `SCHEMA.md`.
- New routes, background jobs, caches, and cross-module data flow belong in
  `ARCHITECTURE.md`.
- New TFL/Fitch OCR behavior, flags, canary results, and artifact expectations
  belong in `FITCH_OCR_AGENT.md`.
- New command recipes belong in `DEV_WORKFLOWS.md` or `AI_WORKFLOW.md`,
  depending on whether they are human/operator recipes or coding-agent recipes.
- Avoid adding narrow one-off markdown files for every investigation. Prefer a
  short section in the current owner doc plus links to ignored `working/`
  artifacts when needed.

## Documentation Cleanup Roadmap

The highest-value cleanup is consolidation, not deletion:

- Label old planning docs such as `STRATEGY_FITCH_PROOFS.md`, `DESIGN_v2.md`,
  and older probe-result files as historical when they are next edited.
- Move durable route/data maps into `ARCHITECTURE.md` rather than duplicating
  them in feature notes.
- Keep historical experiment details searchable, but do not let them override
  current source-of-truth docs.

## Cleanup Candidates

These should not be deleted by an agent without explicit user approval, but
they are likely clutter from the perspective of future coding sessions:

- Root-level generated/media artifacts: `Gemini_Generated_*.png`,
  `icon_*.png`, `icon_*.ico`, `temp_root_zoom_strip.jpg`,
  `P251_Exam_1-solutions.pdf`, `deduct-master.zip`.
- Historical debug folders: `debug_output/`, `debug_output_segment_all/`, and
  most timestamped or one-off subdirectories under `working/`.
- Script clutter: many `scripts/probe_*`, icon-processing scripts, and old
  segmentation utilities are intentionally indexed in `scripts/README.md`, but
  could later be split into `scripts/ocr/`, `scripts/canvas/`, and
  `scripts/legacy/` if the repo needs a stronger cleanup.

## Privacy Note

Do not add real student handwriting, names, LMS IDs, timestamps, or original
file paths to docs. Use sanitized case IDs and aggregate metrics only.
