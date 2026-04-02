# Refactoring Status

This is the canonical status document for `app/server.py` refactoring.

## Current direction

Use **small, workflow-based extractions** from `app/server.py` into `app/routes/`,
starting with the most isolated features first.

This is the preferred approach because it:

- reduces file size without changing behavior aggressively,
- keeps shared state in one place,
- avoids risky whole-file reordering,
- is easier to verify after each step.

## Completed

### 2026-04-02: Fitch debug extracted

The Fitch debug workflow now lives in:

- `app/routes/fitch_debug.py`

The following routes were moved out of `app/server.py` without changing paths:

- `GET /debug/fitch`
- `POST /debug/fitch/save`
- `POST /debug/fitch/rerun`
- `POST /debug/fitch/run_grader`

### 2026-04-02: Flashcards extracted

The flashcards workflow now lives in:

- `app/routes/flashcards.py`

### 2026-04-02: Problem strips extracted

The per-problem strip grading workflow now lives in:

- `app/routes/problem_strips.py`

Shared session state for extracted workflows now lives in:

- `app/session_state.py`

### 2026-04-02: Classwork extracted

The classwork workflow now lives in:

- `app/routes/classwork.py`

Classwork-specific helper logic now lives in:

- `app/classwork_core.py`

### 2026-04-02: Exam exports extracted

The exam export/finalize workflow now lives in:

- `app/routes/exam_exports.py`

Shared exam-side helper logic now lives in:

- `app/exam_shared.py`
- `app/exam_exports_core.py`

## Current rules

1. Do not do whole-file reordering of `app/server.py`.
2. Do not split session storage yet.
3. Extract one clearly bounded workflow at a time.
4. Keep route paths stable during extraction.
5. Verify imports and route registration after every extraction.

## Current structure

The broad code layout is now:

- `app/server.py`
  App creation, mounts, setup/import routes, and remaining legacy exam workflow.
- `app/routes/fitch_debug.py`
  Fitch debug workflow.
- `app/routes/flashcards.py`
  Flashcards workflow.
- `app/routes/problem_strips.py`
  Per-problem strip grading and segmentation routes.
- `app/routes/classwork.py`
  Classwork routes, including Canvas classwork integration.
- `app/routes/exam_exports.py`
  Finish/export/finalize routes for the exam workflow.
- `app/classwork_core.py`
  Classwork-specific helper and file/gradebook logic.
- `app/exam_shared.py`
  Shared exam-side helpers still used by setup/import and exports.
- `app/exam_exports_core.py`
  Export-only exam helper logic.
- `app/session_state.py`
  Shared session loading/saving and locks.

## Likely next candidates

These are the next most plausible extractions, in rough order:

1. Exam setup/import routes
2. Name-review handoff pipeline
3. Remaining legacy grading-page/work-unit routes
4. Shared exam helper cleanup after those moves

## Superseded material

The following documents still contain useful background, but should not be treated
as the current plan:

- `docs/REFACTORING_GUIDE.md`
- `scripts/README_SERVER_REFACTORING.md`

They remain in the repo for context, but this file is the source of truth for the
active refactoring strategy.
