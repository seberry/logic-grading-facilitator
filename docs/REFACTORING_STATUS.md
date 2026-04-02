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

## Current rules

1. Do not do whole-file reordering of `app/server.py`.
2. Do not split session storage yet.
3. Extract one clearly bounded workflow at a time.
4. Keep route paths stable during extraction.
5. Verify imports and route registration after every extraction.

## Likely next candidates

These are the next most plausible extractions, in rough order:

1. Exam strip grading and segmentation routes
2. Flashcards workflow
3. Legacy exam grading/export routes
4. Setup wizard and gradebook merge routes
5. Classwork routes, keeping manual and Canvas classwork together for now

## Superseded material

The following documents still contain useful background, but should not be treated
as the current plan:

- `docs/REFACTORING_GUIDE.md`
- `scripts/README_SERVER_REFACTORING.md`

They remain in the repo for context, but this file is the source of truth for the
active refactoring strategy.
