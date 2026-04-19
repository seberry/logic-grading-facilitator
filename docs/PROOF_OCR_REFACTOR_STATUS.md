# Proof OCR Refactor Status

## Purpose

This note is the short restart brief for the current proof OCR state. Read this
first if you need to know what is live, what is still provisional, and what the
next work should optimize for.

For the stable end-to-end mental model of the pipeline, also read
`docs/PROOF_OCR_PIPELINE_OVERVIEW.md`.

## Current Reality

The repo now has two live modal OCR paths:

- Old reference implementation:
  - `scripts/debug_modal_proof_ocr.py`
- New baseline path:
  - `proofs_ocr/pipelines/modal_v2.py`

The old script is still the richer reference implementation. The new
`proofs_ocr/` path is no longer a scaffold. It runs end to end, writes debug
artifacts, and is selectable in `/debug/modal-proof`.

The `/debug/modal-proof` route surface has also now been split into smaller
neighboring modules, so future OCR/debug work no longer needs to accumulate in
one giant route file.

## What Is Stable Enough To Build On

- `proofs_ocr/common/`
  - shared Fitch-style mechanics already extracted:
    - `tree.py`
    - `masking.py`
    - `rows.py`
    - `stitch.py`
- `proofs_ocr/domains/modal.py`
  - minimal modal-domain normalization and line-resolution helpers
- `proofs_ocr/pipelines/`
  - `modal_v2_core.py` for recursive baseline orchestration
  - `modal_v2_runtime.py` for the line-reader / VLM adapter
  - `modal_v2_debug.py` for artifact writing and debug-state assembly
  - `modal_v2.py` as the public facade / CLI entry point
- `proofs_ocr/regression/fixtures/modal/`
  - committed modal typeset and handwritten fixtures
- `/debug/modal-proof`
  - supports both:
    - `Legacy modal OCR`
    - `modal_v2 baseline`
- `app/routes/`
  - `modal_proof_debug.py` keeps endpoint wiring and route decorators
  - `modal_proof_debug_state.py` owns state persistence and form normalization
  - `modal_proof_debug_fixtures.py` owns regression fixture loading and lookup
  - `modal_proof_debug_artifacts.py` owns artifact loading / URL helpers
  - `modal_proof_debug_page.py` owns page/view-model assembly helpers

## What Is Still Provisional

- The old scripts remain the broader behavioral reference.
- The new modal baseline is intentionally old-script-free by default.
- Optional legacy-style layers are feature-flagged for later work and mostly
  off by default:
  - `enable_strict_opener_classifier`
  - `enable_short_formula_reread`
  - `enable_retry_on_suspicious_line`
  - `enable_unary_verifier`
  - `enable_boundary_repair`
- Some handwritten gold cases still reflect difficult OCR targets and should be
  treated as active evaluation targets, not as evidence that the baseline is
  already strong on handwriting.

## What The New Baseline Already Does

The `modal_v2` baseline now follows the intended recursive shape:

1. detect vertical bars
2. build a subproof tree
3. recurse into child scopes first
4. mask child scopes with placeholders
5. segment rows in the current node
6. OCR per-line crops through an injected line reader
7. stitch the result back into a nested proof structure

That structure is the main refactor win. It is now visible in code and in the
debug viewer.

The repo also now has:

- a curated 5-case modal gold set under `proofs_ocr/regression/fixtures/modal/`
- a slow gold regression runner:
  - `python -m proofs_ocr.regression run-gold ...`

## Recent Baseline Win

The most recent concrete modal_v2 fix was the strict-opener recovery on the
`modal_s4_typeset_r4_corrected` fixture.

That issue turned out to be two separate failures:

- the tiny top `□` opener row in a deep child crop was being stripped out of
  row segmentation as weak "structure"
- once the row was recovered, the per-line OCR result still came back empty and
  needed a narrow modal-domain recovery pass

The current baseline now:

- preserves compact top-left opener components during row-band detection
- records row-band debug artifacts so rejected vs kept bands are inspectable in
  `/debug/modal-proof`
- recovers a tiny strict opener as `□` only when the crop is visually
  consistent with a lone opener and the parsed line has no line label

So the strict-opener problem is no longer the main blocker. The next debugging
work should focus on tree/bar geometry.

There is also now a cheap runtime safeguard against label-only wide crops, so
false parent ownership is less likely to hallucinate stray formulas when the
tree is wrong. That contains one visible failure mode, but it does not fix the
underlying false-bar geometry issue.

## Current Main Blocker

The next problem is not “pipeline missing.” It is baseline quality.

The clearest known failure is bar geometry / masking on some typeset proofs:

- a left number/formula separator can still be mistaken for a scope bar
- child bars can start too high or run too long
- parent masking then erases formulas that should remain visible
- root crops keep line labels while losing formulas or gain extra wrapper
  structure

That means the next useful work is geometric/debug quality work, not another
large structural rewrite.

## Recommended Near-Term Plan

1. Keep new work inside `proofs_ocr/`.
2. Use `/debug/modal-proof` with the `modal_v2 baseline` toggle to inspect the
   real first-pass artifacts.
3. With the route split in place, go back to improving bar bounds / masking
   before stacking more OCR heuristics.
4. Keep route handlers in `app/routes/modal_proof_debug.py`, and keep further
   debug-viewer helpers in the neighboring route modules instead of regrowing
   the endpoint file.
5. Keep legacy behavior behind explicit adapters or feature flags, not as the
   default architecture.
6. Do not rewrite the TFL OCR path in place while the modal baseline is still
   settling.

## Architectural Intent

The intended split remains:

- `proofs_ocr/common/` for shared proof mechanics
- `proofs_ocr/domains/` for domain-specific normalization and heuristics
- `proofs_ocr/pipelines/` for orchestration and experiment paths
- `proofs_ocr/regression/` for evaluation targets

Refactoring is good enough for now. The next phase should be baseline behavior
debugging, not broad package churn.
