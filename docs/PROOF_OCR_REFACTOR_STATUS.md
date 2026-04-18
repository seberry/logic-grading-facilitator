# Proof OCR Refactor Status

## Purpose

This note is the short restart brief for the current proof OCR state. Read this
first if you need to know what is live, what is still provisional, and what the
next work should optimize for.

## Current Reality

The repo now has two live modal OCR paths:

- Old reference implementation:
  - `scripts/debug_modal_proof_ocr.py`
- New baseline path:
  - `proofs_ocr/pipelines/modal_v2.py`

The old script is still the richer reference implementation. The new
`proofs_ocr/` path is no longer a scaffold. It runs end to end, writes debug
artifacts, and is selectable in `/debug/modal-proof`.

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
- The modal regression `expected_structure.json` files are regression targets,
  not uniformly gold truth.
- Handwritten expected structures are especially provisional.

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
3. Before adding another wave of debug-viewer behavior, split
   `app/routes/modal_proof_debug.py` so route wiring, state/fixture loading,
   artifact loading, and page/view-model assembly stop living in one file.
4. After that split is stable, go back to improving bar bounds / masking before
   stacking more OCR heuristics.
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
