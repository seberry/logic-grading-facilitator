# Proof OCR Refactor Status

## Purpose

This note is the short restart brief for the proof OCR cleanup project. Read
this first if the codebase feels confusing and you want the current intended
direction without reconstructing the full conversation.

## The Short Version

The project now has two different layers:

- The old reference implementations:
  - `scripts/debug_fitch_ocr.py`
  - `scripts/debug_modal_proof_ocr.py`
- The new clean refactor workspace:
  - `proofs_ocr/`

The old scripts are still the real implementations. The new `proofs_ocr/`
package is a scaffold for a cleaner second pass. It is not wired into the app
yet.

## What Is Actually Committed

The important committed refactor snapshot is:

- `d173dbb` — `Scaffold proofs OCR workspace`

That commit adds:

- `docs/PROOF_OCR_LLM_README.yaml`
- `proofs_ocr/common/`
- `proofs_ocr/domains/`
- `proofs_ocr/pipelines/`
- `proofs_ocr/regression/`

It also adds a tiny runner:

```bash
python -m proofs_ocr.regression
```

That command reads `proofs_ocr/regression/manifest.yaml` and reports whether
each seeded regression case exists locally.

## What To Treat As Stable

Treat these as the stable starting point for the refactor:

- `proofs_ocr/` as the new workspace
- `proofs_ocr/regression/manifest.yaml` as the seed evaluation target
- `docs/PROOF_OCR_LLM_README.yaml` as the design/guardrail document
- `scripts/debug_fitch_ocr.py` and `scripts/debug_modal_proof_ocr.py` as the
  reference behavior to copy from, not as libraries to import from

## What Is Still Provisional

The modal regression manifest currently contains:

- 4 modal typeset cases from `working/modal-proof-tex-suite`
- 2 handwritten modal cases from repo-local debug crops

Important caveat:

- The handwritten expected structures are provisional debug outputs, not
  hand-curated gold truth.

That is still useful for restart context, but it should not be mistaken for a
finished benchmark suite.

## What Is Confusing Right Now

There may be uncommitted leftovers in the older modal OCR path, including:

- `modal_ocr/tree.py`
- `modal_ocr/masking.py`
- edits in `scripts/debug_modal_proof_ocr.py`
- edits in `modal_ocr/verifier.py`

Those are not part of the committed `proofs_ocr/` scaffold. Unless they are
cleaned up and deliberately committed later, they should be treated as drift,
not direction.

## Recommended Fresh-Context Starting Plan

1. Read `docs/PROOF_OCR_LLM_README.yaml`.
2. Run `python -m proofs_ocr.regression`.
3. Inspect `proofs_ocr/regression/manifest.yaml`.
4. Keep new work inside `proofs_ocr/`.
5. Copy only stable shared primitives from the old scripts.
6. Do not add new repair/verifier layers before the baseline clean pipeline is
   understandable.

## Architectural Intent

The long-term split is:

- `proofs_ocr/common/` for shared Fitch-style proof mechanics
- `proofs_ocr/domains/` for modal/TFL-specific normalization and heuristics
- `proofs_ocr/pipelines/` for concrete experiment pipelines
- `proofs_ocr/regression/` for evaluation targets

The key durable idea is recursive decomposition:

1. detect subproof bars
2. build a tree
3. recurse into child scopes
4. mask children with placeholders
5. OCR the current scope
6. stitch everything back together

That workflow is the real core. Prompt experiments and repair heuristics should
remain secondary to it.
