# Proof OCR Pipeline Overview

## Purpose

This is the stable mental model for the proof OCR work in this repo. It is the
short document to hand to another LLM, or to reread after context loss, before
changing OCR behavior.

Read this together with:

- `docs/PROOF_OCR_REFACTOR_STATUS.md`
- `docs/PROOF_OCR_LLM_README.yaml`

## Main Goal

The project is not trying to make every proof perfect in one OCR pass. The
near-term goal is:

- get the recursive proof structure right
- make final OCR errors small, local, and easy for a human to correct
- avoid adding heuristics that silently damage other fixtures

That means the most important failures are final-answer failures, not ugly
intermediate artifacts by themselves.

## Current Architecture

The live modal baseline is `proofs_ocr/pipelines/modal_v2.py`.

The key boundaries are:

- `proofs_ocr/common/`
  - shared proof mechanics
  - bar detection
  - tree construction
  - masking
  - row segmentation
  - stitching
- `proofs_ocr/domains/modal.py`
  - modal-specific prompt wording
  - normalization
  - narrow modal repair heuristics
- `proofs_ocr/pipelines/modal_v2_core.py`
  - recursive orchestration
- `proofs_ocr/pipelines/modal_v2_runtime.py`
  - runtime OCR adapter
  - per-line artifact writing
- `proofs_ocr/pipelines/modal_v2_debug.py`
  - debug-state assembly for `/debug/modal-proof`
- `proofs_ocr/regression/`
  - curated fixture metadata
  - gold targets
  - regression runner

## Pipeline Shape

The durable idea is recursive decomposition of a Fitch-style proof image:

1. Detect vertical bars.
2. Build a nested subproof tree from those bars.
3. Recurse into children first.
4. Mask each child inside its parent with a placeholder token.
5. Segment rows in the current node.
6. OCR each visible line crop.
7. Stitch child placeholders back into a nested proof structure.
8. Normalize and renumber the final structure.

If the code or a proposed heuristic makes this flow harder to see, that is a
warning sign.

## Current modal_v2 Story

The current `modal_v2` path is no longer just "one OCR prompt per line." It has
three layers:

1. Structure pass
   - detect bars
   - build the recursive subproof tree
   - segment rows inside each node
2. Runtime line OCR
   - do the ordinary whole-line read for each visible row
   - optionally use the formula/justification split path for the same row
3. Narrow runtime rereads
   - contradiction reread for suspicious `¬E` / `~E` rows that may really be
     `⊥`
   - boxed justification-head reread for suspicious or blank justifications
   - formula-only consensus reread, where the verifier model rereads only the
     formula side and the runtime only rewrites the formula when the reread
     attempts agree

That distinction matters:

- runtime rereads can change the saved OCR result
- debugger review tools can only suggest alternatives unless a human applies
  them

The current best-performing content path is therefore:

- keep structure and baseline line ownership stable
- keep the baseline whole-line read conservative
- use stronger verifier rereads only on narrow formula/justification targets

## Visual ROOT Convention

There are two different "roots" in play:

- visual parsing `ROOT`
  - the outer image/container crop used for recursive parsing
- final JSON top level
  - the stitched semantic proof structure after child placeholders are resolved

These are not always the same thing.

In ordinary Fitch proofs with premises, the visible proof can sit entirely
inside an outer proof rail. In that layout, the visual parsing `ROOT` may have
no visible sentences of its own and may only contain a masked child proof
environment token. That is normal and is not by itself a tree bug.

The theorem-style case is different:

- the visual parsing `ROOT` may contain a visible line outside the proof rail,
  usually below one or more cited proof environments

Practical consequence:

- do not treat "empty ROOT rows" as suspicious by default
- when `ROOT` has no visible lines, the parser may legitimately stitch from the
  first child proof environment instead

## What The Debug Viewer Is Showing

The `/debug/modal-proof` page is useful because it exposes several layers of
the pipeline separately:

- `debug_bars.jpg`
  - raw bar detections
- `intermediate_<id>.json`
  - tree structure for a node
- `crop_<id>.jpg`
  - node crop
- `node_row_bands_<id>.json`
  - raw and filtered row bands
- `node_rows_<id>.json`
  - final row records for that node
- `node_line_<id>_<row>.jpg`
  - actual per-line OCR crop
- `node_line_<id>_<row>.json`
  - raw VLM response, parse result, and any recovery/skip reason
- `final_structure_baseline.json`
  - final nested proof output used for regression comparison

For current `modal_v2` runs, the debugger also has two different review layers:

- runtime artifacts inside `node_line_<id>_<row>.json`
  - these show what the baseline OCR path actually applied, including
    `recovery_applied` and any `formula_consensus_reread` or justification
    reread payloads
- checker-guided review artifacts
  - `checker_formula_rereads.json`
  - these are post-checker suggestions used for human review in the debugger
  - they do not silently rewrite the proof JSON

Important debugging principle:

- strange-looking `ROOT` rows are not automatically a final-answer bug
- a bug matters most when it survives into `final_structure_baseline.json`

## Failure Ladder

When a fixture is wrong, check the stages in this order:

1. Bar geometry
   - wrong bar set
   - wrong tree depth
   - false left separator treated as a scope bar
2. Parent masking / child ownership
   - parent crop loses formulas it should still own
   - label-only rows remain visible in the parent
3. Row segmentation
   - missed row
   - merged rows
   - child token slot mistaken for text
   - post-child lower rows may need a justification-side split hint when the
     formula-side band stays merged
4. Line OCR / parsing
   - symbol loss
   - unary operator loss such as `¬`
   - justification corruption
5. Final stitching / normalization
   - wrapper depth wrong
   - harmless formatting differences
   - line numbering differences

Fixes should target the earliest stage that actually explains the bug.

## Current Regression Strategy

The repo now has a curated 5-case modal gold set:

- `modal_k_typeset_mixed_subproofs`
- `modal_s4_typeset_r4_corrected`
- `modal_s5_typeset_example`
- `modal_handwritten_user_box_q`
- `modal_handwritten_user_negated_modal_conjunction`

Those fixtures live under `proofs_ocr/regression/fixtures/modal/`.

There are two useful test layers:

- fast tests
  - mechanics
  - route/debug tests
  - deterministic regression-runner tests
- slow gold runs
  - run the real modal_v2 OCR on the 5 fixture images
  - compare against curated gold JSON

Useful command:

```powershell
$env:PYTHONIOENCODING='utf-8'
python -m proofs_ocr.regression run-gold --output-root working/modal_v2_gold_regressions
```

## Change Policy

When adding a heuristic:

- prefer the narrowest place that can explain the failure
- prefer a contained safeguard before a broad semantic rewrite
- do not silently change the TFL path while modal_v2 is still settling
- keep optional behavior behind explicit flags if it is speculative
- rerun the gold fixtures after the change

When deciding whether a change is good, distinguish:

- final-answer correctness improvements
- debug-view cosmetic improvements
- normalization-only differences

Those are not equally important.

## Current Lessons

Recent example: the false left separator bar in
`modal_s4_typeset_r4_corrected`.

What happened:

- a false extra bar created an extra tree node
- `ROOT` then owned label-only rows
- those crops were sent to the VLM
- the VLM hallucinated a stray `⊥`

What helped:

- adding a runtime safeguard for label-only wide crops

What it did not solve:

- the false extra bar still exists

That is a good example of the general rule:

- contain visible damage first if cheap

Another current example is the handwritten contradiction step in
`modal_handwritten_user_negated_modal_conjunction`.

What happened:

- after child-token filtering, the parent node kept one merged post-child lower
  band
- the justification-side projection still showed two lower rows
- the contradiction step therefore never reached OCR as its own line crop

What helped:

- a narrow post-child split recovery in `proofs_ocr/common/rows.py`
- that recovery only runs when one post-child band remains and the right-side
  projection offers a credible two-row lower-half split

What it did not solve:

- the recovered bottom rows are still OCRed poorly
- so the failure has moved from row segmentation to line reading
- still fix the earlier geometric cause afterwards

## How To Decide What To Work On Next

Prefer this order:

1. final-answer bugs on the curated typeset gold cases
2. structural geometry bugs that affect multiple cases
3. broader handwritten OCR improvements
4. viewer/debug niceties

At the moment, the cleanest remaining semantic typeset target is loss or
corruption of `¬` in `modal_s5_typeset_example`.
