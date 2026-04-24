# Cross-Pipeline Investigation: TFL Proofs Through Modal Pipeline

**Date:** 2026-04-23
**Status:** Investigation complete → refactoring in progress

## Background

We have **two separate OCR pipelines** for Fitch proofs, both using chop-and-mask:

### TFL Pipeline (`proofs_ocr/pipelines/tfl_fitch.py`)
- `run_fitch_ocr_pipeline()` — the TFL-specific entry point
- Detects vertical bars via OpenCV → builds a `SubproofNode` tree → recursively crops/masks child subproofs → VLM per crop → stitches with `<SUBPROOF_ID_X>` tokens
- Uses TFL-specific OCR prompt (no modal references)
- Uses TFL-specific normalization (ASCII→Unicode symbol map, TFL justification snapping)
- Invoked via: `python -m proofs_ocr.regression.tfl_runner run-gold`

### Modal v2 Pipeline (`proofs_ocr/pipelines/modal_v2_core.py`)
- `transcribe_image_to_modal_v2_debug_state()` — the modal entry point
- Also detects bars → builds tree → crops → segments rows within each node → VLM per row
- Uses modal-specific prompts (mentions □, ◇, strict subproofs, etc.)
- Has more complex row segmentation (`_segment_node_rows`)
- Has feature flags for contradiction reread, formula consensus, etc.
- Invoked via: `python -m proofs_ocr.regression.runner run-gold`

## What Happened

We accidentally ran 4 TFL handwritten proof images through the **modal** pipeline (using `runner.py`) instead of the TFL pipeline (using `tfl_runner.py`). The results were dramatically worse:

| Case | TFL Pipeline (F/J/B) | Modal Pipeline (F/J/B) | Modal Lines |
|------|---------------------|------------------------|-------------|
| Q3 `(P∧Q)→R` | 8/8 / 8/8 / 8/8 | 2/8 / 3/8 / 2/8 | 10 (2 extra) |
| Q4 `N∨(K∧W)` | 5/7 / 5/7 / 4/7 | 0/7 / 3/7 / 0/7 | 7 |
| Q5 `(P∨K)↔(Z∧W)` | 4/5 / 4/5 / 3/5 | 1/5 / 1/5 / 0/5 | 6 (1 extra) |
| Q6 `S↔W` | 11/11 / 4/11 / 4/11 | 0/11 / 4/11 / 0/11 | 18 (7 extra!) |
| **Total** | **28/31 / 21/31 / 19/31** | **3/31 / 11/31 / 2/31** | |

## Key Observations

1. **Both pipelines use chop-and-mask** — the structure detection approach is the same (CV bar detection → recursive crop/mask)
2. **The modal pipeline injects modal-specific artifacts** — e.g., `□Q` hallucinated in Q3 line 7, garbage like `(¬8)→(¬<)` 
3. **The modal pipeline produces far more extra lines** — Q6 (11 expected) produced 18 actual lines
4. **100% structural mismatch rate** on Q3 (8/8 lines had wrong nesting paths)
5. **Modal prompts are misaligned for TFL** — they reference □, ◇, strict subproofs, etc., which don't exist in TFL proofs

## Open Questions for Investigation

1. **Why does the modal pipeline hallucinate so badly on TFL proofs?** Is it purely the prompt, or does the row segmentation (`_segment_node_rows`) also contribute?
2. **Could a domain-aware prompt switch help?** If the modal pipeline detected "TFL" vs "modal" and switched prompts, would results improve?
3. **Is the modal pipeline's row segmentation better than TFL's whole-crop approach?** Could TFL benefit from the modal pipeline's `_segment_node_rows` logic?
4. **What would a unified pipeline look like?** Could we merge the best parts of both?
5. **What is the nondeterminism profile of each pipeline on the same images?** We still haven't re-run the same pipeline on the same images to measure VLM nondeterminism.

## How to Re-Run the TFL Pipeline Properly

```bash
# TFL-specific runner (uses tfl_fitch.py pipeline)
python -m proofs_ocr.regression.tfl_runner run-gold --output-root working/tfl_proper_rerun

# Modal runner on TFL images (what we accidentally did)
python -m proofs_ocr.regression.runner run-gold \
  --case tfl_handwritten_q3_P_and_Q_imp_R \
  --case tfl_handwritten_q4_N_or_K_and_W \
  --case tfl_handwritten_q5_P_or_K_biconditional \
  --case tfl_handwritten_q6_S_iff_W \
  --output-root working/tfl_via_modal_pipeline \
  --enable-contradiction-reread \
  --enable-formula-consensus-reread
```

## Refactoring Progress (2026-04-23)

The investigation revealed that both pipelines duplicate substantial logic. We extracted the shared pieces into composable modules that both pipelines can import.

### New shared modules

| Module | Purpose |
|--------|---------|
| `proofs_ocr/tricks/symbol_normalization.py` | `collapse_whitespace()`, `TFL_SYMBOL_MAP`, `MODAL_SYMBOL_REPLACEMENTS` |
| `proofs_ocr/tricks/citation_cleanup.py` | `normalize_citation_dashes()`, `normalize_citation_commas()` |
| `proofs_ocr/tricks/justification_snaps.py` | `snap_justification_rule_head()`, `PREMISE_ALIAS_HEADS` |
| `proofs_ocr/domains/shared.py` | `parse_vlm_json_response()`, `has_visible_formula()`, `build_subproof_token_entry()`, prompt fragments |

### Wiring completed

- **TFL pipeline** (`proofs_ocr/pipelines/tfl_fitch.py`): removed duplicated `renumber_lines` and `recursive_stitch`, now imports from `proofs_ocr.common.stitch` and `proofs_ocr.domains.shared`. Net -56 lines.
- **Modal domain** (`proofs_ocr/domains/modal.py`): replaced `_collapse_whitespace`, `_DASH_BETWEEN_DIGITS_RE`/`_COMMA_BETWEEN_DIGITS_RE`, and `_PREMISE_ALIAS_HEADS` with imports from tricks. Net -2 lines.

### Remaining work

- The TFL `normalize_proof_structure()` closure (~180 lines) has 10+ nested helpers that mirror modal normalization logic. This is the highest-value migration target but requires careful surgical extraction.
- Both pipelines still have their own fence-stripping code; can consolidate to `domains.shared.parse_vlm_json_response`.
- The modal pipeline's row segmentation (`_segment_node_rows`) could potentially benefit the TFL pipeline.

All 381 tests pass after every commit.

## Artifacts

- Modal-on-TFL outputs: `working/tfl_via_modal_pipeline/`
- Previous TFL-on-TFL outputs: `working/tfl_gold_regressions/` (from earlier runs)
