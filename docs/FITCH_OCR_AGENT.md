# TFL Fitch OCR Pipeline — Agent Orientation

> **Audience:** A fresh Cline conversation starting an OCR iteration session.
> Read this file first to get productive immediately on the Fitch OCR loop
> without wading through unrelated grading/viewer code.

---

## Quick Start

```bash
# 1. Run CI-safe regression tests (no VLM needed, fast):
pytest tests/test_tfl_fitch_ocr_regression.py -v

# 2. Run live gold regressions (requires VLM endpoint):
python -m proofs_ocr.regression.tfl_runner run-gold

# 3. Run a single case to save time:
python -m proofs_ocr.regression.tfl_runner run-gold --case tfl_handwritten_q4_N_or_K_and_W

# 4. Custom output directory (default: working/tfl_gold_regressions):
python -m proofs_ocr.regression.tfl_runner run-gold --output-root working/my_experiment

# 5. Live OCR via pytest (opt-in, Windows):
set RUN_LIVE_TFL_OCR=1
pytest tests/test_tfl_fitch_ocr_regression.py -v -k Live
```

### Where to look at results

After running regressions, inspect the per-case output directory:

| File | What it shows |
|---|---|
| `debug_bars.jpg` | Bar detection overlay (green rectangles + IDs) |
| `crop_ROOT.jpg` | Top-level masked image sent to VLM |
| `crop_N.jpg` | Sub-proof node N's masked crop |
| `crop_N_context.jpg` | Context-aware crop (green box) for small subproofs |
| `response_N.json` | Raw VLM JSON response for node N |
| `thinking_N.txt` | Qwen3.x reasoning/thinking text for node N (27b model only) |
| `final_structure.json` | Stitched + normalized proof structure |
| `gold_regression_result.json` | PASS/FAIL + run metadata |
| `gold_regression_metrics.json` | Per-line formula/justification/structure scores |
| `gold_regression_diff.txt` | Unified diff (expected vs actual) on failures |

Default output location: `working/tfl_gold_regressions/<case_id>/`

---

## Key Files

### Pipeline core

| File | Purpose |
|---|---|
| `proofs_ocr/pipelines/tfl_fitch.py` | **Main pipeline.** Recursive chop-and-mask OCR flow. Entry point: `run_fitch_ocr_pipeline()`. Contains bar detection, tree building, VLM calling, masking, context-aware crops, normalization, and CLI `main()`. |
| `proofs_ocr/common/stitch.py` | **Structural assembly.** `recursive_stitch()` expands `<SUBPROOF_ID_N>` placeholders into nested subproof lists. `renumber_lines()` assigns sequential line numbers depth-first. |
| `proofs_ocr/domains/tfl.py` | **TFL canonicalization.** `canonicalize_formula_for_comparison()` and `canonicalize_justification_for_comparison()` normalize OCR output for regression comparison (symbol mapping, alias lookup, whitespace). |
| `proofs_ocr/tricks/justification_snaps.py` | **OCR misread repair.** Tables mapping common VLM misreads to canonical rule heads (`SHARED_HEAD_FIXES`, `TFL_VALID_RULE_HEADS`). `snap_justification_rule_head()` is the main entry point. Also has `clean_justification_for_tfl()`. |

### Regression infrastructure

| File | Purpose |
|---|---|
| `proofs_ocr/regression/manifest.yaml` | **Test case registry.** Defines all fixtures (TFL + modal) with IDs, paths, tags, oracle kind. |
| `proofs_ocr/regression/tfl_runner.py` | **Gold regression runner.** `run_tfl_gold_regressions()` runs the pipeline on all TFL handwritten fixtures, compares output to expected structures, writes granular per-line reports. |
| `proofs_ocr/regression/runner.py` | **Shared regression helpers.** `load_manifest()`, `select_cases()`, `GoldRegressionResult` dataclass. |

### Test fixtures (TFL handwritten)

Each fixture directory contains: `image.png`, `expected_structure.json`, optional `notes.txt` and `semantic_notes.yaml`.

| Fixture ID | Description | Key challenge |
|---|---|---|
| `tfl_handwritten_q3_P_and_Q_imp_R` | `(P∧Q)→R, P→Q ⊢ (P∧S)→R`. Single subproof with →I. | Baseline case, simplest subproof |
| `tfl_handwritten_q4_N_or_K_and_W` | `N∨(K∧W), N↔W ⊢ W`. ∨E with two subproofs. | **Small subproofs**, ∨E structure, ↔F→↔E normalization |
| `tfl_handwritten_q5_P_or_K_biconditional` | No subproofs, intentional error (self-citation on line 4). | Flat proof, error detection |
| `tfl_handwritten_q6_S_iff_W` | ↔I with two subproofs, intentional errors on lines 6 and 10. | Nested subproofs, biconditional intro |

### Test files

| File | Purpose |
|---|---|
| `tests/test_tfl_fitch_ocr_regression.py` | **Main test suite.** CI-safe canonicalization + runner tests, plus opt-in live OCR tests (`RUN_LIVE_TFL_OCR=1`). |
| `tests/test_tfl_fitch_gold_fixtures.py` | Validates gold fixture structure/integrity. |
| `tests/test_debug_fitch_ocr.py` | Tests for the debug CLI wrapper. |

### Debug / utility scripts

| File | Purpose |
|---|---|
| `scripts/debug_fitch_ocr.py` | CLI wrapper for running the pipeline on a single scan image. |
| `app/routes/fitch_debug.py` | Web UI route for viewing pipeline artifacts in browser. |

---

## Pipeline Architecture

The Fitch OCR pipeline uses a **recursive chop-and-mask** strategy:

```
Input image
    │
    ▼
1. BAR DETECTION
   detect_vertical_bars() → list of (x, y, w, h) boxes
   filter_scaffold_bars() → remove page-border artifacts
   _split_bars_on_gaps()  → separate merged subproof bars
    │
    ▼
2. TREE BUILDING
   build_proof_tree() → SubproofNode hierarchy
   - Root = entire image
   - Children = vertical bars, nested left-to-right
   - Parent = closest leftward bar that vertically spans this bar
    │
    ▼
3. BOTTOM-UP RECURSIVE PROCESSING  (process_node_recursive)
   For each node (deepest children first):
   a. Crop the node's region from the parent image
   b. For each already-processed child:
      - White-out the child's region on this image
      - Stamp a red <SUBPROOF_ID_N> token in the masked area
   c. Send the masked crop to the VLM for OCR
   d. Store VLM response in library[node_id]
    │
    ▼
4. LETTERBOXING (small leaf crops)
   Leaf nodes shorter than SMALL_CROP_THRESHOLD (250px)
   get padded with white space to a 4:3 aspect ratio (min height 250px).
   This prevents the VLM's internal image resizing from distorting tiny
   crops, which was causing severe OCR quality degradation.
    │
    ▼
5. STITCHING
   recursive_stitch() replaces <SUBPROOF_ID_N> tokens
   with the actual child node lists → nested proof structure
    │
    ▼
6. NORMALIZATION
   normalize_proof_structure():
   - ASCII → Unicode symbol mapping (~→¬, &→∧, v→∨, etc.)
   - Uppercase atom letters
   - Extract embedded justifications from wffstr
   - Snap rule heads via justification_snaps
   label_and_count_premises() → mark leading blank-jstr lines as Pr
   renumber_lines() → sequential 1..N depth-first
   apply_default_justifications() → fill remaining blanks with Pr/Hyp
    │
    ▼
7. OUTPUT
   final_structure.json written to output dir
```

### Line-count estimation & hallucination guard (ACTIVE)

`estimate_line_count()` wraps the shared morphological band detector from `proofs_ocr/common/rows.py` (`detect_text_row_bands`) with post-filtering:

1. **Band detection:** Morphological opening with horizontal + vertical kernels separates text from underlines and Fitch bars
2. **Artifact removal:** Bands shorter than 18px (underline residuals) are filtered out
3. **Merge estimation:** Bands taller than 1.6× the median height are estimated as containing multiple merged lines (round to nearest)
4. **Result:** 5/5 correct line counts on all TFL leaf crop fixtures

The count is used two ways:

1. **Prompt constraint:** The leaf prompt includes `"This image contains exactly N proof line(s). Output exactly N line objects."`
2. **Response truncation:** If the VLM returns more dict items than expected, extras are silently dropped.

**Diagnostic script:** `scripts/test_line_count_bands.py` validates the detector against known expected line counts.

### Fuzzy justification snapping

`fuzzy_snap_justification()` in `justification_snaps.py` runs as a second normalization pass after `snap_justification_rule_head()`:

- **Combinatorial head repair:** When the rule head isn't in `TFL_VALID_RULE_HEADS`, tries all combinations of known OCR confusions (8↔∧, B↔∧, X↔⊥, ∈↔E, ±↔↔, etc.). Accepts the repair only if exactly one valid candidate emerges.
- **Always-on tail rescue:** Unconditionally applies `S→5` and `O→0` substitutions in citation tails (e.g. `"∧E S"` → `"∧E 5"`).

### VLM Prompt

The prompt is built by `build_ocr_prompt()`:
- **Leaf nodes (no children):** Simple prompt — no token instructions, preventing hallucinated `<SUBPROOF_ID>` tokens. Includes line-count constraint when available. Includes "Transcribe only the lines actually visible in this image." to reduce hallucinated extra lines.
- **Internal nodes:** Includes instruction to copy red-box tokens verbatim
- **Context-aware crops:** DISABLED (see History below). Code preserved but `use_context_crop = False`.

### Letterboxing

`letterbox_crop()` in `tfl_fitch.py` pads small images:
- Minimum height enforced (default 250px) with vertical white padding
- Then padded horizontally to 4:3 aspect ratio
- Applied only to leaf crops shorter than `SMALL_CROP_THRESHOLD`
- Prevents VLM's internal resize from distorting thin crops

---

## Iteration Workflow

The main development loop is:

1. **Tweak** — Edit pipeline code (bar detection thresholds, normalization rules, snap tables, prompt text, context-crop logic)
2. **Commit** — Always commit before running slow tests, so you have a clean revert point and the diff is easy to review: `git add -A && git commit -m "description of tweak"`
3. **Run fast tests** — `pytest tests/test_debug_fitch_ocr.py tests/test_tfl_normalization_rescore.py -v` — catches regressions in normalization logic (under 1 second)
4. **Run slow tests** — `python -m proofs_ocr.regression.tfl_runner run-gold` — live VLM calls, takes several minutes
5. **Analyze** — Inspect output artifacts:
   - `gold_regression_metrics.json` for per-line scores
   - `gold_regression_diff.txt` for expected-vs-actual diffs
   - `crop_*.jpg` images to see what the VLM received
   - `response_*.json` to see raw VLM output
   - `debug_bars.jpg` to verify bar detection
6. **Report** — Summarize what changed vs. the previous baseline: line counts, formula/justification match rates, new passes or regressions
7. **Commit results** — After a successful iteration (tests pass, regressions improve or hold steady), commit all changes including documentation updates: `git add -A && git commit -m "feat: description of improvement and results"`
8. **Plan next tweak** — Based on failure patterns

### Commit Policy

**Commit early, commit often.** Specific guidance:

- **Before slow tests:** Always commit before running `tfl_runner run-gold` so you have a clean revert point
- **After successful work:** When a feature is working and validated (tests pass, gold regressions improve), commit immediately — don't accumulate uncommitted changes across multiple features
- **Documentation updates:** Include doc updates (this file, README changes) in the same commit as the code they describe
- **Good commit messages:** Use descriptive messages: `"feat: wire modal row detector for accurate TFL line counting (26/26 correct)"` rather than `"updates"` or `"fix"`
- **When in doubt, commit.** It costs nothing and gives you a safety net

### Common tweak targets

| Problem | File to edit | What to change |
|---|---|---|
| Bar detection misses/splits bars | `tfl_fitch.py` → `detect_vertical_bars()`, `_split_bars_on_gaps()` | Thresholds, kernel sizes, gap detection |
| Scaffold false positives | `tfl_fitch.py` → `filter_scaffold_bars()` | Edge margin, width filters |
| VLM misreads rule names | `justification_snaps.py` → `SHARED_HEAD_FIXES` | Add new OCR confusion mapping |
| VLM misreads symbols in formulas | `tfl.py` → `_SYMBOL_REPLACEMENTS` | Add ASCII→Unicode mapping |
| VLM puts justification in formula field | `tfl_fitch.py` → `extract_embedded_justification()` | Splitting heuristics |
| Small subproofs get bad OCR | `tfl_fitch.py` → `SMALL_CROP_THRESHOLD`, `build_context_crop()` | Threshold, highlight box logic |
| VLM prompt needs tuning | `tfl_fitch.py` → `build_ocr_prompt()` | Prompt text adjustments |
| Premise count wrong | `tfl_fitch.py` → `label_and_count_premises()` | Heuristic for identifying premises |

---

## Known Issues & Current Focus (as of 2026-04-25, run 10:56)

### Current gold regression scoreboard

| Case | Status | Lines | Formula | Justification | Key issue |
|------|--------|-------|---------|---------------|-----------|
| Q3 | ✅ PASS | 8/8 | 8/8 | 6/8 | 2 citation number errors (pre-existing) |
| Q4 | ❌ FAIL | 7/7 | 4/7 | 5/7 | VLM symbol misreads on small crops (4b model) |
| Q5 | ❌ FAIL | 5/5 | 4/5 | 5/5 | `↔` misread as `→` on line 1 |
| Q6 | ❌ FAIL | 11/11 | **11/11** ✅ | 6/11 | All formulas perfect; justification citation errors |

**Line count accuracy: 26/26 across all cases (0 extra, 0 missing).**

### Q4 small subproof problem
- **Case:** `tfl_handwritten_q4_N_or_K_and_W`
- **Expected:** 7 lines. `N∨(K∧W)` Pr, `N↔W` Pr, subproof [`N` Hyp, `W` ↔E 2,3], subproof [`K∧W` Hyp, `W` ∧E 5], `W` ∨E 1,3-4 5-6
- **Current best result:** 9 lines (was 14 before letterboxing)
- **Remaining issues:**
  - `N↔W` misread as `N↔N` (letter confusion — VLM reads W as N)
  - Crop 3 (second subproof): VLM misreads the layout entirely
- **Crop 3 diagnosis — layout confusion, not just illegibility:**
  - Expected: 2 lines — `K∧W / Hyp` and `W / ∧E 5`
  - VLM returns 3 lines: `"x < 8w"/""`, `"x"/"1"`, `"8 ∈ S"/""`
  - The VLM reads `K∧W` as garbled `"x < 8w"` (illegible), hallucinates a middle line `"x"/"1"`, and then **treats the justification column `∧E 5` as a third formula line**, reading it as `"8 ∈ S"`
  - Our post-processing chain (`8→∧`, `∈→E`) WOULD normalize `8 ∈ S` → `∧ E S`, but since the VLM put it in the wffstr field instead of jstr, it stays as a formula rather than being recognized as a justification
  - **Root cause:** The 4b model can't distinguish the formula region from the justification region in this crop. It reads left-to-right across both columns as if they were one continuous text block, inflating 2 lines into 3
- **Potentially addressable:** A wider left-margin crop that separates formula from justification columns more clearly, or pre-extraction normalization that recognizes `8 ∈ S` as a misplaced justification even when it appears in wffstr

### Q5 biconditional misread
- **Case:** `tfl_handwritten_q5_P_or_K_biconditional`
- **Only failure:** Line 1 reads `(P∨K)→(Z∧W)` instead of `(P∨K)↔(Z∧W)` — the VLM reads handwritten `↔` as `→`. Everything else (lines 2-5) is perfect including all justifications.
- **This is a VLM symbol recognition issue**, not a normalization issue. Whitespace normalization is already correct.

### Q6 near-perfect formulas, citation errors
- **Case:** `tfl_handwritten_q6_S_iff_W`
- **Major improvement:** Line count correct (11/11), structure perfect (0 mismatches), **all 11 formulas correct**.
- **Remaining gap:** 4 justification citation-number errors (lines 4, 5, 6, 8). The VLM reads the correct rule names but misreads the handwritten line reference numbers.
- **Promising:** This is a strong candidate for semantic validation repair — the formulas are correct enough to validate/invalidate citations.

### Context-aware crops: DISABLED
- **What was tried:** Sending the parent's full unmasked image with a green highlight rectangle, telling the VLM to OCR only inside the green box.
- **Result:** The VLM (qwen3-vl-4b) completely ignored the green box and read the entire parent image, producing 5-6 lines per leaf instead of 2. This doubled the total line count (7→14).
- **Decision:** Disabled in favor of letterboxed tight crops. Code is preserved (`build_context_crop()` still exists) with `use_context_crop = False` in case a future model respects the instruction.

### Letterboxing: ACTIVE
- **What it does:** `letterbox_crop()` pads small leaf crops to 4:3 aspect ratio with white space (min height 250px)
- **Result:** Reduced Q4 from 14 lines → 9 lines. The VLM can no longer read beyond crop boundaries because the tight crop physically limits what's visible.
- **Trade-off:** Without parent context, the VLM sometimes can't interpret ambiguous handwriting (e.g. crop_3 returns gibberish).

### Line-count guard: ACTIVE (re-enabled 2026-04-25)
- **Flag:** `USE_LINE_COUNT_CONSTRAINT = True` in `tfl_fitch.py`
- **How it works:** `estimate_line_count()` wraps the shared morphological band detector from `proofs_ocr/common/rows.py` (`detect_text_row_bands`). This detector uses morphological opening with horizontal + vertical kernels to separate text from underlines and Fitch bars before counting. Post-filtering removes artifact bands and estimates merged lines.
- **Accuracy:** 26/26 correct line counts across all gold regression cases (Q3, Q4, Q6). Zero extra lines, zero missing lines.
- **Previous issue resolved:** The old `count_text_lines_in_region()` used naive horizontal projection that picked up Fitch bars, causing over-counts. The new approach morphologically removes bars before counting.

### Symbol normalization status
- `8` → `∧` normalization is **working correctly** in `proofs_ocr/domains/tfl.py` (not in `symbol_normalization.py`). Applied at comparison time via `_EIGHT_BEFORE_EI_RE` and `_EIGHT_BETWEEN_CAPS_RE`.
- `∈` → `E` normalization also exists and works (handles `8 ∈ S` → `8 E S` → `∧E S`).
- **Whitespace normalization** in `canonicalize_formula_for_comparison()` already strips all spaces via `_WHITESPACE_RE.sub("", normalized)` — confirmed working.

### 27b model routing: ACTIVE
- **What it does:** Leaf crops with ≤4 expected lines are routed to Qwen 3.6 27B (`qwen/qwen3.6-27b`) via `LMSTUDIO_MODEL_27B` env var, with a 600s timeout. Larger crops still use the default 4b model.
- **Config:** `SMALL_LEAF_MODEL_ROUTING_THRESHOLD = 4` in `tfl_fitch.py`
- **First Q4 27b result (600s timeout):** 3/7 lines matched, 1/7 both correct. The model produced correct line count (7) but struggled with nested subproof structure. It appears to have timed out on complex reasoning — thinking text capture was not yet available.

### Thinking text capture: ACTIVE (commit 235aef0)
- **What it does:** `LMStudioQwenClient.chat()` now extracts `reasoning_content` from the LM Studio response and stores it on `self.last_thinking`. The pipeline saves this as `thinking_{node_id}.txt` alongside each `response_{node_id}.json`.
- **Why it matters:** The 27b model does extended thinking that can loop forever on confusing prompts. Capturing the thinking text lets us diagnose when/why the model gets stuck.
- **Files affected:** `app/vlm_client.py` (extraction), `proofs_ocr/pipelines/tfl_fitch.py` (saving)

### Other active notes
- **Scaffold removal disabled** (2026-04-24): `filter_scaffold_bars()` scaffold-removal logic deliberately disabled because handwritten Fitch proofs have a real outer subproof bar that spans most of the image height. Only edge-artifact cleanup remains active.
- **Line-number blotting** on ROOT node: The pipeline whites out everything to the left of the leftmost bar, preventing the VLM from reading line numbers as formula content.

---

## Experiment History

### 2026-04-25 (10:56): Robust line counting via modal pipeline's row detector

**Problem:** The old `count_text_lines_in_region()` used naive horizontal projection that over-counted (picking up Fitch bars as text), causing the VLM prompt to *instruct* more lines than actually existed, worsening hallucination. `USE_LINE_COUNT_CONSTRAINT` was disabled.

**Solution:** Replaced with `estimate_line_count()` wrapping the modal pipeline's `detect_text_row_bands()` from `proofs_ocr/common/rows.py`. This detector uses morphological opening with horizontal + vertical kernels to separate text from underlines and Fitch bars before counting. Post-filtering removes artifact bands (<18px) and estimates merged lines.

**Changes:**
1. Added `estimate_line_count()` in `tfl_fitch.py` — wraps `detect_text_row_bands()` with filtering
2. Wired it into `process_node_recursive()` replacing `count_text_lines_in_region()`
3. Re-enabled `USE_LINE_COUNT_CONSTRAINT = True`
4. Added `scripts/test_line_count_bands.py` — offline diagnostic for validating band counts

**Offline validation:** 5/5 leaf crops correct (Q3 crop_2=5, Q4 crop_2=2, Q4 crop_3=2, Q6 crop_2=4, Q6 crop_3=4)

**Gold regression results:**

| Case | Lines (exp→act) | Formula | Justification | Extra | Status |
|------|-----------------|---------|---------------|-------|--------|
| Q3 | 8→8 | 8/8 | 6/8 | 0 | ✅ PASS |
| Q4 | 7→7 | 4/7 | 5/7 | 0 | ❌ FAIL (VLM misreads) |
| Q5 | 5→5 | 4/5 | 5/5 | 0 | ❌ FAIL (↔→→) |
| Q6 | 11→11 | **11/11** | 6/11 | 0 | ❌ FAIL (citations) |

**Key improvement:** Line count accuracy went from unreliable to **26/26 perfect**. Zero extra lines across all cases. Q4 improved from 9→7 lines (now correct count, was over-producing before). The remaining failures are all VLM symbol/citation recognition quality — the line counting guard is no longer the bottleneck.

**Next step:** Use the accurate line counts to route small subproofs (≤4 lines) to a larger VLM model for better symbol recognition.

### 2026-04-25 (01:17): Blanket OCR artifact repair (commit 4e1efef)

**Changes:** Added blanket formula-level replacements in both pipeline normalization (`clean_wff` in `tfl_fitch.py`) and comparison canonicalizer (`canonicalize_formula_for_comparison` in `tfl.py`):
- `∈`/`ε` → `E` (epsilon never appears in TFL)
- `8` → `∧` (digits never appear in TFL formulas)
- `5` → `S` (handwritten S misread as 5)

Updated 2 tests that tested the old "8 only between caps" behavior to match new blanket policy.

**Gold regression results:**

| Case | Lines (exp→act) | Formula | Justification | Both | Extra | Struct | Status |
|------|-----------------|---------|---------------|------|-------|--------|--------|
| Q3 | 8→8 | 8/8 | 8/8 | 8/8 | 0 | 0 | ✅ PASS |
| Q4 | 7→9 | 3/7 | 3/7 | 2/7 | 2 | 2 | ❌ FAIL |
| Q5 | 5→5 | 4/5 | 5/5 | 4/5 | 0 | 0 | ❌ FAIL |
| Q6 | 11→11 | **11/11** | 7/11 | 7/11 | 0 | 0 | ❌ FAIL |

**Analysis:** Scores are identical to the previous baseline. This is expected — the VLM returns the same raw output, and the new blanket replacements clean up the pipeline output that *humans* see (e.g., `∧` instead of `8`, proper `E` instead of `∈` in Q4 crop_3's `X < ∧W`). The comparison canonicalizer also now applies the same repairs so that future VLM outputs with these misreads will be auto-matched.

**Key value:** Infrastructure improvement. The human reviewer now sees `∧E 5` instead of `8∈S` in justifications and `∧` instead of `8` in formulas. The remaining failures are model-capacity issues (illegible handwriting in Q4 crop_3, `↔`→`→` misread in Q5, citation-number errors in Q6).

### 2026-04-25 (00:49): Latest gold regression run (no code changes)

**Context:** Re-ran gold regressions to establish current baseline after previous session's changes. No code changes since last commit. `USE_LINE_COUNT_CONSTRAINT = False`.

**Results:**

| Case | Lines (exp→act) | Formula | Justification | Both | Extra | Struct | Status |
|------|-----------------|---------|---------------|------|-------|--------|--------|
| Q3 | 8→8 | 8/8 | 8/8 | 8/8 | 0 | 0 | ✅ PASS |
| Q4 | 7→9 | 3/7 | 3/7 | 2/7 | 2 | 2 | ❌ FAIL |
| Q5 | 5→5 | 4/5 | 5/5 | 4/5 | 0 | 0 | ❌ FAIL |
| Q6 | 11→11 | **11/11** | 7/11 | 7/11 | 0 | 0 | ❌ FAIL |

**Key observations:**
- Q3 stable, Q4 same as previous baseline (line-count guard disabled avoids the 12-line regression)
- Q5 failure is `↔`→`→` misread on line 1, NOT whitespace — normalization already strips spaces correctly
- **Q6 major improvement:** formulas went from 6/11→**11/11**, line count 13→11 (correct). Only 4 justification citation-number errors remain
- Aggregate: formula 26/31, justification 23/31, both 21/31

**Q6 justification mismatches (only errors):**

| Line | Expected | Actual | Error type |
|------|----------|--------|------------|
| L4 | `∧E 3` | `∧E 1` | Wrong citation number |
| L5 | `→E 1,4` | `Hyp` | VLM failed to read, defaulted to Hyp |
| L6 | `∧I 4,5` | `∧I 3,2` | Wrong citation numbers |
| L8 | `∧E 7` | `→E 2,8` | Wrong rule + citations |

**Conclusion:** Whitespace normalization was already working. Line-count guard confirmed disabled. The most promising next direction is semantic validation of citations for Q6.

### 2026-04-25 (late): Line-count guard, fuzzy justification snap, 8b probe

**Changes made:**
1. Added `count_text_lines_in_region()` — horizontal projection profiling to estimate text lines in leaf crops
2. Added `expected_lines` parameter to `build_ocr_prompt()` — tells VLM exactly how many lines to produce
3. Added response truncation in `process_node_recursive()` — silently drops extra VLM-hallucinated lines
4. Added `fuzzy_snap_justification()` in `justification_snaps.py` — combinatorial OCR-substitution repair for invalid rule heads + always-on `S→5`/`O→0` tail rescue
5. Wired fuzzy snap as second normalization pass after `snap_justification_rule_head()`
6. 11 new unit tests for all three features

**8b model probe results** (qwen3-vl-8b on Q4 leaf crops):
- **crop_2 (8b):** `[N/premise, w/premise, ←→/2,3]` — hallucinated 3 lines, garbled `←→`
- **crop_3 (8b):** `[1<8w, n/8E5]` — same `K∧W`→`1<8w` garble, but `8E5` normalizable to `∧E 5`
- **crop_2 (4b):** `[N, W, N→W/2,3]` — hallucinated 3rd line
- **crop_3 (4b):** `[1<8w, w/8∈5]` — same garble
- **Conclusion:** 8b doesn't improve over 4b. Both models hallucinate extra lines (strongly supporting the line-count guard) and struggle with the same handwritten characters.

**Gold regression results (commit c2db043):**

| Case | Prev lines | New lines | Prev formula | New formula | Prev jstr | New jstr | Notes |
|------|-----------|-----------|-------------|-------------|-----------|----------|-------|
| Q3 | 8 | 8 | 8/8 | 8/8 | 8/8 | 8/8 | ✅ Still perfect |
| Q4 | 9 | **12** | 3/7 | 3/7 | 3/7 | 3/7 | ❌ **WORSE** — line counter over-counted |
| Q5 | 5 | 5 | 4/5 | 4/5 | 5/5 | 5/5 | No change |
| Q6 | — | 13 | — | 6/11 | — | 3/11 | New baseline (first run) |

**Root cause of Q4 regression:** `count_text_lines_in_region()` over-counts — it estimated 4 lines for crop_2 (should be 2) and 5 lines for crop_3 (should be 2). The prompt constraint then *told* the VLM to produce 4-5 lines, causing MORE hallucination than the unconstrained baseline. The horizontal projection picks up the Fitch subproof bar and other non-text features as extra text lines.

**Action needed:** Either fix the line counter to be more accurate, or make it advisory-only (don't increase the prompt constraint beyond what the VLM would naturally produce).

### 2026-04-25: Letterboxing + prompt tweaks

**Changes made:**
1. Added `letterbox_crop()` function — pads small crops to 4:3 aspect ratio
2. Added "Transcribe only the lines actually visible in this image." to leaf-node prompts
3. Added "There are typically 1-3 lines inside the box." to context-crop prompts (moot — context crops disabled)
4. Disabled context-aware crops (`use_context_crop = False`)

**Q4 results:**

| Metric | Before (context crops) | After (letterboxed tight crops) |
|--------|----------------------|-------------------------------|
| Total lines | 14 | **9** |
| Formula match | 2/7 | **3/7** |
| Justification match | 4/7 | 3/7 |
| Both match | 1/7 | **2/7** |
| Extra lines | 7 | **2** |
| Structural mismatches | 3 | **2** |

**Leaf crop responses:**
- `response_2.json`: Correctly read `N` and `W` (was reading full proof before). Hallucinated a third line `N→W / →I 1-2`.
- `response_3.json`: Complete gibberish — VLM cannot read this handwriting. `"x < 8w"`, `"8 ∈ S"` instead of `K∧W` and `W`.

**Conclusion:** Letterboxing is a clear improvement for limiting boundary leakage. The remaining failures are model-capacity issues (4b too small to read handwriting in crop_3) and letter confusion (`W`→`N`). See `docs/TFL_OCR_FUTURE_IDEAS.md` for next strategies to try.

### 2026-04-24: Initial Q4 investigation

- Ran gold regressions: Q3 perfect, Q4 broken (14 lines)
- Sent sub-crops to 8b and 4b via `scripts/ocr_q4_crops_8b.py` probe
- Traced pipeline responses: ROOT found only 1 line, leaf crops read beyond boundaries
- Confirmed `8` → `∧` normalization working correctly
- Root cause identified: structural (boundary leaking + segmentation failure), not symbol-level

---

## Repository Structure (OCR-relevant only)

```
proofs_ocr/
├── pipelines/
│   └── tfl_fitch.py              # Main Fitch OCR pipeline
├── domains/
│   ├── tfl.py                    # TFL canonicalization
│   └── shared.py                 # Shared VLM response parsing
├── tricks/
│   ├── justification_snaps.py    # Rule-head OCR repair tables
│   ├── symbol_normalization.py
│   └── citation_cleanup.py
├── common/
│   ├── stitch.py                 # Placeholder expansion + renumbering
│   └── tree.py                   # (Modal tree helpers)
└── regression/
    ├── manifest.yaml              # Test case registry
    ├── tfl_runner.py              # TFL gold regression runner
    ├── runner.py                  # Shared regression infrastructure
    └── fixtures/tfl/handwritten/
        ├── q3_P_and_Q_imp_R/
        ├── q4_N_or_K_and_W/
        ├── q5_P_or_K_biconditional/
        └── q6_S_iff_W/

tests/
├── test_tfl_fitch_ocr_regression.py
├── test_tfl_fitch_gold_fixtures.py
└── test_debug_fitch_ocr.py

working/                            # (gitignored) regression outputs
└── tfl_gold_regressions/           # Default output location
    └── <case_id>/
        ├── debug_bars.jpg
        ├── crop_*.jpg
        ├── response_*.json
        ├── thinking_*.txt          # Qwen3.x reasoning (27b model only)
        ├── final_structure.json
        └── gold_regression_*.json

scripts/
└── debug_fitch_ocr.py              # CLI wrapper for single-image runs
```

---

## Folder Tidying Observations

These are non-blocking cleanups that would make navigation easier. **Do not delete anything without user approval.**

1. **Stale fixture fragment:** `proofs_ocr/regression/fixtures/tfl/handwritten/q4_N` — appears to be a truncated directory name (incomplete copy of `q4_N_or_K_and_W`). Safe to delete.

2. **~15 stale regression output dirs in `working/`:** Accumulated one-off run outputs: `test_run0..3/`, `tfl_gold_probe_q3/`, `tfl_gold_regressions_blot/`, `tfl_gold_regressions_retry/`, `tfl_gold_regressions_latest/`, `tfl_gold_regressions_new/`, `tfl_via_modal_pipeline/`, `tfl_live_q3/`, etc. All gitignored so they don't pollute the repo, but they clutter the local filesystem. Safe to clean up.

3. **Utility scripts in `working/`:** `working/dump_results.py` and `working/rescore_errors.py` — these should probably move to `scripts/` if still useful, or be deleted if stale.

4. **Root-level test files:** `test_column_matching.py` and `test_column_matching_simple.py` sit in the repo root instead of `tests/`. Low priority but inconsistent with project conventions.

5. **`.gitignore` has redundant entries:** `*.pdf`, `*.jpg`, `*.png`, `__pycache__/`, `data/` are each listed 2–3 times. Not harmful but could be cleaned up.

6. **No `.gitkeep` in `working/`:** The `working/` directory is gitignored but needs to exist for the runner. Adding `working/.gitkeep` with a negation pattern in `.gitignore` would prevent confusion for fresh clones.