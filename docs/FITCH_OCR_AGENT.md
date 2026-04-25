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

### Line-count estimation & hallucination guard

`count_text_lines_in_region()` uses horizontal projection profiles (same adaptive-threshold pipeline as bar detection) to count text lines in leaf crops. The count is used two ways:

1. **Prompt constraint:** The leaf prompt includes `"This image contains exactly N proof line(s). Output exactly N line objects."`
2. **Response truncation:** If the VLM returns more dict items than expected, extras are silently dropped.

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
7. **Plan next tweak** — Based on failure patterns

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

## Known Issues & Current Focus (as of 2026-04-25)

### Q4 small subproof problem
- **Case:** `tfl_handwritten_q4_N_or_K_and_W`
- **Expected:** 7 lines. `N∨(K∧W)` Pr, `N↔W` Pr, subproof [`N` Hyp, `W` →E 2,3], subproof [`K∧W` Hyp, `W` ∧E 5], `W` ∨E 1,3-4 5-6
- **Current best result:** 9 lines (was 14 before letterboxing)
- **Remaining issues:**
  - `N↔W` misread as `N↔N` (letter confusion — VLM reads W as N)
  - Crop 3 (second subproof) is unreadable by qwen3-vl-4b — returns gibberish like `"x < 8w"`, `"8 ∈ S"`
  - VLM adds hallucinated conclusion lines to 2-line subproofs (e.g. `N→W / →I 1-2`)
- **Hard floor reached with 4b model:** One of the two leaf crops contains handwriting the model literally cannot decipher

### Context-aware crops: DISABLED
- **What was tried:** Sending the parent's full unmasked image with a green highlight rectangle, telling the VLM to OCR only inside the green box.
- **Result:** The VLM (qwen3-vl-4b) completely ignored the green box and read the entire parent image, producing 5-6 lines per leaf instead of 2. This doubled the total line count (7→14).
- **Decision:** Disabled in favor of letterboxed tight crops. Code is preserved (`build_context_crop()` still exists) with `use_context_crop = False` in case a future model respects the instruction.

### Letterboxing: ACTIVE
- **What it does:** `letterbox_crop()` pads small leaf crops to 4:3 aspect ratio with white space (min height 250px)
- **Result:** Reduced Q4 from 14 lines → 9 lines. The VLM can no longer read beyond crop boundaries because the tight crop physically limits what's visible.
- **Trade-off:** Without parent context, the VLM sometimes can't interpret ambiguous handwriting (e.g. crop_3 returns gibberish).

### Symbol normalization status
- `8` → `∧` normalization is **working correctly** in `proofs_ocr/domains/tfl.py` (not in `symbol_normalization.py`). Applied at comparison time via `_EIGHT_BEFORE_EI_RE` and `_EIGHT_BETWEEN_CAPS_RE`.
- `∈` → `E` normalization also exists and works (handles `8 ∈ 5` → `8 E 5` → `∧E 5`).

### Other active notes
- **Scaffold removal disabled** (2026-04-24): `filter_scaffold_bars()` scaffold-removal logic deliberately disabled because handwritten Fitch proofs have a real outer subproof bar that spans most of the image height. Only edge-artifact cleanup remains active.
- **Line-number blotting** on ROOT node: The pipeline whites out everything to the left of the leftmost bar, preventing the VLM from reading line numbers as formula content.

---

## Experiment History

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

**Gold regression results:*** (pending — run `python -m proofs_ocr.regression.tfl_runner run-gold`)

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