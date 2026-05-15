# TFL Fitch OCR Pipeline — Agent Orientation

> **Audience:** A fresh Cline conversation starting an OCR iteration session.
> Read this file first to get productive immediately on the Fitch OCR loop
> without wading through unrelated grading/viewer code.

## How To Use This Doc

This file is currently both the TFL/Fitch OCR operating guide and the running
experiment log. Treat the early sections as current guidance:

- `Quick Start`
- `Key Files`
- current status notes before `Suggested next-conversation prompt`
- `Pipeline Architecture`
- test/debug command sections

Treat later dated experiment history as historical evidence. It is useful for
understanding why a choice was made, but it should not override the current
status notes near the top. Future cleanup should split this file into a compact
operating guide plus an archived experiment log; until then, keep new current
facts near the top and avoid adding another one-off OCR markdown file.

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
| `crop_N_fullwidth.jpg` | Full-width crop sent to 27b model (preserves line-number gutter) |
| `response_N.json` | Raw VLM JSON response for node N |
| `thinking_N.txt` | Qwen3.x reasoning/thinking text for node N (27b model only) |
| `final_structure.json` | Stitched + normalized proof structure |
| `gold_regression_result.json` | PASS/FAIL + run metadata |
| `gold_regression_metrics.json` | Per-line formula/justification/structure scores |
| `gold_regression_diff.txt` | Unified diff (expected vs actual) on failures |

Default output location: `working/tfl_gold_regressions/<case_id>/`

---

## Private Real-Scan Evaluation Plan

Goal: build a FERPA-safe local benchmark loop for real student TFL/Fitch proof
scans. The benchmark should answer whether the OCR pipeline generalizes beyond
fake/instructor-authored samples, while keeping raw scans, crops, names,
rosters, LMS ids, local paths, and handwriting images outside tracked fixtures
and outside agent conversations.

### Overall design target

1. Use the existing strip-view segmentation and Fitch debug/correction pages to
   create proof crops and corrected `final_structure.json` files.
2. Seed an external private eval corpus with anonymous case ids, copied proof
   crops, and human-checked gold structures.
3. Split private cases into `discovery`, `dev_regression`, and `blind_holdout`
   so OCR changes can be debugged without overfitting the holdout set.
4. Run the normal TFL/Fitch OCR pipeline against that external corpus.
5. Write detailed local reports for the instructor and a redacted
   `shareable_summary.txt/json` that contains only aggregate metrics, anonymous
   case ids, split labels, and failure categories.
6. Later: add a local-model case narrator that describes geometry/OCR failures
   in words without exposing handwriting images or PII to cloud tools.

### Implemented so far

- `proofs_ocr/regression/private_tfl_runner.py`
  - loads an external private manifest
  - enforces an outside-repo path guard by default
  - runs private gold regressions through `run_fitch_ocr_pipeline()`
  - compares against human-checked gold using the same TFL canonicalization as
    the tracked gold runner
  - writes local detailed artifacts plus redacted `shareable_summary.txt/json`
- `proofs_ocr/regression/private_tfl_exporter.py`
  - seeds private manifest cases from existing strip-page Fitch OCR review state
  - exports accepted/corrected debug folders into an external eval root
  - copies the proof crop and saved `final_structure.json`
  - appends anonymous case ids such as `exam1_spring2026_q6_case_0001`
- Tests:
  - `tests/test_tfl_private_runner.py`
  - `tests/test_tfl_private_exporter.py`
- Schema/operation docs:
  - `docs/SCHEMA.md`, section `9.1`
  - `docs/AI_WORKFLOW.md`, private runner/exporter commands

### Instructor workflow

1. Keep real data outside the repo, e.g.
   `C:\grading-data\live` for active grading and
   `C:\grading-data\ocr-eval\exam1` for the private benchmark.
2. In the strip view, tag the problem as `fitch_proof`, run TFL/Fitch OCR, open
   the debug page for a proof, and manually correct `final_structure.json`.
3. Only click/record `accepted` after the crop and JSON are good enough to serve
   as gold. Gold should transcribe what is visible; do not infer clipped or
   unreadable citations just to make a proof verify.
4. Seed accepted cases into the private corpus:

```powershell
python -m proofs_ocr.regression.private_tfl_exporter seed-from-strip `
  --eval-root C:\grading-data\ocr-eval\exam1 `
  --assignment-id exam1_spring2026 `
  --problem-id Q6 `
  --grades-dir C:\grading-data\live\grades `
  --debug-fitch-dir C:\grading-data\live\debug_fitch `
  --split dev_regression
```

5. Check the external `manifest.yaml`, `crops/<problem>/`, and
   `gold/<problem>/` folders. Move especially useful cases between
   `discovery`, `dev_regression`, and `blind_holdout` by editing only the
   manifest split label.
6. Run private regressions locally:

```powershell
python -m proofs_ocr.regression.private_tfl_runner `
  --manifest C:\grading-data\ocr-eval\exam1\manifest.yaml `
  run-gold --split dev_regression
```

7. Share only `shareable_summary.txt/json` back with coding agents, unless the
   underlying case is fake or fully sanitized.

### Not built yet

- A one-click UI action in strip view to export accepted OCR directly into a
  private eval corpus.
- A human review queue that distinguishes `accepted_for_grading_view` from
  `gold_human_checked`; for now, treat `accepted` as the export gate.
- Local-model narrative reports for private failures.
- A holdout discipline helper that warns before exporting `blind_holdout` cases
  from material you have already used for debugging.

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
| `proofs_ocr/regression/private_tfl_runner.py` | **Private real-scan runner.** Runs external, anonymous, FERPA-local TFL/Fitch cases and writes redacted shareable summaries. |
| `proofs_ocr/regression/private_tfl_exporter.py` | **Private corpus seeder.** Exports accepted strip-page Fitch OCR debug artifacts into an external private eval manifest. |
| `proofs_ocr/regression/runner.py` | **Shared regression helpers.** `load_manifest()`, `select_cases()`, `GoldRegressionResult` dataclass. |

### Test fixtures (TFL handwritten)

Each fixture directory contains: `image.png`, `expected_structure.json`, optional `notes.txt` and `semantic_notes.yaml`.

| Fixture ID | Description | Key challenge |
|---|---|---|
| `tfl_handwritten_q3_P_and_Q_imp_R` | `(P∧Q)→R, P→Q ⊢ (P∧S)→R`. Single subproof with →I. | Baseline case, simplest subproof |
| `tfl_handwritten_q4_N_or_K_and_W` | `N∨(K∧W), N↔W ⊢ W`. ∨E with two subproofs. | **Small subproofs**, ∨E structure, ↔F→↔E normalization |
| `tfl_handwritten_q5_P_or_K_biconditional` | No subproofs, intentional error (self-citation on line 4). | Flat proof, error detection |
| `tfl_handwritten_q6_S_iff_W` | ↔I with two subproofs, intentional errors on lines 6 and 10. | Nested subproofs, biconditional intro |
| `tfl_fake_implication_intro` | Fake instructor-authored scan: `A→(B→C) ⊢ (A∧B)→C`. | Clean one-subproof case from phone crop |
| `tfl_fake_neg_intro_error` | Fake instructor-authored scan: `(A∧B)→C, ¬C, A ⊢ ¬B`. | Short phone crop with a long outer proof spine and one nested `B...⊥` subproof |
| `tfl_fake_nested_disjunction_scratch_completed` | Completed fake handwritten scan for the same `A→(B∨C), A→¬B ⊢ A→C` target. | Harder nested ∨E phone crop with curated gold answers |

The fake-data fixtures are deliberately not real student submissions. They were
created as phone scans of instructor-authored messy proofs and cropped to the
proof region, matching the intended human-corrected crop boundary for the
grading workflow. An earlier incomplete scratch-only nested-disjunction crop was
moved out of the active tracked fixtures to
`working/tfl_candidate_scans/deferred_fake_nested_disjunction_scratch/`.

### Fake phone-crop slant repair probe (2026-05-12)

The first fake fixture exposed a new input-class problem: phone-cropped lined
paper differs sharply from the older clean PNG fixtures.

Observed failure modes:

- high-resolution phone crops caused severe over-detection of vertical bars
  from handwriting strokes
- notebook ruling and shadows confused row/line detection
- slightly slanted Fitch bars were detected too low, so assumption/opening
  rows were placed outside their subproof
- repaired slanted bars could then extend too far downward and swallow a
  post-subproof conclusion line

Current opt-in experiment:

```powershell
$env:TFL_FITCH_SLANTED_BAR_REPAIR='1'
```

When enabled, `detect_vertical_bars()` uses a slant-tolerant repair path:

1. wider horizontal smear before vertical morphology
2. upward extension to nearby opener/assumption rows
3. nested-bar trimming before large text-row gaps

This flag is **off by default**. It is meant for normalized phone crops on
lined paper, not as a replacement for the standard detector. A synthetic unit
test covers opener-row extension, but this path still needs live-regression
work before becoming a default.

Phone-crop preprocessing now defaults to:

```powershell
$env:TFL_FITCH_SLANTED_PHONE_PREPROCESS_MODE='auto'
```

Modes:

- `global` / `legacy`: original downscale + global threshold + ruling cleanup
- `shadow`: background-normalized foreground extraction + ruling cleanup
- `auto`: use `global` unless the left/mid-lower crop region has a high dark
  density that `shadow` materially reduces

The auto switch was added after the long/scratch fake phone crops showed large
peppery black regions from shadows after global thresholding. Keep inspecting
`normalized.png`/`debug_bars.jpg`; shadow cleanup fixes those blobs but can
shorten or split scope bars on cleaner crops if forced manually.

Best probe result on the easiest fake proof:

- geometry became essentially correct after downscale + rough ruled-line
  cleanup + slanted-bar repair + gap trim
- with the correct 5-line child count forced, the inner subproof OCR produced
  the intended formulas and justifications
- remaining miss: the post-subproof conclusion justification was read as
  `2-6`/blank instead of `→I 2-6`

Follow-up implemented after `62d4c68`:

- chose the 5-line child-count repair as the next lowest-risk step
- when `TFL_FITCH_SLANTED_BAR_REPAIR=1`, leaf line counting now uses a
  ruled-paper-aware text mask that removes horizontal ruling and vertical
  scope bars, then drops clipped top-edge carryover from the previous row
- the fake implication child crop now estimates 5 lines instead of the
  default detector's 7-line count; this keeps the separated-list prompt on
  the intended shape
- the final `→I 2-6` miss remains a visual reread/forced-choice candidate;
  do not silently infer that rule head from checker context alone

Slow canary result:

- command attempted:
  `python -m proofs_ocr.regression.tfl_runner run-gold --case tfl_fake_implication_intro --output-root working/tfl_fake_implication_slanted_linecount`
  with `TFL_FITCH_SLANTED_BAR_REPAIR=1`
- the run was stopped after it spent too long OCRing over-detected crop nodes
- `debug_bars.jpg` showed about 50 detected bar-like boxes, mostly handwriting
  strokes and rule/letter artifacts, so this did not reach the clean two-bar
  geometry needed for the new 5-line child-count repair to matter end-to-end
- partial OCR evidence still matched the known direction: a large whole-proof
  crop could read most lines, but the geometry split the proof into many
  irrelevant child crops
- next live work should add a cheap geometry-only canary before VLM calls:
  detect/filter bars, write `debug_bars.jpg`, report bar count/tree shape and
  candidate child line counts, then only run VLM when geometry is plausible
- reason: small leaf crops can trigger the slower larger-model routing, so
  over-detected bars can multiply expensive LLM calls before the proof-level
  result has any chance of improving

Follow-up geometry port:

- the slow run was broken because it used the raw tracked fixture directly,
  while the good probe used a normalized phone crop (`no_hlines...scale...`):
  downscaled, thresholded, and rough horizontal ruling removed
- `TFL_FITCH_SLANTED_BAR_REPAIR=1` now also normalizes high-resolution phone
  crops before geometry unless `TFL_FITCH_SLANTED_PHONE_PREPROCESS=0`
- a pre-OCR guard writes `debug_bars.jpg` and aborts if slanted mode still
  detects more than `TFL_FITCH_SLANTED_MAX_BARS_BEFORE_OCR` bars
- geometry-only fake implication preflight now recovers two bars, one nested
  child, and a 5-line child crop before any VLM calls
- bounded live rerun after geometry preflight:
  `python -m proofs_ocr.regression.tfl_runner run-gold --case tfl_fake_implication_intro --output-root working/tfl_fake_implication_slanted_preprocessed`
  with `TFL_FITCH_SLANTED_BAR_REPAIR=1`
- result: completed quickly with 2 bars, 7 lines, 1 premise; structure 7/7,
  formulas 7/7, justifications 5/7
- remaining misses: child line 5 justification read as `∧I 3,4` instead of
  `∧E 2`, and final line's visible `→I 2-6` was still only read as `2-`/blank
  before defaulting to `Hyp`

Important caution for grading: do not silently infer a missing rule head unless
there is enough visual/textual evidence. This may be a good candidate for a
review flag or forced-choice reread rather than an automatic proof repair.

Recommended next step:

Investigate what the local model is seeing for the two remaining justification
misses before moving to the next fixture or trying to make this path automatic.
The current fake implication run has good geometry and perfect formula/structure
metrics, so the remaining uncertainty is now crop/prompt/justification-reading
quality, not tree construction. Use saved artifacts from
`working/tfl_fake_implication_slanted_preprocessed/tfl_fake_implication_intro/`
or rerun the single case after a targeted reread/probe change.

Targeted probe result after `1a52a25`:

- visual inspection of `crop_2_4b_fullwidth.jpg` shows child line 5's printed
  justification as an `8E 2`-shaped handwritten `∧E 2`; this is visible in the
  saved crop and was not a geometry miss
- whole-child rereads were unstable: one ordered-justification prompt returned
  `→E 2` for that row, while a separated-list prompt omitted one justification
- a tight line-5 justification crop returned `8 ∈ Z`; normalization now repairs
  this kind of already-read glyph confusion to `∧E 2` (`59900ef`), but the main
  pipeline does not yet perform that row-level reread
- visual inspection of `crop_1.jpg` shows the final line's printed justification
  as only `→I 2-`; the terminal `6` is not visible in the saved parent crop, so
  `→I 2-6` would require inference from proof structure rather than direct OCR
- a tight final-line reread saw the rule head in one crop (`→I 2-`) but still
  could not recover the missing endpoint; keep this as a review/forced-choice
  candidate rather than a silent repair

Wide fake-crop follow-up (`93cd800` and live canaries):

- the original fake fixture was a same-size crop from `IMG_2146` with the right
  edge too far left; the terminal `6` in the final `→I 2-6` was outside the
  tracked image
- the fixture is now a wider crop that keeps the left proof context and includes
  that terminal citation endpoint; after preprocessing it normalizes to a
  smaller, wider image (`714x573`)
- line counting initially regressed because the phone-crop counter discarded the
  real top child row as clipped carryover; the carryover drop now keys on a
  large gap before the next row, and geometry-only checks recover 2 bars, one
  nested child, and 5 child rows
- run old-fixture canaries with `LMSTUDIO_MODEL_27B=''` unless explicitly testing
  the 27b route. Accidental 27b routing can make a quick Q4 canary spend minutes
  in LM Studio generation even when model-template thinking is reported as off
- no-27b live canary on Q4 completed quickly and kept the expected 7-line,
  structurally correct shape, so the wider fake-crop geometry change did not
  show an obvious old-fixture regression
- no-27b live canary on the wider fake implication crop exposed the next
  bottleneck: the child crop fallback can now read line 5 as normalizable
  `8E 2`, and the parent crop sees the final `2-6`, but the parent OCR omits
  `<SUBPROOF_ID_2>` and still misses the final `→I` rule head
- compacting the red subproof token and placing it nearer the scope bar made
  the parent OCR preserve `<SUBPROOF_ID_2>` again; adding a prompt hint for
  handwritten implication-introduction heads recovered the final `→I 2-6`
- fake implication no-27b live canary after that change passed: 2 bars, 7
  lines, 7/7 formulas, 7/7 justifications, 0 structural mismatches
- old Q4 no-27b canary still completed quickly with 7 lines and 0 structural
  mismatches; remaining formula/justification misses matched the known
  handwriting/OCR quality pattern rather than a structural regression

Next fake crop: negation-introduction phone crop (`IMG_2148`):

- the original tight crop clipped the right-side justifications; a wider,
  natural-aspect crop was generated from `working/tfl_candidate_scans/IMG_2148.png`
  and tracked as `tfl_fake_neg_intro_error`
- avoid including the far-right page edge because it becomes a false vertical
  bar after phone-crop preprocessing
- gold treats the long left vertical line from 1-8 as the ordinary outer proof
  spine, not as a nested semantic subproof; the stored nested block is only
  lines 4-7 (`B`, `A∧B`, `C`, `⊥`)
- slanted geometry needed a relaxed fallback vertical opening because the
  nested scope line is fragmented after ruled-paper cleanup; preflight now
  recovers 2 bars, one child node, and a 4-line child crop
- fixed parent masking coordinate translation so child whiteout starts at the
  true child top even when parent-crop padding is clipped at image edge
- top-anchored red subproof tokens improved parent OCR ordering on this crop
- latest no-27b canary was structurally aligned after correcting gold, but the
  model still misread the final contradiction row in the inner child crop as
  justification text (`¬E 2,6`) rather than formula `⊥` with justification
  `¬E 2,6`; treat this as the next OCR-quality/reread issue, not geometry

Shadow-preprocess probe on remaining fake candidate crops:

- current/global preprocessing created large peppery black artifacts in the
  shadowed lower-left regions of the long and scratch crops
- `auto` keeps the known-good implication and neg-intro geometry on the global
  path, but switches the long/scratch crops to the shadow-normalized path
- the shadow-normalized images are visually cleaner, so judge future bar/line
  failures from those cleaned preflight artifacts rather than from the old
  global-threshold blobs

Completed scratch crop follow-up after `eefa14b`:

- refreshed preflight artifacts are under
  `working/tfl_candidate_scans/geometry_preflight_auto_after_linecount_fix/`
- `fake_long_disjunction_elim` is still reserve-only: it detects 2 bars but
  the line-count signal is implausible (`ROOT`/full-width count 44)
- `fake_nested_disjunction_scratch_completed` now has plausible geometry after
  making slanted gap-splitting use the same wider smear as slanted detection:
  4 bars, tree `ROOT -> 1 -> 2 -> {3,4}`, and child crop counts 4 and 3
- the completed scratch crop has been promoted as
  `tfl_fake_nested_disjunction_scratch_completed` with sanitized gold and
  semantic notes for `X 7` and the intentional `∧E 9` citation error
- no-27b canary still fails structurally:
  `working/tfl_fake_nested_disjunction_scratch_completed_no27b_final/`
  accepted the two inner leaf OCR calls at 4 and 3 lines, but parent OCR
  omitted `<SUBPROOF_ID_2>` and collapsed the result to the two premises plus
  final `A→C`; treat this as a parent-token/prompt issue, not a bar-geometry
  issue

Follow-up after `1bce994`:

- parent-token/stitch collapse is fixed for
  `tfl_fake_nested_disjunction_scratch_completed` in opt-in slanted mode with
  no 27b route
- the pipeline now top-anchors red child tokens, restores missing immediate
  child placeholders from the detected tree, trims top carryover from nested
  leaf full-width OCR crops, adjusts leaf masking so parent rows above child
  openers remain visible, and drops the narrow trailing `E` citation fragment
  caused by bottom bleed from the next parent line
- latest no-27b canary:
  `working/tfl_fake_nested_disjunction_scratch_completed_no27b_bleed_filter/`
  reports 4 bars, 13 actual lines, 0 missing, 0 extra, 0 structural
  mismatches, and 13/13 formulas
- remaining misses are OCR-quality issues in justifications: missing rule
  heads on `→E`/`∨E`, `X 7` read as a cross glyph, two off-by-one conjunction
  citations, and final `→I 3-12` still read as `→I 3-2`

Verifier-guided small-model reread scaffold:

- `proofs_ocr/pipelines/tfl_line_reread.py` now contains an opt-in probe path
  for replacing broad/large-model retries with targeted single-line small-model
  rereads
- `process_node_recursive()` tags stitched line dictionaries with transient OCR
  provenance; `final_structure.json` is still scrubbed before writing
- set `TFL_FITCH_VERIFIER_REREAD=1` to run the verifier after stitching and
  write `verifier_line_reread_candidates.json` plus `verifier_line_rereads/`
  row crops for suspicious leaf-derived lines
- the probe does not auto-apply candidate rereads; verifier errors can be real
  student mistakes, so use the output to evaluate visual reread quality before
  wiring any guarded replacement policy
- current limitation: parent-derived lines are identified as suspicious but
  skipped because robust final-line-to-parent-row cropping is not implemented
  yet; the first useful targets are leaf justifications such as off-by-one
  citation rows or `X 7`-style OCR misses

Strip-view OCR review integration:

- `app/proof_ocr_review.py` provides the first UI integration slice for
  handwritten TFL/Fitch OCR inside exam strip grading.
- `POST /problem/{problem_id}/fitch_ocr/start` crops all selected proof strips
  for the question into `data/debug_fitch/<slug>/`, writes queue state under
  `data/grades/proof_ocr/tfl_fitch_{problem_id}.json`, and submits OCR jobs to
  a process-local single-worker queue.
- `GET /problem/{problem_id}/fitch_ocr/status` returns per-strip status for
  browser polling. `POST /problem/{problem_id}/fitch_ocr/{unit_id}/accept`
  records human acceptance in the review state.
- The strip page renders the states `queued`, `processing`, `needs_review`,
  `verifier_issue`, `accepted`, and `ocr_failed`. It also links each finished
  item back to `/debug/fitch?scan=<slug>` for full artifact inspection/editing.
- When `verifier_line_reread_summary.json` and
  `verifier_line_reread_candidates.json` are present, the review card surfaces
  original OCR, candidate reread, `candidate_class`, reason text, and row-crop
  thumbnails when the crop path is under `data/debug_fitch/`.
- The UI deliberately does not auto-apply reread candidates or assign scores.
  Verifier errors can be real student proof mistakes, so the first integration
  is review/confirmation only.
- Test coverage for this slice lives in
  `tests/test_problem_strips_routes.py`; OCR pipeline and reread helper coverage
  remains in `tests/test_debug_fitch_ocr.py`.

Suggested next-conversation prompt:

> Continue the TFL/Fitch OCR fake phone-crop work after commit `5fed00c`.
> Read `docs/FITCH_OCR_AGENT.md`, especially "Fake phone-crop slant repair
> probe (2026-05-12)". Current baseline: `tfl_fake_implication_intro` uses a
> wider crop from `IMG_2146`, opt-in slanted mode recovers 2 bars / 7 lines,
> and the no-27b live canary passes with 7/7 formulas and 7/7 justifications.
> Old Q4 no-27b canary remains structurally correct. For the next step, pick
> the next easiest instructor-authored phone crop under
> `working/tfl_candidate_scans/crops/` (likely a short/simple proof before the
> scratch-heavy nested disjunction case), make sure the crop is human-like and
> not forced into a square, then run geometry-only/preflight checks first:
> bar count, tree shape, and child line counts. Keep `LMSTUDIO_MODEL_27B=''`
> for quick canaries unless explicitly testing the 27b route. If geometry is
> plausible, add or refresh the sanitized gold fixture/manifest entry and run
> one no-27b live case. Keep FERPA-safe boundaries; these fake scans are
> instructor-authored, but real student handwriting should remain external.

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
| `scripts/probe_tfl_row_reread.py` | Side experiment: split a saved full-width crop into generous row bands and reread each row's justification with the default VLM. |
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

Default slow-test habit: after implementing an OCR fix/change and passing fast
tests, run the smallest relevant live gold case unless the user says not to.
Prefer `python -m proofs_ocr.regression.tfl_runner run-gold --case <case_id>
--output-root working/<descriptive_run_name>` before spending time on the full
suite. Report whether the slow run improved, held steady, or regressed.

Before invoking the live VLM on a new crop class, do a geometry-only preflight:
write/inspect `debug_bars.jpg`, confirm the subproof tree shape, and check leaf
line counts. This matters because small detected leaf crops can route to the
super-slow larger-model path, so over-detection can create many expensive calls.

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

## Known Issues & Current Focus (as of 2026-04-26)

### Current priority: Remaining formula-only OCR errors

**Status (2026-04-26):** Q6 citation handling is now solved in the current gold set. The 27b full-width routing is **wired and working** for ≤3-line leaf subproofs. Nested leaf subproofs that stay on 4b use full-width horizontal slices and separated-list OCR by default. This preserves the line-number gutter and uses Fitch structure to avoid cross-column row-association drift.

**Q6 failure analysis:**
- **11/11 formulas correct** — the VLM reads every formula perfectly
- Baseline tight/letterboxed 4b crops had **6/11 justifications correct** because the model often used crop-local citations (`∧I 2,3`) instead of absolute proof citations (`∧I 4,5`)
- 4b full-width nested leaf crops improve Q6 to **9/11 justifications correct**
- Remaining Q6 errors are row-association errors in the second 4-line subproof: L8 receives L9's `→E 2,8`, and L9 receives L10's `∧I 8,9`

**Next experiment:** Use `scripts/probe_tfl_row_reread.py` on `crop_3_4b_fullwidth.jpg` to split the problematic full-width subproof crop into generous full-width row bands and reread one row at a time. This keeps scale/context while reducing cross-row citation drift.

**Row reread probe result (2026-04-26):**
- `detect_text_row_bands()` found only 3 bands for the 4-line Q6 node 3 crop, merging the top rows. The probe script therefore supports `--expected-lines 4` to create overlapping equal-height row windows.
- On Q6 node 3, default 4b row windows fixed line 9's justification (`→E 2,8`) and recovered line 10's citation tail (`8,9`), but line 8 still misread `∧E 7` as an arrow-style rule and line 10 dropped the rule head.
- Asking 4b to read only the derived-line justifications as an ordered list from the whole full-width crop worked on Q6 node 3: `["∧E 7", "→E 2,8", "∧I 8,9"]` in ~2s. The prompt explicitly told the model that the first/top line is an assumption with no printed justification.
- Asking 4b to return separated lists also worked on both Q6 4-line nested leaf crops:
  - Node 2: `{"formulas": ["S∧A", "S", "W", "W∧A"], "justifications": ["∧E 3", "→E 1,4", "∧I 4,5"]}`
  - Node 3: `{"formulas": ["W∧A", "W", "S", "S∧A"], "justifications": ["∧E 7", "→E 2,8", "∧I 8,9"]}`
- Conclusion: row isolation helps row association, but single-row crops are not stable enough for automatic correction yet. The strongest path is likely "separated-list nested leaf OCR": when a nested leaf has the expected line count, ask for formulas and derived-line justifications as separate lists, require `len(formulas) == len(justifications) + 1`, then stitch with a blank opening assumption.

### Separated-list nested leaf OCR: WIRED (2026-04-26)

**What it does:**
1. Applies only to nested leaf subproofs on the default 4b path.
2. Uses the full-width crop and asks the model for `{"formulas": [...], "justifications": [...]}`.
3. Requires `len(formulas) == expected_lines` and `len(justifications) == len(formulas) - 1`.
4. Stitches the first formula with blank justification, then assigns the nth justification to the n+1th formula.
5. Saves the raw separated response as `response_<node>_separated_raw.json`.
6. Falls back to the normal row-object OCR prompt if the separated response has the wrong shape.

**Gold regression result (2026-04-26):**

| Case | Formula | Justification | Both | Status |
|------|---------|---------------|------|--------|
| Q3 | 8/8 | 8/8 | 8/8 | PASS |
| Q4 | 6/7 | 7/7 | 6/7 | FAIL: existing formula error `N↔W`→`N↔N` |
| Q5 | 4/5 | 5/5 | 4/5 | FAIL: existing formula error `↔`→`→` on line 1 |
| Q6 | 11/11 | 11/11 | 11/11 | PASS |

**Aggregate: 29/31 formulas, 31/31 justifications, 29/31 both, 0 structural mismatches.**

### 4b full-width nested leaf routing: WIRED (2026-04-26)

**What it does:**
1. For nested leaf subproofs not routed to 27b, the default 4b path now uses `crop_full_width()` instead of the tight/letterboxed crop.
2. It preserves the line-number gutter while keeping the crop vertically restricted to the target subproof.
3. It deliberately skips outer/flat proof leaves, because applying full-width to Q5's flat proof regressed the biconditional on line 1 from `↔` to `→`.
4. `TFL_FITCH_4B_FULLWIDTH_MAX_LINES` controls the maximum detected line count for this path; default is `99`. Set it to `0` to disable.

**Experiment results (2026-04-26):**

| Mode | Aggregate formula | Aggregate justification | Notes |
|------|-------------------|-------------------------|-------|
| 27b ≤3 only | 29/31 | 24/31 | Q6 stayed on tight 4b crops and had local citation errors |
| 4b full-width for Q6 4-line leaves | 11/11 Q6 | 9/11 Q6 | First Q6 subproof became perfect; second still had row association drift |
| 4b full-width for all non-27b leaves | 29/31 | 28/31 | Q3 perfect and Q6 improved, but Q5 flat proof regressed one biconditional rule |
| 4b full-width for nested non-27b leaves only | 29/31 | 29/31 | Default path: keeps Q3/Q6 gains and avoids the Q5 flat-proof justification regression |

### 27b full-width routing: WIRED AND VALIDATED (2026-04-26)

**What was done:**
1. Added `crop_full_width()` — generates full-width horizontal slices preserving the line-number gutter
2. Added `build_27b_fullwidth_prompt()` — the proven manual prompt from probe experiments
3. Added `prompt_override` parameter to `call_vlm()` for bypassing the auto-built prompt
4. Wired routing in `process_node_recursive()`: when a leaf has ≤3 detected lines and 27b is available, the pipeline generates a full-width crop and uses the manual prompt
5. Set `SMALL_LEAF_MODEL_ROUTING_THRESHOLD = 3`

**Gold regression results (2026-04-26):**

| Case | Lines | Formula | Justification | Both | Notes |
|------|-------|---------|---------------|------|-------|
| Q3 | 8/8 | **8/8** | 6/8 | 6/8 | 2 citation errors in 5-line subproof (stayed on 4b) |
| Q4 | 7/7 | 6/7 | **7/7** | 6/7 | **Subproofs perfect via 27b full-width**; 1 error in ROOT (N↔N vs N↔W) |
| Q5 | 5/5 | 4/5 | **5/5** | 4/5 | ↔→→ misread on L1 (no subproofs, no routing change) |
| Q6 | 11/11 | **11/11** | 6/11 | 6/11 | Perfect formulas! 5 citation errors in 4-line subproofs (stayed on 4b) |

**Aggregate: 29/31 formulas (94%), 24/31 justifications (77%), 0 structural mismatches**

**Key evidence the 27b routing works:**
- Q4 nodes 2 and 3 (2 lines each) were routed to 27b with full-width crops
- Both returned perfect results after normalization
- Log output confirmed: `[ModelRoute] Node 2: routing to qwen/qwen3.6-27b (<=3 lines)`

### Semantic citation repair: DEFERRED

Post-OCR justification validation (checking whether `∧E 3` is consistent with line 3's formula, and repairing citations that reference wrong lines) is a powerful technique but **deferred** because this is a grading system — it is critical to read and recognize wrong student answers as wrong, not silently repair them. A future use could be **flagging** suspect citations for re-examination with a different model, but auto-repair is off the table for now.

### 27b local model status: usable, not fully integrated

**Current state:** The 27b model (`qwen/qwen3.6-27b`) now works through the LM Studio OpenAI-compatible API and gives high-quality answers on the hard Q4 crop_3 case. The apparent "API timeout" blocker was a mix of two issues:

1. LM Studio thinking was enabled, causing very long `reasoning_content` generations.
2. During one retest, the LM Studio server had been stopped while unloading/reloading the model, so the client got `WinError 10061` connection-refused errors. That was not a model failure.

**Server sanity check:** Before interpreting any probe failure, verify the LM Studio server is actually listening:

```powershell
Test-NetConnection 127.0.0.1 -Port 1234
```

`TcpTestSucceeded : True` means the API server is reachable. If LM Studio is using a different port, set `LMSTUDIO_BASE_URL`, e.g. `http://127.0.0.1:PORT/v1`.

**Best current 27b setup:**
- In LM Studio, load `qwen/qwen3.6-27b`, open the model/runtime settings for
  that loaded model, and turn off thinking/reasoning. If the setting was changed
  while the model was already loaded, unload and reload the model before probing.
- Keep the LM Studio local server running after reload. In current LM Studio
  builds this is typically under the Developer/API server area; confirm the
  OpenAI-compatible endpoint is listening on `http://127.0.0.1:1234/v1` or set
  `LMSTUDIO_BASE_URL` to the port shown in the app.
- Use the GUI-like single-user message shape for direct crop probes.
- Send the API-level no-thinking controls too: append `/no_think`, send
  `chat_template_kwargs.enable_thinking=false`, and keep `max_tokens` low enough
  to prevent long generations (`512` is enough for the successful Q4 crop_3
  response).
- Verify the setup in saved metadata: `thinking_chars` should be `0`, and
  `usage.completion_tokens_details.reasoning_tokens` should be `0` or absent.

**Probe command for Q4 crop_3:**

```bash
python -u scripts/probe_27b_crop.py --manual-prompt --no-think --enable-thinking-false --max-tokens 512 --timeout 900
```

**Prompt that works well manually and through API:**

```
Role: You are a logic quiz transcriber performing OCR on subproof chunks cropped out from the middle of handwritten Fitch-style natural deduction proofs.
Context: You will be given cropped images of specific subproofs (beginning with an assumption with no justification above a horizontal line, and deriving various consequences from that line and earlier lines). So don't worry that those are missing, and remember that the top line in the crop you are seeing might not be line 1 in the proof.
Task: Extract the logical formulas and their justifications.
Handwriting Translation Guide (CRITICAL):
Students have sloppy handwriting. Use the following known distortions to correct your visual reading:
A sloppy K frequently looks like 1 < or a vertical line next to a <.
A sloppy ampersand & (conjunction) frequently looks like an 8.
```

**Successful answer shape:** The 27b crop_3 response regularizes to the correct proof lines:

```json
[{"line": 5, "wffstr": "K & W", "jstr": ""}, {"line": 6, "wffstr": "W", "jstr": "& E 5"}]
```

After normalization (`&` -> `∧`, blank assumption justification -> `Hyp`), this is the desired `K∧W`/Hyp and `W`/`∧E 5`.

**Diagnostic probe script:** `scripts/probe_27b_crop.py` has flags for testing the important call-shape and generation controls:

```bash
python -u scripts/probe_27b_crop.py --manual-prompt
python -u scripts/probe_27b_crop.py --manual-prompt --no-think
python -u scripts/probe_27b_crop.py --manual-prompt --max-tokens 512
python -u scripts/probe_27b_crop.py --manual-prompt --enable-thinking-false
python -u scripts/probe_27b_crop.py --single-user-message
python -u scripts/probe_27b_crop.py --no-max-tokens
```

**27b pipeline controls:** The pipeline can now test 27b call behavior via environment variables:

```powershell
$env:TFL_FITCH_27B_SINGLE_USER_MESSAGE='1'
$env:TFL_FITCH_27B_NO_THINK='1'
$env:TFL_FITCH_27B_ENABLE_THINKING_FALSE='1'
$env:TFL_FITCH_27B_MAX_TOKENS='512'
$env:TFL_FITCH_27B_TIMEOUT_SECONDS='900'
```

Then run a focused regression:

```bash
python -m proofs_ocr.regression.tfl_runner run-gold --case tfl_handwritten_q4_N_or_K_and_W --output-root working/tfl_27b_probe
```

**Current 2026-05-14 probe result:** Turning off thinking now works through LM
Studio plus the API payload controls. A completed nested-disjunction reread with
`--manual-prompt --chunk-aware --no-think --enable-thinking-false --max-tokens
512` produced `thinking_chars=0` and `reasoning_tokens=0`. A large full-width
parent-block crop timed out at 90s, but a full-width single-row crop preserving
the line-number gutter returned in about 20s and kept the visible absolute line
number (`12`). It read `VE 4, 5-8 9-11`, which is normalizable to `∨E 4,5-8,9-11`.

**Remaining 27b issue:** The model can read the hard crops, but line numbering
is brittle when the crop does not show the gutter. On tight crops it may infer
absolute line numbers from local justification text, or it may assume
crop-relative numbering depending on prompt and visible context. Downstream
renumbering usually makes the raw `line` fields less important, but citation
justifications (`∧E 5`, `↔E 2,3`, etc.) depend on absolute proof line numbers.
For verifier-guided rereads, prefer full-width leaf or single-row crops that
preserve the left line-number gutter over large parent-block crops.

**Recommended next direction:**
1. Keep 27b as a targeted fallback/reread model, not as the default for every
   crop.
2. For checker-suspicious rows, generate a full-width row crop from the
   node/full-width crop so the left line-number gutter remains visible.
3. Ask 27b to reread only the suspicious row or very small section. Avoid whole
   parent-block rereads unless there is no row-level alternative.
4. Do not undo ROOT line-number blotting for the normal 4b path; only the
   targeted 27b reread/full-width path should expose line numbers.
5. Keep `max_tokens=512` or lower for these tiny rereads unless a probe proves
   it truncates useful output.
6. Record elapsed time, thinking chars, content chars, and
   `usage.completion_tokens_details.reasoning_tokens` from each probe. A
   successful no-thinking setup should show zero or near-zero reasoning tokens.

**Relevant files:**
- `scripts/probe_27b_crop.py` — direct crop diagnostic probe.
- `app/vlm_client.py` — `LMStudioQwenClient.chat()` sends the API call and records metadata.
- `proofs_ocr/pipelines/tfl_fitch.py` — 27b routing and prompt construction.
- `working/thinking_captures/` — saved probe records and reasoning captures.

### Q4 small subproof problem: SOLVED (27b full-width routing)
- **Case:** `tfl_handwritten_q4_N_or_K_and_W`
- **Expected:** 7 lines. `N∨(K∧W)` Pr, `N↔W` Pr, subproof [`N` Hyp, `W` ↔E 2,3], subproof [`K∧W` Hyp, `W` ∧E 5], `W` ∨E 1,3-4 5-6
- **Solution:** Both subproofs (2 lines each) now routed to 27b with full-width crops. Results are **perfect** after normalization.
- **Remaining issue:** ROOT crop still read by 4b, which misreads `N↔W` as `N↔N` (letter confusion). This is in the parent node, not the subproof leaves.

### Q5 biconditional misread
- **Case:** `tfl_handwritten_q5_P_or_K_biconditional`
- **Only failure:** Line 1 reads `(P∨K)→(Z∧W)` instead of `(P∨K)↔(Z∧W)` — the VLM reads handwritten `↔` as `→`. Everything else (lines 2-5) is perfect including all justifications.
- **This is a VLM symbol recognition issue**, not a normalization issue. Whitespace normalization is already correct.

### Q6 near-perfect formulas, citation errors — NEXT INVESTIGATION TARGET
- **Case:** `tfl_handwritten_q6_S_iff_W`
- **Major improvement:** Line count correct (11/11), structure perfect (0 mismatches), **all 11 formulas correct**.
- **Remaining gap:** 5 justification citation-number errors (lines 4, 5, 6, 8, 10). The VLM reads the correct rule names but misreads the handwritten line reference numbers.
- **Key observation:** Both subproofs have 4 lines each, so they **don't trigger** the ≤3 27b routing threshold. They stay on 4b with tight/letterboxed crops.
- **Investigation needed:** Look at crop images (`working/tfl_gold_regressions/tfl_handwritten_q6_S_iff_W/crop_2.jpg`, `crop_3.jpg`) and VLM responses to decide whether to raise the threshold or take a different approach.
- **Promising:** This is also a strong candidate for semantic validation repair — the formulas are correct enough to validate/invalidate citations.

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

### 27b model routing: ACTIVE (with full-width crops)
- **What it does:** Leaf crops with ≤3 expected lines are routed to Qwen 3.6 27B (`qwen/qwen3.6-27b`) via `LMSTUDIO_MODEL_27B` env var. Routed crops use **full-width images** (preserving line-number gutter) with a dedicated manual prompt. Non-routed crops continue to use tight/letterboxed crops with the auto-built prompt.
- **Config:** `SMALL_LEAF_MODEL_ROUTING_THRESHOLD = 3` in `tfl_fitch.py`
- **Crop generation:** `crop_full_width()` — full image width, node Y-range only, no letterboxing
- **Prompt:** `build_27b_fullwidth_prompt()` — the validated manual prompt with handwriting translation guide
- **Runtime controls:** `TFL_FITCH_27B_TIMEOUT_SECONDS`, `TFL_FITCH_27B_MAX_TOKENS`, `TFL_FITCH_27B_NO_THINK`, `TFL_FITCH_27B_ENABLE_THINKING_FALSE`, and `TFL_FITCH_27B_SINGLE_USER_MESSAGE`.
- **Validated:** Q4 subproofs (2 lines each) produce perfect OCR via 27b full-width routing (2026-04-26 regression).

### Thinking/metadata capture: ACTIVE
- **What it does:** `LMStudioQwenClient.chat()` extracts `reasoning_content` into `self.last_thinking` and records call metadata (`elapsed_s`, thinking/content chars, finish reason, usage, token caps). The pipeline saves thinking text as `thinking_{node_id}.txt` alongside each `response_{node_id}.json`.
- **Why it matters:** It separates transport failures, prompt-shape failures, and real model latency. For 27b no-thinking verification, check both `thinking_chars` and LM Studio usage details such as `reasoning_tokens`.
- **Files affected:** `app/vlm_client.py` (extraction + metadata), `proofs_ocr/pipelines/tfl_fitch.py` (saving + logging), `scripts/probe_27b_crop.py` (probe records).

### Other active notes
- **Scaffold removal disabled** (2026-04-24): `filter_scaffold_bars()` scaffold-removal logic deliberately disabled because handwritten Fitch proofs have a real outer subproof bar that spans most of the image height. Only edge-artifact cleanup remains active.
- **Line-number blotting** on ROOT node: The pipeline whites out everything to the left of the leftmost bar, preventing the VLM from reading line numbers as formula content.
- **Outer pytest timeout:** `tests/test_tfl_fitch_ocr_regression.py` still uses a 180s subprocess timeout for live cases. That is fine for 4b canary runs but can misclassify 27b experiments as timeouts. Prefer `python -m proofs_ocr.regression.tfl_runner run-gold ...` for 27b probes unless the test timeout is deliberately raised.
- **Do not overfit line numbers from crop OCR:** The pipeline renumbers final structures depth-first after stitching. Raw OCR `line` fields are secondary; citation strings inside `jstr` are the harder problem because they must refer to absolute proof lines.

---

## Experiment History

### 2026-04-26: 27b API timeout resolved; remaining issue is context/line numbers

**Problem:** 27b appeared to time out on API calls even though LM Studio GUI responses worked. A later probe also failed with `WinError 10061`.

**Resolution:**
1. The connection-refused error was not a model issue: the LM Studio server had been stopped while unloading/reloading the model.
2. Long API calls were caused by model thinking. Turning off thinking in LM Studio's GUI/model settings is the important control; prompt-level `/no_think` and `chat_template_kwargs.enable_thinking=false` alone were not sufficient in earlier tests.
3. With the server running and thinking disabled, 27b returns Q4 crop_3 answers that regularize to the correct formulas/justification.

**Code added for investigation (commit `933c6cb`):**
- `LMStudioQwenClient.chat()` records call metadata and accepts `max_tokens` plus extra payload knobs.
- `scripts/probe_27b_crop.py` supports manual prompt, single-user message mode, low max-token probes, and `enable_thinking=false` payload probes.
- `tfl_fitch.py` exposes 27b runtime controls through `TFL_FITCH_27B_*` environment variables.

**Current recommendation:** Use 27b as a targeted tiny-crop fallback while testing two integration strategies:
1. A 27b-only crop variant that includes the line-number gutter.
2. A prompt variant that provides absolute crop-start line metadata without widening the image.

The first strategy gives the model visual evidence for citations; the second avoids visual leakage. Test both on Q4 crop_2/crop_3 before changing full-pipeline defaults.

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
