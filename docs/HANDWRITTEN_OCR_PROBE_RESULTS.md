# Handwritten OCR Probe Results

**Date**: 2026-04-23 (updated)  
**Pipeline**: TFL Fitch OCR (VLM-based crop-and-transcribe)  
**Fixture images**: 4 TFL handwritten proofs (FERPA-safe)

## Current Scores (after Phase 1+2+2a normalization)

| Case | Lines | Formula Acc. | Justification Acc. | Both Correct |
|------|-------|-------------|-------------------|-------------|
| Q3 `(P∧Q)→R` | 8/8 | **8/8 (100%)** | **8/8 (100%)** | **8/8 (100%)** |
| Q4 `N∨(K∧W)` | 7/7 | 5/7 (71%) | 5/7 (71%) | 4/7 (57%) |
| Q5 `(P∨K)↔(Z∧W)` | 5/5 | 4/5 (80%) | 4/5 (80%) | 3/5 (60%) |
| Q6 `S↔W` | 11/11 | **11/11 (100%)** | 4/11 (36%) | 4/11 (36%) |
| **TOTAL** | **31/31** | **28/31 (90%)** | **21/31 (68%)** | **19/31 (61%)** |

### Score progression

| Phase | Justification | Both Correct | Delta |
|-------|--------------|-------------|-------|
| Baseline (raw OCR) | 18/31 (58%) | 16/31 (52%) | — |
| Phase 1 (space→comma) | 19/31 (61%) | 17/31 (55%) | +1 J, +1 B |
| Phase 2 (`8`→`∧` repair) | 21/31 (68%) | 19/31 (61%) | +2 J, +2 B |
| Phase 2a (epsilon→E) | 21/31 (68%) | 19/31 (61%) | +0 (defensive) |

## Remaining Errors (10 lines, post-normalization)

### Q3 — 0 errors ✅ (was 2 J errors, fixed by Phase 2)

### Q4 — 3 lines with errors

| Line | Field | Gold | OCR (normalized) | Error Type |
|------|-------|------|-------------------|------------|
| 4 | J | `↔E 2,3` | `←F 2,3` | `↔E` misread as `←F` |
| 5 | F | `K∧W` | `1<8W` | Severe: `K∧` → `1<8` |
| 6 | F+J | `W` / `∧E 5` | `N` / `∧E S` | `W`→`N`, digit `5`→`S` |

### Q5 — 2 lines with errors

| Line | Field | Gold | OCR (normalized) | Error Type |
|------|-------|------|-------------------|------------|
| 1 | F | `(P∨K)↔(Z∧W)` | `(P∨K)→(Z∧W)` | `↔` → `→` confusion |
| 4 | J | `↔E 1,4` | `→E 1,4` | `↔E` → `→E` (same root cause) |

### Q6 — 7 lines with errors (all justification)

| Line | Gold J | OCR J | Error Type |
|------|--------|-------|------------|
| 4 | `∧E 3` | `∧E 1` | Wrong line number (3→1) |
| 5 | `→E 1,4` | `∧E 1` | Wrong rule (`→E`→`∧E`) + wrong refs |
| 6 | `∧I 4,5` | `∧I 2,3` | Wrong line numbers |
| 8 | `∧E 7` | `∧E 5` | Wrong line number |
| 9 | `→E 2,8` | `∧E 5` | Wrong rule + wrong refs |
| 10 | `∧I 8,9` | `∧I 6,7` | Wrong line numbers |
| 11 | `↔I 3-6,7-10` | `3-6 7-10` | Missing rule head `↔I` |

## Low-Hanging Fruit Assessment

### Potential Phase 3a: `←F` → `↔E` justification head repair
- **Impact**: +1 justification (Q4 L4), bringing J to 22/31 (71%)
- **Risk**: Very low — `←F` is never a valid TFL justification rule
- **Implementation**: In `canonicalize_justification_for_comparison`, replace `←F` with `↔E` in the head

### Potential Phase 3b: `→E`/`→I` → `↔E`/`↔I` contextual disambiguation
- **Impact**: Could fix Q5 L4 (`→E 1,4` → `↔E 1,4`) if the formula also contains `↔`
- **Risk**: HIGH — `→E` and `→I` are valid rules; blind replacement would break correct `→E` uses
- **Assessment**: Not feasible without formula-level context (knowing the goal formula has `↔`)

### Not fixable by normalization (fundamental OCR errors)
- **Q4 L5**: `K∧W` → `1<8W` — letter/digit confusion at VLM level
- **Q4 L6**: `W` → `N`, `5` → `S` — letter/digit confusion
- **Q6 L4-10**: Wrong line numbers in tails — VLM misreads handwritten digits
- **Q6 L11**: Missing `↔I` head — VLM dropped the rule name entirely

## Error Pattern History (for reference)

### Pattern 1: `∧` → `8` confusion ✅ FIXED (Phase 2)
The VLM reads handwritten `∧` (written as `&`) as `8`. Fixed by post-OCR heuristic replacing `8` with `∧` before `I` or `E` in justification heads, and between capital letters in formulas.

### Pattern 2: `↔` → `→` confusion (OPEN)
Biconditional arrow read as conditional. Affects both formulas and justifications. Not safely normalizable without context.

### Pattern 3: `↔E` → `←F` substitution (Phase 3a candidate)
Single occurrence (Q4 L4). Safe to normalize: `←F` is never a valid TFL rule.

### Pattern 4: Subproof nesting / line number errors (VLM-level, not normalizable)
Q6 has 7 lines with wrong justification line numbers. These are fundamental VLM misreads of handwritten digits.

### Pattern 5: Space-between-digits → comma ✅ FIXED (Phase 1)

## Normalization Pipeline (current)

The comparison pipeline now applies these normalizations in order:

**Formula**: strip whitespace → symbol aliases (`&`→`∧`, `v`→`∨`, `~`→`¬`, `->`→`→`, `<->`→`↔`) → `8` between caps → `∧`

**Justification**: strip whitespace → epsilon→E (`∈`→`E`, `ε`→`E`) → head aliases (`ai`→`∧I`, `&e`→`∧E`, etc.) → `8`+`I`/`E` → `∧I`/`∧E` → tail space-between-digits → comma

## Planned Improvements

1. ~~**Phase 1**: Fix space-between-digits → comma normalization~~ ✅ DONE
2. ~~**Phase 2**: `8`→`∧` post-OCR repair heuristic~~ ✅ DONE
3. ~~**Phase 2a**: epsilon→E normalization before 8→∧ chain~~ ✅ DONE (defensive)
4. **Phase 3a** (easy): `←F` → `↔E` justification head repair (+1 J)
5. **Phase 3b** (hard): `↔` vs `→` contextual disambiguation — needs formula awareness
6. **VLM prompt improvements**: Better handling of handwritten `&`/`∧`, line numbers, biconditional arrows

## Artifacts

Full diagnostic output saved to `working/handwritten_ocr_probe/tfl/`:
- Per-case `gold_regression_metrics.json` with line-by-line comparisons
- Per-case `gold_regression_diff.txt` with unified diffs
- Per-case `final_structure.json` with OCR output
- Per-case `response_*.json` with raw VLM responses
- Per-case `crop_*.jpg` and `debug_bars.jpg` for visual inspection