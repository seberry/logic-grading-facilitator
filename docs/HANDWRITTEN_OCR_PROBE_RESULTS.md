# Handwritten OCR Probe Results

**Date**: 2026-04-23  
**Pipeline**: TFL Fitch OCR (VLM-based crop-and-transcribe)  
**Fixture images**: 4 TFL handwritten proofs (FERPA-safe)

## Summary

| Case | Lines | Formula Acc. | Justification Acc. | Both Correct | Structure Mismatches |
|------|-------|-------------|-------------------|-------------|---------------------|
| Q3 `(P∧Q)→R` | 8/8 | **8/8 (100%)** | **6/8 (75%)** | 6/8 (75%) | 0 |
| Q4 `N∨(K∧W)` | 7/7 | **5/7 (71%)** | **4/7 (57%)** | 3/7 (43%) | 2 |
| Q5 `(P∨K)↔(Z∧W)` | 5/5 | **4/5 (80%)** | **4/5 (80%)** | 3/5 (60%) | 0 |
| Q6 `S↔W` | 11/11 | **11/11 (100%)** | **4/11 (36%)** | 4/11 (36%) | 4 |
| **TOTAL** | **31/31** | **28/31 (90%)** | **18/31 (58%)** | **16/31 (52%)** | **6** |

## Error Pattern Analysis

### 1. `∧` → `8` confusion (most common, caused by badly-written `&`)

The VLM reads handwritten `∧` (written as `&`) as `8`. Affects both formulas and justifications.

| Case | Line | Gold | OCR Output |
|------|------|------|------------|
| Q3 | 4 | `∧E 3` | `8 E 3` |
| Q3 | 6 | `∧I 4,5` | `8 I 4,5` |
| Q4 | 5 | `K∧W` | `1 < 8 W` |
| Q4 | 6 | `∧E 5` | `8 ∈ S` |

**Planned fix**: Post-OCR heuristic replacing `8` with `∧` before `I` or `E` in justifications, and in formulas where contextually appropriate.

### 2. `↔` → `→` confusion (biconditional read as conditional)

| Case | Line | Gold | OCR Output |
|------|------|------|------------|
| Q5 | 1 | `(P∨K)↔(Z∧W)` | `(P∨K)→(Z∧W)` |
| Q5 | 4 | `↔E 1,4` | `→E 1,4` |

### 3. `↔E` → `←F` substitution

| Case | Line | Gold | OCR Output |
|------|------|------|------------|
| Q4 | 4 | `↔E 2,3` | `←F 2,3` |

### 4. Subproof nesting (structure) errors

| Case | Lines | Description |
|------|-------|-------------|
| Q4 | 5-6 | Nested subproof at depth 4 misread as depth 3 |
| Q6 | 4 lines | Subproof boundaries confused |

### 5. Missing comma in justification tail (normalization gap)

| Case | Line | Gold | OCR Output |
|------|------|------|------------|
| Q4 | 7 | `∨E 1,3-4,5-6` | `∨E 1, 3-4 5-6` |

**Fix**: Add space-between-digits → comma normalization in `canonicalize_justification_for_comparison`.

## Normalization Already Applied

The comparison pipeline already normalizes:
- **Formula whitespace**: All spaces stripped (`"P ∧ Q"` → `"P∧Q"`)
- **Symbol aliases**: `&` → `∧`, `v` → `∨`, `~` → `¬`, `->` → `→`, `<->` → `↔`
- **Justification head aliases**: `"ai"` → `"∧I"`, `"&e"` → `"∧E"`, etc.
- **Dash/comma between digits**: standardized

## What's Not Available

- **Modal handwritten images**: 0/2 modal handwritten fixture images exist on disk (only gold structures committed)

## Planned Improvements

1. **Phase 1 ✅ DONE**: Fix space-between-digits → comma normalization in justification comparison
   - Result: justification 18/31 → **19/31** (58% → 61%), both-correct 16/31 → **17/31** (52% → 55%)
2. **Phase 2**: `∧` → `8` post-OCR repair heuristic
3. **Phase 3**: `↔` vs `→` disambiguation strategies

## Artifacts

Full diagnostic output saved to `working/handwritten_ocr_probe/tfl/`:
- Per-case `gold_regression_metrics.json` with line-by-line comparisons
- Per-case `gold_regression_diff.txt` with unified diffs
- Per-case `final_structure.json` with OCR output
- Per-case `response_*.json` with raw VLM responses
- Per-case `crop_*.jpg` and `debug_bars.jpg` for visual inspection