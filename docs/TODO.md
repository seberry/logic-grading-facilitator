# TODO

## Problem strips issues
- Improve prompts for Fitch proof problems so the model targets the proof body (not the problem statement).
- Add user guidance: rubric labels should include the exact problem text (or a close paraphrase) to improve segmentation accuracy.

## Fitch proof mode
- Wire “fitch_proof” tag into a dedicated UI path in the problem strips view.
- Reuse the debug pipeline (`docs/STRATEGY_FITCH_PROOFS.md`) to show OCR + verifier outputs in-place.

## Exports
- Consider adding crops/OCR outputs into the feedback ZIP once those artifacts are stable; Canvas ignores comment columns in CSV imports, so richer feedback needs files rather than CSV.

## UX polish
- Consider row/cell shortcuts in the finish overview (jump to strip grader at a specific student/problem).
