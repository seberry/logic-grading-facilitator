# TODO

## Problem strips issues
- Improve prompts for Fitch proof problems so the model targets the proof body (not the problem statement).
- Add user guidance: rubric labels should include the exact problem text (or a close paraphrase) to improve segmentation accuracy.

## Fitch proof mode
- Wire “fitch_proof” tag into a dedicated UI path in the problem strips view.
- Reuse the debug pipeline (`docs/STRATEGY_FITCH_PROOFS.md`) to show OCR + verifier outputs in-place.

## Exports
- Add a "feedback ZIP" export that bundles per-student attachments (comments, per-problem crops/OCR) for manual drag-and-drop into Canvas; Canvas ignores comment columns in CSV imports, so richer feedback needs files rather than CSV.

## UX polish
- Clicking a problem in the finish overview should jump to that problem's strip grader (currently only the header links; consider row/cell shortcuts too).
- If you cancel the "ungraded entries" warning, jump to an ungraded entry and highlight all ungraded inputs.
- Add a navbar button that takes you directly to the finish/overview screen.
