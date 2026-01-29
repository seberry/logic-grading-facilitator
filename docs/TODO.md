# TODO

## Problem strips issues
- Blank strips: confirm image files exist, verify `page_filenames` in `session.json`, check 404s in the browser console, and log missing `seg.present` / bbox values.
- Bad Q1 localization: log the exact prompt + image region sent to the model, preview the crop, and tighten or pre-crop the input if needed.
- Improve prompts for Fitch proof problems so the model targets the proof body (not the problem statement).
- Add user guidance: rubric labels should include the exact problem text (or a close paraphrase) to improve segmentation accuracy.

## Fitch proof mode
- Wire “fitch_proof” tag into a dedicated UI path in the problem strips view.
- Reuse the debug pipeline (`docs/STRATEGY_FITCH_PROOFS.md`) to show OCR + verifier outputs in-place.
