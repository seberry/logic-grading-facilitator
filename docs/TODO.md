# TODO

## Server refactoring (2026-03-21)
- **COMPLETED**: Created scripts for improving server.py readability
  - `scripts/organize_server.py` - Full reorganization script (BROKEN - caused syntax errors)
  - `scripts/add_server_markers.py` - Safe marker-only script (better approach)
  - `scripts/README_SERVER_REFACTORING.md` - Documentation for both scripts
- **LESSON LEARNED**: Full reorganization is risky. Better to use incremental approach:
  1. Add section markers only (preserve code order and dependencies)
  2. Test to ensure system still works
  3. Only then consider extracting sections to separate modules
- **NEXT STEPS**: When ready, run `python scripts/add_server_markers.py` to add markers without breaking anything
- **RECOVERY**: System was rolled back to commit `014c692` after breaking changes; working version restored via git

## FERPA guardrails

## FERPA guardrails
- Investigate hard guardrails for live grading data so the CLI AI can do metadata-only operations on FERPA-sensitive paths without ever reading raw file contents into model context.
- Preferred direction: keep live grading data outside the normal AI-readable workspace and expose a tiny local helper/tool for safe operations only (`exists`, `list filenames`, `rename`, `copy`, `write manifest`), with no raw content-read command.
- Apply stronger warnings/approval expectations before touching `DATA_DIR` in production mode; distinguish clearly between path/metadata operations and file-content reads.
- For Canvas ZIP imports, preserve the untouched original ZIP plus a manifest that maps original Canvas filenames to canonical grading files; reupload should use that manifest automatically rather than asking the user twice.

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
