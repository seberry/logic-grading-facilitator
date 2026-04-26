# Working Directory

This directory is local scratch space for live OCR runs, VLM probes, generated
crops, and regression artifacts.

The directory is intentionally ignored by git except for this file and
`.gitkeep`. Existing subdirectories may be useful for comparing model behavior;
do not delete them unless the user explicitly asks.

Common output roots:

- `working/tfl_gold_regressions/` - default TFL handwritten gold regression run.
- `working/tfl_*` - focused TFL/Fitch OCR experiments.
- `working/fullwidth_crops/` - direct full-width crop probe artifacts.
- `working/test_tmp/` - temporary directories created by tests.

Real student handwriting/crops should stay in protected local grading-data
folders unless a workflow intentionally creates local-only debug artifacts.
