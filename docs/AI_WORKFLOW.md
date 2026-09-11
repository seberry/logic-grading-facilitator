# AI Workflow

This note is for coding agents and slower local models that need a stable map of
the repo before changing OCR or regression code.

## Fast entry points

- Read [`docs/QUIZ_VARIANT_TF_WORKFLOW.md`](QUIZ_VARIANT_TF_WORKFLOW.md) for
  versioned paper-quiz intake, local-model preparation, model-free TA/TF
  handoff, and code-release work. This workflow is separate from proof OCR.
- Read [`proofs_ocr/regression/manifest.yaml`](../proofs_ocr/regression/manifest.yaml) for fixture ids, domains, and paths.
- Read [`docs/FITCH_OCR_AGENT.md`](FITCH_OCR_AGENT.md) before changing handwritten TFL/Fitch OCR behavior.
- Use [`proofs_ocr/regression/runner.py`](../proofs_ocr/regression/runner.py) for modal gold-regression helpers.
- Use [`proofs_ocr/regression/tfl_runner.py`](../proofs_ocr/regression/tfl_runner.py) for handwritten TFL/Fitch gold-regression helpers.
- Use [`proofs_ocr/pipelines/tfl_fitch.py`](../proofs_ocr/pipelines/tfl_fitch.py) for the importable Fitch OCR pipeline.
- Keep [`scripts/debug_fitch_ocr.py`](../scripts/debug_fitch_ocr.py) as a CLI wrapper, not the primary implementation.

## Safe and fast commands

- `pytest tests/test_debug_fitch_ocr.py -q`
- `pytest tests/test_tfl_fitch_gold_fixtures.py -q`
- `pytest tests/test_tfl_fitch_ocr_regression.py -q`
- `pytest tests/test_tfl_private_runner.py tests/test_tfl_private_exporter.py -q`
- `pytest tests/test_tfl_private_narrator.py -q`
- `pytest tests/test_proofs_ocr_regression_runner.py -q`
- `python -m proofs_ocr.regression summary`
- `python -m proofs_ocr.regression.tfl_runner summary`

These are the preferred commands for CI-safe edits because they do not require a
live vision model. They validate harness logic and fixture plumbing, not the
quality of live handwritten OCR.

For synthetic FOL model fixture work, the fast offline checks are:

- `pytest tests/test_synthetic_logic_exam_generator.py tests/test_synthetic_fol_ocr_eval.py -q`
- `python scripts/generate_synthetic_logic_exam.py --family fol_model --style printed --count 4 --output-dir working/synthetic_fol_smoke --force`

The synthetic OCR evaluation runner also has unit coverage with a fake
transcriber. Live runs against LM Studio are manual workflows, not default tests.

## Slow or live-model workflows

- `python -m proofs_ocr.regression run-gold ...`
- `python -m proofs_ocr.regression.tfl_runner run-gold ...`
- `python -m proofs_ocr.regression.private_tfl_runner --manifest <external_manifest.yaml> run-gold ...`
- `RUN_LIVE_TFL_OCR=1 pytest tests/test_tfl_fitch_ocr_regression.py -q -k Live`
- `python scripts/debug_fitch_ocr.py <scan>.jpg`

Treat these as local/manual workflows. They may require LM Studio or another
running local VLM endpoint. When a live OCR run is enabled, expect strict
gold-match failures until the handwritten pipeline actually improves.

For TFL/Fitch runs that exercise the 27b reread/fallback path, confirm LM
Studio has thinking/reasoning disabled for the loaded 27b model before timing
or judging failures. The API payload also sends no-thinking controls when the
`TFL_FITCH_27B_*` flags are set, but the LM Studio runtime setting is the
important control. Check `response_metadata_*.json`: successful no-thinking
probes should show `thinking_chars=0` and `reasoning_tokens=0`.

For OCR behavior changes, assume the user usually wants a bounded slow run
after the fast tests pass. Default to the smallest relevant live case first
using `--case ... --output-root working/<descriptive_run_name>`, then summarize
whether the change helped before proposing broader runs.

Before a live OCR run on new proof geometry, confirm the pre-VLM structure first:
bar count, subproof tree shape, and leaf line counts should be plausible. Small
leaf crops can route to slower verifier/larger-model paths, so bar
over-detection can multiply expensive calls quickly.

For real student proof crops, use the private TFL runner instead of adding
fixtures to the repo manifest. The manifest, crops, gold files, and run outputs
must stay outside the repo, for example under `C:\grading-data\ocr-eval\...`.
The runner writes local detailed reports plus `shareable_summary.txt/json`,
which omit paths, images, line text, diffs, and timestamps:

To bootstrap a new private curation pass from a Canvas ZIP, the setup wizard can
also be used as an intake UI with an external production `DATA_DIR`. If no
`roster.csv` exists, the Canvas ZIP importer creates anonymous work units from
filename prefixes (`anon_0001`, etc.) so strip OCR can preserve page/problem
grouping without loading a real roster. Use this for OCR-corpus curation, not
for actual grade export. Keep `LMSTUDIO_MODEL_27B` unset or empty unless you are
explicitly testing the larger small-crop route.

```powershell
$env:DATA_DIR='C:\grading-data\live-ocr-curation'
$env:DATA_ARCHIVE_DIR='C:\grading-data\archive'
$env:ENV_MODE='PRODUCTION'
$env:LMSTUDIO_MODEL_27B=''
python -m uvicorn app.server:app --port 5001

python -m proofs_ocr.regression.private_tfl_exporter seed-from-strip --eval-root C:\grading-data\ocr-eval\exam1 --assignment-id exam1_spring2026 --problem-id Q6 --grades-dir C:\grading-data\live\grades --debug-fitch-dir C:\grading-data\live\debug_fitch --split dev_regression
python -m proofs_ocr.regression.private_tfl_runner --manifest C:\grading-data\ocr-eval\exam1\manifest.yaml summary
python -m proofs_ocr.regression.private_tfl_runner --manifest C:\grading-data\ocr-eval\exam1\manifest.yaml run-gold --split dev_regression
python -m proofs_ocr.regression.private_tfl_narrator suggest-cases --summary C:\grading-data\ocr-eval\exam1\runs\tfl_private_gold_regressions\shareable_summary.json --category structure
python -m proofs_ocr.regression.private_tfl_narrator list-artifacts --manifest C:\grading-data\ocr-eval\exam1\manifest.yaml --run-root C:\grading-data\ocr-eval\exam1\runs\tfl_private_gold_regressions --case exam1_spring2026_q6_case_0007
python -m proofs_ocr.regression.private_tfl_narrator ask --manifest C:\grading-data\ocr-eval\exam1\manifest.yaml --run-root C:\grading-data\ocr-eval\exam1\runs\tfl_private_gold_regressions --case exam1_spring2026_q6_case_0007 --artifact debug_bars --model qwen/qwen3-vl-27b --disable-thinking --question "Why might the bar detector or line grouping be failing here?"
```

Current private Q1 Fitch starter corpus:

```powershell
python -m proofs_ocr.regression.private_tfl_runner --manifest C:\grading-data\ocr-eval\real-fitch-q1\manifest.yaml summary
python -m proofs_ocr.regression.private_tfl_runner --manifest C:\grading-data\ocr-eval\real-fitch-q1\manifest.yaml run-gold --split dev_regression
```

Only paste the shareable summary back into agent conversations unless the case is
fake or fully sanitized. For protected real scans, agents should not directly
open or visually inspect images/crops or paste raw gold/OCR text; use
`private_tfl_narrator ask` to have the local vision model inspect protected
artifacts and return redacted observations.

For synthetic FOL model OCR runs, use the generated fake fixture and report
runner:

```powershell
python scripts/generate_synthetic_logic_exam.py --family fol_model --style phone_scan_light --count 8 --output-dir data/samples/synthetic_fol_models --force
python scripts/run_synthetic_fol_ocr_eval.py --fixture-root data/samples/synthetic_fol_models --output-dir working/synthetic_fol_ocr_report --limit 6
```

Inspect `working/synthetic_fol_ocr_report/summary.html` locally. The summary is
safe to discuss because the fixture is synthetic.

## Synthetic Feature Playbook

Use this pattern when adding a new autograded problem family or answer shape.
It keeps fake-data work useful without pretending synthetic OCR equals real
student performance.

1. **Start with semantic gold.** Generate answer data from the checker contract:
   the target rubric config, the intended answer text/structure, and the
   expected pass/fail or score. The gold file should be independent of the image
   renderer.
2. **Render a clean baseline.** Produce complete `DATA_DIR` fixtures with
   `roster.csv`, rubric YAML, `raw_scans/`, `grades/session.json`,
   precomputed `grades/segments_*.json`, and a README. The clean version should
   prove the session, strip view, and checker plumbing.
3. **Add controlled visual styles.** Add one degradation axis at a time:
   handwriting-like font, line jitter, paper texture, faint ink, blur, shadow,
   rotation, crop padding. Keep these deterministic by seed.
4. **Separate OCR stress from segmentation stress.** First keep crop boxes
   correct and measure OCR/checker behavior. Then add a separate crop-stress mode
   for cut-off answers, extra neighboring text, off-box writing, or shifted
   boxes.
5. **Write an evaluation report.** The runner should crop from the segment cache,
   run OCR, run the deterministic checker, compare to gold, and write JSON/HTML
   reports under `working/`.
   Add coarse failure labels so OCR runs can be skimmed before reading every
   artifact. For FOL models the current labels include missing domain/predicate
   sections, bad constants section, checker parse failure, invented default
   constants, and verdict mismatch.
6. **Keep tests offline.** Unit tests should use fake transcribers and checker
   calls. Live VLM runs belong in manual commands and should write only ignored
   artifacts.
7. **Document the family-specific plug-ins.** For FOL models this means model
   text and formula targets. For truth tables it will mean table schemas and
   expected rows. For Fitch/modal proofs it will need proof geometry, bar/line
   structure, and verifier gold.

## OCR architecture

Handwritten Fitch/TFL OCR currently flows through:

1. image load
2. vertical-bar detection
3. subproof tree construction
4. recursive crop and mask
5. VLM transcription of each crop
6. token stitch-back into nested proof structure
7. normalization and blank-justification repair
8. final structure write
9. optional verifier pass

The importable implementation lives in
[`proofs_ocr/pipelines/tfl_fitch.py`](../proofs_ocr/pipelines/tfl_fitch.py).

## Fixture source of truth

- Modal and TFL handwritten fixtures are declared in [`proofs_ocr/regression/manifest.yaml`](../proofs_ocr/regression/manifest.yaml).
- Curated expected structures are the source of truth for OCR regressions.
- Semantic caveats belong in the sidecar `semantic_notes.yaml` file next to the fixture.

## Working conventions

- Prefer importable modules under `proofs_ocr/` for new OCR logic.
- Keep shell scripts thin wrappers around importable functions.
- Write long-running debug artifacts under `working/` for regression runs.
- Avoid live-model requirements in default test targets.
