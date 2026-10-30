# TFL/Fitch OCR Operating Guide

This is the current source of truth for handwritten TFL/Fitch OCR work. The
long dated history was moved to [FITCH_OCR_EXPERIMENT_LOG.md](FITCH_OCR_EXPERIMENT_LOG.md).
Use that log when you need old experiment evidence; use this file when you need
to change or run the current pipeline.

## Working Boundary

- Main pipeline: `proofs_ocr/pipelines/tfl_fitch.py`
- Thin CLI wrapper: `scripts/debug_fitch_ocr.py`
- Gold regression runner: `proofs_ocr/regression/tfl_runner.py`
- Fixture manifest: `proofs_ocr/regression/manifest.yaml`
- Private real-scan tools:
  - `proofs_ocr/regression/private_tfl_exporter.py`
  - `proofs_ocr/regression/private_tfl_runner.py`
  - `proofs_ocr/regression/private_tfl_narrator.py`

Keep behavior in importable modules under `proofs_ocr/`. Scripts should remain
thin wrappers or local probes.

## Privacy Rules

Real student scans, crops, handwriting, rosters, LMS IDs, timestamps, and local
paths are protected local data.

- Do not add real scans/crops to tracked fixtures or docs.
- Do not paste raw private gold structures, OCR diffs, path names, or image
  contents into agent conversations.
- Use anonymous case IDs and aggregate/redacted summaries.
- For visual inspection of protected OCR artifacts, prefer
  `private_tfl_narrator ask`; it asks a local model and writes a redacted
  observation.
- Keep private eval corpora outside the repo, for example
  `C:\grading-data\ocr-eval\...`.

## Fast Offline Commands

These are CI-safe and do not require a live VLM:

```powershell
$env:PYTHONPATH='.'
pytest tests/test_debug_fitch_ocr.py -q
pytest tests/test_tfl_fitch_gold_fixtures.py -q
pytest tests/test_tfl_fitch_ocr_regression.py -q
pytest tests/test_tfl_private_runner.py tests/test_tfl_private_exporter.py -q
pytest tests/test_tfl_private_narrator.py -q
python -m proofs_ocr.regression.tfl_runner summary
```

Use these before live OCR unless the change is docs-only.

## Live OCR Commands

Live runs now default to IU REALLMS Gemma through `app/vlm_provider.py` and
require `REALLMS_API_KEY`. Set `REALLMS_MODEL=qwen3.8-27b` for an IU Qwen
comparison, or `GRADING_VLM_PROVIDER=lmstudio` for the existing local baseline.
Provider selection changes the transport/model, not the chopping, masking,
stitching, normalization, or regression comparison. Name-recognition success is
not proof-OCR validation; run bounded instructor-authored/fake gold cases first.

```powershell
python -m proofs_ocr.regression.tfl_runner run-gold
python -m proofs_ocr.regression.tfl_runner run-gold --case tfl_handwritten_q4_N_or_K_and_W
python -m proofs_ocr.regression.tfl_runner run-gold --output-root working/my_experiment
```

For one cropped proof:

```powershell
python scripts/debug_fitch_ocr.py path\to\proof_crop.jpg
```

The optional `LMSTUDIO_MODEL_27B` reread route is for explicitly configured local
experiments; leave it unset for IU runs. For local 27B probes, confirm LM Studio has thinking/reasoning
disabled for the loaded model. The pipeline writes `response_metadata_*.json`;
healthy no-thinking probes should show `thinking_chars=0` and
`reasoning_tokens=0`.

## Pipeline Shape

The handwritten TFL/Fitch OCR flow is:

1. Load image.
2. Detect vertical Fitch bars.
3. Build a subproof tree.
4. Recursively crop and mask child subproofs.
5. Ask the VLM to transcribe each crop.
6. Stitch `<SUBPROOF_ID_N>` placeholders into nested proof structure.
7. Normalize formulas, justifications, depth, and blank assumptions/premises.
8. Write `final_structure.json`.
9. Optionally run verifier/reread reporting.

Typical artifacts are written beside the crop under `DATA_DIR/debug_fitch/<slug>/`
or under a regression output root:

| Artifact | Purpose |
|---|---|
| `debug_bars.jpg` | Bar detection overlay. |
| `crop_ROOT.jpg`, `crop_N.jpg` | Images sent to the VLM. |
| `crop_N_fullwidth.jpg` | Full-width small-leaf crop used by the optional 27B route. |
| `response_N.json` | Raw VLM response. |
| `response_metadata_N.json` | Runtime and token metadata. |
| `thinking_N.txt` | Captured Qwen reasoning text when present. |
| `final_structure.json` | Stitched and normalized proof JSON. |
| `gold_regression_*.json/txt` | Regression result, metrics, and diff files. |

## Current OCR Status

- Line-count estimation is active and uses `proofs_ocr/common/rows.py`.
- Letterboxed tight crops are the default for small leaves.
- Context-aware green-box crops are not the default because earlier Qwen runs
  ignored the highlighted region.
- ROOT line-number blotting is active to prevent line numbers from entering
  formula text.
- Outer Fitch proof frame handling treats the ordinary printed spine as page
  scaffold; logical proof depth starts at 0.
- The 27B small-crop route is opt-in through `LMSTUDIO_MODEL_27B`; leave it
  unset for normal fast runs.
- Reread candidates are report-only in strip view. The grader must accept or
  correct by hand.

## Private Real-Scan Evaluation

Use the private runner/exporter for real student proof crops. The manifest,
crops, gold files, and detailed runs must remain outside the repo.

Seed cases from accepted strip OCR/debug state:

```powershell
python -m proofs_ocr.regression.private_tfl_exporter seed-from-strip `
  --eval-root C:\grading-data\ocr-eval\exam1 `
  --assignment-id exam1_spring2026 `
  --problem-id Q6 `
  --grades-dir C:\grading-data\live\grades `
  --debug-fitch-dir C:\grading-data\live\debug_fitch `
  --split dev_regression
```

Run private regressions:

```powershell
python -m proofs_ocr.regression.private_tfl_runner `
  --manifest C:\grading-data\ocr-eval\exam1\manifest.yaml `
  run-gold --split dev_regression
```

Share only `shareable_summary.txt/json` and anonymous case IDs back into agent
workflows.

## Maintenance Notes

- New durable OCR behavior, flags, artifact expectations, and redacted current
  findings belong here.
- Dated experiment narratives belong in
  [FITCH_OCR_EXPERIMENT_LOG.md](FITCH_OCR_EXPERIMENT_LOG.md).
- Broader command recipes belong in [AI_WORKFLOW.md](AI_WORKFLOW.md) or
  [DEV_WORKFLOWS.md](DEV_WORKFLOWS.md).
- Do not create another narrow OCR markdown file unless these owner docs cannot
  reasonably hold the information.

## Initial IU model comparison and next experiment

The provider migration is committed at `8701bce`. A bounded comparison used the
registered instructor-authored Q3, Q4, and Q5 handwritten gold fixtures (20 proof
lines total), including genuine subproofs and an intentionally invalid citation.
No real student proofs were used. The production chop/mask/stitch algorithm was
left unchanged during this comparison.

| Model / strategy | Exact gold structures | Total time for three proofs |
|---|---:|---:|
| Local Qwen3-VL-4B, existing pipeline | 1/3 | 50.8 s |
| IU Gemma, existing pipeline | 1/3 | 9.5 s |
| IU Qwen, existing pipeline | 0/3 | 12.3 s |
| IU Gemma, constrained whole-page experiment | 3/3 | 3.8 s |
| IU Qwen, constrained whole-page experiment | 3/3 | 8.4 s |

The existing pipeline produced an extra line and incorrect nesting on Q3 with
both IU models. Both changed Q5's visibly written invalid citation `1,4` to the
logically appropriate `1,3`. These are unacceptable transcription changes even
when a formula is otherwise read correctly. Gemma read all 20 aligned expected
formulas correctly, but the extra line still makes the complete result wrong.

A first unconstrained whole-page prompt produced inconsistent list nesting,
letter case, and rule formatting. A second version used a literal-transcription
instruction, uppercase TFL atoms, compact Unicode rule heads, a generic nested
array example, and a JSON schema constraining lines versus subproof arrays.
Both IU models then matched formulas, justifications, and nested structure for
all three fixtures, preserving the intentional bad citation. The schema defines
output shape, not the correct proof; gold answers were not supplied to the model.

This is a three-proof exploratory result, with changed prompts/output contracts
and single timing samples. It does not validate general handwriting accuracy,
deep nesting, FOL, or unattended grading. The whole-page strategy is **not yet
integrated into production OCR**; only provider selection has changed there.

The retained local probe is `working/iu-fitch-comparison.py`; detailed outputs
and the combined summary are under `working/iu-fitch-comparison/`. `gemma` and
`qwen` contain the existing-pipeline runs; `gemma_whole_page_v2` and
`qwen_whole_page_v2` contain the constrained whole-page runs. These are fake/
instructor-authored artifacts safe for agent inspection. Do not generalize that
permission to protected student corpora.

Suggested fresh-session prompt:

```text
Continue the IU Fitch whole-page OCR experiment in this repository. Read AGENTS.md,
docs/FITCH_OCR_AGENT.md, and docs/QUIZ_VARIANT_TF_WORKFLOW.md. IU Gemma is now the
production default; Qwen is an explicit alternative. Provider migration commit:
8701bce. Preserve the pre-existing unrelated working-tree changes.

Inspect working/iu-fitch-comparison.py, its summary.json, and the three instructor-
authored gold fixtures. Constrained whole-page Gemma and Qwen each matched 3/3;
the existing chopping pipeline scored local 1/3, Gemma 1/3, and Qwen 0/3.

Evaluate the whole-page contract on the remaining registered fake/instructor-
authored cases, especially nested subproofs and intentional errors. Keep literal
citations and errors; never repair a student's reasoning during transcription.
Check explicit line numbers as well as formulas, rules, nesting, and extra/missing
lines. Gold answers must never be included in model inputs. Keep runs bounded and
record transport/model metadata. Use no protected student scans without an
explicitly authorized private workflow.

If the results support it, implement an importable opt-in whole-page strategy
under proofs_ocr/, with the current pipeline retained for comparison. Add meaningful
offline tests, verify strip-view integration and human acceptance behavior, and
run a small live canary. Report the evidence before changing the production OCR
strategy default. Do not infer correctness from model confidence or verifier
success alone. Commit a concrete, reviewable implementation if successful.
```

## Expanded instructor tests and private bundle pilot

Gemma's same constrained whole-proof contract has now been evaluated on all
seven registered instructor-authored/fake TFL fixtures: **5/7 exact** under the
existing normalized structure comparison. The four additional fixtures produced
two exact results, one correct line-content result with incorrect subproof
structure (the biconditional-introduction example), and one omitted line in the
nested/disjunction example with scratch-outs. Explicit line numbers were also
compared. These failures were retained; the prompt was not tuned to their gold.

The opt-in implementation is `proofs_ocr/pipelines/tfl_whole_page.py`. It keeps
the whole proof region intact, uses the tested literal-copy prompt and constrained
JSON output, validates representation without logical repair, and records prompt/
schema hashes plus model runtime metadata. The current experimental schema allows
up to three nested subproof levels. It does not replace production OCR.

`scripts/private_fitch_pilot.py` and
`proofs_ocr/regression/private_fitch_pilot.py` prepare a read-only TF-bundle pilot
without showing protected handwriting or transcriptions to a coding agent:

```powershell
python scripts/private_fitch_pilot.py inventory
python scripts/private_fitch_pilot.py prepare --quiz 6 --samples 20 --output-dir C:\grading-data\ocr-eval\fitch-pilot
python scripts/private_fitch_pilot.py run --limit 1 --output-dir C:\grading-data\ocr-eval\fitch-pilot
python scripts/private_fitch_pilot.py run --output-dir C:\grading-data\ocr-eval\fitch-pilot
python scripts/private_fitch_pilot.py serve --output-dir C:\grading-data\ocr-eval\fitch-pilot --port 5030
```

Preparation uses the bundle filename's quiz number before the saved session's
potentially stale label and reports label conflicts in inventory. `--bundle`
can select an explicit ZIP. Before preparing the sample, IU privately classifies
representative answer-image layouts from each main question, without returning
student text. Only questions verified as TFL proof layouts enter the pilot;
rubric keyword matches alone are insufficient. Preparation then
balances questions/versions before inference, uses at most one response per
student, and excludes extra-credit and explicitly quantified questions from the
initial pilot. Whole proof regions are cropped from the saved normalized boxes
with modest padding; no subproof chopping is performed. The source ZIP is never
written. The prompt receives the selected image plus generic instructions, not
the roster, session grades, old model responses, or rubric answer-key fields.

All snapshots, crops, source-page previews, raw responses, human labels, and
detailed results stay outside the repository. The coding agent may inspect
metadata and `shareable_summary.json` only, not real images or proof text. Test
the review UI using synthetic fixtures and never take a real-student screenshot
for agent inspection. The loopback-only panel uses
`app/templates/private_fitch_pilot.html`; it is separate from grading and does
not change session assignments or grades.

The reviewer can confirm an exact transcription, mark errors for later repair,
edit formulas/citations/line numbers/depths, add or remove lines, and distinguish
adjacent sibling subproofs using **New block**. Unreadable writing and incorrect
crops have separate labels. Counts distinguish technical completion from human
accuracy; model confidence and proof validity never establish OCR correctness.
Assess correctness against the written response, including invalid reasoning.
Corrected labels do not overwrite the original model output. The panel's anonymous
download contains counts and error categories only, suitable for agent feedback.

The first proposed Quiz 7 pilot was invalidated after instructor inspection:
its ZIP was actually labeled Quiz 9, while its saved session still said Quiz 7
and carried obsolete proof-related rubric text. Its outputs were retained as an
invalid source-selection experiment, not proof-OCR accuracy evidence.

The replacement pilot uses an older Quiz 6 bundle with 36 submissions. Its three
main question layouts were independently verified as Fitch proofs by IU's private
categorical layout check. It contains 20 sampled proof regions from 20 distinct
submissions; all 20 model responses passed representation checks. Accuracy remains
unmeasured until the instructor labels them. Keep the remaining submissions for a subsequent holdout rather than tuning
the prompt and reporting accuracy on the same reviewed sample.

For separate IU-backed coding work, `scripts/start_iu_opencode.ps1 -ProjectDir
<coding-worktree>` launches the existing workspace-local OpenCode installation
using `scripts/opencode_iu_glm.json` and the catalog ID `glm-5.3-flash`. The key is
read from the private `REALLMS_API_KEY` environment; no literal key belongs in the
JSON. Use a separate code/fake-fixture worktree to avoid concurrent edits to the
grading workspace. The configuration asks before shell commands, denies external
directory access, and denies direct reads of `.env` files; these agent permission
rules are not an operating-system sandbox. This coding setup does not change the
Gemma model used by the private proof pilot.

## Working notation reference: scope, header, and the sideways T

The instructor's ASCII conventions are in the adjacent
`fitch-protocol-experiment/prompts/fitch_rules_prompt.md` and
`prompts/ascii_to_json_conversion.md`. The latter defines logical depth as the
number of visible vertical bars minus one: the main proof has a visible spine
but logical depth zero. The main vertical stroke begins at the first premise,
continues alongside all premises and the remaining proof, and has its horizontal
header separator below the last premise. A subproof's vertical stroke similarly
includes its hypothesis header, with the separator below that header. A header
containing multiple lines has one separator after its final line.

Use this registered instructor-written example as a shared, non-FERPA reference:
[handwritten Q4 proof](../proofs_ocr/regression/fixtures/tfl/handwritten/q4_N_or_K_and_W/image.png),
[curated JSON](../proofs_ocr/regression/fixtures/tfl/handwritten/q4_N_or_K_and_W/expected_structure.json).
The standardized rule labels below follow the curated JSON; the original scan's
handwritten rule-head glyph is discussed in its semantic notes.

```text
1 | N ∨ (K ∧ W)            Pr
2 | N ↔ W                  Pr
  |----------------------------
3 |  | N                   Hyp
  |  |-------------------------
4 |  | W                   ↔E 2,3
  |
5 |  | K ∧ W               Hyp
  |  |-------------------------
6 |  | W                   ∧E 5
  |
7 | W                      ∨E 1,3-4,5-6
```

The block structure is `main: [line1, line2, [line3, line4],
[line5, line6], line7]`. The two inner arrays are sibling subproofs, both inside
the main proof; the second is not nested in the first.

| Block | Depth | Header lines | Body | Scope interval |
|---|---:|---|---|---|
| Main proof | 0 | 1–2 | First case, second case, line 7 | 1–7 |
| First case | 1 | 3 | 4 | 3–4 |
| Second case | 1 | 5 | 6 | 5–6 |

The flat editor needs `start_subproof` (currently labeled **New block**) because
depth alone loses the boundary between consecutive sibling subproofs:

| Written line | Depth | New block |
|---|---:|---|
| 1 | 0 | No |
| 2 | 0 | No |
| 3 | 1 | Yes |
| 4 | 1 | No |
| 5 | 1 | Yes |
| 6 | 1 | No |
| 7 | 0 | No |

New block means **this row begins a distinct subproof**. It is not the horizontal
stroke itself, does not mark every premise or hypothesis line in a shared header,
and does not imply increased depth: line 5 starts a new sibling at depth 1.
No New block flag is needed for the main proof's premise separator. The nested
JSON arrays already carry this block identity; the flag is an editor bridge.

Two additional safe comparison examples are the
[single conditional subproof](../proofs_ocr/regression/fixtures/tfl/handwritten/q3_P_and_Q_imp_R/image.png)
with header 3 and scope 3–7, and the
[three-premise example](../proofs_ocr/regression/fixtures/tfl/handwritten/fake_neg_intro_error/image.png)
with main header 1–3 and hypothesis scope 4–7.
Logical mistakes in a source example must be preserved; these references explain
notation and scope, not automatic grading or proof repair.

The current pilot editor is not a faithful renderer of these conventions: it
omits the depth-zero spine and puts a new-block border above a hypothesis row.
It also needs a visible break between sibling strokes. These are presentation
limitations; do not infer incorrect OCR nesting solely from those drawn bars.
The intended rendering model is a scope tree whose blocks each have a header
and body. Block identity controls nesting/sibling breaks; the end of the header
controls the horizontal stroke. If literal, misplaced student header separators
must eventually be recorded independently of logical rule labels, that requires
explicit graphical metadata rather than silently repairing the written proof.
