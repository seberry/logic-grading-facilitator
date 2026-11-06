This document defines the core data structures that all parts of the system agree on.

## Incremental question-location preparation

Instructor preparation checkpoints each problem under
`data/grades/segments_<problem_id>.partial_<revision>.json`. This is a JSON
object keyed by string work-unit ID; each value uses the same location row shape
as the final `segments_<problem_id>.json` list, including `unit_id`, `present`,
`bbox`, and recognition model/provider. The revision is a hash of the current
question description and work-unit scan file metadata. Partial files are
private preparation state and are never included in a TF bundle. Once all
work units have valid rows, the app writes the standard full list cache.

## Quiz 9 FOL truth assistance (schema 1)

`grading_key.fol_truth_assistance` is absent for ordinary quizzes. The Quiz 9
opt-in shape is `{ "schema": 1, "assessment": "quiz9_monadic_fol_truth" }`.
The optional `scoring` map has per-question `truth` and, for Q2–5, `object`
points plus `blank_na_credit` (false in the approved Quiz 9 rubric). Each
rule's parts must sum to that question's `points`.
`work_units[].fol_truth_reviews` maps question IDs to records with optional
`proposed_truth`, `proposed_object`, `proposed_raw_object`, `confirmed_truth`, `confirmed_object`,
`raw_object`, and `fingerprint`. Proposed readings are distinct from human
confirmations. Truth readings use `true`, `false`, `blank`, or `unclear`;
written answers use `1`, `2`, `3`, `na`, `blank`, or `unclear`. Derived verdicts
and draft feedback are recalculated, never authoritative persisted grades.
The ordinary `per_problem` score and `per_problem_comments` contain the human
result. Opted-in TF manifests declare schema 1, required grader feature
`quiz9_fol_truth_assistance_v1`, and upstream module commit. Legacy
answer-key fields remain stripped.
All examples are canonical; when writing new code, verify output conforms to these formats.

---

# 1. Roster Schema

Derived from Canvas gradebook CSV.

```json
[
  {
    "student_id": "12345",
    "name": "Berry, Sharon"
  },
  {
    "student_id": "67890",
    "name": "Johnson, Alex"
  }
]
```

Invariants:

- `student_id` must be unique.
- Names are stored exactly as they appear in Canvas (for import matching).
- The fuzzy-search UI matches on name but canonicalization happens via `student_id`.

# 2. Grading Key (YAML)

Example (`data/rubric/quiz3.yaml`):

```yaml
quiz_id: "quiz3"

problems:
  - id: "Q1"
    label: "Truth value + witness"
    points: 3
    segmentation_hint: "Include the Q1 label and answer area."
    variants:
      A:
        prompt: "Version A printed prompt."
        question_image: "quiz3/Q1_A.png"
        rubric_notes: "Human grader chooses partial credit."
      B:
        prompt: "Version B printed prompt."

  - id: "Q2"
    label: "Counterexample"
    points: 4

  - id: "Q3"
    label: "Explanation"
    points: 3
```

Invariants:

- `id` values must be unique.
- All per-problem score entries in WorkUnits must refer to these ids exactly.
- The problem `label` is also used as a natural-language hint when the model searches for the answer region in strip view, so match the wording on the exam if possible.
- `variants` is optional. Keys are normalized printed version identifiers such as
  `A`, `B`, or `C`; each value may override `label`, `prompt`,
  `question_image`, `rubric_notes`, and `segmentation_hint`.
- `question_image` is an optional relative path below
  `DATA_DIR/rubric_assets/`. It must be a crop of the blank assessment, not a
  student scan or solution. If omitted, the strip page checks the conventional
  path `<quiz_id>/<problem_id>_<version>.png` and otherwise uses prompt text.
- `answer_key` remains accepted only for backward compatibility. Strip grading
  does not display it, the PDF-to-YAML prompt forbids generating it, and TF
  bundle export removes it.

Totals are computed as:

```text
regular_max = sum(problem.points for non-extra-credit problems)
regular_score = sum(scores for non-extra-credit problems)
extra_credit_score = sum(scores for extra-credit problems)
total_score = min(regular_max, regular_score + extra_credit_score)
```

Thus extra credit can restore points missed on regular questions, but cannot
raise the exported quiz score above the regular maximum. There is no separate
quiz-level maximum: total/export code derives the authoritative maximum from
the non-extra-credit problem definitions. Older rubric files may still contain
`max_points`; it is ignored when loaded and omitted the next time the rubric or
session is saved.

# 3. WorkUnit Schema (v0)

WorkUnits are written to JSON after each grading action.

```json
{
  "unit_id": 7,
  "page_filenames": ["IMG_4024.JPG", "IMG_4025.JPG"],
  "student_id": "12345",
  "quiz_version": "B",

  "per_problem": {
    "Q1": 3,
    "Q2": 2,
    "Q3": 3
  },

  "per_problem_comments": {
    "Q1": "missing witness",
    "Q2": null,
    "Q3": null
  },

  "comment": "missing witness in Q2"
}
```

Invariants:

- `page_filenames` lists raw images in display order.
- Exactly one `student_id` per WorkUnit.
- `quiz_version` is optional for old/single-version sessions. Versioned handoffs
  should store a normalized value matching a rubric `variants` key.
- Every key in `per_problem` must correspond to an `id` in the GradingKey.
- `per_problem` values are floats or `null` (`null` means ungraded).
- Missing values are allowed (e.g., ungraded problems) but must be resolved before export.
- Final CSV/PDF and gradebook merge require explicit scores on regular problems
  in every identified WorkUnit, including retained duplicate submissions. Blank
  optional EC counts as zero. Ungraded sessions may still be exported as TF bundles.
- `per_problem_comments` keys mirror `per_problem`; values are strings or `null`.

# 4. WorkUnit Schema (future segmentation)

After OCR segmentation:

```json
{
  "unit_id": 7,
  "page_filenames": ["IMG_4024.JPG", "IMG_4025.JPG"],
  "student_id": "12345",

  "answers": [
    {
      "problem_id": "Q1",
      "page": 1,
      "bbox": [100, 400, 320, 800],
      "text": "student's OCR'd answer to Q1..."
    }
  ],

  "per_problem": {
    "Q1": {
      "human_score": 3,
      "auto_score": 3,
      "flag_disagreement": false
    },
    "Q2": {...}
  }
}
```

Invariants:

- `answers[].problem_id` must match the GradingKey.
- `answers[].bbox` is `[x1, y1, x2, y2]` in pixel coordinates.
- Future modules may populate or modify `text`, `auto_score`, etc.
- Human scores remain canonical for exported grades.

# 5. Export CSV Schema (Canvas)

The final CSV looks like:

```csv
student_id,quiz3_score
12345,8
67890,9.5
```

If integrating with full Canvas gradebook:

- A merge step inserts a `quiz3_score` column into a Canvas-exported gradebook CSV.
- Canvas then imports the updated CSV without changing other columns.

## 5.1 Related Export: Renamed Scans (ZIP)

The HTML UI provides a **"Download renamed scans (ZIP)"** action.

This endpoint:

- Reads the current `Session`:
  - `grading_key.quiz_id`
  - `work_units[*].student_id`
  - `work_units[*].page_filenames` (in display order)
- Builds a ZIP archive containing **all** pages that belong to some WorkUnit
  with a non-null `student_id`.
- For each page, constructs a canonical filename, e.g.:

```text
<quiz_id>_<student_id>_p<page_index>.<ext>
# examples:
#   quiz1_12345_p01.jpg
#   quiz1_12345_p02.jpg
```

Invariants:

- Page order within a WorkUnit is preserved (`page_index` is 1-based in the order given by `WorkUnit.page_filenames`).
- Only WorkUnits with a `student_id` are included; unassigned scans are skipped.
- The original images in `data/raw_scans/` are unchanged; the renaming happens only in the ZIP archive.

This export is intended for:

- handback (e.g., uploading named scans to another system),
- offline archival of per-student scan bundles,
- training/evaluation datasets for future OCR and local-model experiments.

# 6. Session File Format (optional)

Incremental grading sessions may be stored as:

```json
{
  "roster": [...],
  "grading_key": {...},
  "work_units": [...],
  "unclaimed_pages": ["IMG_4031.JPG", "IMG_4032.JPG"],
  "pinned_note": null,
  "problem_notes": {
    "Q1": "Truth table for ¬(P→Q)"
  },
  "reference_pins": {
    "Q1": {"A": 7, "B": 12}
  },
  "problem_types": {
    "Q1": "fitch_proof"
  }
}
```

This file is reloadable at any time.

`reference_pins` maps each problem and normalized printed version to the
`unit_id` of a human-selected visual comparison response. Pins are not
machine-verified answer keys. A pin is discarded if its WorkUnit is later
reassigned to a different version.

# 7. Session Extensions (v0.2)

The Session model now includes an optional `pinned_note` field:

```json
{
  "pinned_note": "string or null"
}
```

- This is a legacy session-wide grading-note field. New strip grading should use
  neutral `problem_notes` plus human-selected, version-aware `reference_pins`.
- TF bundle export clears this field so it cannot carry a supplied solution key.
- Shared across all WorkUnits.
- Preserved when creating archival snapshots.
- `null` is stored instead of an empty string for backwards compatibility.

# 8. Session Extensions (v0.3)

`problem_types`:

```json
{
  "problem_types": {
    "Q1": "fitch_proof",
    "Q2": "modal_proof"
  }
}
```

Stores per-problem type tags used for future autograde workflows (e.g., Fitch proofs).
These can be set from the Problem Strips UI via the "Problem type wheel."

# 9. TFL/Fitch OCR Review State (current)

For `fitch_proof` strip grading, asynchronous OCR review state is stored outside
`session.json` so scores remain human-authored and the proof OCR queue can be
refreshed independently.

Path:

```text
data/grades/proof_ocr/tfl_fitch_<problem_id>.json
```

Example:

```json
{
  "problem_id": "Q5",
  "kind": "tfl_fitch",
  "updated_at": "2026-05-14T12:00:00+00:00",
  "items": {
    "7": {
      "unit_id": 7,
      "page_filename": "quiz1_student_p01.jpg",
      "slug": "q5_7_quiz1_student_p01",
      "status": "verifier_issue",
      "message": "OCR ready with verifier/reread flags.",
      "debug_url": "/debug/fitch?scan=q5_7_quiz1_student_p01",
      "proof_image_url": "/debug_fitch_assets/q5_7_quiz1_student_p01/q5_7_quiz1_student_p01.jpg",
      "run_summary": "Fitch OCR pipeline: 2 bars, 7 lines, 2 premises, geometry=standard",
      "line_count": 7,
      "preview_lines": [
        {"line": 1, "depth": 0, "wffstr": "A", "jstr": "Pr"},
        {"line": 2, "depth": 0, "wffstr": "B", "jstr": "Pr"}
      ],
      "checker_issues": ["Line 5: Rule does not apply."],
      "reread_summary": {
        "record_count": 1,
        "candidate_class_counts": {"changed_justification": 1},
        "gold_metrics_available": false
      },
      "reread_records": [
        {
          "line": 5,
          "target_kind": "rejected",
          "candidate_class": "changed_justification",
          "reason": "Candidate changes the normalized justification.",
          "original": {"wffstr": "C", "jstr": "Hyp"},
          "candidate": {"wffstr": "C", "jstr": "→E 1,4"},
          "row_crop_url": "/debug_fitch_assets/q5_7_quiz1_student_p01/verifier_line_rereads_27b/rejected_line_05_node_2_row_3.jpg"
        }
      ]
    }
  }
}
```

Invariants:

- `items` is keyed by `WorkUnit.unit_id` as a string.
- Valid `status` values are `queued`, `processing`, `needs_review`,
  `verifier_issue`, `accepted`, and `ocr_failed`.
- `debug_url` and asset URLs are local web URLs only; do not write absolute
  live-data paths into tracked docs or fixtures.
- `accepted` records human confirmation of the displayed OCR result. It does
  not update `WorkUnit.per_problem` or apply reread candidates.
- The canonical proof artifacts remain under `data/debug_fitch/<slug>/`,
  including `final_structure.json`, `debug_bars.jpg`, VLM responses, and
  verifier reread files.

## 9.1 Private TFL/Fitch OCR Evaluation Manifest

Real student proof crops and human-checked gold structures must stay outside the
repo. The private runner uses an external YAML manifest with anonymous case ids:

```yaml
suite:
  name: "exam1_spring2026_private_tfl"

cases:
  - id: "exam1_q6_case_0001"
    split: "dev_regression"
    image_path: "crops/q6/exam1_q6_case_0001.jpg"
    expected_structure_path: "gold/q6/exam1_q6_case_0001.json"
    gold_status: "human_checked"
    tags: ["student_scan", "tfl", "nested_subproofs"]
```

Path rules:

- `image_path` and `expected_structure_path` may be absolute, or relative to the
  manifest file.
- The manifest, crops, gold files, and run outputs should live under an external
  private folder such as `C:\grading-data\ocr-eval\...`.
- Default CLI behavior refuses private paths inside the repo. Use
  `--allow-repo-paths` only for fake tests.

Gold structure rules:

- `expected_structure_path` may contain either the plain nested proof list or the
  debug-page shape `{ "lines": [...], "meta": {...} }`.
- Gold should transcribe what is visible. Do not infer clipped or unreadable
  citation endpoints just to make a proof verify.
- Use `gold_status: human_checked` only after manual review. Candidate OCR output
  that still needs review should use a different status such as
  `needs_human_check`.

Split rules:

- `discovery`: cases you actively inspect while developing.
- `dev_regression`: recurring private regression cases.
- `blind_holdout`: cases not used for debugging; report aggregate results only.

The private runner writes detailed local artifacts under the selected external
run directory, including diffs and per-line records. It also writes
`shareable_summary.txt` and `shareable_summary.json`, which are designed to omit
paths, images, line text, diffs, and timestamps so their aggregate results can be
reported back safely.

## 9.2 FOL Model OCR Review State

For `fol_model` strip grading, inline OCR/checker review state is stored outside
`session.json` so the grader can edit proposed model text per strip without
turning checker suggestions into authoritative grades.

Path:

```text
data/grades/fol_model_ocr/fol_model_<problem_id>.json
```

Example:

```json
{
  "problem_id": "Q3",
  "kind": "fol_model",
  "updated_at": "2026-05-15T12:00:00+00:00",
  "items": {
    "7": {
      "unit_id": 7,
      "page_filename": "quiz1_student_p01.jpg",
      "slug": "q3_7_quiz1_student_p01",
      "status": "checker_passed",
      "message": "Checker passed.",
      "debug_url": "/debug/fol-model?scan=q3_7_quiz1_student_p01",
      "image_url": "/debug_fol_model_assets/q3_7_quiz1_student_p01/q3_7_quiz1_student_p01.jpg",
      "model_text": "Domain: 1,2 P: 1",
      "requirement_text": "Need: the two target sentences have different truth values.",
      "passed": true,
      "summary": "The two formulas differ in truth value in this model, so they are not logically equivalent.",
      "suggested_score": 3.0,
      "suggested_comment": "Question: FOL model\n...",
      "trace_blocks": [
        {
          "title": "Sentence 1: \u2200x Px",
          "result": "False",
          "body": "\u2200xPx\nx=1: P1 = TRUE\n..."
        }
      ]
    }
  }
}
```

Valid statuses:

- `not_started`: strip panel placeholder only; no persisted OCR/checker result required
- `queued`, `processing`: OCR job state
- `checker_passed`, `checker_failed`: deterministic checker ran
- `invalid_input`: checker could not parse or evaluate the proposed model
- `ocr_failed`: local model or crop preparation failed

The normal `WorkUnit.per_problem` score and `per_problem_comments` remain the
authoritative grading fields. The FOL panel's score/comment are suggestions
until the grader applies or copies them into the regular strip score/comment
controls.

## 10. Proposed Rubric Extension (draft)

This section documents a proposed rubric/schema change for future autograde workflows.
It is a design target, not yet the current implementation.

### 10.1 Motivation

Right now `ProblemSpec` is minimal:

```yaml
- id: "Q1"
  label: "Problem 1"
  points: 5
```

The current app stores per-problem type tags separately in session state (`problem_types`),
and the local model only receives the problem `id` and `label` as segmentation hints.

For richer workflows such as:

- Fitch proof OCR/checking
- FOL model tasks
- future Kripke model tasks

the rubric needs a place to store:

- the broad problem family (`problem_type`)
- task-specific fixed logical inputs (`prompt_data`)
- OCR/perception hints for the local model (`ocr`)

### 10.2 Proposed `ProblemSpec` extension

Add these optional fields:

```yaml
- id: "Q3"
  label: "Give a model showing these formulas are not logically equivalent"
  points: 4
  extra_credit: false
  problem_type: "fol_model"
  autograde: {}
```

Conceptually:

- `problem_type`: broad family such as `plain`, `fitch_proof`, `fol_model`
- `autograde`: family-specific metadata preserved in the rubric

### 10.2a Implementation note

The shared autograde scaffolding now exists in `app/autograde/`, but only `fol_model`
is currently wired through a real deterministic engine. The presence of other engine
modules should be read as reserved extension points, not as completed grading flows.

### 10.3 Proposed shape for `fol_model`

Use one broad family plus a task-kind inside `autograde`.

```yaml
- id: "Q3"
  label: "Give a model showing these formulas are not logically equivalent"
  points: 4
  problem_type: "fol_model"
  autograde:
    task_kind: "non_equivalence"
    default_score_correct: 4
    default_score_incorrect: 0
    prompt_data:
      formulas:
        - "∀x Px"
        - "∃x Px"
    ocr:
      expected_answer_shape: "model_only"
      constants: []
      predicates: ["P"]
      extra_prompt: "The student answer should mainly be a model, not a prose explanation."
```

The FOL-model OCR prompt infers expected constants and predicates from
`prompt_data.formulas` by default. Use `ocr.constants` and `ocr.predicates` only
when the rubric needs to pin the OCR vocabulary more explicitly. This prevents
the OCR pass from seeding default names for problems whose target sentences use
only quantified variables.

Recommended `task_kind` values:

- `truth_in_given_model`
- `true_false_in_given_model`
- `make_true`
- `make_false`
- `joint_consistency`
- `non_equivalence`
- `non_validity`
- maybe later `non_logical_truth`
- maybe later `non_logical_falsehood`

### 10.4 Why not `left` / `right`?

For binary tasks such as non-equivalence, prefer:

```yaml
prompt_data:
  formulas:
    - "∀x Px"
    - "∃x Px"
```

instead of:

```yaml
prompt_data:
  left: "∀x Px"
  right: "∃x Px"
```

Using `formulas` avoids building order assumptions into OCR, trace wording,
or the student-facing interpretation of the task.

### 10.5 Example task shapes

Truth in a fixed model:

```yaml
problem_type: "fol_model"
autograde:
  task_kind: "true_false_in_given_model"
  prompt_data:
    given_model_text: |
      DOMAIN: {1,2}
      CONSTANTS:
      a: 1
      PREDICATES:
      P: {1}
    formulas:
      - "∀x Px"
  ocr:
    expected_answer_shape: "truth_value_only"
```

Make a sentence true:

```yaml
problem_type: "fol_model"
autograde:
  task_kind: "make_true"
  prompt_data:
    formulas:
      - "Pa"
  ocr:
    expected_answer_shape: "model_only"
```

Joint consistency:

```yaml
problem_type: "fol_model"
autograde:
  task_kind: "joint_consistency"
  prompt_data:
    formulas:
      - "∃x Px"
      - "∃x Qx"
      - "∀x (Px → ¬Qx)"
```

Non-validity:

```yaml
problem_type: "fol_model"
autograde:
  task_kind: "non_validity"
  prompt_data:
    premises:
      - "∀x (Px → Qx)"
      - "∃x Px"
    conclusion: "∀x Qx"
```

### 10.6 Initial shape for `kripke_model`

Use `kripke_model` when the answer is a Kripke model/countermodel rather than a
FOL structure. The initial checker accepts already-transcribed model text; OCR
for drawn graph answers is still a separate future step.

Canonical model text:

```text
W: {0,1}
R: <0,1>, <1,1>
P: {1}
Q: {}
```

The parser also accepts `w0` labels in `W`/`R` lines and worksheet-style
valuation entries such as `ν: ν0(P) = F, ν1(P) = T`.

Non-entailment / non-validity countermodel:

```yaml
problem_type: "kripke_model"
autograde:
  task_kind: "non_entailment"
  default_score_correct: 4
  default_score_incorrect: 0
  prompt_data:
    world: 0
    premises:
      - "<>P"
    conclusion: "[]Q"
```

Worksheet-style constraints can be specified directly:

```yaml
problem_type: "kripke_model"
autograde:
  task_kind: "different"
  prompt_data:
    constraints:
      - type: "different"
        world: 0
        formulaA: "<>P"
        formulaB: "[]P"
      - type: "relation"
        world: 0
        property: "reflexive"
        expected: true
```

Supported `task_kind` values in the first checker pass:

- `truth`, `truth_value`, `make_true`
- `make_false`
- `different`, `non_equivalence`
- `non_entailment`, `non_validity`

Supported formula syntax includes `[]`, `<>`, `~`, `&`, `|`, `->`, and the
Unicode variants `□`, `◇`, `¬`, `∧`, `∨`, `→`.

### 10.7 Proposed shape for `fitch_proof`

Do not force Fitch proofs into the same low-level structure as FOL model tasks.
Only share the top-level `problem_type` / `autograde` pattern.

```yaml
- id: "Q5"
  label: "Fitch proof"
  points: 6
  problem_type: "fitch_proof"
  autograde:
    conclusion: "∀x Px"
    premise_count: 2
    ocr:
      expected_answer_shape: "fitch_proof"
      pipeline: "fitch_debug"
```

### 10.7 Intended division of responsibility

For these richer workflows:

- the rubric supplies fixed logical inputs and OCR hints
- the student answer supplies the variable content to interpret and check

For example:

- in `make_true`, the rubric supplies the target sentence; OCR supplies the model
- in `true_false_in_given_model`, the rubric supplies the model and sentence; OCR supplies only the student's verdict
- in `non_validity`, the rubric supplies premises and conclusion; OCR supplies the countermodel

This separation is intentional and should guide future route, prompt, and checker design.

# 11. Model provenance and private name benchmarks

New name-prediction cache entries and question-location entries may include
`recognition_model` (the requested model ID) and `recognition_provider`
(`reallms` or `lmstudio`). Older entries without those fields remain valid.
These fields do not change human-reviewed student assignments or crop geometry.

Private name benchmarks must live outside the repository. `inputs.json` stores
the roster, original local predictions, per-case scan hashes, isolated scan
filenames, and optional saved-session references matched by image bytes.
`results.json` stores `(case_id, model, status, prediction, elapsed_seconds,
responses)` records. `comparison.json` supplies the local review panel.
`reviews.json` maps case IDs to `{student_id: string | null}`; an absent key
means unreviewed and explicit null means the reviewer could not identify a name.
Saved session assignments are provisional references, not automatically accepted
human labels. `summary.json` and the panel's downloaded counts contain aggregate
metrics only. All other benchmark files, including thumbnails, are protected.

# 12. Private whole-proof review pilot

These files live in an external pilot directory and never enter a TF grading
bundle or the repository. `manifest.json` freezes the source ZIP hash, sampling
seed/policy, prompt/schema hashes, and anonymous cases with their original image
member, normalized crop bounds, and image/crop hashes. No roster or grades are
copied into the model request. `results/<case>.json` stores immutable original
`status`, nested `proof`, `raw_response`, and model `metadata` on success, or a
safe error type on failure. The proof shape is the existing nested list of
`{line, wffstr, jstr}` objects.

`reviews.json` maps case IDs to statuses `exact`, `corrected`, `needs_correction`,
`unreadable`, or `bad_crop`; absence means unreviewed. Exact/corrected records
include the human-approved `proof`; needs-correction records can include a list
of error-category strings. Gold files are written separately for exact/corrected
reviews. Review status is authoritative: a historical gold file after clearing
or excluding a review must not be treated as currently accepted gold.

`shareable_summary.json` contains counts only. The accuracy denominator is
`exact + corrected + needs_correction`; unreadable cases, bad crops, pending
requests, and request failures are reported separately. Error-category counts
describe proofs affected, may overlap, and are not character-level accuracy.
The editor uses `depth` plus `start_subproof` at a sibling block's first row to
round-trip nested arrays without merging adjacent subproofs. Those UI fields
are not additional keys in the persisted proof-line schema.
