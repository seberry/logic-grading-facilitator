This document defines the core data structures that all parts of the system agree on.
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
max_points: 10

problems:
  - id: "Q1"
    label: "Truth value + witness"
    points: 3

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

Totals are computed as:

```bash
total_score = sum(WorkUnit.per_problem[id] for id in problems)
```

# 3. WorkUnit Schema (v0)

WorkUnits are written to JSON after each grading action.

```json
{
  "unit_id": 7,
  "page_filenames": ["IMG_4024.JPG", "IMG_4025.JPG"],
  "student_id": "12345",

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
- Every key in `per_problem` must correspond to an `id` in the GradingKey.
- `per_problem` values are floats or `null` (`null` means ungraded).
- Missing values are allowed (e.g., ungraded problems) but must be resolved before export.
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
  "problem_types": {
    "Q1": "fitch_proof"
  }
}
```

This file is reloadable at any time.

# 7. Session Extensions (v0.2)

The Session model now includes an optional `pinned_note` field:

```json
{
  "pinned_note": "string or null"
}
```

- This stores the assignment's answer key or grading notes.
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
      extra_prompt: "The student answer should mainly be a model, not a prose explanation."
```

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

### 10.6 Proposed shape for `fitch_proof`

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
