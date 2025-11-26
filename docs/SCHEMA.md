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

student_id must be unique.

Names are stored exactly as they appear in Canvas (for import matching).

The fuzzy-search UI matches on name but canonicalization happens via student_id.

# 2. Grading Key (YAML)
Example (data/rubric/quiz3.yaml):

yaml
Copy code
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
Invariants:

id values must be unique.

All per-problem score entries in WorkUnits must refer to these ids exactly.

Totals are computed as:

bash
Copy code
total_score = sum(WorkUnit.per_problem[id] for id in problems)


3. WorkUnit Schema (v0)
WorkUnits are written to JSON after each grading action.

json
Copy code
{
  "unit_id": 7,
  "page_filenames": ["IMG_4024.JPG", "IMG_4025.JPG"],
  "student_id": "12345",

  "per_problem": {
    "Q1": 3,
    "Q2": 2,
    "Q3": 3
  },

  "comment": "missing witness in Q2"
}
Invariants:

page_filenames lists raw images in display order.

Exactly one student_id per WorkUnit.

Every key in per_problem must correspond to an id in the GradingKey.

Missing values are allowed (e.g., ungraded problems) but must be resolved before export.

4. WorkUnit Schema (future segmentation)
After OCR segmentation:

json
Copy code
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

Invariants:

answers[].problem_id must match the GradingKey.

answers[].bbox is [x1, y1, x2, y2] in pixel coordinates.

Future modules may populate or modify text, auto_score, etc.
Human scores remain canonical for exported grades.

5. Export CSV Schema (Canvas)
The final CSV looks like:

csv
Copy code
student_id,quiz3_score
12345,8
67890,9.5
If integrating with full Canvas gradebook:

A merge step inserts a quiz3_score column into a Canvas-exported gradebook CSV.

Canvas then imports the updated CSV without changing other columns.


## 5.1 Related Export: Renamed Scans (ZIP)

The HTML UI provides a **“Download renamed scans (ZIP)”** action.

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

Page order within a WorkUnit is preserved (page_index is 1-based in the order
given by WorkUnit.page_filenames).

Only WorkUnits with a student_id are included; unassigned scans are skipped.

The original images in data/raw_scans/ are unchanged; the renaming happens
only in the ZIP archive.

This export is intended for:

handback (e.g., uploading named scans to another system),

offline archival of per-student scan bundles,

training/evaluation datasets for future OCR and local-model experiments

# 6. Session File Format (optional)
Incremental grading sessions may be stored as:

json
Copy code
{
  "roster": [...],
  "grading_key": {...},
  "work_units": [...],
  "unclaimed_pages": ["IMG_4031.JPG", "IMG_4032.JPG"]
}
This file is reloadable at any time.


7. Session Extensions (v0.2)

The Session model now includes an optional pinned_note field:

{
  "pinned_note": "string or null"
}


This stores the assignment’s answer key or grading notes.

Shared across all WorkUnits.

Preserved when creating archival snapshots.

Null is stored instead of an empty string for backwards compatibility.