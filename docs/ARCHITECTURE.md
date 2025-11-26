# System Architecture Overview

This document describes the high-level architecture of the Logic Grading Facilitator as of v0, with explicit notes about where and how future OCR/autograding modules will plug in.

---

# 1. Core Concept: WorkUnits

A **WorkUnit** is the central abstraction of the entire system.

It represents:

- one student submission,
- containing one or more scanned pages,
- with per-problem scores and optional comments,
- keyed to a unique student ID from the Canvas roster.

All downstream workflows (Canvas export, per-question views, OCR segmentation, autograding) operate on `WorkUnit`.

---

# 2. Backend Model

The backend service (Flask or FastAPI) maintains:

- A list of **unclaimed pages** (filenames) from `data/raw_scans/`.
- A growing list of **WorkUnits**, each with:
  ```json
  {
    "unit_id": 3,
    "page_filenames": ["IMG_4027.JPG", "IMG_4028.JPG"],
    "student_id": "12345",
    "per_problem": {
      "Q1": 3,
      "Q2": 2,
      "Q3": 3
    },
    "comment": "missing witness in Q2"
  }
A Roster, derived from the Canvas gradebook CSV:

[
  {"student_id": "12345", "name": "Berry, Sharon"},
  {"student_id": "67890", "name": "Johnson, Alex"}
]


A GradingKey, loaded from YAML (see docs/SCHEMA.md).

The backend persists incremental grading in data/grades/session.json (or multiple session files).

# 3. UI Flow (v0)

The frontend (basic HTML/JS or a light framework) interacts with the backend via endpoints:

Page: /grade/<unit_id>

Displays:

All pages in the WorkUnit (stacked or side-by-side).

Name picker with fuzzy search.

Per-problem score inputs (based on GradingKey).

Optional comment textbox.

Buttons:

Append next scan

Remove last scan

Save & Next

Page: /export

Backend synthesizes quizX_scores.csv.

Output conforms to Canvas import format (student_id → score).

# 3.6 Pinned Notes (Session-wide)

The grading UI includes a pinned note editor for storing the answer key or general grading instructions.

Stored in Session.pinned_note

Updated via the update_pinned_note action in /grade/{unit_id}/page

Displayed for all WorkUnits

Saved inside archival snapshots

## 3.7 Renamed scans export

Once you have:

- run `scripts/init_session.py`,
- created WorkUnits from `data/raw_scans/`,
- and assigned `student_id` values in the grading UI,

the HTML UI exposes a **“Download renamed scans (ZIP)”** control.

Conceptually:

- For each `WorkUnit` with a `student_id`, the backend walks `page_filenames`
  in order and computes a canonical filename using:

  - `Session.grading_key.quiz_id`
  - `WorkUnit.student_id`
  - page index within that WorkUnit

- These files are written into a ZIP archive grouped by student (flat namespace
  by default; folder-per-student is a possible future extension).

This feature is deliberately **purely derived** from the `Session` data model:

- No additional metadata is stored in `session.json`.
- The export can be recomputed at any time as long as the underlying scans and
  session file exist.
- It provides a clean on-ramp for future OCR / clustering experiments by
  giving you a labeled, per-student image dataset.

# 4. Multi-Page Support

The UI allows construction of multi-page WorkUnits with:

Append next scan

Remove last appended scan

The backend updates:

WorkUnit.page_filenames

unclaimed_pages

This approach avoids brittle automatic inference while remaining fast.

# 5. Future Automation Integration Points

Although v0 includes no OCR or autograding, the architecture is explicitly prepared for:

## 5.1 OCR-based segmentation

After pages are part of a WorkUnit, a segmentation module may:

run OCR,

detect headings (Q1, 1., (a)),

infer bounding boxes per problem,

store:

"answers": [
  { "problem_id": "Q1", "page": 1, "bbox": [...], "text": "..." }
]

## 5.2 Per-question grading modes

UI mode:

/grade-by-problem/<problem_id>

Show all answer regions for that problem

Enter one score per answer or per cluster

## Per-Problem Strip View (Segmentation UI)

Route: /problem/{problem_id}/strips

For each WorkUnit:

Runs segment_page_for_problem on its pages using SINGLE_QUESTION_PROMPT.

Picks the first present=True segment.

Renders the underlying scan with an overlaid bbox.

Shows a score field for that problem_id.

This is the main UI for grading a single problem across all students.

## 5.3 Clustering

Embedding + clustering for similar answers

UI provides drag-and-drop reclassification

Cluster-level scoring

5.4 Autograding

Modules may plug into:

symbolization equivalence

truth-value checking

Fitch-proof validation

LLM second-opinion classifier

Stored alongside human score:

```
{
  "human_score": 3,
  "auto_score": 3,
  "flag_disagreement": false
}
```
# 6. FERPA/Compliance Design Constraints

Never auto-upload scans to Canvas.

Feedback/comments may be auto-uploaded safely.

Human scores override autograding unless explicitly approved.


# 7. Assignment Reset & Archival (v0.2)

The system includes an explicit end-of-assignment archival flow:

The /finalize endpoint triggers a helper in app/storage.py that:

Copies the entire data/ directory into
data_archive/<timestamp>_<label>/

Writes a metadata.json containing the label and timestamp

Clears:

data/raw_scans/

data/grades/session.json

Leaves roster.csv and rubric/ untouched

This preserves a complete snapshot of the grading state while preparing the system for the next assignment.

After archival, scripts/init_session.py must be run before new grading begins.

# 7. Summary

The architecture is intentionally minimal but designed so each future phase (segmentation, clustering, autograding) snaps cleanly into place without refactoring core concepts. Everything revolves around the WorkUnit abstraction and the GradingKey specification.


## Components

- **Backend (FastAPI)**  
  Orchestrates ingestion, VLM calls, logic checks, and serves the grading UI.

- **Database (SQLite)**  
  Stores roster, WorkUnits (submissions), per-question segments, transcriptions, and logic check results. This replaces earlier flat JSON as the primary source of truth.

- **AI Service (Local VLM)**  
  A local HTTP endpoint (e.g. LM Studio running Qwen-VL) that:
  - Takes page images and prompts.
  - Returns JSON (presence flags, bounding boxes, draft transcriptions).

- **Logic Engine**  
  Pure-Python deterministic modules, e.g.:
  - `kripke_checker.py`: evaluate frame properties, sentence truth in models.
  - `proof_checker.py`: verify TFL/FOL proof lines given a structured representation.

- **Frontend (Strip View UI)**  
  A web UI optimized for grading **one question across all students**. Each “strip” for a student shows:
  1. A crop of the handwritten answer with tools for manual re-cropping.
  2. A machine interpretation + visualization (e.g. graph of the Kripke model, structured proof steps with pass/fail indicators).
  3. Logic check results, feedback selection, and score entry.


  # Repository Layout (Code vs Data)

The repo is intentionally split into:

- `app/` – core application code
  - `server.py` – FastAPI app entrypoint and routes
  - `models.py` – domain models (Session, WorkUnit, GradingKey, etc.)
  - `storage.py` – load/save helpers for sessions, rosters, rubrics
  - `auto_group_core.py` – VLM-driven name recognition + grouping logic
  - `vlm_client.py` – LM Studio + Qwen client (stateless, OpenAI-style)
  - `vlm_classifier.py` – page-level name classifier using `NAME_SYSTEM_PROMPT`
  - `segmentation_core.py` – per-question segmentation helper
  - `templates/` – Jinja2 templates:
    - `grade.html` – per-WorkUnit grading UI (old v0 UI, still used)
    - `name_review.html` – page-level name review / approval UI
    - `problem_strips.html` – per-problem “one strip per student” grading UI

- `scripts/` – CLI utilities for offline / batch operations
  - `init_session.py` – initialize `data/` for a new quiz
  - `auto_group_scans.py` – run VLM name recognition + auto-group scans
  - `preview_page_predictions.py` – print page-level name predictions
  - `debug_segment_view.py` – playground for single-question segmentation

- `docs/` – design docs (no student data)
  - `ARCHITECTURE.md` – this document
  - `DESIGN_v2.md` – more detailed UI/flow design
  - `SCHEMA.md` – JSON schema for sessions / work units
  - `FUTURE_AUTOMATION.md` – roadmap for more automation

- `data/` – **live quiz data** (FERPA-sensitive)
  - `roster.csv` – canonical roster for the *current* run
  - `rubric/quiz1.yaml` – rubric / grading key (OK to share)
  - `raw_scans/` – raw scanned PDFs/JPGs from the current quiz
  - `auto_grouped/` – model outputs for auto-grouping (predictions + renamed scans)
  - `grades/session.json` – current grading session data
  - `name_review_cache/` – cached VLM predictions for `/name_review`

- `data_archive/` – snapshots of previous runs
  - preserved for reproducibility, but should never be committed to a public repo

- `debug_output/`, `debug_output_segment_all/` – local debug artifacts and overlays
