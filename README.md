# Logic Grading Facilitator

A lightweight local tool for grading scanned student submissions with minimal friction, clean data structures, and a clear path toward future automation (OCR, clustering, autograding). Designed to eliminate the mechanical overhead of: matching scans to names, managing multi-page submissions, entering per-problem scores, and exporting grades back to Canvas.

This project is built around a simple core abstraction:

> **WorkUnit** = one student submission, possibly spanning multiple scanned pages, with per-problem scores and metadata.

The goal is to make **v0 grading** extremely easy while guaranteeing that the data structures and UI flow will support **v1–v4 automation** without refactoring.

---

## High-Level Features (v0)

- Local browser-based UI (Flask/FastAPI backend).
- Navigate through raw scans and assemble them into **WorkUnits**.
- Handle multi-page submissions interactively:
  - “Append next scan”
  - “Remove last appended scan”
- Fast **name assignment**:
  - fuzzy-search name field (type 2–4 chars → auto-filtered list)
  - “same as previous submission” hotkey for multi-page cases
- Structured **per-problem grading** driven by a YAML grading key.
- Persist incremental grading to disk (JSON/CSV).
- Export a Canvas-importable CSV mapping student IDs → scores.

---

## Future Extensions (planned, not required for v0)

See `docs/FUTURE_AUTOMATION.md` for details, but the major phases include:

- OCR-based detection of problem headings in handwritten scans.
- Automated answer segmentation into per-problem regions.
- Per-question grading views.
- Clustering/grouping of similar answers for consistent partial credit.
- Deterministic autograding (symbolization equivalence checker, truth-table evaluator, Fitch checker).
- LLM-based “second-opinion” disagreement flags.
- Optional programmatic upload of feedback comments to Canvas (safe; scans are **never** uploaded automatically).

---

## Repository Structure (initial draft)

```text
logic-grading-facilitator/
│
├── app/ # backend & UI (to be implemented)
│
├── data/
│   ├── raw_scans/         # current assignment’s scans
│   ├── grades/            # current session.json
│   ├── rubric/            # grading keys (YAML)
│   ├── roster.csv
│   └── schema.md
├── data_archive/          # snapshots created by Finalize & Archive
│   └── 2025-11-16T14-05-23_quiz1/
│       ├── data/          # full snapshot of previous state
│       └── metadata.json
├── docs/
│ ├── ARCHITECTURE.md
│ ├── FUTURE_AUTOMATION.md
│ └── SCHEMA.md
│
└── README.md
```




## Quick Start (for future implementers)

1. Export the Canvas gradebook CSV for the course.
2. Place all raw scan images in `data/raw_scans/`.
3. Create a grading key file in `data/rubric/quizX.yaml`.
4. Start the app → local browser UI opens.
5. Grade:
   - assign name,
   - assemble pages into WorkUnits,
   - mark per-problem scores.
6. Export CSV → import into Canvas.


## Assignment Lifecycle (v0.2)

The system now supports a full assignment lifecycle, including archiving and resetting between quizzes or exams.

Flow:

Grade normally using the WorkUnit UI.

Download renamed scans for Canvas or record-keeping:

The grading page includes a link “Download renamed scans (ZIP)”.

Generates clean filenames such as berry_sharon_1.jpg, berry_sharon_2.jpg, etc.

Finalize & Archive (end of grading):

Use the “Finalize & Archive” form at the top of the grading page.

Enter a label (e.g., quiz1_sectionA).

The system will:

Copy the entire data/ directory into
data_archive/<timestamp>_<label>/

Write a small metadata.json file into the snapshot

Clear only:

data/raw_scans/

data/grades/session.json

Leave data/roster.csv and data/rubric/ intact

Start a new assignment:

Place new scans into data/raw_scans/

Update the YAML rubric if needed

Run:

python -m scripts.init_session


Then use /work_units/create once to create the first WorkUnit

Continue grading as usual (Save & Next auto-creates subsequent units)

---

## Core Invariants

- **WorkUnit is the canonical object.**  
- **Human scores are authoritative** even when autograding exists.
- **Per-problem keys MUST match the grading key IDs.**
- **A submission can have multiple pages**, but exactly one student.
- **Scans are never auto-uploaded** to Canvas (FERPA risk).
- **Feedback is safe to auto-upload** (not student data from others).

---

## For the RA / future contributor

Start by reading:

- `ARCHITECTURE.md` for system overview,
- `docs/SCHEMA.md` for data contracts,
- `docs/FUTURE_AUTOMATION.md` for planned extensibility.

Then implement the v0 UI flow as described there.

## Implementation Status (v0)

Current working pieces:

- **Session management**:
  - `scripts/init_session.py` builds `data/grades/session.json` from:
    - `data/roster.csv` (trimmed Canvas export)
    - `data/rubric/quiz1.yaml` (grading key)
    - `data/raw_scans/` (JPG/PNG files)
- **Backend API** (FastAPI, see `/docs` when server is running):
  - `POST /work_units/create` — create a new WorkUnit from the next unclaimed page.
  - `POST /work_units/{unit_id}/append_next_scan` — attach another page.
  - `POST /work_units/{unit_id}/remove_last_scan` — undo last page.
  - `GET  /grade/{unit_id}` — JSON view of one WorkUnit + rubric + roster.
  - `POST /grade/{unit_id}` — JSON update of scores, student_id, comment.
  - `GET  /export` — CSV (`student_id,quiz_id_score`) for Canvas upload.
  - `GET  /session/summary` — debug summary.
- **HTML grading UI**:
  - `GET  /grade/{unit_id}/page` — renders `app/templates/grade.html`:
    - shows scan image(s),
    - roster dropdown,
    - per-problem score fields,
    - comment box.
  - `POST /grade/{unit_id}/page` — form submission:
    - updates student_id, per-problem scores, and comment,
    - then redirects back to the same page.

    - **Renamed scans ZIP export**:
  - The HTML grading UI includes a “Download renamed scans (ZIP)” action.
  - This uses the current `Session` (WorkUnits + roster + grading key) to:
    - group pages by WorkUnit,
    - order pages by their position in `page_filenames`,
    - and rename them using a canonical convention based on `quiz_id` and `student_id`.
  - The original files in `data/raw_scans/` are not modified; the feature only
    generates a ZIP archive with new filenames.
  - The exact endpoint and filename pattern are documented in `docs/SCHEMA.md`
    under “Related Export: Renamed Scans (ZIP)”.

To run the server locally:

```bash
python -m uvicorn app.server:app --reload --port 5000
# then open http://127.0.0.1:5000/docs or /grade/1/page
```

### Exporting grades

Once you’ve finished grading:

1. Make sure the app is running (`uvicorn app.server:app --reload`).
2. In your browser, open: `http://127.0.0.1:5000/docs`.
3. Find the `GET /export` endpoint in the list.
4. Click **Try it out → Execute**.
5. Your browser will download a file named like `quiz1_scores.csv`  
   (or you can navigate directly to `http://127.0.0.1:5000/export`).

This CSV has columns:

- `student_id`
- `<quiz_id>_score`  (e.g. `quiz1_score`)


