# Logic Grading Facilitator

A lightweight local tool for grading scanned student submissions with minimal friction, clean data structures, and a clear path toward future automation (OCR, clustering, autograding). Designed to eliminate the mechanical overhead of: matching scans to names, managing multi-page submissions, entering per-problem scores, and exporting grades back to Canvas.

This project is built around a simple core abstraction:

> **WorkUnit** = one student submission, possibly spanning multiple scanned pages, with per-problem scores and metadata.

The goal is to make **v0 grading** extremely easy while guaranteeing that the data structures and UI flow will support **v1–v4 automation** without refactoring.

Logic Grading Facilitator (v2 Hybrid Workflow)
A privacy-first, local tool for grading handwritten logic exams. It uses a "Cyborg" workflow: a local Vision Language Model (VLM) handles the mechanical overhead (grouping scans, recognizing names, segmenting answers), while the human instructor focuses purely on grading logic via a high-speed "Strip View."

Core Philosophy:

Local-Only: Scans never leave your machine (FERPA compliant).

Human Authority: AI suggests groupings and bounding boxes; the human makes the final score decisions.


WorkUnit: The core data structure representing one student's grouped pages and scores.

Core Features

Auto-Grouping: scripts.auto_group_scans uses a local VLM (Qwen via LM Studio) to read names and group widely scattered pages into single student submissions.


Name Review UI: A web interface to audit the AI's name guesses before finalizing the session (copies canonical filenames into `data/raw_scans/`).

Strip View Grading: A vertical interface showing one problem across all students (e.g., "Grade Q1 for everyone"). Includes auto-segmentation to zoom in on the specific answer.



Canvas Export: Generates an import-ready CSV for the Canvas gradebook.

Architecture Overview
Ingestion: Raw scans are placed in data/raw_scans/.

AI Processing:

Step 1 (Grouping): The VLM reads names and creates "WorkUnits" from raw pages.


Step 2 (Segmentation): On-the-fly segmentation detects where "Q1" is on the page.

Storage:


session.json: The single source of truth for the active grading session.


data_archive/: Snapshots of finished assignments.

## Data location (FERPA-safe)

All `data/...` paths in this README are relative to `DATA_DIR` (defaults to `data/samples`).
Set `DATA_DIR` to a folder outside the repo for real student data. In production mode,
the app refuses to start if `DATA_DIR` points inside the repo.

Example `.env`:
```
DATA_DIR=C:\grading-data\live
DATA_ARCHIVE_DIR=C:\grading-data\archive
ENV_MODE=PRODUCTION
```

Dev workflow (safe):
- Leave `DATA_DIR` unset or point it at `data/samples`.
- Use fake roster/scans/rubric.

Grading workflow (real data):
- Set `ENV_MODE=PRODUCTION`.
- Point `DATA_DIR` and `DATA_ARCHIVE_DIR` to external folders.

## Setup Wizard (new)

Run the server (`python -m uvicorn app.server:app --reload --port 5000`), then open `http://127.0.0.1:5000/setup` to configure a fresh assignment without touching the CLI. The wizard lets you:

- Archive any existing session before starting (protects against accidental data loss).
- Pick an existing roster CSV from `data/rosters/` **or** upload a new one; the selected file is copied into `data/roster.csv`.
- Or upload a full Canvas gradebook CSV; the wizard extracts `Student` + `SIS Login ID` into `data/roster.csv`.
- Upload scans as individual JPG/PNG files **or** as a PDF (auto-splits into per-page JPEGs under `data/raw_scans/`).
- Edit and save the grading rubric YAML; it writes to `data/rubric/<quiz_id>.yaml` and pre-fills the quiz_id for downstream routes. The per-problem `label` is also used as a natural-language hint when the model tries to locate each answer region in strip view, so match the wording on the exam if possible.
- Kick off auto-grouping/name recognition; the wizard writes the name review cache and redirects you straight to `/name_review` for confirmation.

## Flashcards Dataset Builder (new)

Use the flashcards builder to curate a dataset of student photos + facts for a name-learning game.

Start the server, then visit `http://127.0.0.1:5000/flashcards` to:

- Optionally select or upload a roster CSV (used only to guess names from filenames).
- Upload a ZIP of student files (PDFs, DOCX, or images).
- Manually crop each student photo, enter the name, and paste free-text facts.
- Export `dataset.json` plus cropped images under `DATA_DIR/flashcards/<session_id>/`.

Notes:
- The roster is optional; if you skip it, you can type names manually.
- All files are stored under `DATA_DIR`, so keep that folder outside the repo for FERPA.

## Future Extensions (planned, not required for v0)

Quality of life:
- Avoid slugish loading of grading pages (after the first one) by having the local model segment question n+1 (if this has not already been done) while you are working on grading question n in the per unit grader.

More automation: See `docs/FUTURE_AUTOMATION.md` for details, but the major phases include:
- Clustering/grouping of similar answers for consistent partial credit.
- Deterministic autograding: connect to known existing code for: symbolization equivalence checker, truth-table evaluator, Fitch proof checker (I could use the special ocr->ascii->json->verification stuff I've already written for my fitch proof project), kripke model checker.
- LLM-based “second-opinion” disagreement flags.
- Optional attachment of per question comments to scans which will be manually uploaded to canvas. 

## Fitch proof debug + human-correction page
Use the in-browser review page to inspect OCR artifacts, hand-edit the stitched proof JSON, and run the verifier without leaving the UI:

1. Generate artifacts once for a cropped proof scan:
   ```bash
   python scripts/debug_fitch_ocr.py quiz1_studentA_p1.jpg
   # writes to data/debug_fitch/quiz1_studentA_p1/
   ```
2. Start the app as usual:
   ```bash
   uvicorn app.server:app --reload --port 5000
   ```
3. Visit `http://127.0.0.1:5000/debug/fitch`, pick your scan slug, and use:
   - **Re-run OCR/parse** to refresh `final_structure.json` + logs.
   - **Save edits** after tweaking the JSON in the textarea (persists back to `data/debug_fitch/<slug>/final_structure.json`).
   - **Run grader** to push the saved structure through `app/verifier.py` and see rule issues + inferred conclusion inline.


---

## Repository Structure (initial draft)

```text
logic-grading-facilitator/
│
├── app/ # backend & UI
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




## Assignment Lifecycle (v0.2)

The system now supports a full assignment lifecycle, including archiving and resetting between quizzes or exams.

Flow:

Grade normally using the WorkUnit UI.

Download renamed scans for Canvas or record-keeping:

The grading page includes a link “Download renamed scans (ZIP)”.

Generates clean filenames such as quiz1_12345_p01.jpg, quiz1_12345_p02.jpg, etc.

Finalize & Archive (end of grading):

Use the “Finalize & Archive” form at the top of the grading page.

Enter a label (e.g., quiz1_sectionA).

The system will:

Copy the entire data/ directory into
data_archive/<timestamp>_<label>/

Write a small metadata.json file into the snapshot

Clear live grading state:

data/raw_scans/

data/grades/ (including session.json and segments_*.json caches)

Leave data/roster.csv and data/rubric/ intact

Start a new assignment:

- **Recommended:** Use the `/setup` wizard to archive the previous session (if any), choose or upload a roster, upload scans (images or PDF), edit/save the rubric YAML, and run auto-grouping. The wizard redirects straight to `/name_review` so you can confirm groups and start grading.
- **Manual fallback:** Place new scans into `data/raw_scans/`, update the YAML rubric if needed, run `python -m scripts.init_session`, then hit `/work_units/create` once to create the first WorkUnit. Continue grading as usual (Save & Next auto-creates subsequent units).

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

### Fake data generator (safe dev)
Create a minimal fake roster/rubric/scans set under `DATA_DIR`:
```
python scripts/generate_fake_data.py
```

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
  - `GET  /work_units` — summary view (unit_id, page_count, scored status).
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
3. Visit `http://127.0.0.1:5000/finish` to download:
   - grades CSV (`/export`)
   - renamed scans ZIP (`/export_renamed_scans`)
4. Or use the `GET /export` endpoint in `/docs`.

This CSV has columns:

- `student_id`
- `<quiz_id>_score`  (e.g. `quiz1_score`)

