# Logic Grading Facilitator

A lightweight local tool for grading scanned student submissions with minimal friction, clean data structures, and a clear path toward future automation (OCR, clustering, autograding). Designed to eliminate the mechanical overhead of: matching scans to names, managing multi-page submissions, entering per-problem scores, and exporting grades back to Canvas.

This project is built around a simple core abstraction:

> **WorkUnit** = one student submission, possibly spanning multiple scanned pages, with per-problem scores and metadata.

The goal is to make **v0 grading** extremely easy while guaranteeing that the data structures and UI flow will support **v1–v4 automation** without refactoring.

---

## Documentation map

- **This README**: best place to start for the current (FastAPI + Jinja) implementation and the end-to-end grading/archival flow.
- **`docs/ARCHITECTURE.md`**: details on current components, API endpoints, templates, and how WorkUnits flow through the system.
- **`docs/SCHEMA.md`**: canonical data contracts (Session/WorkUnit schema, CSV export, renamed-scan ZIP conventions).
- **`docs/FUTURE_AUTOMATION.md`**: ideas and aspirational v1–v4 automation (OCR, clustering, autograding, VLM helpers); not required for v0 usage.

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

## Day 1 flow (quick start)

1. Export the Canvas gradebook CSV for the course → place as `data/roster.csv`.
2. Put all raw scan images in `data/raw_scans/`.
3. Add a grading key in `data/rubric/quizX.yaml`.
4. Initialize the session once per assignment:
   ```bash
   python -m scripts.init_session
   ```
5. Start the server and open the grading UI:
   ```bash
   python -m uvicorn app.server:app --reload --port 5000
   # browse http://127.0.0.1:5000/grade/1/page
   ```
6. Grade submissions (assign names, assemble pages, enter per-problem scores) and export CSV for Canvas when done.


## Key local links & stability

- `/docs` — FastAPI docs for all endpoints (current v0 surface).
- `/grade/{unit_id}/page` — main grading UI; stable.
- `/name_review` — optional ML-assisted name review; experimental.
- `/problem/{unit_id}/strips` — per-problem strip view; experimental.


## Assignment Lifecycle (v0.2)

The system supports archiving and resetting between quizzes/exams.

- Grade normally using the WorkUnit UI.
- Download renamed scans (ZIP) from the grading page for Canvas/records; filenames follow `docs/SCHEMA.md` conventions.
- Finalize & Archive (form at top of grading page):
  - Provide a label (e.g., `quiz1_sectionA`).
  - Copies `data/` → `data_archive/<timestamp>_<label>/` and writes `metadata.json`.
  - Clears only `data/raw_scans/` and `data/grades/session.json`; keeps `data/roster.csv` and `data/rubric/` intact.
- Start the next assignment:
  - Place new scans into `data/raw_scans/`.
  - Update the YAML rubric if needed.
  - Run `python -m scripts.init_session`, then visit `/work_units/create` once to seed the first WorkUnit. Subsequent “Save & Next” actions auto-create more units.

---

## Repository Structure (current)

```text
logic-grading-facilitator/
│
├── app/
│   ├── server.py          # FastAPI app + HTML routes/templates
│   ├── templates/         # Jinja templates (grading UI, name review)
│   ├── static/            # CSS/JS for the UI
│   ├── models.py          # Session/WorkUnit dataclasses
│   ├── storage.py         # Session loading, saving, archival utilities
│   └── ...                # VLM helpers, segmentation, auto-grouping
├── data/
│   ├── raw_scans/         # current assignment’s scans (input)
│   ├── grades/            # `session.json` + incremental grading state
│   ├── rubric/            # grading keys (YAML)
│   └── roster.csv         # Canvas roster export
├── data_archive/          # snapshots created by Finalize & Archive
│   └── 2025-11-16T14-05-23_quiz1/
│       ├── data/          # full snapshot of previous state
│       └── metadata.json
├── docs/
│   ├── ARCHITECTURE.md
│   ├── FUTURE_AUTOMATION.md
│   └── SCHEMA.md
├── scripts/               # session init and helper scripts
└── README.md
```

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

---

## Organizing `app/server.py` for readability (no code changes required)

Suggested structure for keeping the long module easy to navigate:

1. **Module banner & invariants**: A short comment/docstring near the top summarizing the WorkUnit-centric flow, data directories used, and the “human scores are authoritative” invariant.
2. **Constants & helpers**: Group configuration (paths, models, cache dirs) and pure helpers (e.g., `natural_sort_key`, `canonical_filename`) under a header like `# === Config & helpers ===`.
3. **Session accessors**: Keep `get_session()` and `persist_session()` together with a note about the threading lock and single-session expectation.
4. **Route groups with headers**: Use Markdown-style separators to cluster related endpoints:
   - `# === WorkUnit creation & page assembly ===`
   - `# === Grading (JSON + HTML forms) ===`
   - `# === Name review & auto-grouping tools ===`
   - `# === Exports (CSV, renamed scans ZIP) ===`
   - `# === Assignment lifecycle (finalize/archive/reset) ===`
5. **Inline schema pointers**: Within each route block, add a 1–2 line comment referencing the relevant schema section (e.g., `docs/SCHEMA.md` → “WorkUnit JSON view”, “Renamed scans export”).
6. **Side-effect callouts**: For routes that mutate disk state (archive/reset), include short comments summarizing which directories are written/cleared to align with the Assignment Lifecycle section above.
7. **Template expectations**: Above HTML-rendering routes, note the template name and key context variables to help template edits stay in sync with backend changes.

These headers and breadcrumbs keep search and scrolling manageable while preserving the single-file FastAPI app for now.


