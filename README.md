# Logic Grading Facilitator

## Third-party notices

The Fitch proof parser/verifier in `app/syntax.py` and `app/verifier.py`
includes code adapted from `OpenLogicProject/fitch-checker` under `GPL-3.0`.
See `THIRD_PARTY_NOTICES.md` for provenance details and `LICENSES/GPL-3.0.txt`
for the included license text for that reused component.

The repository also includes `deduct-master.zip`, a bundled archive of the
separate `Colonial-Dev/deduct` project under `AGPL-3.0`. It is currently kept
as a vendored reference/tool archive rather than as Python code adapted into
this codebase. See `VENDORED_COMPONENTS.md`, `THIRD_PARTY_NOTICES.md`, and
`LICENSES/AGPL-3.0.txt` for details.

## Collaboration note

When an agent task would be significantly faster, simpler, or less token-heavy
if a human manually fetches or pastes a file, the agent should say so before
doing a slow workaround. Prefer asking for the file directly when that is
likely to save substantial time or tokens.

For coding-agent work, start with [AGENTS.md](AGENTS.md). For OCR/regression
work, agents should also read [docs/AI_WORKFLOW.md](docs/AI_WORKFLOW.md) and
the active TFL/Fitch guide [docs/FITCH_OCR_AGENT.md](docs/FITCH_OCR_AGENT.md)
before making structural changes.

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

## Codebase Layout

Three semi-independent domains live side by side in this repo:

- **`app/`** — Flask/FastAPI web server, grading UI, Canvas export, session management
- **`proofs_ocr/`** — TFL Fitch proof OCR pipeline (recursive crop-mask-transcribe), regression fixtures, symbol normalization tricks
- **`modal_ocr/`** — Modal Fitch proof OCR scaffold (experimental sibling of the TFL path)

Supporting scripts live in `scripts/`; shared tests in `tests/`. For the full development workflow recipes, see [docs/DEV_WORKFLOWS.md](docs/DEV_WORKFLOWS.md).

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

Documentation reminder for future work:
- Before changing or inspecting live-data workflows again, review the **FERPA & Compliance** section in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md#5-ferpa--compliance).
- Planned investigation: enforce hard guardrails so AI-facing tools can do metadata-only operations on live FERPA paths without reading raw file contents into model context.
- Preferred direction: preserve untouched Canvas ZIPs plus an import manifest automatically, and route any live-data file operations through a narrow local helper instead of direct file reads.

## Setup Wizard (new)

Run the server (`python -m uvicorn app.server:app --reload --port 5000`), then open `http://127.0.0.1:5000/setup` to configure a fresh assignment without touching the CLI. The wizard lets you:

- Archive any existing session before starting (protects against accidental data loss).
- Pick an existing roster CSV from `DATA_DIR/rosters/` **or** upload a new one; the selected file is copied into `DATA_DIR/roster.csv`.
- Or upload a full Canvas gradebook CSV; the wizard extracts `Student` + `SIS Login ID` into `DATA_DIR/roster.csv`.
- Upload scans as individual JPG/PNG files **or** as a PDF (auto-splits into per-page JPEGs under `DATA_DIR/raw_scans/`).
- Edit and save the grading rubric YAML; it writes to `DATA_DIR/rubric/<quiz_id>.yaml` and pre-fills the quiz_id for downstream routes. Any rubric you create manually should also be placed in `DATA_DIR/rubric/` with a `.yaml` or `.yml` extension so it appears in the `/setup` rubric dropdown after a page reload.
- In production, `DATA_DIR` is often an external live folder such as `C:\grading-data\live`, so the real rubric location is typically something like `C:\grading-data\live\rubric\`, not the repo’s `data/rubric/`.
- The per-problem `label` is also used as a natural-language hint when the model tries to locate each answer region in strip view, so match the wording on the exam if possible.
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

## Autograde scaffolding status

The repo now includes a small shared autograde package at `app/autograde/`.

- `fol_model` is wired through that shared layer and has both a debug page and
  strip-view OCR/checker review for canonical model answers.
- `kripke_model` now has an initial deterministic checker for typed/canonical
  model text. It checks modal truth constraints, formula-difference tasks,
  non-entailment countermodels, and basic accessibility-relation properties.
  OCR and strip-view review for Kripke drawings are still future work.
- `modal_proof` now has backend checker integration through a separate `deduct` CLI wrapper, controlled by `DEDUCT_CHECKER_BIN`, plus a dedicated `/debug/modal-proof` page for manual/typeset proof debugging.
- `fitch_proof` remains an explicit shared-autograde scaffold; its working
  user-facing path is the separate strip OCR/debug verifier workflow.
- The unfinished engines are present to define extension points and shared result shapes; they are not a claim that the user-facing workflows are finished.

See [docs/MODAL_PROOF_DEDUCT_CLI.md](docs/MODAL_PROOF_DEDUCT_CLI.md)
for the build/run boundary of the separate modal checker component.

See [docs/AUTOGRADE_ARCHITECTURE.md](docs/AUTOGRADE_ARCHITECTURE.md) for the current contract and status table.

## Kripke model checker starter

The `kripke_model` autograde path currently expects already-transcribed model
text, for example:

```text
W: {0,1}
R: <0,1>, <1,1>
P: {1}
Q: {}
```

It also accepts `w0` labels in `W`/`R` lines and worksheet-style valuation
entries such as `ν: ν0(P) = F, ν1(P) = T`.

Rubric examples:

```yaml
problem_type: kripke_model
autograde:
  task_kind: non_entailment
  default_score_correct: 4
  default_score_incorrect: 0
  prompt_data:
    world: 0
    premises:
      - "<>P"
    conclusion: "[]Q"
```

```yaml
problem_type: kripke_model
autograde:
  task_kind: different
  prompt_data:
    constraints:
      - type: different
        world: 0
        formulaA: "<>P"
        formulaB: "[]P"
```

The first implementation ports the worksheet's verdict style into Python:
lines such as `At w0: <>P is T (expected T)` and draft comments that include a
model summary and per-constraint results. It intentionally does not yet OCR a
drawn graph into model text.

## Modal proof debug page

Use `http://127.0.0.1:5000/debug/modal-proof` to exercise the modal checker backend with saved example states or the built-in starter proof. The page is intentionally backend-focused:

- optionally run the modal OCR scaffold against an uploaded/saved proof image
- edit the proof JSON directly
- choose `K`, `T`, `S4`, or `S5`
- choose the LM Studio base URL and vision model
- toggle derived rules
- run the shared `modal_proof` autograde path
- inspect the exact checker request, checker response, normalized lines, and reported parse/rule issues

This remains a debug workflow for typeset/manual proof structures and proof-image OCR; it is not yet a finished end-to-end handwritten grading flow.

## Modal proof OCR scaffold

There is now a separate experimental script at `scripts/debug_modal_proof_ocr.py`
for modal Fitch-style proof OCR. It is intentionally a sibling to the older
Fitch OCR path rather than a full abstraction over both families.

Current status:

- it reuses the recursive vertical-bar / crop / mask strategy
- it asks the local model for a modal-specific intermediate transcription format
- it converts that into the nested proof JSON accepted by `/debug/modal-proof`
- it can optionally run the separate Rust `deduct` checker and save the request
  and response beside the OCR artifacts

This scaffold has been validated on a clean typeset strict-subproof example,
including:

- top-level boxed premises such as `□A` and `□B`
- a strict opener line `□`
- internal citations such as `∧I 4 5`
- a final `□I 3-6`

It has also now been validated on a more complex mixed example from the modal
handout:

- an ordinary subproof containing a strict subproof
- a post-strict boxed result line such as `□B`
- a final ordinary introduction line such as `→I 2-7`

That mixed case required a small amount of modal-specific post-stitch repair at
subproof boundaries, because the model sometimes:

- omits the immediate boxed result line after a strict child subproof
- or duplicates a conclusion line both inside and just outside the parent
  subproof crop

Artifacts from a run are written to a chosen output folder and typically include:

- `ocr_prompt.txt`
- `response_<node>.json`
- `intermediate_<node>.json`
- `final_structure.json`
- `state.json`
- `checker_request.json`
- `checker_response.json`

The same scaffold is now also available through `/debug/modal-proof`, which
lets you upload a proof image, choose the LM Studio model, inspect the OCR
artifacts, hand-correct the JSON if needed, and rerun the modal checker.

This is still a development/debug path, not a finished grading workflow for
real handwritten submissions.

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

You can also start from strip grading for `fitch_proof` problems:

- open `/problem/{problem_id}/strips`
- tag the problem as **Fitch proof** if it is not already set in the rubric
- use the crop controls to confirm the proof region for each strip
- click **Start OCR** to queue all proof strips for that question

The strip page then polls the local server and updates each proof card as it
finishes. This lets a grader review, edit scores/comments, and accept ready OCR
results while later proofs continue processing. The one-strip **OCR Proof**
button is still available when you want to create a cropped debug example and
jump directly into `/debug/fitch`.

The asynchronous review state is stored under:

- `DATA_DIR/grades/proof_ocr/tfl_fitch_{problem_id}.json`

Per-proof crops and OCR artifacts are stored under:

- `DATA_DIR/debug_fitch/<problem_unit_page_slug>/`

The UI may show verifier/reread report details when they exist, including the
original OCR text, candidate reread text, `candidate_class`, concise reason
text, proof-checker issues, and line-crop thumbnails. Reread candidates are
report-only for now; the grader must confirm by hand.


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

DATA_DIR/raw_scans/

DATA_DIR/grades/ (including session.json and segments_*.json caches)

Leave DATA_DIR/roster.csv and DATA_DIR/rubric/ intact

Start a new assignment:

- **Recommended:** Use the `/setup` wizard to archive the previous session (if any), choose or upload a roster, upload scans (images or PDF), edit/save the rubric YAML, and run auto-grouping. The wizard redirects straight to `/name_review` so you can confirm groups and start grading.
- **Manual fallback:** Place new scans into `DATA_DIR/raw_scans/`, update the YAML rubric in `DATA_DIR/rubric/` if needed, run `python -m scripts.init_session`, then hit `/work_units/create` once to create the first WorkUnit. Continue grading as usual (Save & Next auto-creates subsequent units).

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
  - `scripts/init_session.py` builds `DATA_DIR/grades/session.json` from:
    - `DATA_DIR/roster.csv` (trimmed Canvas export)
    - `DATA_DIR/rubric/quiz1.yaml` (grading key)
    - `DATA_DIR/raw_scans/` (JPG/PNG files)
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
   - feedback ZIP (`/export_feedback_zip`)
   - Canvas reupload feedback ZIP (`/export_canvas_reupload_feedback_zip`)
   - renamed scans ZIP (`/export_renamed_scans`)
4. Or use the `GET /export` endpoint in `/docs`.

This CSV has columns:

- `student_id`
- `<quiz_id>_score`  (e.g. `quiz1_score`)

### Canvas reupload feedback ZIP

If you use Canvas "Re-Upload Submissions", place the original Canvas download ZIP under:

- `data/canvas_reupload/original_zips/`

Then use:

- `GET /export_canvas_reupload_feedback_zip`

This export preserves the exact original submission filenames and extensions while replacing content with per-student feedback artifacts (PDF/image). Upload that ZIP back in Canvas re-upload mode, and upload grades separately using the CSV export.
