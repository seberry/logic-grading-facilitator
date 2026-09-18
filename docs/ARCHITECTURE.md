


# System Architecture Overview

This document describes the high-level architecture of the Logic Grading Facilitator (v2), covering the core data models, the "Cyborg" workflow, and the backend data pipelines.

Note: All `data/...` paths in this document are relative to `DATA_DIR` (defaults to `data/samples`).

---

# 1. Core Concept: WorkUnits

A **WorkUnit** is the central abstraction of the entire system.

It represents:
- One student submission.
- Containing one or more scanned pages.
- With per-problem scores and optional comments.
- Keyed to a unique `student_id` from the Canvas roster.
- Optionally tagged with a normalized printed `quiz_version` used to resolve
  version-specific rubric prompts and neutral grading notes.

All downstream workflows (Canvas export, per-question views, OCR segmentation, autograding) operate on the `WorkUnit`.

---

# 2. Backend Model

The backend service (FastAPI) maintains the state of the grading session.

### Key Data Stores
1.  **`session.json` (The Database):**
    * The single source of truth for the active grading session.
    * Contains the list of **WorkUnits**, the **GradingKey**, and the **Roster**.
    * Located at: `data/grades/session.json`.

2.  **`raw_scans/` (The Images):**
    * Contains raw uploads (e.g., `IMG_001.jpg`).
    * After the "Hand-off," canonical copies are added (e.g., `quiz1_berry_sharon_p01.jpg`); originals are left in place.
    * Intake skips byte-identical image uploads using SHA-256. Filename
      collisions with different content are preserved using `_2`, `_3`, etc.
      Auto-grouping quarantines exact duplicates created by older versions under
      `DATA_DIR/duplicate_scans_quarantine/<timestamp>/` before invoking the VLM.

3.  **`segments_*.json` (The Geometry):**
    * Stores bounding boxes for where specific answers are located on a page.
    * Generated lazily by the VLM.
    * Located at: `data/grades/segments_{problem_id}.json`.

4.  **`proof_ocr/*.json` (Per-question proof OCR review state):**
    * Stores asynchronous TFL/Fitch proof OCR status by `WorkUnit`.
    * Located at: `data/grades/proof_ocr/tfl_fitch_{problem_id}.json`.
    * Contains only review state and normalized OCR/checker summaries; cropped
      proof images and OCR artifacts remain under `data/debug_fitch/<slug>/`.

---

# 3. UI Flow & Views

The frontend serves specific interfaces for different phases of the grading lifecycle.

### Setup and rubric drafting (`GET /setup`)

The setup wizard lists saved rubric YAML files from `DATA_DIR/rubric/` and keeps
manual YAML editing as the authoritative review surface. `POST
/setup/generate_rubric` optionally creates an editable draft from a blank PDF,
DOCX, TeX, Markdown, text, or YAML assessment:

* the rubric section uses progressive disclosure: saved rubrics are the default
  starting point, API generation and simple manual generation are alternatives,
  and the exact YAML editor is shared by every path

* configured providers are detected from `OPENAI_API_KEY` and
  `DEEPSEEK_API_KEY`; the corresponding model overrides are `OPENAI_MODEL` and
  `DEEPSEEK_MODEL`
* OpenAI uses the Responses API with a JSON schema and `store: false`; DeepSeek
  uses Chat Completions JSON mode
* the blank assessment is read in memory, bounded by upload/text limits, and is
  not written into `DATA_DIR`
* model output is validated through a restricted draft schema and the canonical
  `GradingKey` model; generated solution/`answer_key` fields are not part of the
  draft contract
* generation never saves or replaces a rubric—the instructor reviews the YAML,
  sees live regular/extra-credit totals, and explicitly submits the existing
  `POST /setup/save_rubric` form

Do not upload student scans to rubric generation. The feature is intended only
for blank assessment text and instructor-authored requests.

### A. Name Review View (`GET /name_review`)
* **Purpose:** rapid audit of VLM-grouped pages before grading begins.
* **Input:** Reads draft predictions from CSV (CLI output) or JSON cache.
* **Action:** User confirms identity and printed quiz-version matches and can use **Save & Auto-Fill Unknowns** to cascade the last chosen student onto later "Unknown" rows without renaming files yet.
* **Draft persistence:** **Save Corrections** stores the exact selected names and
  versions in the prediction cache. Refreshed recognition preserves reviewed
  name choices while updating printed-version guesses.
* **Idempotent review:** If an earlier handoff left both an original upload and
  its byte-identical canonical name-labeled copy, the page displays one merged
  row. It can take the reviewed name from one copy and the printed-version guess
  from the other without deleting either file.
* **Transition:** **Save assignments & grade now** creates the session and opens
  the first Strip View. **Save assignments & prepare TF handoff** creates the
  same session and opens the Assignment overview in TF-preparation mode.

### B. Strip View Grading (`GET /problem/{id}/strips`)
* **Purpose:** High-speed grading of **one question across all students**.
* **Display:**
    * A versioned rubric opens one A/B/C tab at a time. The selected version's
      blank-quiz question image (or text fallback) stays pinned at the top and
      is rendered once, not repeated inside every student card.
    * Shows a large cropped "strip" of the student's answer (via segmentation),
      with the score/comment controls in a narrower side column.
    * Provides full-page toggle and scoring inputs.
    * Filters work by printed version and permits inline version correction.
      Stored grading notes are omitted from this compact grading surface.
    * Saves human-selected visual reference responses independently for each
      problem/version. These pins persist in `session.json` and the TF bundle.
    * Does not display rubric `answer_key` values. TF bundle export removes any
      such legacy values from both its rubric and session copies.
    * For `fitch_proof` problems, shows a TFL/Fitch OCR review card beside
      each strip and polls for status updates while OCR continues in the
      background.
    * For `fol_model` problems, shows the cropped answer, editable OCR/model
      text, per-strip target details, checker verdict, suggested score/comment,
      and a button to save edits and rerun the deterministic checker.
* **Navigation:** "Save & Next" moves through versions before advancing to the
  next problem (for example Q1/A, Q1/B, Q1/C, then Q2/A).

#### TFL/Fitch OCR review in Strip View

For TFL/Fitch proof problems, the strip page exposes a lightweight asynchronous
review lane:

* `POST /problem/{problem_id}/fitch_ocr/start` queues OCR for the current
  question's proof strips. The route uses the cached/manual segmentation boxes,
  crops each proof into `data/debug_fitch/<slug>/`, initializes per-strip state,
  and submits jobs to a single background worker.
* `GET /problem/{problem_id}/fitch_ocr/status` returns per-strip status for
  polling. The browser updates proof cards in place, so a grader can review
  finished strips without waiting for later OCR jobs.
* `POST /problem/{problem_id}/fitch_ocr/{unit_id}/accept` marks a reviewed OCR
  result as accepted. This records human confirmation in the OCR review state;
  it does not currently overwrite scores or auto-apply verifier reread
  candidates.

The review statuses are:

* `queued` - crop prepared and waiting for the background worker.
* `processing` - the importable TFL/Fitch OCR pipeline is running.
* `needs_review` - OCR completed and no verifier/reread flags were surfaced.
* `verifier_issue` - OCR completed with proof-checker issues or reread report
  records that need human attention.
* `accepted` - the grader accepted the displayed OCR result.
* `ocr_failed` - crop preparation or OCR execution failed; the debug folder is
  still the place to inspect available artifacts.

Verifier reread outputs are report-only in this workflow. When available, the
strip card can display original OCR text, candidate reread text,
`candidate_class`, concise reason text, and row-crop thumbnails. Candidates are
not auto-applied because verifier failures may be real student mistakes.

### C. WorkUnit View (Legacy/Detail) (`GET /grade/{unit_id}/page`)
* **Purpose:** Detailed view of a single student's full exam.
* **Use Case:** Spot-checking a specific student or handling complex multi-page overrides.

### D. Fitch Debug & Human-Correction View (`GET /debug/fitch`)
* **Purpose:** Inspect highlighted proof scans, hand-edit stitched `final_structure.json`, and run the verifier inline.
* **Inputs:** Reads artifacts from `data/debug_fitch/<scan_stem>/` (produced by `scripts/debug_fitch_ocr.py`, the strip-page one-proof debug action, or the asynchronous strip OCR queue).
* **Actions:**
  * Re-run the OCR/parse pipeline on demand to refresh the JSON and log output.
  * Edit and save `final_structure.json` directly in the browser; persists to the same folder.
  * Run the Fitch verifier on the saved structure to surface rule issues and the inferred conclusion without leaving the page.

---

# 4. Automation Integration Points

The architecture is explicitly prepared for progressive automation:

1.  **OCR-based Segmentation:**
    * *Current:* VLM identifies bounding boxes for "Q1".
    * *Future:* OCR detects specific headings (1., (a)) to refine these boxes.

2.  **Visual prompt injection:**
    * For structural tasks (Fitch Proofs), we do not rely on the VLM to see geometry.
    * We use OpenCV to detect geometric features (vertical bars) and inject
      them into the image as high-contrast color overlays.
    * This converts a geometric problem ("which line is this text inside?") into a color perception problem ("what background color is this text on?"), which local VLMs handle significantly better.

3.  **Clustering:**
    * Future modules can group similar answers (via embeddings) to allow bulk-grading.

4.  **Autograding:**
    * Deterministic checkers (Truth Table, Fitch) can populate an `auto_score` field in the WorkUnit.
    * Human scores always remain authoritative.

---

# 5. FERPA & Compliance

* **Local-Only:** Scans are processed locally. They are **never** auto-uploaded to Canvas.
* **Feedback:** Text comments are safe to upload (they contain no student PII from others).
* **Archival:** The `finalize` flow moves data to a timestamped archive folder, ensuring no cross-contamination between quizzes.

---

# 6. Data Flow & Pipeline Lifecycle

This section explains how data transforms from "Raw Images" to "Graded Assignments."

### Phase 1: Ingestion & Drafting (CLI)
* **Trigger:** User runs `scripts.auto_group_scans.py`.
* **Input:** Raw images in `data/raw_scans/`.
* **Process:** VLM predicts student names for every page.
* **Output:** A **CSV Report** at `data/auto_grouped/{quiz}/{quiz}_page_predictions.csv`.
* **State:** "Draft Mode." No files are moved; no session exists.

### Phase 2: The Review-to-Grading Hand-off
* **Trigger:** User clicks **Save assignments & grade now** or **Save assignments
  & prepare TF handoff** on the `/name_review` page (the earlier **Save &
  Auto-Fill Unknowns** button only updates the cache and does not rename files).
* **Backend Logic (`POST /name_review`):**
    1.  **Grouping:** Consecutive pages normally form one WorkUnit when their student assignments agree. A confirmed version change or a human-selected **New submission starts here** marker creates a boundary. The browser flags separate groups for the same student and can exclude a duplicate group before session creation.
    2.  **Renaming:** Copies included raw files to `data/raw_scans/` with canonical names (e.g., `quiz1_student_p01.jpg`) and leaves originals intact. Excluded scans remain local but do not enter a WorkUnit or TF bundle.
    3.  **Session Creation:** Constructs `WorkUnit` objects in memory based on the approved groups.
    4.  **Persistence:** Writes the **Source of Truth** to `data/grades/session.json`.
    5.  **Redirect:** Opens either the first Strip View or `/finish?mode=tf-prep`,
        according to the selected goal.
* **State:** "Live Grading Mode." The system now ignores the CSV and relies on `session.json`.

The instructor Assignment overview reports how many per-problem segmentation
caches are complete, prepares only missing problems, provides a crop spot-check
link, and enables TF bundle creation after every problem has a bundle-safe cache.

### Phase 3: Grading & Segmentation
* **Trigger:** User loads `/problem/Q1/strips`.
* **Logic:**
    1.  Server checks `data/grades/segments_Q1.json`.
    2.  **Cache Miss:** If missing, runs VLM on all pages to find "Q1" boxes. Saves to JSON.
    3.  **Cache Hit:** If present, loads boxes instantly.
* **Result:** User sees cropped answers. Scores are written back to `session.json`.

### Phase 4: Optional TFL/Fitch OCR Review
* **Trigger:** For a `fitch_proof` problem, the user clicks **Start OCR** on
  `/problem/{problem_id}/strips`.
* **Logic:**
    1. Server reads the current segmentation cache and selected page for each
       `WorkUnit`.
    2. It writes cropped proof images under `data/debug_fitch/<slug>/`.
    3. It stores queue/review state in
       `data/grades/proof_ocr/tfl_fitch_{problem_id}.json`.
    4. A single background worker runs `proofs_ocr.pipelines.tfl_fitch.run_fitch_ocr_pipeline()`
       for one crop at a time.
    5. The strip page polls the status endpoint and updates cards as each proof
       reaches `needs_review`, `verifier_issue`, `accepted`, or `ocr_failed`.
* **Result:** The grader can scan down the same per-question page and review
  finished proof OCR results while later proofs are still processing.

---

# 7. Caching Strategy

To ensure speed and prevent "double-spending" on VLM tokens, the system uses a tiered caching strategy.

### A. Name Recognition Cache
Used by `GET /name_review` to display guesses.
1.  **Tier 1 (CLI Output):** Looks for `*_page_predictions.csv` generated by the batch script. (Fastest)
2.  **Tier 2 (Server Cache):** Looks for `data/name_review_cache/{quiz}__root.json`.
3.  **Tier 3 (Live Inference):** Runs VLM live (Slow) and writes to Tier 2 cache.

### B. Segmentation Cache
Used by the Strip View to show cropped answers.
* **Location:** `data/grades/segments_{problem_id}.json`.
* **Invalidation:** Automatic on mismatch with the active `session.json` (or manual via refresh/delete).
* **Validation before reuse:** must match current WorkUnits (`unit_id`s), each cached `page_filename` must belong to that WorkUnit, and bbox must be normalized `[x1,y1,x2,y2]` in `[0,1]`.
* **Behavior:** Lazy-loaded. The first time you grade "Q1", you pay the wait time. Subsequent loads are instant.

### C. TFL/Fitch OCR Review State
Used by the Strip View for asynchronous proof OCR review.
* **Location:** `data/grades/proof_ocr/tfl_fitch_{problem_id}.json`.
* **Artifacts:** cropped proof images, `final_structure.json`, `debug_bars.jpg`,
  VLM response files, verifier reread reports, and row-crop thumbnails live
  under `data/debug_fitch/<slug>/`.
* **Invalidation:** currently explicit. Re-clicking **Start OCR** skips queued,
  running, ready, flagged, or accepted items unless the backend is called with
  `force=1`.
* **Concurrency:** a process-local single-worker queue runs one TFL/Fitch OCR
  job at a time. Restarting the server does not resume in-memory jobs, but the
  JSON state and debug artifacts remain for inspection or requeueing.

### D. FOL Model OCR Review State
Used by the Strip View for asynchronous model OCR and inline checker review.
* **Location:** `data/grades/fol_model_ocr/fol_model_{problem_id}.json`.
* **Artifacts:** cropped answer images, OCR prompt/response files, proposed
  canonical model text, and debug-page state live under
  `data/debug_fol_model/<slug>/`.
* **Invalidation:** explicit. Starting OCR skips queued/running/reviewed items
  unless the route is called with `force=1`.
* **Concurrency:** a process-local single-worker queue runs one FOL-model OCR
  job at a time. Manual edits in the strip card can be checked immediately
  through `app/autograde/fol_model.py`.

---

# 8. Repository Layout (Code vs Data)

## Active Route/Data Map

This map is intentionally narrower than the full repository. It lists the
current exam-grading surfaces most likely to be touched by UI, OCR, and export
work.

| Workflow | Routes / modules | Primary persisted state | Notes |
|---|---|---|---|
| Setup and name review | `app/server.py` (`/setup`, `/name_review`) | `data/name_review_cache/*.json`, `data/grades/session.json` | Creates canonical `WorkUnit`s and redirects to strip grading or TF preparation. |
| Per-question strip grading | `app/routes/problem_strips.py`, `app/templates/problem_strips.html` | `data/grades/session.json`, `data/grades/segments_{problem_id}.json` | Main "one question across all students" grading surface. |
| TFL/Fitch strip OCR review | `app/proof_ocr_review.py`, `app/routes/problem_strips.py` (`/problem/{problem_id}/fitch_ocr/*`) | `data/grades/proof_ocr/tfl_fitch_{problem_id}.json`, `data/debug_fitch/<slug>/` | Async review lane for `fitch_proof` problems. Uses polling and a process-local single-worker queue. |
| FOL model strip OCR/checker review | `app/fol_model_review.py`, `app/routes/problem_strips.py` (`/problem/{problem_id}/fol_model_ocr/*`) | `data/grades/fol_model_ocr/fol_model_{problem_id}.json`, `data/debug_fol_model/<slug>/` | Async OCR lane plus inline deterministic checker for `fol_model` problems. |
| Fitch debug / correction | `app/routes/fitch_debug.py`, `app/templates/debug_fitch.html` | `data/debug_fitch/<slug>/final_structure.json` and sibling artifacts | Deep inspection/editing path for one proof crop. |
| Modal proof debug | `app/routes/modal_proof_debug*.py`, `app/templates/debug_modal_proof.html` | `data/debug_modal_proof/<slug>/state.json` and checker artifacts | Separate modal proof scaffold/debug path. |
| Exam exports/finalize | `app/routes/exam_exports.py`, `app/exam_exports_core.py` | `data/grades/session.json`, export ZIP/CSV responses, archive snapshots | Human scores in `session.json` remain authoritative. `/export_tf_bundle` validates and packages scans, session, rubric, and every `segments_*.json` needed for model-free grading. |
| Classwork workflows | `app/routes/classwork.py`, `app/classwork_core.py` | `data/classwork/`, `data/canvas_cache/`, classwork state in session | Semi-independent from exam strip grading. |

When more than one WorkUnit has the same `student_id`, the Finish page compares
all units and their problem scores. CSV and feedback exports use the
highest-total unit (first in session order on a tie) rather than summing
attempts. Instructor-side TF bundle creation requires an explicit choice to
exclude a duplicate or keep both. Excluding a unit prunes its prepared segment
rows and any reference pins; the underlying local scan files are retained.

* `app/server.py` – FastAPI app shell plus the remaining exam setup/import and legacy grading routes.
* `app/routes/classwork.py` – Classwork workflow routes, including Canvas classwork integration.
* `app/routes/exam_exports.py` – Exam finish/export/finalize routes.
* `app/routes/problem_strips.py` – Per-problem strip grading routes and segmentation entry points.
* `app/proof_ocr_review.py` – TFL/Fitch strip-review queue, state file helpers, proof crop preparation, status summaries, and acceptance persistence.
* `app/routes/flashcards.py` – Student-name flashcards workflow.
* `app/routes/fitch_debug.py` – Fitch debug / human-correction workflow.
* `app/classwork_core.py` – Classwork-specific helper and export logic.
* `app/exam_shared.py` – Shared exam-side helpers used by setup/import and exports.
* `app/exam_exports_core.py` – Export-specific exam helper logic.
* `app/session_state.py` – Shared session load/save functions and locks.
* `scripts/` – Offline CLI utilities (auto-grouping, debugging, code-map generation).
* `docs/` – Design documentation.
* `data/` – **Live quiz data** (FERPA-sensitive).
    * `raw_scans/` – Images (both raw `IMG_*.jpg` and renamed `quiz_*.jpg`).
    * `grades/` – The database (`session.json`) and segmentation cache.
    * `auto_grouped/` – Intermediate CSV outputs from the CLI.
* `data_archive/` – Snapshots of finished assignments.

---

> **Historical note:** Earlier v0/v1 architecture descriptions (SQLite references,
> thick-client UI, Flask mentions, duplicated sections) have been removed to avoid
> confusion. The current architecture is FastAPI + `session.json` + Jinja2 templates
> as described in sections 1–8 above. See git history for the deprecated sections if needed.
