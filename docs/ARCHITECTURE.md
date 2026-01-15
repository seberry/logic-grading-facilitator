


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

3.  **`segments_*.json` (The Geometry):**
    * Stores bounding boxes for where specific answers are located on a page.
    * Generated lazily by the VLM.
    * Located at: `data/grades/segments_{problem_id}.json`.

---

# 3. UI Flow & Views

The frontend serves specific interfaces for different phases of the grading lifecycle.

### A. Name Review View (`GET /name_review`)
* **Purpose:** rapid audit of VLM-grouped pages before grading begins.
* **Input:** Reads draft predictions from CSV (CLI output) or JSON cache.
* **Action:** User confirms identity matches and can use **Save & Auto-Fill Unknowns** to cascade the last chosen student onto later "Unknown" rows without renaming files yet.
* **Transition:** Submitting this form triggers the **Hand-off** (see Section 8).

### B. Strip View Grading (`GET /problem/{id}/strips`)
* **Purpose:** High-speed grading of **one question across all students**.
* **Display:**
    * Shows a cropped "strip" of the student's answer (via segmentation).
    * Provides full-page toggle and scoring inputs.
* **Navigation:** "Save & Next" moves linearly through the rubric.

### C. WorkUnit View (Legacy/Detail) (`GET /grade/{unit_id}/page`)
* **Purpose:** Detailed view of a single student's full exam.
* **Use Case:** Spot-checking a specific student or handling complex multi-page overrides.

### D. Fitch Debug & Human-Correction View (`GET /debug/fitch`)
* **Purpose:** Inspect highlighted proof scans, hand-edit stitched `final_structure.json`, and run the verifier inline.
* **Inputs:** Reads artifacts from `data/debug_fitch/<scan_stem>/` (produced by `scripts/debug_fitch_ocr.py` against a cropped proof scan).
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

2.  **Visual "Prompt Injection" (New):**
    * For structural tasks (Fitch Proofs), we do not rely on the VLM to see geometry.
    * [cite_start]We use OpenCV to detect geometric features (vertical bars) and "inject" them into the image as high-contrast **Color Overlays**[cite: 928].
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
* **Trigger:** User clicks **"Save & Start Grading"** or **"Save & Download Scans (ZIP)"** on the `/name_review` page (the earlier **"Save & Auto-Fill Unknowns"** button only updates the cache and does not rename files).
* **Backend Logic (`POST /name_review`):**
    1.  **Renaming:** Copies raw files to `data/raw_scans/` with canonical names (e.g., `quiz1_student_p01.jpg`) and leaves originals intact.
    2.  **Session Creation:** Constructs `WorkUnit` objects in memory based on the approved groups.
    3.  **Persistence:** Writes the **Source of Truth** to `data/grades/session.json`.
    4.  **Redirect:** Sends the user immediately to the Strip View grader.
* **State:** "Live Grading Mode." The system now ignores the CSV and relies on `session.json`.

### Phase 3: Grading & Segmentation
* **Trigger:** User loads `/problem/Q1/strips`.
* **Logic:**
    1.  Server checks `data/grades/segments_Q1.json`.
    2.  **Cache Miss:** If missing, runs VLM on all pages to find "Q1" boxes. Saves to JSON.
    3.  **Cache Hit:** If present, loads boxes instantly.
* **Result:** User sees cropped answers. Scores are written back to `session.json`.

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
* **Invalidation:** Currently manual (delete file to re-run).
* **Behavior:** Lazy-loaded. The first time you grade "Q1", you pay the wait time. Subsequent loads are instant.

---

# 8. Repository Layout (Code vs Data)

* `app/` – Core application code (Server, Models, UI Logic).
* `scripts/` – Offline CLI utilities (Auto-grouping, Debugging).
* `docs/` – Design documentation.
* `data/` – **Live quiz data** (FERPA-sensitive).
    * `raw_scans/` – Images (both raw `IMG_*.jpg` and renamed `quiz_*.jpg`).
    * `grades/` – The database (`session.json`) and segmentation cache.
    * `auto_grouped/` – Intermediate CSV outputs from the CLI.
* `data_archive/` – Snapshots of finished assignments.



## Historical (Deprecated)
The sections below describe earlier v0/v1 ideas (SQLite, thick-client UI, etc.). They are kept for reference but do not match the current implementation.

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

# 8Data Flow & Caching Strategy (v2)
The system uses a Hybrid Caching Strategy to bridge offline scripts and the live server.

A. The "Name Review" Pipeline
This pipeline ensures we never run the VLM twice on the same data.

Producer (CLI):

Script: scripts.auto_group_scans.py

Action: Runs VLM on all raw scans.

Output: data/auto_grouped/{quiz_id}/{quiz_id}_page_predictions.csv

Role: The canonical source of "batch" inference.

Consumer (Server):

Endpoint: GET /name_review

Logic:

Check CLI Output: First, looks for the CSV from the CLI. If found, loads it (Fast).

Check Internal Cache: If no CSV, looks for data/name_review_cache/{quiz_id}__root.json.

Fallback: If neither exists, runs VLM live (Slow) and saves to JSON cache.

B. The Session Hand-off
Trigger: User clicks **Save & Start Grading** or **Save & Download Scans (ZIP)**. The **Save & Auto-Fill Unknowns** option is a pre-flight cache save only and does not touch the filesystem.

Action: Server creates a Session object in memory and writes data/grades/session.json.

Critical Note: This session.json is the Source of Truth for the grading phase. Running scripts/init_session.py after this step is dangerous as it will overwrite the grouped session with a blank one.

# 9. Summary

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

### Per-Problem Grading Flow (v0.3 update)

The strip view now supports full grading interaction:
- Each row (strip) displays a student’s cropped answer and scoring widget.
- A dedicated POST endpoint saves scores across all WorkUnits.
- Navigation buttons (“Save & Next”, “Save & Back”) step through the rubric order.
- Per-problem rubric notes are stored in Session.problem_notes and displayed at the top of each strip view.


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
