


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
* **Invalidation:** Automatic on mismatch with the active `session.json` (or manual via refresh/delete).
* **Validation before reuse:** must match current WorkUnits (`unit_id`s), each cached `page_filename` must belong to that WorkUnit, and bbox must be normalized `[x1,y1,x2,y2]` in `[0,1]`.
* **Behavior:** Lazy-loaded. The first time you grade "Q1", you pay the wait time. Subsequent loads are instant.

---

# 8. Repository Layout (Code vs Data)

* `app/server.py` – FastAPI app shell plus the remaining exam setup/import and legacy grading routes.
* `app/routes/classwork.py` – Classwork workflow routes, including Canvas classwork integration.
* `app/routes/exam_exports.py` – Exam finish/export/finalize routes.
* `app/routes/problem_strips.py` – Per-problem strip grading routes and segmentation entry points.
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
