# Classwork grading backend — architecture map

> **Purpose:** Quick reference for vibecoding sessions. Explains the backend plumbing
> (what calls what, where state lives, what mutates) so you can make changes without
> re-reading server.py from scratch.
>
> Last updated from code review: March 2026

---

## 1. Module dependency map

Everything runs inside one FastAPI process. `server.py` is the only file with routes — it imports everything else as libraries.

```
server.py (all routes, all orchestration)
├── storage.py          State I/O: load/save session, roster, rubric.
│                        Defines DATA_DIR, all path constants.
│                        Owns archive_current_data().
│
├── models.py           Pydantic models: Session, WorkUnit, GradingKey,
│                        ClassworkModeState, ClassworkBatchState,
│                        ClassworkEntry, ClassworkFileRef.
│
├── canvas_client.py    Canvas REST API wrapper (httpx).
│                        Caching layer (canvas_cache/ dir, TTL-based).
│                        Methods: get_courses, get_assignments,
│                        get_gradebook_csv, download_assignment_submissions,
│                        bulk_post_comments.
│
├── vlm_client.py       LM Studio / Qwen HTTP client (OpenAI-style).
│                        Used by segmentation and name recognition.
│
├── vlm_classifier.py   Page-level name classifier. Uses vlm_client.
│
├── auto_group_core.py  Name recognition + page grouping pipeline.
│                        Uses vlm_classifier + vlm_client.
│
├── segmentation_core.py  Per-question answer cropping via VLM.
│                          Uses vlm_client.
│
└── verifier.py + syntax.py   Fitch proof checker (deterministic, no VLM).
```

**Key insight:** The classwork workflow does NOT use vlm_client, vlm_classifier, auto_group_core, or segmentation_core at all. It's a simpler path through the system: ingestion → review → finalize.

---

## 2. Session state anatomy (classwork mode)

`session.json` is the single source of truth. For classwork, the relevant fields are:

```
Session (models.py)
├── roster, grading_key, work_units, ...    ← quiz grading (not used by classwork)
│
├── classwork_mode: ClassworkModeState?     ← pointer to the ACTIVE assignment
│   (this is literally one of the objects inside classwork_batch.assignments)
│
└── classwork_batch: ClassworkBatchState?   ← the batch container
    ├── gradebook_token: str                ← lookup key for tmp CSV on disk
    ├── id_column: str                      ← which CSV column has student IDs
    └── assignments: List[ClassworkModeState]
        └── [0]: ClassworkModeState
            ├── assignment_id: str          ← slug like "wk04_attendance_q"
            ├── max_points: float           ← e.g. 10.0 or 1.0
            ├── gradebook_column: str?      ← target column in Canvas CSV
            ├── canvas_course_id: int?      ← set only if Canvas API path
            ├── canvas_assignment_id: int?  ← set only if Canvas API path
            ├── gradebook_token: str        ← same as batch-level token
            ├── id_column: str
            ├── entries: List[ClassworkEntry]
            │   └── ClassworkEntry
            │       ├── student_id, student_name
            │       ├── has_submission: bool
            │       ├── files: List[ClassworkFileRef]
            │       │   └── stored_filename, original_filename, ext
            │       ├── score: float
            │       └── comment: str?
            └── unmatched_files: List[str]
```

**The classwork_mode ↔ batch.assignments relationship:**
`classwork_mode` is a **pointer** — it points to one of the `ClassworkModeState` objects inside `batch.assignments`. When you switch assignments (via `batch/open` or `batch/open_next`), it just reassigns `session.classwork_mode` to a different element of the list. When you save scores during review, `_classwork_sync_mode_into_batch()` copies the modified mode object back into the batch's list.

---

## 3. On-disk state (classwork-related files)

All paths relative to DATA_DIR:

| Path | Written by | Purpose |
|------|-----------|---------|
| `grades/session.json` | persist_session() | All classwork state lives here |
| `gradebook_merge/tmp/{token}.csv` | classwork setup routes | Original gradebook CSV, kept for merge at finalize |
| `classwork/files/{uuid}_{original}` | _classwork_build_mode_from_zip() | Extracted student submission files |
| `classwork/exports/{export_id}/` | finalize routes | Merged CSV + comment ZIPs + manifest.json |
| `canvas_cache/` | canvas_client.py | Cached API responses (courses, assignments, gradebook) |
| `canvas_cache/submissions/` | server.py (canvas_setup) | Downloaded submission files (temp) |

---

## 4. Route map — classwork workflow

### Phase 1: Ingestion (two alternative paths)

**Manual path (working):**

| Route | What it does |
|-------|-------------|
| `GET /classwork` | Renders setup page (classwork_setup.html) |
| `POST /classwork/batch/start` | Accepts gradebook CSV upload + first submissions ZIP. Parses gradebook, builds roster, calls `_classwork_build_mode_from_zip()`, creates `ClassworkBatchState`. |
| `POST /classwork/batch/add` | Accepts additional ZIP for another assignment. Same processing, appends to batch. |

**Canvas API path (buggy):**

| Route | What it does |
|-------|-------------|
| `GET /classwork/canvas_setup` | Renders canvas setup page |
| `POST /classwork/canvas_setup` | **Does everything in one request:** fetches gradebook, fetches all assignments, downloads ALL submission files for ALL selected assignments, creates ZIP in memory, runs `_classwork_build_mode_from_zip()` for each. This is the problematic handler. |

**Both paths converge on the same internal functions:**
1. `_parse_gradebook_csv(text)` → fieldnames, rows, id_column
2. `_classwork_resolve_roster(source, rows, id_col)` → List[RosterEntry]
3. `_classwork_build_mode_from_zip(zip_bytes, ...)` → ClassworkModeState
   - Extracts ZIP, matches files to roster via `_classwork_match_file_to_roster()`
   - Writes files to `classwork/files/`
   - Creates ClassworkEntry per roster student (has_submission = True/False)

### Phase 2: Review

| Route | What it does |
|-------|-------------|
| `GET /classwork/review` | Reads `session.classwork_mode`, renders review page |
| `POST /classwork/review` | Calls `_classwork_apply_form_updates()`: loops form fields `score_{sid}` and `comment_{sid}`, updates entries. Then `_classwork_sync_mode_into_batch()` + `persist_session()`. |
| `POST /classwork/batch/open` | Switches classwork_mode to a specific assignment_id |
| `POST /classwork/batch/open_next` | Advances to next assignment in batch |
| `POST /classwork/batch/set_column` | Sets gradebook_column for an assignment |
| `GET /classwork/preview_text/{file}` | Returns text preview of DOCX/TXT submission files |

### Phase 3: Finalize

| Route | What it does |
|-------|-------------|
| `GET /classwork/batch/finalize` | Renders finalize page with assignment stats |
| `POST /classwork/batch/finalize` | Reads tmp gradebook, merges scores into correct columns, checks for conflicts (non-empty cells). Writes merged CSV + per-assignment comment ZIPs to `classwork/exports/{id}/`. |
| `POST /classwork/finalize` | Single-assignment finalize (non-batch). Similar but simpler. |
| `POST /classwork/batch/close_session` | Nulls out classwork_batch and classwork_mode. |
| `POST /classwork/canvas_post_comments` | Reads batch state, posts comments to Canvas API. |

### Export / download routes

| Route | Returns |
|-------|---------|
| `GET /classwork/export/{id}/gradebook` | Merged gradebook CSV |
| `GET /classwork/export/{id}/comments` | Comments reupload ZIP |
| `GET /classwork/batch/export/{id}/gradebook` | Batch merged CSV |
| `GET /classwork/batch/export/{id}/comments/{assignment_id}` | Per-assignment comment ZIP |

---

## 5. Key internal functions (classwork)

These are the private functions in server.py that do the actual work:

| Function | Purpose |
|----------|---------|
| `_parse_gradebook_csv(text)` | Parse Canvas gradebook CSV. Returns (fieldnames, rows, id_column). Auto-detects "SIS Login ID" or "student_id" column. |
| `_classwork_resolve_roster(source, rows, id_col)` | Build roster from gradebook rows. |
| `_classwork_match_file_to_roster(filename, roster_index, sid_index)` | Match a ZIP filename to a student. Tries: Canvas prefix key → student ID token → normalized name. |
| `_classwork_build_mode_from_zip(...)` | Core ingestion: extract ZIP → match files → create ClassworkModeState. |
| `_classwork_apply_form_updates(form, mode, action)` | Apply score/comment changes from the review form to entries. |
| `_classwork_sync_mode_into_batch(session)` | Copy current classwork_mode back into its slot in batch.assignments. |
| `_build_classwork_score_map(mode)` | Dict of student_id → score for merge. |
| `_classwork_open_next_batch_assignment(session)` | Advance classwork_mode to next in batch. |
| `_classwork_batch_assignment_stats(batch, rows, id_col)` | Compute stats for finalize page. |
| `_classwork_gradebook_tmp_path(token)` | Resolve gradebook_merge/tmp/{token}.csv path. |

---

## 6. The Canvas API bug

**Current behavior of `POST /classwork/canvas_setup`:**

1. Fetch gradebook CSV (~2s)
2. Parse gradebook + resolve roster (instant)
3. Fetch all assignments (~2s)
4. **FOR EACH selected assignment: `client.download_assignment_submissions()`** ← THE PROBLEM
   - Downloads ALL files for ALL students for EACH selected assignment
   - N assignments × M students × K files = potentially hundreds of HTTP requests
   - All synchronous, all blocking, all in one handler
5. Pack everything into in-memory ZIP
6. Run `_classwork_build_mode_from_zip()` for each assignment
7. Save session, redirect to review

**Why the manual path works:** You upload one ZIP at a time via `batch/start` then `batch/add`. Each is a small, bounded operation.

**What the Canvas path does wrong:** Step 4 downloads everything eagerly. If you select 5 assignments with 30 students each, that's 150+ file download requests before you see any UI.

---

## 7. Concurrency model

- Single-threaded FastAPI with `_session_lock` (threading.Lock) guarding all session reads/writes.
- `get_session()` checks file mtime to detect external changes; caches in `_session` global.
- `persist_session()` writes to disk immediately.
- No async I/O for Canvas API calls — `canvas_client.py` uses synchronous `httpx.Client`.
- The VLM client is also synchronous (only used in quiz workflow, not classwork).

---

## 8. Static file serving

| Mount path | Directory | Used for |
|------------|----------|----------|
| `/scans/{filename}` | DATA_DIR/raw_scans/ | Quiz scan images |
| `/classwork_files/{filename}` | DATA_DIR/classwork/files/ | Classwork submission files |
| `/debug_fitch_assets/` | DATA_DIR/debug_fitch/ | Fitch OCR debug images |
| `/flashcards_assets/` | DATA_DIR/flashcards/ | Flashcard images |
