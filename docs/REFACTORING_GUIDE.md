# Refactoring guide: making the grading facilitator less confusing

> **Status:** Superseded as the primary plan on 2026-04-02.
> Use [docs/REFACTORING_STATUS.md](/c:/Users/seber/logic-grading-facilitator-MAIN/docs/REFACTORING_STATUS.md)
> as the current source of truth.
>
> This guide still contains useful inventory work and function-grouping notes, but
> parts of it assume a larger and more ambitious split than the project currently
> wants. Treat it as background reference, not as the active implementation plan.

> **Who this is for:** Future-Sharon vibecoding, and any coding AI working on the project.
> Written while the full codebase was loaded in context, so the function-by-function
> classifications should be accurate. Verify any specific line numbers before using
> them — they'll drift as code changes.

---

## Table of contents

1. [Refactor 1: Split server.py into route modules](#refactor-1-split-serverpy-into-route-modules)
2. [Refactor 2: Separate Session models for each workflow](#refactor-2-separate-session-models)
3. [Refactor 3: Extend LLM_README.yaml with classwork coverage](#refactor-3-extend-llm_readmeyaml)

These are ordered by impact. Refactor 1 is the biggest win. Refactor 3 is the safest
(pure documentation). You can do them in any order or independently.

---

## Refactor 1: Split server.py into route modules

### Why

`server.py` is ~5,500 lines with ~80 route handlers for five unrelated workflows
(exam grading, classwork grading, Fitch proof debug, flashcards, setup wizard)
interleaved with their private helper functions. When working on classwork, you
scroll past Fitch proof handlers and strip-view segmentation code. An LLM working
on classwork has all of that in its context window competing for attention.

### Target structure

```
app/
  server.py                  # App creation, middleware, mounts, includes routers
  session_state.py           # NEW: shared session globals + get/persist helpers
  helpers.py                 # NEW: cross-workflow utility functions
  routes/
    __init__.py
    classwork.py             # /classwork/* routes (manual CSV path)
    classwork_canvas.py      # /classwork/canvas_* routes (Canvas API path)
    exam_grading.py          # /grade/*, /problem/*, /work_units/*, /name_review
    exam_export.py           # /export*, /finish, /finalize, gradebook_merge
    setup.py                 # /setup/* routes
    fitch_debug.py           # /debug/fitch/* routes
    flashcards.py            # /flashcards/* routes
```

### Critical safety rules

**Rule 1: Shared state must live in ONE place.**

These globals currently live in `server.py`:

```python
_session_lock = threading.Lock()
_session: Optional[Session] = None
_session_mtime_ns: Optional[int] = None
_segmentation_lock = threading.Lock()
```

Along with `get_session()` and `persist_session()`.

Move ALL of these to a new `app/session_state.py`. Every route module imports from there:

```python
# app/session_state.py
import threading
from typing import Optional
from .models import Session, GradingKey
from .storage import load_session, save_session, DEFAULT_SESSION_PATH

_session_lock = threading.Lock()
_session: Optional[Session] = None
_session_mtime_ns: Optional[int] = None
_segmentation_lock = threading.Lock()

def get_session() -> Session:
    # ... existing implementation ...

def persist_session() -> None:
    # ... existing implementation ...
```

```python
# In any route module:
from app.session_state import get_session, persist_session, _session_lock
```

**DO NOT copy these into each route file.** If two files each define their own
`_session_lock = threading.Lock()`, you get two independent locks, and concurrent
requests can corrupt `session.json`. This is the #1 risk of this refactor.

**Rule 2: Never duplicate a helper function — extract to a shared module.**

Some helpers are used by multiple workflows. These must go in `app/helpers.py`
(or stay in whatever module they're already in), not be copied into each route file.

**Rule 3: Use FastAPI APIRouter.**

Each route module creates a router and the main `server.py` includes it:

```python
# app/routes/classwork.py
from fastapi import APIRouter
router = APIRouter()

@router.get("/classwork", response_class=HTMLResponse)
def classwork_setup_page(request: Request) -> Response:
    ...

# app/server.py
from .routes.classwork import router as classwork_router
app.include_router(classwork_router)
```

### Function-by-function classification

Every function and route handler in `server.py`, classified by where it should go.

#### → `app/session_state.py` (shared mutable state)

| Function | Line | Notes |
|----------|------|-------|
| `_session_lock`, `_session`, `_session_mtime_ns` | 440-442 | Module globals |
| `_segmentation_lock` | 443 | Used only by exam but keep with other locks |
| `get_session()` | 446 | |
| `persist_session()` | 467 | |

#### → `app/helpers.py` (cross-workflow utilities)

These are used by more than one workflow. Extract them here and have all route
modules import from `app.helpers`.

| Function | Line | Used by |
|----------|------|---------|
| `_parse_gradebook_csv()` | 858 | classwork, gradebook_merge, setup, canvas |
| `_validate_roster_csv()` | 764 | setup |
| `_normalize_headers()` | 773 | called by _parse_gradebook_csv and _extract_canvas_roster |
| `_extract_canvas_roster()` | 781 | setup |
| `_candidate_assignment_columns()` | 884 | classwork, gradebook_merge |
| `_assignment_column_stats()` | 912 | gradebook_merge (but could be used by classwork) |
| `_unique_path()` | 2622 | classwork, setup, flashcards, exam |
| `_unique_pdf_stem()` | 2636 | setup |
| `_normalize_canvas_key()` | 2796 | setup, exam export |
| `_filename_prefix_key()` | 2801 | classwork, setup |
| `_build_roster_prefix_index()` | 2808 | classwork, setup, exam export |
| `_lookup_student_name()` | 528 | exam grading |
| `_render_feedback_image_bytes()` | 4360 | classwork finalize, exam export |
| `_render_feedback_pdf_bytes()` | 4381 | classwork finalize, exam export |
| `natural_sort_key()` | 190 | setup, exam |
| `canonical_filename()` | 202 | setup, exam |
| `_is_image_filename()` | 233 | exam |
| `_save_image_upload()` | 2645 | setup |
| `_save_pdf_pages()` | 2652 | setup |
| `_save_pdf_pages_with_paths()` | 2669 | setup |
| `_ingest_zip_bytes()` | 2686 | setup |

#### → `app/routes/fitch_debug.py`

| Item | Line |
|------|------|
| `list_fitch_debug_scans()` | 263 |
| `final_structure_path()` | 269 |
| `load_structure_text()` | 273 |
| `find_scan_image_path()` | 280 |
| `scan_image_url()` | 301 |
| `_clean_meta()` | 316 |
| `normalize_structure_payload()` | 337 |
| `load_structure_json()` | 366 |
| `ensure_raw_scan_available()` | 379 |
| `run_subprocess_with_logs()` | 397 |
| `structure_to_proof_lines()` | 409 |
| `label_and_count_premises()` | 425 |
| `GET /debug/fitch` | 604 |
| `POST /debug/fitch/save` | 626 |
| `POST /debug/fitch/rerun` | 646 |
| `POST /debug/fitch/run_grader` | 674 |

#### → `app/routes/classwork.py` (manual CSV path)

| Item | Line |
|------|------|
| `_classwork_gradebook_tmp_path()` | 943 |
| `_sanitize_assignment_id()` | 947 |
| `_classwork_make_assignment_id_from_column()` | 953 |
| `_classwork_unique_assignment_id()` | 957 |
| `_normalize_match_key()` | 969 |
| `_classwork_resolve_roster()` | 973 |
| `_classwork_match_file_to_roster()` | 995 |
| `_classwork_build_gradebook_roster()` | 1027 |
| `_classwork_entry_default_score()` | 1043 |
| `_build_classwork_score_map()` | 1047 |
| `_classwork_zip_unique_name()` | 1054 |
| `_classwork_comment_text()` | 1070 |
| `_classwork_apply_form_updates()` | 1080 |
| `_classwork_analysis()` | 1119 |
| `_classwork_build_mode_from_zip()` | 1139 |
| `_classwork_sync_mode_into_batch()` | 1232 |
| `_same_numeric_score()` | 1243 |
| `_classwork_open_next_batch_assignment()` | 1250 |
| `_classwork_batch_assignment_stats()` | 1268 |
| `_classwork_resolve_stored_file()` | 1305 |
| `_classwork_extract_docx_text()` | 1315 |
| `_classwork_extract_text_preview()` | 1326 |
| `GET /classwork` | 1381 |
| `POST /classwork/setup` | 1409 |
| `POST /classwork/batch/start` | 1476 |
| `POST /classwork/batch/add` | 1509 |
| `POST /classwork/batch/open` | 1577 |
| `POST /classwork/batch/open_next` | 1593 |
| `POST /classwork/batch/remove` | 1603 |
| `POST /classwork/batch/set_column` | 1621 |
| `POST /classwork/batch/clear` | 1660 |
| `GET /classwork/batch/finalize` | 1670 |
| `POST /classwork/batch/finalize` | 1700 |
| `POST /classwork/batch/close_session` | 1843 |
| `GET /classwork/review` | 1873 |
| `POST /classwork/review` | 1899 |
| `POST /classwork/finalize` | 1920 |
| `GET /classwork/preview_text/...` | 2077 |
| `GET /classwork/export/...` | 2091 |
| `GET /classwork/batch/export/.../gradebook` | 2122 |
| `GET /classwork/batch/export/.../comments/...` | 2142 |

Note: `_classwork_apply_form_updates`, `_classwork_sync_mode_into_batch`,
`_classwork_analysis`, `_build_classwork_score_map`, and a few other helpers
are also used by the Canvas path. When `classwork_canvas.py` is a separate file,
it should import these from `classwork.py`. Alternatively, extract the shared
classwork helpers into `app/classwork_helpers.py` and have both route files
import from there.

#### → `app/routes/classwork_canvas.py` (Canvas API path)

| Item | Line |
|------|------|
| `_create_submissions_zip_from_downloads()` | 92 |
| `GET /classwork/canvas_courses` | 1340 |
| `GET /classwork/canvas_assignments` | 1351 |
| `GET /classwork/canvas_submission_counts` | 1362 |
| `GET /classwork/canvas_setup` | 2163 |
| `POST /classwork/canvas_setup` | 2178 |
| `POST /classwork/canvas_post_comments` | 2304 |

#### → `app/routes/setup.py`

| Item | Line |
|------|------|
| `_has_existing_session_state()` | 555 |
| `_read_default_rubric()` | 567 |
| `GET /setup` | 712 |
| `POST /setup/archive_reset` | 755 |
| `POST /setup/roster` | 2396 |
| `POST /setup/roster_gradebook` | 2437 |
| `POST /setup/upload_scans` | 2716 |
| `POST /setup/import_canvas_zip` | 2824 |
| `POST /setup/save_rubric` | 3026 |
| `POST /setup/select_rubric` | 3045 |
| `POST /setup/run_auto_grouping` | 3064 |

#### → `app/routes/flashcards.py`

| Item | Line |
|------|------|
| `_flashcards_session_dir()` | 3104 |
| `_flashcards_state_path()` | 3108 |
| `_load_flashcards_state()` | 3112 |
| `_save_flashcards_state()` | 3120 |
| `_normalize_for_match()` | 3127 |
| `_guess_name_from_filename()` | 3133 |
| `_safe_extract_zip()` | 3139 |
| `_render_pdf_to_images()` | 3158 |
| `_extract_docx_assets()` | 3173 |
| `_slugify_name()` (flashcards version) | 3191 |
| `_parse_bbox_value()` | 3198 |
| `GET /flashcards` | 3212 |
| `POST /flashcards/upload` | 3232 |
| `GET /flashcards/review` | 3366 |
| `POST /flashcards/review` | 3425 |
| `POST /flashcards/export` | 3462 |

#### → `app/routes/exam_grading.py`

| Item | Line |
|------|------|
| `_select_display_filename()` | 237 |
| `_build_page_options()` | 249 |
| `_build_page_rows_for_template()` | 475 |
| `next_unit_id()` | 536 |
| `name_review_cache_path()` | 541 |
| `find_work_unit()` | 3564 |
| `WorkUnitSummary` (Pydantic model) | near 3571 |
| `GradeUpdate` (Pydantic model) | near 3737 |
| `GET /session/summary` | 3538 |
| `GET /work_units` | 3571 |
| `POST /work_units/create` | 3592 |
| `GET /name_review` | 3629 (and duplicate at 4635) |
| `POST /name_review` | 4795 |
| `POST /grade/{unit_id}` (JSON API) | 3736 |
| `POST /work_units/{unit_id}/append_next_scan` | 3785 |
| `POST /work_units/{unit_id}/remove_last_scan` | 3804 |
| `GET /grade/{unit_id}/page` | 4021 |
| `POST /grade/{unit_id}/page` | 4466 |
| `GET /problem/{problem_id}/strips` | 5208 |
| `POST /problem/{problem_id}/strips` | 5313 |
| `POST /problem/{problem_id}/type` | 5465 |
| `POST /problem/{problem_id}/prefetch` | 5194 |
| `_compute_segments_data()` | 5037 |
| `_is_normalized_bbox()` | 5091 |
| `_segments_cache_matches_session()` | 5103 |
| `_clear_segment_caches()` | 5138 |
| `ensure_segments_cached()` | 5148 |
| `PROBLEM_TYPE_OPTIONS` constant | 182 |
| `IMAGE_EXTENSIONS` constant | 230 |

**Note:** There is a duplicate `GET /name_review` at lines 3629 and 4635.
FastAPI will use whichever is registered last. During the split, determine
which one is current and drop the other.

#### → `app/routes/exam_export.py`

| Item | Line |
|------|------|
| `_build_total_scores_by_student()` | 817 |
| `_required_total_max()` | 831 |
| `_score_summary_for_work_unit()` | 835 |
| `slugify_name()` (exam version) | 3827 |
| `roster_lookup()` | 3839 |
| `GET /finish` | 3844 |
| `GET /export_renamed_scans` | 3892 |
| `GET /export` | 3947 |
| `POST /finalize` | 3978 |
| `GET /export_feedback_zip` | 4051 |
| `_list_canvas_original_zips()` | 4145 |
| `_canvas_import_manifest_path()` | 4170 |
| `_load_canvas_import_manifest()` | 4175 |
| `_save_canvas_import_manifest()` | 4185 |
| `_find_canvas_original_zip_for_quiz()` | 4218 |
| `_build_best_rows_by_student()` | 4288 |
| `_feedback_text_from_row()` | 4324 |
| `GET /export_canvas_reupload_feedback_zip` | 4396 |

#### → `app/routes/gradebook_merge.py`

| Item | Line |
|------|------|
| `GET /gradebook_merge` | 2474 |
| `POST /gradebook_merge/analyze` | 2489 |
| `POST /gradebook_merge/merge` | 2531 |

#### → `app/server.py` (what stays)

After the split, `server.py` becomes very small:

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from .storage import RAW_SCANS_DIR, DEBUG_FITCH_DIR, FLASHCARDS_DIR, DATA_DIR

app = FastAPI(title="Logic Grading Facilitator")

# Middleware
app.add_middleware(CORSMiddleware, ...)

# Static file mounts
app.mount("/scans", StaticFiles(directory=str(RAW_SCANS_DIR)), name="scans")
app.mount("/debug_fitch_assets", StaticFiles(directory=str(DEBUG_FITCH_DIR)), ...)
app.mount("/flashcards_assets", StaticFiles(directory=str(FLASHCARDS_DIR)), ...)
app.mount("/classwork_files", StaticFiles(directory=str(CLASSWORK_FILES_DIR)), ...)

# Templates (shared by all routes)
TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# Include routers
from .routes.classwork import router as classwork_router
from .routes.classwork_canvas import router as canvas_router
from .routes.exam_grading import router as exam_router
from .routes.exam_export import router as export_router
from .routes.setup import router as setup_router
from .routes.fitch_debug import router as fitch_router
from .routes.flashcards import router as flashcards_router
from .routes.gradebook_merge import router as merge_router

app.include_router(classwork_router)
app.include_router(canvas_router)
app.include_router(exam_router)
app.include_router(export_router)
app.include_router(setup_router)
app.include_router(fitch_router)
app.include_router(flashcards_router)
app.include_router(merge_router)

@app.get("/")
def root():
    return RedirectResponse(url="/setup")
```

The `templates` object needs to be accessible to all route modules. Either:
- Import it from `server.py` in each route module (circular import risk — be careful), or
- Create it in a shared `app/template_config.py` and import from there (safer).

### How to verify the split worked

After the refactor, run these checks:

1. `python -m uvicorn app.server:app --port 5000` starts without import errors.
2. Visit `/setup` — setup wizard loads.
3. Visit `/classwork` — classwork setup loads.
4. Visit `/debug/fitch` — Fitch debug page loads.
5. Visit `/flashcards` — flashcards page loads.
6. If you have a live session.json, visit `/finish` — finish page loads with data.
7. Check that no function is defined in more than one file (`grep -rn "def function_name" app/`).

---

## Refactor 2: Separate Session models

### Why

The `Session` model is a god object. It holds:

- Quiz grading state: `roster`, `grading_key`, `work_units`, `unclaimed_pages`, `pinned_note`, `problem_notes`, `problem_types`
- Classwork state: `classwork_mode`, `classwork_batch`

When the classwork workflow creates a session, it has to provide dummy values for
`grading_key` (with `quiz_id="classwork"` and `max_points=0`) and empty `work_units` /
`unclaimed_pages`. These fields are meaningless for classwork. Similarly, when the
exam workflow is active, `classwork_mode` and `classwork_batch` are None but still
present in the JSON.

For the Canvas API workflow (see `CANVAS_WORKFLOW_REDESIGN_SPEC.md`), even the
classwork-specific fields like `gradebook_token` and `gradebook_column` are
irrelevant — the Canvas path doesn't use the CSV intermediary at all.

### Approach: incremental, not revolutionary

Don't try to split `session.json` into separate files right away. That would break
every existing function that calls `get_session()`. Instead:

**Phase 1: Extract workflow-specific sub-models (safe, backwards-compatible).**

Create focused interfaces that wrap the Session:

```python
# app/classwork_state.py

from .models import Session, ClassworkModeState, ClassworkBatchState

class ClassworkSessionView:
    """
    Thin wrapper providing a classwork-only view of the Session.
    Keeps all state in the underlying Session object (and therefore session.json)
    but gives classwork code a focused API that doesn't expose exam fields.
    """
    def __init__(self, session: Session):
        self._session = session

    @property
    def mode(self) -> Optional[ClassworkModeState]:
        return self._session.classwork_mode

    @mode.setter
    def mode(self, value: Optional[ClassworkModeState]):
        self._session.classwork_mode = value

    @property
    def batch(self) -> Optional[ClassworkBatchState]:
        return self._session.classwork_batch

    @batch.setter
    def batch(self, value: Optional[ClassworkBatchState]):
        self._session.classwork_batch = value

    @property
    def is_canvas_batch(self) -> bool:
        """True if the current batch came from Canvas API (not manual CSV)."""
        if self.batch and self.batch.assignments:
            return self.batch.assignments[0].canvas_course_id is not None
        return False
```

Classwork route handlers would then do:

```python
session = get_session()
cw = ClassworkSessionView(session)
if cw.batch is None:
    raise HTTPException(...)
cw.mode = cw.batch.assignments[0]
persist_session()  # still writes the whole session.json
```

This is a clarity improvement, not a data model change. `session.json` format
doesn't change at all, so existing data is fully compatible.

**Phase 2 (future, optional): Split into separate JSON files.**

If you later want to fully decouple:
- `data/grades/session.json` → exam-only state
- `data/grades/classwork_session.json` → classwork state
- Separate `get_exam_session()` / `get_classwork_session()` functions
- Each writes to its own file with its own lock

This is a bigger change and requires migrating any code that checks both workflows
(e.g., the archive function that copies both). Only do this if the wrapper approach
from Phase 1 still feels confusing after living with it for a while.

---

## Refactor 3: Extend LLM_README.yaml with classwork coverage

### Why

The existing `LLM_README.yaml` has detailed `user_flows` entries for the exam
workflow (setup wizard, name review, strip grading, export, archive) but nothing
for classwork. When an LLM works on classwork features, it has to read templates
and server.py to figure out what exists. Adding classwork entries to the YAML gives
the same "map" coverage that already exists for exams.

### What to add

Append these entries to the `user_flows` section in `LLM_README.yaml`. They follow
the exact same format as the existing entries.

```yaml
  # ---------------------------------------------------------------
  # CLASSWORK WORKFLOW (effort-based grading, no VLM)
  # ---------------------------------------------------------------

  - id: flow_classwork_setup
    human_name: "Classwork queue setup (manual CSV)"
    user_action: "Instructor uploads gradebook CSV and submissions ZIPs for effort-based grading."
    status: "live"

    entrypoints:
      - method: "GET"
        path: "/classwork"
        file: "app/server.py"
        template: "app/templates/classwork_setup.html"
      - method: "POST"
        path: "/classwork/batch/start"
        file: "app/server.py"
        template: "redirect → /classwork"
      - method: "POST"
        path: "/classwork/batch/add"
        file: "app/server.py"
        template: "redirect → /classwork"

    data_flow:
      reads:
        - path: "data/gradebook_merge/tmp/{token}.csv"
          purpose: "Previously stored gradebook CSV (for column list in batch/add)."
      writes:
        - path: "data/gradebook_merge/tmp/{token}.csv"
          what: "Uploaded gradebook CSV, stored for later merge at finalize."
        - path: "data/classwork/files/{uuid}_{original}"
          what: "Extracted submission files from uploaded ZIP."
        - path: "data/grades/session.json"
          what: "ClassworkBatchState with assignments list; ClassworkModeState per assignment."

    current_behavior:
      description: |
        Two-step flow: batch/start creates an empty ClassworkBatchState from a gradebook CSV.
        batch/add processes one submissions ZIP per call: extracts files, matches them to the
        roster via Canvas filename conventions / student ID tokens / normalized name matching,
        creates ClassworkEntry per roster student, and appends a ClassworkModeState to the batch.

        The setup page (/classwork) shows the queue overview: queued assignments with stats
        (submitted/missing counts), per-assignment "Open in Review" / "Remove" / "Save Column"
        buttons, and a "Finalize Queue" link.

    key_internal_functions:
      - name: "_parse_gradebook_csv(content)"
        returns: "(fieldnames, rows, id_column)"
        notes: "Auto-detects 'SIS Login ID' or 'student_id' as the ID column."
      - name: "_classwork_resolve_roster(source, rows, id_col)"
        returns: "List[RosterEntry]"
      - name: "_classwork_build_mode_from_zip(zip_bytes, ...)"
        returns: "ClassworkModeState with populated entries and file refs."
        notes: "Core ingestion function. Extracts ZIP, matches files to roster, writes files to classwork/files/."
      - name: "_classwork_match_file_to_roster(filename, roster_index, sid_index)"
        returns: "RosterEntry | None"
        notes: "Tries Canvas prefix key → student ID token → normalized name, in order."

    assumptions:
      - "Gradebook CSV is a standard Canvas export with 'Student' and 'SIS Login ID' columns."
      - "Submissions ZIP uses Canvas naming conventions (student name prefix in filename)."

  - id: flow_classwork_review
    human_name: "Classwork review (effort-based scoring)"
    user_action: "Instructor scrolls through submissions, adjusts scores, leaves comments."
    status: "live"

    entrypoints:
      - method: "GET"
        path: "/classwork/review"
        file: "app/server.py"
        template: "app/templates/classwork_review.html"
      - method: "POST"
        path: "/classwork/review"
        file: "app/server.py"
        template: "redirect → /classwork/review"

    data_flow:
      reads:
        - path: "data/grades/session.json"
          purpose: "session.classwork_mode (the active ClassworkModeState)."
        - path: "data/classwork/files/{stored_filename}"
          purpose: "Submission files for inline preview (images served via /classwork_files/ mount; text via /classwork/preview_text/)."
      writes:
        - path: "data/grades/session.json"
          what: "Updated scores and comments on ClassworkEntry objects."

    current_behavior:
      description: |
        Renders one row per student from session.classwork_mode.entries.
        Each row shows: student name, submission status, file preview (images inline,
        PDFs as links, text files via AJAX preview), score input (pre-filled: max_points
        if submitted, 0 if missing), and a comment textarea.

        Bulk actions available: "Credit All Submitted", "Set Missing to Zero".
        Keyboard shortcuts: f (full credit), z (zero), j/k (next/prev row), c (focus comment).

        POST calls _classwork_apply_form_updates() which loops over form fields
        score_{student_id} and comment_{student_id}, updates the entries, then calls
        _classwork_sync_mode_into_batch() to copy changes back to the batch list.

    ui_controls:
      - name: "Credit All Submitted"
        behavior: "Sets score = max_points for all entries where has_submission is True."
      - name: "Set Missing to Zero"
        behavior: "Sets score = 0 for all entries where has_submission is False."
      - name: "Filter toggles"
        behavior: "Show all / submitted only / missing only / commented only."
      - name: "Finalize button"
        behavior: "Posts to /classwork/finalize (single) or links to /classwork/batch/finalize (batch)."

    key_internal_functions:
      - name: "_classwork_apply_form_updates(form, mode, action)"
        notes: "Loops over form fields, updates ClassworkEntry.score and .comment."
      - name: "_classwork_sync_mode_into_batch(session)"
        notes: "Copies session.classwork_mode back into its slot in batch.assignments."
      - name: "_classwork_analysis(mode)"
        returns: "Dict with submitted_count, missing_count, commented_count, gradebook columns, etc."

  - id: flow_classwork_finalize
    human_name: "Classwork finalize (CSV merge + export)"
    user_action: "Instructor exports merged gradebook CSV and optional comments ZIP."
    status: "live"

    entrypoints:
      - method: "GET"
        path: "/classwork/batch/finalize"
        file: "app/server.py"
        template: "app/templates/classwork_batch_finalize.html"
      - method: "POST"
        path: "/classwork/batch/finalize"
        file: "app/server.py"
        template: "app/templates/classwork_batch_done.html"
      - method: "POST"
        path: "/classwork/finalize"
        file: "app/server.py"
        template: "app/templates/classwork_done.html"

    data_flow:
      reads:
        - path: "data/gradebook_merge/tmp/{token}.csv"
          purpose: "Original gradebook CSV to merge scores into."
        - path: "data/grades/session.json"
          purpose: "Batch state with per-assignment scores and comments."
        - path: "data/classwork/files/{stored_filename}"
          purpose: "Original submission files for comment ZIP (preserves filenames)."
      writes:
        - path: "data/classwork/exports/{export_id}/"
          what: "Merged gradebook CSV, per-assignment comment ZIPs, manifest.json."

    current_behavior:
      description: |
        Reads the stored gradebook CSV, merges classwork scores into the correct
        columns (one column per assignment, set via gradebook_column on each
        ClassworkModeState). Checks for non-empty cell conflicts and warns before
        overwriting.

        For each assignment with comments, generates a Canvas reupload ZIP
        preserving original filenames. Comment artifacts are rendered as
        feedback PDFs or images depending on original file type.

        batch/finalize handles multiple assignments at once; /classwork/finalize
        handles a single assignment.

    current_limitations:
      - "gradebook_column must be set on each assignment before finalize — no auto-matching."
      - "No preview of what will be written — goes straight to export."

  - id: flow_classwork_canvas
    human_name: "Classwork Canvas API integration"
    user_action: "Instructor selects Canvas course/assignments; grades posted via API."
    status: "broken — being redesigned (see CANVAS_WORKFLOW_REDESIGN_SPEC.md)"

    entrypoints:
      - method: "GET"
        path: "/classwork/canvas_setup"
        file: "app/server.py"
        template: "app/templates/classwork_canvas_setup.html"
      - method: "POST"
        path: "/classwork/canvas_setup"
        file: "app/server.py"
        template: "redirect → /classwork/review"
      - method: "GET"
        path: "/classwork/canvas_courses"
        file: "app/server.py"
        template: "JSON API (no template)"
      - method: "GET"
        path: "/classwork/canvas_assignments"
        file: "app/server.py"
        template: "JSON API (no template)"
      - method: "GET"
        path: "/classwork/canvas_submission_counts"
        file: "app/server.py"
        template: "JSON API (no template)"
      - method: "POST"
        path: "/classwork/canvas_post_comments"
        file: "app/server.py"
        template: "app/templates/classwork_batch_done.html"

    current_behavior:
      description: |
        CURRENT (BROKEN): The canvas_setup page lets you pick a course and check
        assignments. The "Fetch & Setup" button triggers POST /classwork/canvas_setup
        which downloads ALL files for ALL selected assignments in one synchronous
        handler, packs them into an in-memory ZIP, runs _classwork_build_mode_from_zip
        for each, and redirects to review. This is too slow and never sets
        gradebook_column, so finalize fails.

        PLANNED REDESIGN: See CANVAS_WORKFLOW_REDESIGN_SPEC.md. The new flow splits
        into: (1) lightweight queue creation from assignment metadata, (2) per-assignment
        lazy file download + current grade fetch, (3) review with discrepancy highlighting,
        (4) direct API grade posting per assignment.

    canvas_api_endpoints_used:
      - "GET /courses (paginated, cached 1hr)"
      - "GET /courses/{id}/assignments (paginated, cached 30min)"
      - "GET /courses/{id}/assignments/{id}/submissions (with user + submission_history)"
      - "PUT /courses/{id}/assignments/{id}/submissions/{student_id} (comment posting)"
      - "Planned: grade posting via submissions/update_grades or individual PUT"
```

Also add these entries to the `data_files` section:

```yaml
  - path: "data/classwork/files/{uuid}_{original}"
    role: "classwork_submissions"
    authority: "Extracted submission files; served via /classwork_files/ static mount."
    notes: "Created by _classwork_build_mode_from_zip() or canvas download. Referenced by ClassworkFileRef.stored_filename."

  - path: "data/classwork/exports/{export_id}/"
    role: "classwork_exports"
    authority: "Finalized gradebook CSVs and comment ZIPs."
    notes: "Contains manifest.json (or batch_manifest.json) linking export artifacts."

  - path: "data/gradebook_merge/tmp/{token}.csv"
    role: "gradebook_tmp"
    authority: "Stored copy of uploaded Canvas gradebook CSV."
    notes: "Referenced by ClassworkBatchState.gradebook_token. Used at finalize to merge scores into correct columns. Not used by Canvas API path."

  - path: "data/canvas_cache/"
    role: "canvas_api_cache"
    authority: "TTL-based cache of Canvas API responses."
    notes: |
      courses_cache.json (1hr TTL), {course_id}_assignments_*.json (30min),
      {course_id}_submission_counts.json (5min). Managed by canvas_client.py.
```

And add to the `code_structure` section:

```yaml
  - file: "app/canvas_client.py"
    purpose: "Canvas REST API client with caching and rate limiting."
    key_functions:
      - "get_courses() → List of courses where user can grade."
      - "get_assignments(course_id) → Filtered, sorted assignments list."
      - "get_assignment_submission_counts(course_id, assignment_ids) → Per-assignment submitted/graded/ungraded counts."
      - "get_gradebook_csv(course_id) → CSV string (deprecated — uses enrollment API fallback)."
      - "download_assignment_submissions(course_id, assignment_id, output_dir) → Dict[student_id, List[Path]]."
      - "bulk_post_comments(course_id, assignment_id, comments) → success/failed summary."
    notes: |
      Uses httpx (synchronous). All responses cached to data/canvas_cache/ with TTL.
      Rate limiting: 0.5s sleep between individual API calls in loops; 10s backoff on 429.
      Known bug: download_assignment_submissions uses _make_request (non-paginated),
      silently drops students beyond page 1 (~100). Should use _make_paginated_request.

  - file: "app/templates/classwork_setup.html"
    purpose: "Queue overview for manual classwork grading. Shows batch status, add-assignment form, queued assignment list."
  - file: "app/templates/classwork_review.html"
    purpose: "Per-student review page. Shows submission preview, score input, comment field. Bulk actions and keyboard shortcuts."
  - file: "app/templates/classwork_canvas_setup.html"
    purpose: "Canvas course/assignment picker. JS fetches courses and assignments from API, renders checkbox list."
  - file: "app/templates/classwork_batch_finalize.html"
    purpose: "Pre-finalize summary showing per-assignment stats and conflict warnings."
  - file: "app/templates/classwork_batch_done.html"
    purpose: "Post-finalize page with download links for merged CSV and comment ZIPs."
  - file: "app/templates/classwork_done.html"
    purpose: "Post-finalize page for single-assignment mode."
```
