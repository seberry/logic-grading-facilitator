# Canvas classwork workflow redesign — implementation spec

> **Context:** Read `CLASSWORK_BACKEND_MAP.md` first for the full backend architecture.
> This document specifies changes to replace the broken `POST /classwork/canvas_setup`
> handler with a clean per-assignment workflow that downloads files lazily and posts
> grades directly via the Canvas API (no CSV intermediary).

---

## 1. Design overview

### Current broken flow (to be replaced)
```
canvas_setup page → one "Fetch & Setup" button →
  downloads ALL files for ALL selected assignments in one synchronous handler →
  redirects to review
```

### New flow
```
canvas_setup page → select course + check assignments → "Create Queue" button →
  fetches assignment metadata only (no file downloads) →
  redirects to queue overview page

queue overview → click "Open" on one assignment →
  downloads files + fetches current grades for THAT assignment only →
  redirects to review page (enhanced with current-grade column)

review page → inspect submissions, adjust scores, leave comments → "Apply to Canvas" →
  posts grades + comments directly via Canvas API →
  shows results, returns to queue overview
```

### Key principles
- **Lazy downloads:** Files are only downloaded when you open a specific assignment for review.
- **No CSV intermediary:** Grades are posted directly to Canvas via API, not merged into a gradebook CSV.
- **Per-assignment cycle:** Open → review → apply → next. Each assignment is self-contained.
- **Existing review page reuse:** The `/classwork/review` template and `_classwork_apply_form_updates()` still work — we're adding to them, not replacing them.

---

## 2. Data model changes

### 2.1 Add field to `ClassworkEntry` (in `models.py`)

```python
class ClassworkEntry(BaseModel):
    student_id: str
    student_name: str
    has_submission: bool
    files: List[ClassworkFileRef] = Field(default_factory=list)
    score: float
    comment: Optional[str] = None
    # NEW: current grade in Canvas (None = no grade posted yet)
    current_canvas_grade: Optional[float] = None
```

This lets the review page show "Canvas: 8 → 10" or highlight discrepancies.

### 2.2 Add field to `ClassworkModeState` (in `models.py`)

```python
class ClassworkModeState(BaseModel):
    # ... existing fields ...
    canvas_course_id: Optional[int] = None
    canvas_assignment_id: Optional[int] = None
    # NEW: track whether files have been downloaded for this assignment
    canvas_files_loaded: bool = False
    # NEW: track whether grades have been applied to Canvas
    canvas_grades_applied: bool = False
```

### 2.3 `ClassworkBatchState` — no changes needed

The existing structure works. `gradebook_token` and `id_column` become unused for the Canvas path (they're still needed for the manual CSV path, so don't remove them — just leave them empty/null for Canvas batches).

---

## 3. New `canvas_client.py` methods

### 3.1 `get_submissions_with_grades(course_id, assignment_id)`

Fetches all submissions for an assignment, returning per-student: submission status, current grade, student name, student ID, and attachment metadata.

```python
def get_submissions_with_grades(
    self,
    course_id: int,
    assignment_id: int,
) -> List[Dict[str, Any]]:
    """
    Fetch submissions with current grades and student info.
    
    Returns list of dicts:
        {
            "student_id": str (sis_user_id or login_id),
            "student_name": str,
            "canvas_user_id": int,
            "workflow_state": str ("submitted", "graded", "unsubmitted", ...),
            "current_grade": float | None,
            "submitted_at": str | None,
            "attachments": [{"url": str, "filename": str, "content_type": str}, ...],
            "body": str | None  (for text submissions),
        }
    """
```

**Implementation notes:**
- Use endpoint: `GET /courses/{course_id}/assignments/{assignment_id}/submissions`
- Include params: `include[]=user&include[]=submission_history&per_page=100`
- Use `_make_paginated_request` to handle pagination (classes can have 100+ students).
- `current_grade` comes from `submission["score"]`.
- `student_id` should prefer `user["sis_user_id"]` then `user["login_id"]` then `str(user["id"])` — same logic as existing `download_assignment_submissions`.

### 3.2 `update_grades(course_id, assignment_id, grades)`

Posts grades for multiple students to a specific assignment.

```python
def update_grades(
    self,
    course_id: int,
    assignment_id: int,
    grades: Dict[str, float],
) -> Dict[str, Any]:
    """
    Post grades to Canvas for multiple students.
    
    Args:
        grades: Dict mapping student_id (sis_user_id) to score.
    
    Returns:
        {"success": [...], "failed": [...]} same shape as bulk_post_comments.
    """
```

**Implementation notes:**
- Canvas has a bulk grade update endpoint: `POST /courses/{course_id}/assignments/{assignment_id}/submissions/update_grades`
  with body `{"grade_data": {"sis_user_id:XXXX": {"posted_grade": score}, ...}}`
- If the bulk endpoint doesn't work reliably, fall back to per-student PUT:
  `PUT /courses/{course_id}/assignments/{assignment_id}/submissions/sis_user_id:{student_id}`
  with body `{"submission": {"posted_grade": score}}`
- Use same rate-limiting pattern as `bulk_post_comments` (0.5s sleep between requests).
- Return same success/failed structure for consistency.

### 3.3 Modify `download_assignment_submissions` (existing method)

The existing method works but has a bug: it uses `_make_request` (non-paginated) to fetch submissions, so it only gets the first page (up to 100 students). Change to use `_make_paginated_request`. Also, the method currently doesn't return the student name — add it to the return value or use `get_submissions_with_grades` to get that info separately.

---

## 4. Route changes in `server.py`

### 4.1 Redesign `POST /classwork/canvas_setup`

**What it does now (broken):** Downloads all files, builds full ClassworkModeState with entries, redirects to review.

**What it should do:** Fetch assignment metadata only, create lightweight batch with empty assignment shells, redirect to queue overview.

```python
@app.post("/classwork/canvas_setup", response_class=HTMLResponse)
async def classwork_canvas_setup(
    request: Request,
    course_id: int = Form(...),
    assignment_ids: List[int] = Form(None),
) -> Response:
    # 1. Validate
    if not assignment_ids or not any(assignment_ids):
        raise HTTPException(status_code=400, detail="Select at least one assignment.")
    
    client = get_canvas_client()
    
    # 2. Fetch assignment metadata (already cached, fast)
    all_assignments = client.get_assignments(course_id)
    assignments_map = {a["id"]: a for a in all_assignments}
    
    # 3. Build lightweight batch — NO file downloads, NO roster parsing
    batch_assignments = []
    for aid in assignment_ids:
        if aid not in assignments_map:
            raise HTTPException(status_code=400, detail=f"Assignment {aid} not found.")
        a = assignments_map[aid]
        
        mode = ClassworkModeState(
            assignment_id=_sanitize_assignment_id(a["name"]).lower(),
            max_points=float(a["max_points"] or 0),
            gradebook_token="",       # not used for Canvas path
            id_column="",             # not used for Canvas path
            gradebook_column=None,    # not used for Canvas path
            entries=[],               # populated lazily when opened
            canvas_course_id=course_id,
            canvas_assignment_id=aid,
            canvas_files_loaded=False,
        )
        batch_assignments.append(mode)
    
    # 4. Save batch to session
    with _session_lock:
        session = get_session()
        session.classwork_batch = ClassworkBatchState(
            gradebook_token="",
            id_column="",
            assignments=batch_assignments,
        )
        session.classwork_mode = None  # nothing open yet
        persist_session()
    
    client.close()
    
    # 5. Redirect to queue overview (the existing /classwork setup page)
    return RedirectResponse(
        url="/classwork?status=success&message=" + quote("Canvas assignments queued."),
        status_code=303,
    )
```

### 4.2 New route: `POST /classwork/canvas_open`

This replaces the old "Open in Review" action for Canvas assignments. It downloads files and fetches current grades for ONE assignment, then redirects to review.

```python
@app.post("/classwork/canvas_open", response_class=HTMLResponse)
async def classwork_canvas_open(assignment_id: str = Form(...)) -> Response:
    assignment_clean = _sanitize_assignment_id(assignment_id)
    
    with _session_lock:
        session = get_session()
        batch = session.classwork_batch
        if batch is None:
            raise HTTPException(status_code=400, detail="No batch active.")
        
        # Find the assignment in the batch
        mode = None
        for item in batch.assignments:
            if item.assignment_id == assignment_clean:
                mode = item
                break
        if mode is None:
            raise HTTPException(status_code=404, detail="Assignment not found in batch.")
        
        # If files already loaded, just switch to it and go to review
        if mode.canvas_files_loaded:
            session.classwork_mode = mode
            persist_session()
            return RedirectResponse(url="/classwork/review", status_code=303)
    
    # Outside the lock: do the slow API calls
    client = get_canvas_client()
    course_id = mode.canvas_course_id
    canvas_aid = mode.canvas_assignment_id
    
    # 1. Fetch submissions with current grades + student info
    submissions = client.get_submissions_with_grades(course_id, canvas_aid)
    
    # 2. Download submission files (for this assignment only)
    import tempfile
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        student_files = client.download_assignment_submissions(
            course_id=course_id,
            assignment_id=canvas_aid,
            output_dir=temp_path,
        )
        
        # 3. Build entries from submissions + downloaded files
        #    (This replaces _classwork_build_mode_from_zip for the Canvas path)
        entries = _build_canvas_entries(
            submissions=submissions,
            student_files=student_files,
            max_points=mode.max_points,
        )
    
    client.close()
    
    # 4. Update mode and persist
    with _session_lock:
        session = get_session()
        # Re-find mode in case session was modified
        for item in batch.assignments if session.classwork_batch else []:
            if item.assignment_id == assignment_clean:
                item.entries = entries
                item.canvas_files_loaded = True
                session.classwork_mode = item
                break
        persist_session()
    
    return RedirectResponse(url="/classwork/review", status_code=303)
```

### 4.3 New helper: `_build_canvas_entries()`

Builds `ClassworkEntry` list from Canvas API data instead of from a ZIP file.

```python
def _build_canvas_entries(
    submissions: List[Dict[str, Any]],
    student_files: Dict[str, List[Path]],
    max_points: float,
) -> List[ClassworkEntry]:
    """
    Build ClassworkEntry list from Canvas submissions data + downloaded files.
    
    Unlike _classwork_build_mode_from_zip, this uses Canvas submission metadata
    to determine who submitted (instead of filename matching), and includes
    current Canvas grades for each student.
    """
    entries = []
    
    for sub in submissions:
        student_id = str(sub["student_id"])
        student_name = sub.get("student_name", "")
        has_submission = sub["workflow_state"] not in ("unsubmitted",)
        current_grade = sub.get("current_grade")
        
        # Find downloaded files for this student
        file_refs = []
        downloaded = student_files.get(student_id, [])
        for file_path in downloaded:
            if file_path.exists():
                # Copy to classwork files dir (same pattern as ZIP extraction)
                stored_name = f"{uuid4().hex}_{file_path.name}"
                stored_path = CLASSWORK_FILES_DIR / stored_name
                shutil.copy2(file_path, stored_path)
                file_refs.append(ClassworkFileRef(
                    stored_filename=stored_name,
                    original_filename=file_path.name,
                    ext=file_path.suffix.lower(),
                ))
        
        # Pre-fill score: max if submitted, 0 if not
        score = float(max_points) if has_submission else 0.0
        
        entries.append(ClassworkEntry(
            student_id=student_id,
            student_name=student_name,
            has_submission=has_submission,
            files=file_refs,
            score=score,
            comment=None,
            current_canvas_grade=current_grade,
        ))
    
    return entries
```

### 4.4 New route: `POST /classwork/canvas_apply`

Posts grades and comments to Canvas for the currently active assignment.

```python
@app.post("/classwork/canvas_apply", response_class=HTMLResponse)
async def classwork_canvas_apply(request: Request) -> Response:
    form = await request.form()
    confirm_overwrites = bool(form.get("confirm_overwrites"))
    
    with _session_lock:
        session = get_session()
        mode = session.classwork_mode
        if mode is None or mode.canvas_course_id is None:
            raise HTTPException(status_code=400, detail="No Canvas assignment active.")
        
        # Apply form updates (scores, comments) — reuse existing function
        _classwork_apply_form_updates(form, mode, "save")
        _classwork_sync_mode_into_batch(session)
        persist_session()
    
    # Check for overwrites (where current Canvas grade differs from proposed)
    overwrites = []
    for entry in mode.entries:
        if entry.current_canvas_grade is not None and entry.current_canvas_grade != 0:
            if not math.isclose(entry.current_canvas_grade, entry.score, abs_tol=0.01):
                overwrites.append({
                    "student_id": entry.student_id,
                    "student_name": entry.student_name,
                    "current": entry.current_canvas_grade,
                    "proposed": entry.score,
                })
    
    if overwrites and not confirm_overwrites:
        # Render review page with overwrite warning
        # (Template needs to show the warning and a "Confirm & Apply" button)
        analysis = _classwork_analysis(mode)
        return templates.TemplateResponse(
            "classwork_review.html",
            {
                "request": request,
                "mode": mode,
                "analysis": analysis,
                "submitted_count": sum(1 for e in mode.entries if e.has_submission),
                "missing_count": sum(1 for e in mode.entries if not e.has_submission),
                "commented_count": sum(1 for e in mode.entries if (e.comment or "").strip()),
                "canvas_overwrites": overwrites,
                "is_canvas_mode": True,
            },
        )
    
    # Post grades to Canvas
    client = get_canvas_client()
    
    # Build grades dict
    grades = {e.student_id: e.score for e in mode.entries}
    grade_result = client.update_grades(
        course_id=mode.canvas_course_id,
        assignment_id=mode.canvas_assignment_id,
        grades=grades,
    )
    
    # Build comments dict (only entries with comments)
    comments = {}
    for entry in mode.entries:
        if entry.comment and entry.comment.strip():
            comments[entry.student_id] = {"text": entry.comment.strip()}
    
    comment_result = {"success": [], "failed": []}
    if comments:
        comment_result = client.bulk_post_comments(
            course_id=mode.canvas_course_id,
            assignment_id=mode.canvas_assignment_id,
            comments=comments,
        )
    
    client.close()
    
    # Mark as applied
    with _session_lock:
        session = get_session()
        if session.classwork_mode and session.classwork_mode.assignment_id == mode.assignment_id:
            session.classwork_mode.canvas_grades_applied = True
            _classwork_sync_mode_into_batch(session)
            persist_session()
    
    # Render results page
    # (Could be a new template or reuse classwork_done.html with Canvas-specific info)
    return templates.TemplateResponse(
        "classwork_canvas_applied.html",  # NEW template
        {
            "request": request,
            "assignment_id": mode.assignment_id,
            "assignment_name": mode.assignment_id,  # TODO: store human name
            "grade_result": grade_result,
            "comment_result": comment_result,
            "total_grades": len(grades),
            "total_comments": len(comments),
        },
    )
```

### 4.5 Modify `GET /classwork` (setup page)

The existing setup page template (`classwork_setup.html`) already shows the queue overview for the manual path. For Canvas batches, it needs to:

1. Show queued assignments with Canvas metadata (name, max_points, submission counts).
2. Show "Open" button per assignment → posts to `/classwork/canvas_open` instead of `/classwork/batch/open`.
3. Show status indicators: "Not loaded" / "Reviewed" / "Grades applied".
4. Hide the "Add Assignment" ZIP upload form (not needed for Canvas path).
5. Hide the "Finalize Queue" button (Canvas path uses per-assignment "Apply" instead).

The simplest approach: detect whether the batch is a Canvas batch (check if `assignments[0].canvas_course_id is not None`) and conditionally render the appropriate UI.

### 4.6 Modify `GET /classwork/review` (review page)

The review page template (`classwork_review.html`) needs small additions:

1. **Current grade column:** For each entry, if `current_canvas_grade is not None`, show it next to the score input.
2. **Discrepancy highlighting:** If `current_canvas_grade != score` (proposed), highlight the row.
3. **"Apply to Canvas" button:** When `is_canvas_mode`, show this button (posts to `/classwork/canvas_apply`) instead of / in addition to the "Finalize" button.
4. **Overwrite warning panel:** When `canvas_overwrites` is set, show the list of conflicts with a "Confirm & Apply" button.

---

## 5. Template changes summary

| Template | Changes |
|----------|---------|
| `classwork_setup.html` | Detect Canvas batch, show Canvas-specific queue UI (open → canvas_open, hide ZIP upload, hide finalize, show per-assignment status) |
| `classwork_review.html` | Add current_canvas_grade column, discrepancy highlighting, "Apply to Canvas" button, overwrite warning panel |
| `classwork_canvas_setup.html` | No changes needed (course/assignment picker works as-is) |
| `classwork_canvas_applied.html` | **NEW** — results page showing grades/comments posted, link back to queue |

---

## 6. What to keep, what to remove

### Keep (still used by manual CSV path)
- `POST /classwork/batch/start` — gradebook CSV upload
- `POST /classwork/batch/add` — ZIP upload per assignment
- `POST /classwork/batch/finalize` — CSV merge + export
- `_classwork_build_mode_from_zip()` — ZIP → entries pipeline
- All `gradebook_token` / `id_column` / `gradebook_column` machinery
- `POST /classwork/batch/open` — for manual batch assignments

### Remove or deprecate
- The entire body of the current `POST /classwork/canvas_setup` handler (replace with lightweight version above)
- `_create_submissions_zip_from_downloads()` — this was a hack to feed Canvas downloads into the ZIP-based pipeline; no longer needed since `_build_canvas_entries()` works directly from API data

### New
- `POST /classwork/canvas_open` — lazy per-assignment download + grade fetch
- `POST /classwork/canvas_apply` — direct API grade + comment posting
- `_build_canvas_entries()` — helper to build entries from Canvas API data
- `client.get_submissions_with_grades()` — new canvas_client method
- `client.update_grades()` — new canvas_client method
- `classwork_canvas_applied.html` — results template

---

## 7. Edge cases to handle

1. **Student ID format mismatch:** Canvas returns `sis_user_id` (the SIS Login ID) for some students and `login_id` for others. The download method and the submissions method must use the same ID resolution logic, otherwise files won't match to students. Use a consistent helper that tries `sis_user_id → login_id → str(id)` in order.

2. **Students with no submission but an existing grade:** This can happen if the instructor posted grades manually. The `current_canvas_grade` will be non-null but `has_submission` will be false. Show this clearly in the review UI.

3. **Rate limiting on apply:** The grade posting and comment posting loops need the same 0.5s sleep pattern as `bulk_post_comments`. If Canvas returns 429, back off for 10 seconds.

4. **Re-opening an already-loaded assignment:** If `canvas_files_loaded` is True, `canvas_open` should skip the download and just set `classwork_mode`. But the user might want to refresh (re-fetch current grades). Consider a `?refresh=1` query param or a "Refresh from Canvas" button.

5. **Large classes:** `download_assignment_submissions` currently uses `_make_request` (non-paginated). This silently drops students beyond the first page (~100). Must switch to `_make_paginated_request`.

6. **Text-only submissions:** Canvas text submissions (`body` field) should be saved as `.txt` files in `classwork/files/`, same as the existing `download_assignment_submissions` method does.

7. **Assignment with 0 max_points:** Some Canvas assignments have `points_possible = 0` or null. Pre-fill as 0 and let the instructor override.

---

## 8. Suggested implementation order

1. **Add `current_canvas_grade` and `canvas_files_loaded` fields to models.** (Tiny change, backwards-compatible since both have defaults.)
2. **Add `get_submissions_with_grades()` to canvas_client.** (Self-contained, testable with `scripts/test_canvas_api.py`.)
3. **Add `update_grades()` to canvas_client.** (Same — self-contained, testable.)
4. **Fix pagination bug in `download_assignment_submissions`.** (Change `_make_request` to `_make_paginated_request`.)
5. **Rewrite `POST /classwork/canvas_setup`** to the lightweight metadata-only version.
6. **Add `_build_canvas_entries()` helper and `POST /classwork/canvas_open` route.**
7. **Add `POST /classwork/canvas_apply` route.**
8. **Update `classwork_setup.html`** to detect Canvas batch and show appropriate UI.
9. **Update `classwork_review.html`** to show current Canvas grade column + Apply button.
10. **Create `classwork_canvas_applied.html`** results template.
