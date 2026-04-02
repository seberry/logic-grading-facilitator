from __future__ import annotations

from typing import Any, Dict, List, Optional
import csv
import io
import json
import math
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path
from urllib.parse import quote
from uuid import uuid4

from fastapi import APIRouter, File, Form, HTTPException, Query, Request, Response, UploadFile
from fastapi.responses import HTMLResponse, PlainTextResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from ..canvas_client import CanvasClientError, get_canvas_client
from ..classwork_core import (
    CLASSWORK_EXPORTS_DIR,
    CLASSWORK_FILES_DIR,
    GRADEBOOK_MERGE_TMP_DIR,
    _build_classwork_score_map,
    _candidate_assignment_columns,
    _classwork_analysis,
    _classwork_apply_form_updates,
    _classwork_batch_assignment_stats,
    _classwork_build_mode_from_downloads,
    _classwork_build_mode_from_zip,
    _classwork_comment_text,
    _classwork_extract_text_preview,
    _classwork_gradebook_tmp_path,
    _classwork_make_assignment_id_from_column,
    _classwork_open_next_batch_assignment,
    _classwork_resolve_roster,
    _classwork_resolve_stored_file,
    _classwork_sync_mode_into_batch,
    _classwork_unique_assignment_id,
    _classwork_zip_unique_name,
    _parse_gradebook_csv,
    _render_feedback_image_bytes,
    _render_feedback_pdf_bytes,
    _same_numeric_score,
    _sanitize_assignment_id,
)
from ..models import ClassworkBatchState, ClassworkModeState
from ..session_state import _session_lock, get_session, persist_session

router = APIRouter()
templates = Jinja2Templates(
    directory=str(Path(__file__).resolve().parents[1] / 'templates')
)

@router.get("/classwork/canvas_courses")
def get_canvas_courses():
    """Fetch list of Canvas courses for the user."""
    try:
        client = get_canvas_client()
        courses = client.get_courses()
        return {"courses": courses}
    except CanvasClientError as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/classwork/canvas_assignments")
def get_canvas_assignments(course_id: int = Query(...)):
    """Fetch assignments for a Canvas course, sorted by due date."""
    try:
        client = get_canvas_client()
        assignments = client.get_assignments(course_id)
        return {"course_id": course_id, "assignments": assignments}
    except CanvasClientError as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/classwork/canvas_submission_counts")
def get_canvas_submission_counts(
    course_id: int = Query(...),
    limit: int = Query(6, ge=1, le=20, description="Only count submissions for first N assignments")
):
    """Fetch submission counts for a limited number of Canvas assignments.
    
    Limits to first N assignments (default 6) to improve page load performance.
    Counts are only shown for ungraded assignments, so limiting is acceptable.
    """
    try:
        client = get_canvas_client()
        # Get all assignments (cached, fast)
        assignments = client.get_assignments(course_id)
        
        # Limit to first N assignments for counting
        limited_assignments = assignments[:limit]
        assignment_ids = [a["id"] for a in limited_assignments]
        
        # Get submission counts for limited set only
        counts = client.get_assignment_submission_counts(course_id, assignment_ids)
        return {
            "course_id": course_id,
            "assignment_counts": counts,
            "total_assignments": len(assignments),
            "counted_assignments": len(assignment_ids)
        }
    except CanvasClientError as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/classwork", response_class=HTMLResponse)
def classwork_setup_page(request: Request) -> Response:
    session = get_session()
    existing = session.classwork_mode
    batch = session.classwork_batch
    status = str(request.query_params.get("status") or "").strip()
    message = str(request.query_params.get("message") or "").strip()
    batch_columns: List[str] = []
    if batch is not None:
        tmp_path = _classwork_gradebook_tmp_path(batch.gradebook_token)
        if tmp_path.exists():
            decoded = tmp_path.read_text(encoding="utf-8")
            fieldnames, _rows, id_col = _parse_gradebook_csv(decoded)
            batch_columns = _candidate_assignment_columns(fieldnames, id_col)
    return templates.TemplateResponse(
        "classwork_setup.html",
        {
            "request": request,
            "existing": existing,
            "batch": batch,
            "batch_columns": batch_columns,
            "status": status,
            "message": message,
            "error_message": "",
        },
    )


@router.post("/classwork/setup", response_class=HTMLResponse)
async def classwork_setup(
    request: Request,
    submissions_zip: UploadFile = File(...),
    gradebook_csv: UploadFile = File(...),
    assignment_id: Optional[str] = Form(None),
    assignment_column: str = Form(...),
    max_points: float = Form(...),
) -> Response:
    if max_points < 0:
        raise HTTPException(status_code=400, detail="max_points must be >= 0.")
    if not submissions_zip.filename:
        raise HTTPException(status_code=400, detail="Please upload a submissions ZIP.")
    if not gradebook_csv.filename:
        raise HTTPException(status_code=400, detail="Please upload a gradebook CSV.")

    gradebook_bytes = await gradebook_csv.read()
    try:
        gradebook_text = gradebook_bytes.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="Gradebook CSV must be UTF-8 encoded.")

    fieldnames, rows, id_col = _parse_gradebook_csv(gradebook_text)
    active_roster = _classwork_resolve_roster("gradebook", rows, id_col)

    zip_bytes = await submissions_zip.read()
    if not zip_bytes:
        raise HTTPException(status_code=400, detail="Uploaded submissions ZIP was empty.")

    GRADEBOOK_MERGE_TMP_DIR.mkdir(parents=True, exist_ok=True)
    token = uuid4().hex
    _classwork_gradebook_tmp_path(token).write_text(gradebook_text, encoding="utf-8")
    selected_col = (assignment_column or "").strip()
    if not selected_col:
        raise HTTPException(status_code=400, detail="Please choose a target gradebook column.")
    if selected_col not in fieldnames:
        raise HTTPException(status_code=400, detail="Selected gradebook column not found.")

    raw_assignment_id = (assignment_id or "").strip()
    if raw_assignment_id:
        resolved_assignment_id = _sanitize_assignment_id(raw_assignment_id).lower()
    else:
        resolved_assignment_id = _classwork_make_assignment_id_from_column(selected_col)

    mode = _classwork_build_mode_from_zip(
        zip_bytes=zip_bytes,
        assignment_id=resolved_assignment_id,
        max_points=max_points,
        active_roster=active_roster,
        gradebook_token=token,
        id_col=id_col,
        gradebook_column=selected_col,
    )

    with _session_lock:
        session = get_session()
        session.classwork_mode = mode
        session.classwork_batch = ClassworkBatchState(
            gradebook_token=token,
            id_column=id_col,
            assignments=[mode],
        )
        persist_session()

    return RedirectResponse(url="/classwork/review", status_code=303)


@router.post("/classwork/batch/start", response_class=HTMLResponse)
async def classwork_batch_start(
    gradebook_csv: UploadFile = File(...),
) -> Response:
    if not gradebook_csv.filename:
        raise HTTPException(status_code=400, detail="Please upload a gradebook CSV.")
    gradebook_bytes = await gradebook_csv.read()
    try:
        gradebook_text = gradebook_bytes.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="Gradebook CSV must be UTF-8 encoded.")
    fieldnames, rows, id_col = _parse_gradebook_csv(gradebook_text)
    _classwork_resolve_roster("gradebook", rows, id_col)
    if not fieldnames:
        raise HTTPException(status_code=400, detail="Gradebook CSV has no headers.")

    GRADEBOOK_MERGE_TMP_DIR.mkdir(parents=True, exist_ok=True)
    token = uuid4().hex
    _classwork_gradebook_tmp_path(token).write_text(gradebook_text, encoding="utf-8")

    with _session_lock:
        session = get_session()
        session.classwork_batch = ClassworkBatchState(
            gradebook_token=token,
            id_column=id_col,
            assignments=[],
        )
        session.classwork_mode = None
        persist_session()

    return RedirectResponse(url="/classwork?status=success&message=" + quote("Queue created from gradebook."), status_code=303)


@router.post("/classwork/batch/add", response_class=HTMLResponse)
async def classwork_batch_add(
    submissions_zip: UploadFile = File(...),
    assignment_id: Optional[str] = Form(None),
    max_points: float = Form(...),
    assignment_column: str = Form(...),
) -> Response:
    if max_points < 0:
        raise HTTPException(status_code=400, detail="max_points must be >= 0.")
    if not submissions_zip.filename:
        raise HTTPException(status_code=400, detail="Please upload a submissions ZIP.")
    zip_bytes = await submissions_zip.read()
    if not zip_bytes:
        raise HTTPException(status_code=400, detail="Uploaded submissions ZIP was empty.")

    with _session_lock:
        session = get_session()
        batch = session.classwork_batch
        if batch is None:
            raise HTTPException(status_code=400, detail="Batch mode is not initialized. Start a batch first.")
        tmp_path = _classwork_gradebook_tmp_path(batch.gradebook_token)
        if not tmp_path.exists():
            raise HTTPException(status_code=400, detail="Batch gradebook token missing. Restart batch.")
        gradebook_text = tmp_path.read_text(encoding="utf-8")
        fieldnames, rows, id_col = _parse_gradebook_csv(gradebook_text)
        selected_col = (assignment_column or "").strip()
        if not selected_col:
            raise HTTPException(status_code=400, detail="Please choose a target gradebook column.")
        if selected_col not in fieldnames:
            raise HTTPException(status_code=400, detail="Selected gradebook column not found.")
        existing_ids = {(a.assignment_id or "").lower() for a in batch.assignments}
        raw_assignment_id = (assignment_id or "").strip()
        if raw_assignment_id:
            resolved_assignment_id = _sanitize_assignment_id(raw_assignment_id).lower()
        else:
            base_from_col = _classwork_make_assignment_id_from_column(selected_col)
            resolved_assignment_id = _classwork_unique_assignment_id(base_from_col, existing_ids)

        active_roster = _classwork_resolve_roster("gradebook", rows, id_col)
        mode = _classwork_build_mode_from_zip(
            zip_bytes=zip_bytes,
            assignment_id=resolved_assignment_id,
            max_points=max_points,
            active_roster=active_roster,
            gradebook_token=batch.gradebook_token,
            id_col=id_col,
            gradebook_column=selected_col,
        )
        for item in batch.assignments:
            if item.assignment_id != mode.assignment_id and (item.gradebook_column or "").strip() == selected_col:
                raise HTTPException(
                    status_code=400,
                    detail=f"Column {selected_col!r} is already assigned to {item.assignment_id}.",
                )
        replaced = False
        for idx, item in enumerate(batch.assignments):
            if item.assignment_id == mode.assignment_id:
                batch.assignments[idx] = mode
                replaced = True
                break
        if not replaced:
            batch.assignments.append(mode)
        session.classwork_mode = mode
        persist_session()

    return RedirectResponse(url="/classwork", status_code=303)


@router.post("/classwork/batch/open", response_class=HTMLResponse)
async def classwork_batch_open(assignment_id: str = Form(...)) -> Response:
    assignment_clean = _sanitize_assignment_id(assignment_id)
    with _session_lock:
        session = get_session()
        batch = session.classwork_batch
        if batch is None:
            raise HTTPException(status_code=400, detail="Batch mode is not initialized.")
        for item in batch.assignments:
            if item.assignment_id == assignment_clean:
                session.classwork_mode = item
                persist_session()
                return RedirectResponse(url="/classwork/review", status_code=303)
    raise HTTPException(status_code=404, detail="Batch assignment not found.")


@router.post("/classwork/batch/open_next", response_class=HTMLResponse)
async def classwork_batch_open_next() -> Response:
    with _session_lock:
        session = get_session()
        if not _classwork_open_next_batch_assignment(session, include_current=True):
            raise HTTPException(status_code=400, detail="No queued assignments to review.")
        persist_session()
    return RedirectResponse(url="/classwork/review", status_code=303)


@router.post("/classwork/batch/remove", response_class=HTMLResponse)
async def classwork_batch_remove(assignment_id: str = Form(...)) -> Response:
    assignment_clean = _sanitize_assignment_id(assignment_id)
    with _session_lock:
        session = get_session()
        batch = session.classwork_batch
        if batch is None:
            raise HTTPException(status_code=400, detail="Batch mode is not initialized.")
        before = len(batch.assignments)
        batch.assignments = [a for a in batch.assignments if a.assignment_id != assignment_clean]
        if len(batch.assignments) == before:
            raise HTTPException(status_code=404, detail="Batch assignment not found.")
        if session.classwork_mode and session.classwork_mode.assignment_id == assignment_clean:
            session.classwork_mode = None
        persist_session()
    return RedirectResponse(url="/classwork", status_code=303)


@router.post("/classwork/batch/set_column", response_class=HTMLResponse)
async def classwork_batch_set_column(
    assignment_id: str = Form(...),
    assignment_column: str = Form(...),
) -> Response:
    assignment_clean = _sanitize_assignment_id(assignment_id)
    selected_col = (assignment_column or "").strip()
    if not selected_col:
        raise HTTPException(status_code=400, detail="Please choose a target gradebook column.")
    with _session_lock:
        session = get_session()
        batch = session.classwork_batch
        if batch is None:
            raise HTTPException(status_code=400, detail="Batch mode is not initialized.")
        tmp_path = _classwork_gradebook_tmp_path(batch.gradebook_token)
        if not tmp_path.exists():
            raise HTTPException(status_code=400, detail="Batch gradebook token missing. Restart batch.")
        decoded = tmp_path.read_text(encoding="utf-8")
        fieldnames, _rows, _id_col = _parse_gradebook_csv(decoded)
        if selected_col not in fieldnames:
            raise HTTPException(status_code=400, detail="Selected gradebook column not found.")
        target: Optional[ClassworkModeState] = None
        for item in batch.assignments:
            if item.assignment_id == assignment_clean:
                target = item
            elif (item.gradebook_column or "").strip() == selected_col:
                raise HTTPException(
                    status_code=400,
                    detail=f"Column {selected_col!r} is already assigned to {item.assignment_id}.",
                )
        if target is None:
            raise HTTPException(status_code=404, detail="Batch assignment not found.")
        target.gradebook_column = selected_col
        if session.classwork_mode and session.classwork_mode.assignment_id == assignment_clean:
            session.classwork_mode.gradebook_column = selected_col
        persist_session()
    return RedirectResponse(url="/classwork", status_code=303)


@router.post("/classwork/batch/clear", response_class=HTMLResponse)
async def classwork_batch_clear() -> Response:
    with _session_lock:
        session = get_session()
        session.classwork_batch = None
        session.classwork_mode = None
        persist_session()
    return RedirectResponse(url="/classwork?status=success&message=" + quote("Queue cleared."), status_code=303)


@router.get("/classwork/batch/finalize", response_class=HTMLResponse)
def classwork_batch_finalize_page(request: Request) -> Response:
    session = get_session()
    batch = session.classwork_batch
    if batch is None:
        return RedirectResponse(url="/classwork", status_code=303)
    if not batch.assignments:
        raise HTTPException(status_code=400, detail="Batch has no assignments yet.")
    tmp_path = _classwork_gradebook_tmp_path(batch.gradebook_token)
    if not tmp_path.exists():
        raise HTTPException(status_code=400, detail="Batch gradebook token missing. Restart batch.")
    decoded = tmp_path.read_text(encoding="utf-8")
    _fieldnames, rows, id_col = _parse_gradebook_csv(decoded)
    assignment_stats = _classwork_batch_assignment_stats(batch, rows, id_col)
    missing_columns = [a["assignment_id"] for a in assignment_stats if not a["gradebook_column"]]
    return templates.TemplateResponse(
        "classwork_batch_finalize.html",
        {
            "request": request,
            "batch": batch,
            "assignment_stats": assignment_stats,
            "error_message": (
                "Missing target gradebook column for: " + ", ".join(missing_columns)
                if missing_columns else ""
            ),
            "conflicts": [],
        },
    )


@router.post("/classwork/batch/finalize", response_class=HTMLResponse)
async def classwork_batch_finalize_submit(request: Request) -> Response:
    form = await request.form()
    overwrite = bool(form.get("overwrite_non_empty"))

    with _session_lock:
        session = get_session()
        batch = session.classwork_batch
        if batch is None:
            raise HTTPException(status_code=400, detail="Batch mode is not initialized.")
        _classwork_sync_mode_into_batch(session)
        if not batch.assignments:
            raise HTTPException(status_code=400, detail="Batch has no assignments yet.")

        tmp_path = _classwork_gradebook_tmp_path(batch.gradebook_token)
        if not tmp_path.exists():
            raise HTTPException(status_code=400, detail="Batch gradebook token missing. Restart batch.")
        decoded = tmp_path.read_text(encoding="utf-8")
        fieldnames, rows, id_col = _parse_gradebook_csv(decoded)
        selected_cols: Dict[str, str] = {}
        used_cols: set[str] = set()
        for mode in batch.assignments:
            col = (mode.gradebook_column or "").strip()
            if not col:
                raise HTTPException(status_code=400, detail=f"Missing gradebook column for {mode.assignment_id}.")
            if col in used_cols:
                raise HTTPException(status_code=400, detail="Each batch assignment must use a different column.")
            used_cols.add(col)
            selected_cols[mode.assignment_id] = col

        conflicts: List[Dict[str, str]] = []
        for mode in batch.assignments:
            score_map = _build_classwork_score_map(mode)
            col = selected_cols[mode.assignment_id]
            for row in rows:
                sid = (row.get(id_col) or "").strip()
                if sid not in score_map:
                    continue
                old_val = (row.get(col) or "").strip()
                if old_val == "" or overwrite:
                    continue
                new_score = float(score_map[sid])
                if _same_numeric_score(old_val, new_score) or old_val == f"{new_score:g}":
                    continue
                conflicts.append(
                    {
                        "assignment_id": mode.assignment_id,
                        "column": col,
                        "student_id": sid,
                        "student": (row.get("Student") or row.get("student") or ""),
                        "existing": old_val,
                        "new": f"{new_score:g}",
                    }
                )

        if conflicts and not overwrite:
            assignment_stats = _classwork_batch_assignment_stats(batch, rows, id_col)
            return templates.TemplateResponse(
                "classwork_batch_finalize.html",
                {
                    "request": request,
                    "batch": batch,
                    "assignment_stats": assignment_stats,
                    "error_message": (
                        f"Found {len(conflicts)} non-empty conflict(s). "
                        "Edit queue column mappings or enable overwrite."
                    ),
                    "conflicts": conflicts[:40],
                },
            )
        
        # CRITICAL FIX: Save session to disk before creating exports
        persist_session()

    for mode in batch.assignments:
        score_map = _build_classwork_score_map(mode)
        col = selected_cols[mode.assignment_id]
        for row in rows:
            sid = (row.get(id_col) or "").strip()
            if sid in score_map:
                row[col] = f"{score_map[sid]:g}"

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(row)

    export_id = uuid4().hex
    export_dir = CLASSWORK_EXPORTS_DIR / export_id
    export_dir.mkdir(parents=True, exist_ok=True)
    gradebook_filename = "classwork_batch_gradebook_merged.csv"
    gradebook_path = export_dir / gradebook_filename
    gradebook_path.write_text(output.getvalue(), encoding="utf-8")

    comments_files: Dict[str, str] = {}
    for mode in batch.assignments:
        commented_entries = [e for e in mode.entries if (e.comment or "").strip()]
        if not commented_entries:
            continue
        comments_filename = f"{mode.assignment_id}_comments_reupload.zip"
        comments_path = export_dir / comments_filename
        used_names: set[str] = set()
        with zipfile.ZipFile(comments_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for entry in commented_entries:
                if not entry.files:
                    continue
                feedback_text = _classwork_comment_text(mode, entry)
                for ref in entry.files:
                    ext = (ref.ext or "").lower()
                    stored_path = CLASSWORK_FILES_DIR / ref.stored_filename
                    original_name = ref.original_filename or ref.stored_filename
                    zip_name = _classwork_zip_unique_name(used_names, original_name)
                    if ext == ".pdf":
                        payload = _render_feedback_pdf_bytes(feedback_text)
                        zf.writestr(zip_name, payload)
                    elif ext in {".jpg", ".jpeg", ".png"}:
                        payload = _render_feedback_image_bytes(feedback_text, ext)
                        zf.writestr(zip_name, payload)
                    else:
                        if stored_path.exists():
                            zf.writestr(zip_name, stored_path.read_bytes())
                        sidecar_name = _classwork_zip_unique_name(used_names, f"{zip_name}.comment.txt")
                        zf.writestr(sidecar_name, feedback_text.encode("utf-8"))
        comments_files[mode.assignment_id] = comments_filename

    manifest = {
        "gradebook_file": gradebook_filename,
        "comments_files": comments_files,
        "assignment_ids": [m.assignment_id for m in batch.assignments],
        "created_at": datetime.now().isoformat(),
    }
    (export_dir / "batch_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return templates.TemplateResponse(
        "classwork_batch_done.html",
        {
            "request": request,
            "export_id": export_id,
            "comments_files": comments_files,
            "assignment_ids": [m.assignment_id for m in batch.assignments],
            "closed": False,
        },
    )


@router.post("/classwork/batch/close_session", response_class=HTMLResponse)
async def classwork_batch_close_session(request: Request) -> Response:
    form = await request.form()
    export_id = str(form.get("export_id") or "").strip()
    if not export_id:
        raise HTTPException(status_code=400, detail="Missing export id.")
    export_dir = CLASSWORK_EXPORTS_DIR / export_id
    manifest_path = export_dir / "batch_manifest.json"
    if not manifest_path.exists():
        raise HTTPException(status_code=404, detail="Batch export not found.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    comments_files = manifest.get("comments_files") or {}
    assignment_ids = manifest.get("assignment_ids") or sorted(comments_files.keys())
    with _session_lock:
        session = get_session()
        session.classwork_batch = None
        session.classwork_mode = None
        persist_session()
    return templates.TemplateResponse(
        "classwork_batch_done.html",
        {
            "request": request,
            "export_id": export_id,
            "comments_files": comments_files,
            "assignment_ids": assignment_ids,
            "closed": True,
        },
    )


@router.get("/classwork/review", response_class=HTMLResponse)
def classwork_review_page(request: Request) -> Response:
    session = get_session()
    mode = session.classwork_mode
    if mode is None:
        return RedirectResponse(url="/classwork", status_code=303)

    analysis = _classwork_analysis(mode)
    batch_mode = bool(session.classwork_batch)
    submitted_count = sum(1 for e in mode.entries if e.has_submission)
    commented_count = sum(1 for e in mode.entries if (e.comment or "").strip())
    return templates.TemplateResponse(
        "classwork_review.html",
        {
            "request": request,
            "mode": mode,
            "batch_mode": batch_mode,
            "analysis": analysis,
            "submitted_count": submitted_count,
            "missing_count": len(mode.entries) - submitted_count,
            "commented_count": commented_count,
            "error_message": "",
        },
    )


@router.post("/classwork/review", response_class=HTMLResponse)
async def classwork_review_submit(request: Request) -> Response:
    form = await request.form()
    action = str(form.get("action") or "save")
    redirect_url = "/classwork/review"

    with _session_lock:
        session = get_session()
        mode = session.classwork_mode
        if mode is None:
            raise HTTPException(status_code=400, detail="Classwork mode is not initialized.")
        _classwork_apply_form_updates(form, mode, action)
        _classwork_sync_mode_into_batch(session)
        if action == "save_next_assignment":
            moved = _classwork_open_next_batch_assignment(session, include_current=False)
            redirect_url = "/classwork/review" if moved else "/classwork"
        persist_session()

    return RedirectResponse(url=redirect_url, status_code=303)


@router.post("/classwork/finalize", response_class=HTMLResponse)
async def classwork_finalize(request: Request) -> Response:
    form = await request.form()
    assignment_column = str(form.get("assignment_column") or "").strip()
    overwrite = bool(form.get("overwrite_non_empty"))

    with _session_lock:
        session = get_session()
        mode = session.classwork_mode
        if mode is None:
            raise HTTPException(status_code=400, detail="Classwork mode is not initialized.")
        if session.classwork_batch is not None:
            raise HTTPException(
                status_code=400,
                detail="Batch mode is active. Use /classwork/batch/finalize for export.",
            )

        _classwork_apply_form_updates(form, mode, "save")
        _classwork_sync_mode_into_batch(session)
        persist_session()

        analysis = _classwork_analysis(mode)
        if not assignment_column:
            return templates.TemplateResponse(
                "classwork_review.html",
                {
                    "request": request,
                    "mode": mode,
                    "analysis": analysis,
                    "submitted_count": sum(1 for e in mode.entries if e.has_submission),
                    "missing_count": sum(1 for e in mode.entries if not e.has_submission),
                    "commented_count": sum(1 for e in mode.entries if (e.comment or "").strip()),
                    "error_message": "Please select a gradebook assignment column.",
                },
            )

        tmp_path = _classwork_gradebook_tmp_path(mode.gradebook_token)
        if not tmp_path.exists():
            raise HTTPException(
                status_code=400,
                detail="Classwork gradebook token missing. Re-run classwork setup.",
            )
        decoded = tmp_path.read_text(encoding="utf-8")
        fieldnames, rows, id_col = _parse_gradebook_csv(decoded)
        if assignment_column not in fieldnames:
            raise HTTPException(status_code=400, detail="Selected assignment column not found.")

        score_map = _build_classwork_score_map(mode)
        conflicts: List[Dict[str, str]] = []
        for row in rows:
            sid = (row.get(id_col) or "").strip()
            if sid not in score_map:
                continue
            old_val = (row.get(assignment_column) or "").strip()
            if old_val != "" and not overwrite:
                new_score = float(score_map[sid])
                same_numeric = False
                try:
                    same_numeric = math.isclose(float(old_val), new_score, rel_tol=0.0, abs_tol=1e-9)
                except ValueError:
                    same_numeric = False
                if same_numeric or old_val == f"{new_score:g}":
                    continue
                conflicts.append(
                    {
                        "student_id": sid,
                        "student": (row.get("Student") or row.get("student") or ""),
                        "existing": old_val,
                        "new": f"{new_score:g}",
                    }
                )

        if conflicts and not overwrite:
            analysis_with_conflicts = dict(analysis)
            analysis_with_conflicts["conflicts"] = conflicts[:30]
            return templates.TemplateResponse(
                "classwork_review.html",
                {
                    "request": request,
                    "mode": mode,
                    "analysis": analysis_with_conflicts,
                    "submitted_count": sum(1 for e in mode.entries if e.has_submission),
                    "missing_count": sum(1 for e in mode.entries if not e.has_submission),
                    "commented_count": sum(1 for e in mode.entries if (e.comment or "").strip()),
                    "error_message": (
                        f"Selected column has {len(conflicts)} existing grade(s) among matched students. "
                        "Choose a blank column or enable overwrite."
                    ),
                },
            )

    for row in rows:
        sid = (row.get(id_col) or "").strip()
        if sid in score_map:
            row[assignment_column] = f"{score_map[sid]:g}"

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(row)

    export_id = uuid4().hex
    export_dir = CLASSWORK_EXPORTS_DIR / export_id
    export_dir.mkdir(parents=True, exist_ok=True)
    gradebook_filename = f"{mode.assignment_id}_gradebook_merged.csv"
    gradebook_path = export_dir / gradebook_filename
    gradebook_path.write_text(output.getvalue(), encoding="utf-8")

    comments_filename: Optional[str] = None
    commented_entries = [e for e in mode.entries if (e.comment or "").strip()]
    if commented_entries:
        comments_filename = f"{mode.assignment_id}_comments_reupload.zip"
        comments_path = export_dir / comments_filename
        used_names: set[str] = set()
        with zipfile.ZipFile(comments_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for entry in commented_entries:
                if not entry.files:
                    continue
                feedback_text = _classwork_comment_text(mode, entry)
                for ref in entry.files:
                    ext = (ref.ext or "").lower()
                    stored_path = CLASSWORK_FILES_DIR / ref.stored_filename
                    original_name = ref.original_filename or ref.stored_filename
                    zip_name = _classwork_zip_unique_name(used_names, original_name)

                    if ext == ".pdf":
                        payload = _render_feedback_pdf_bytes(feedback_text)
                        zf.writestr(zip_name, payload)
                    elif ext in {".jpg", ".jpeg", ".png"}:
                        payload = _render_feedback_image_bytes(feedback_text, ext)
                        zf.writestr(zip_name, payload)
                    else:
                        if stored_path.exists():
                            zf.writestr(zip_name, stored_path.read_bytes())
                        sidecar_name = _classwork_zip_unique_name(used_names, f"{zip_name}.comment.txt")
                        zf.writestr(sidecar_name, feedback_text.encode("utf-8"))

    manifest = {
        "gradebook_file": gradebook_filename,
        "comments_file": comments_filename,
        "created_at": datetime.now().isoformat(),
    }
    (export_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return templates.TemplateResponse(
        "classwork_done.html",
        {
            "request": request,
            "assignment_id": mode.assignment_id,
            "export_id": export_id,
            "gradebook_file": gradebook_filename,
            "comments_file": comments_filename,
        },
    )


@router.get("/classwork/preview_text/{stored_filename:path}", response_class=PlainTextResponse)
def classwork_preview_text(stored_filename: str) -> Response:
    path = _classwork_resolve_stored_file(stored_filename)
    ext = path.suffix.lower()
    allowed = {
        ".txt", ".md", ".csv", ".json", ".log", ".text", ".rtf",
        ".html", ".htm", ".xml", ".yaml", ".yml", ".docx",
    }
    if ext not in allowed:
        raise HTTPException(status_code=400, detail="Inline preview not supported for this file type.")
    text = _classwork_extract_text_preview(path)
    return PlainTextResponse(text)


@router.get("/classwork/export/{export_id}/{kind}")
def classwork_download_export(export_id: str, kind: str):
    export_dir = CLASSWORK_EXPORTS_DIR / export_id
    manifest_path = export_dir / "manifest.json"
    if not manifest_path.exists():
        raise HTTPException(status_code=404, detail="Export not found.")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if kind == "gradebook":
        filename = manifest.get("gradebook_file")
        media_type = "text/csv"
    elif kind == "comments":
        filename = manifest.get("comments_file")
        media_type = "application/zip"
    else:
        raise HTTPException(status_code=404, detail="Unknown export kind.")

    if not filename:
        raise HTTPException(status_code=404, detail="Requested export file not available.")

    path = export_dir / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Export file missing.")

    return Response(
        content=path.read_bytes(),
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{quote(filename)}"'},
    )


@router.get("/classwork/batch/export/{export_id}/gradebook")
def classwork_batch_download_gradebook(export_id: str):
    export_dir = CLASSWORK_EXPORTS_DIR / export_id
    manifest_path = export_dir / "batch_manifest.json"
    if not manifest_path.exists():
        raise HTTPException(status_code=404, detail="Batch export not found.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    filename = manifest.get("gradebook_file")
    if not filename:
        raise HTTPException(status_code=404, detail="Batch gradebook export missing.")
    path = export_dir / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Batch gradebook file missing.")
    return Response(
        content=path.read_bytes(),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{quote(filename)}"'},
    )


@router.get("/classwork/batch/export/{export_id}/comments/{assignment_id}")
def classwork_batch_download_comments(export_id: str, assignment_id: str):
    export_dir = CLASSWORK_EXPORTS_DIR / export_id
    manifest_path = export_dir / "batch_manifest.json"
    if not manifest_path.exists():
        raise HTTPException(status_code=404, detail="Batch export not found.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    comments_files = manifest.get("comments_files") or {}
    filename = comments_files.get(assignment_id)
    if not filename:
        raise HTTPException(status_code=404, detail="No comments ZIP for that assignment.")
    path = export_dir / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Batch comments file missing.")
    return Response(
        content=path.read_bytes(),
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{quote(filename)}"'},
    )


@router.get("/classwork/canvas_setup", response_class=HTMLResponse)
def classwork_canvas_setup_page(request: Request) -> Response:
    """Render Canvas setup page."""
    status = str(request.query_params.get("status") or "").strip()
    message = str(request.query_params.get("message") or "").strip()
    return templates.TemplateResponse(
        "classwork_canvas_setup.html",
        {
            "request": request,
            "status": status,
            "message": message,
        },
    )


@router.post("/classwork/canvas_setup", response_class=HTMLResponse)
async def classwork_canvas_setup(
    request: Request,
    course_id: int = Form(...),
    gradebook_csv: UploadFile = File(...),
) -> Response:
    """Set up classwork using Canvas API to fetch assignments and uploaded CSV for gradebook."""
    import traceback
    
    form = await request.form()
    
    # Extract assignment_ids from form data (may be sent as multiple values)
    assignment_ids_raw = form.getlist("assignment_ids")
    assignment_ids = [int(aid) for aid in assignment_ids_raw if aid.strip()]
    
    print(f"[DEBUG] Form data - assignment_ids_raw: {assignment_ids_raw}")
    print(f"[DEBUG] Form data - assignment_ids: {assignment_ids}")
    print(f"[DEBUG] Gradebook CSV file: {gradebook_csv.filename}")
    
    if not assignment_ids:
        raise HTTPException(status_code=400, detail="Please select at least one assignment.")
    
    if not gradebook_csv.filename:
        raise HTTPException(status_code=400, detail="Please upload a gradebook CSV file.")

    try:
        # Initialize Canvas client
        client = get_canvas_client()
        
        # Read uploaded CSV
        gradebook_bytes = await gradebook_csv.read()
        try:
            gradebook_text = gradebook_bytes.decode("utf-8-sig")
        except UnicodeDecodeError:
            raise HTTPException(status_code=400, detail="Gradebook CSV must be UTF-8 encoded.")
        
        print(f"[DEBUG] Gradebook text length: {len(gradebook_text)}")
        
        # Parse gradebook
        print(f"[DEBUG] Parsing gradebook...")
        try:
            fieldnames, rows, id_col = _parse_gradebook_csv(gradebook_text)
            print(f"[DEBUG] Gradebook parsed - fieldnames: {fieldnames[:5]}..., rows: {len(rows)}, id_col: {id_col}")
            print(f"[DEBUG] rows type: {type(rows)}, first row type: {type(rows[0]) if rows else 'N/A'}")
        except Exception as e:
            print(f"[ERROR] Failed to parse gradebook: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        try:
            active_roster = _classwork_resolve_roster("gradebook", rows, id_col)
            print(f"[DEBUG] Roster resolved - {len(active_roster)} students")
        except Exception as e:
            print(f"[ERROR] Failed to resolve roster: {e}")
            import traceback
            traceback.print_exc()
            raise
        print(f"[DEBUG] Roster resolved - {len(active_roster)} students")
        
        # Fetch assignments and get metadata
        print(f"[DEBUG] Fetching assignments...")
        all_assignments = client.get_assignments(course_id)
        print(f"[DEBUG] All assignments fetched: {len(all_assignments)} assignments")
        print(f"[DEBUG] First assignment structure: {all_assignments[0] if all_assignments else 'None'}")
        
        assignments_map = {a["id"]: a for a in all_assignments}
        print(f"[DEBUG] Assignments map created: {list(assignments_map.keys())[:5]}...")
        
        # Validate selected assignments
        selected_assignments = []
        for aid in assignment_ids:
            if aid not in assignments_map:
                raise HTTPException(status_code=400, detail=f"Assignment {aid} not found.")
            selected_assignments.append(assignments_map[aid])
        print(f"[DEBUG] Selected assignments: {[a['name'] for a in selected_assignments]}")
        
        # Store gradebook
        GRADEBOOK_MERGE_TMP_DIR.mkdir(parents=True, exist_ok=True)
        token = uuid4().hex
        _classwork_gradebook_tmp_path(token).write_text(gradebook_text, encoding="utf-8")
        
        # Download submissions for each selected assignment separately
        import tempfile
        batch_assignments = []
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Accumulate Canvas user ID mappings across all assignments
            canvas_user_ids: Dict[str, int] = {}
            
            for assignment_data in selected_assignments:
                canvas_aid = assignment_data["id"]
                assignment_name = assignment_data["name"]
                
                print(f"[AUTO DOWNLOAD] Downloading submissions for: {assignment_name} (ID: {canvas_aid})")
                
                # Download submissions to temp directory for this assignment only
                # Now returns tuple: (student_files, canvas_user_ids)
                student_files, canvas_ids_for_assignment = client.download_assignment_submissions(
                    course_id=course_id,
                    assignment_id=canvas_aid,
                    output_dir=temp_path / f"assignment_{canvas_aid}"
                )
                
                # Accumulate Canvas user IDs across all assignments
                canvas_user_ids.update(canvas_ids_for_assignment)
                
                # Check if we got any submissions
                if not student_files:
                    print(f"[AUTO DOWNLOAD] No submissions found for assignment {assignment_name}")
                    continue
                
                print(f"[AUTO DOWNLOAD] Downloaded submissions for {len(student_files)} students in {assignment_name}")
                
                # Build mode from downloads IMMEDIATELY while temp directory still exists
                # This copies files to CLASSWORK_FILES_DIR (permanent storage)
                # Pass gradebook fieldnames and original assignment name for column matching
                mode = _classwork_build_mode_from_downloads(
                    student_files=student_files,
                    assignment_id=_sanitize_assignment_id(assignment_name).lower(),
                    assignment_name=assignment_name,  # Pass original name for matching
                    max_points=float(assignment_data["max_points"]),
                    active_roster=active_roster,
                    gradebook_token=token,
                    id_col=id_col,
                    gradebook_column=None,  # Will be auto-matched
                    gradebook_fieldnames=fieldnames,  # Pass fieldnames for matching
                )
                
                # Add Canvas metadata
                mode.canvas_course_id = course_id
                mode.canvas_assignment_id = canvas_aid
                
                batch_assignments.append(mode)
                
                print(f"[AUTO DOWNLOAD] Created ClassworkModeState for {assignment_name}")
        
        # Verify we have at least one successful download
        if not batch_assignments:
            raise HTTPException(
                status_code=400,
                detail="No submissions found for the selected assignments."
            )
        
        # Create batch state
        batch = ClassworkBatchState(
            gradebook_token=token,
            id_column=id_col,
            assignments=batch_assignments,
            canvas_user_ids=canvas_user_ids,  # Store Canvas user ID mapping
        )
        
        # Save session
        with _session_lock:
            session = get_session()
            session.classwork_batch = batch
            session.classwork_mode = batch_assignments[0] if batch_assignments else None
            persist_session()
        
        return RedirectResponse(url="/classwork/review", status_code=303)
        
    except CanvasClientError as e:
        return RedirectResponse(
            url=f"/classwork?status=error&message={quote(f'Canvas API error: {e}')}",
            status_code=303,
        )
    except Exception as e:
        return RedirectResponse(
            url=f"/classwork?status=error&message={quote(f'Error: {e}')}",
            status_code=303,
        )


@router.post("/classwork/canvas_post_comments", response_class=HTMLResponse)
async def classwork_canvas_post_comments(
    request: Request,
    export_id: str = Form(...),
    overwrite: bool = Form(False),
) -> Response:
    """Post text comments from classwork to Canvas via API."""
    export_dir = CLASSWORK_EXPORTS_DIR / export_id
    manifest_path = export_dir / "batch_manifest.json"
    if not manifest_path.exists():
        raise HTTPException(status_code=404, detail="Export not found.")
    
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    gradebook_file = manifest.get("gradebook_file")
    
    print(f"[DEBUG] Canvas comment posting started for export_id: {export_id}")
    
    with _session_lock:
        session = get_session()
        batch = session.classwork_batch
        if batch is None:
            raise HTTPException(status_code=400, detail="No batch session found.")
        
        print(f"[DEBUG] Batch has {len(batch.assignments)} assignments")
        
        results_summary = []
        success_count = 0
        failed_count = 0
        
        try:
            client = get_canvas_client()
            
            for mode in batch.assignments:
                print(f"[DEBUG] Processing assignment: {mode.assignment_id}")
                
                if not mode.canvas_course_id or not mode.canvas_assignment_id:
                    results_summary.append({
                        "assignment_id": mode.assignment_id,
                        "status": "skipped",
                        "message": "No Canvas assignment mapping"
                    })
                    continue
                
                # Build comments dict for this assignment
                # Map SIS login ID (student_id) to Canvas user ID
                comments_dict = {}
                for entry in mode.entries:
                    if entry.comment and entry.comment.strip():
                        # Look up Canvas user ID from batch mapping
                        canvas_user_id = batch.canvas_user_ids.get(entry.student_id)
                        if canvas_user_id:
                            comments_dict[canvas_user_id] = {
                                "text": entry.comment.strip()
                            }
                            print(f"[DEBUG] Found comment for student {entry.student_id}: {entry.comment[:50]}...")
                        else:
                            # Log warning if Canvas user ID not found
                            print(f"[COMMENT POST] Warning: No Canvas user ID for student {entry.student_id}")
                
                print(f"[DEBUG] Assignment {mode.assignment_id}: {len(comments_dict)} comments to post")
                
                if not comments_dict:
                    results_summary.append({
                        "assignment_id": mode.assignment_id,
                        "status": "skipped",
                        "message": "No comments to post"
                    })
                    continue
                
                # Post comments to Canvas
                print(f"[DEBUG] Posting comments to Canvas: {mode.canvas_course_id}, assignment {mode.canvas_assignment_id}")
                result = client.bulk_post_comments(
                    course_id=mode.canvas_course_id,
                    assignment_id=mode.canvas_assignment_id,
                    comments=comments_dict
                )
                
                success_count += len(result.get("success", []))
                failed_count += len(result.get("failed", []))
                
                results_summary.append({
                    "assignment_id": mode.assignment_id,
                    "status": "posted",
                    "success": len(result.get("success", [])),
                    "failed": len(result.get("failed", [])),
                    "errors": [f["error"] for f in result.get("failed", [])]
                })
            
            # Close client
            client.close()
            
        except CanvasClientError as e:
            raise HTTPException(status_code=500, detail=f"Canvas API error: {e}")
        
        print(f"[DEBUG] Canvas posting complete: {success_count} success, {failed_count} failed")
        
        return templates.TemplateResponse(
            "classwork_batch_done.html",
            {
                "request": request,
                "export_id": export_id,
                "canvas_results": results_summary,
                "total_success": success_count,
                "total_failed": failed_count,
                "comments_files": manifest.get("comments_files", {}),
                "assignment_ids": manifest.get("assignment_ids", []),
                "closed": False,
            },
        )


