from __future__ import annotations

from typing import Any, Dict, Optional
import csv
import io
import zipfile
from pathlib import Path
from urllib.parse import quote

from fastapi import APIRouter, Form, HTTPException, Request, Response
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from ..exam_exports_core import (
    _build_best_rows_by_student,
    _feedback_text_from_row,
    _find_canvas_original_zip_for_quiz,
    roster_lookup,
)
from ..exam_shared import (
    _build_roster_prefix_index,
    _build_total_scores_by_student,
    _parse_scale_to,
    _filename_prefix_key,
    _required_total_max,
    _score_summary_for_work_unit,
)
from ..session_state import _session_lock, get_session
from ..storage import RAW_SCANS_DIR, archive_current_data
from ..classwork_core import _render_feedback_image_bytes, _render_feedback_pdf_bytes

router = APIRouter()
templates = Jinja2Templates(directory=str(Path(__file__).resolve().parents[1] / 'templates'))

@router.get("/finish", response_class=HTMLResponse)
def finish_page(request: Request, scale_to: Optional[float] = None) -> Response:
    session = get_session()
    grading_key = session.grading_key
    problems = grading_key.problems if grading_key else []
    total_max = _required_total_max(grading_key) if problems else 0.0
    scale_to = _parse_scale_to(scale_to)

    roster_map = {r.student_id: r.name for r in session.roster}
    rows_by_student: Dict[str, Dict] = {}

    for wu in session.work_units:
        if not wu.student_id:
            continue
        per_problem_scores = {}
        per_problem_comments = {}
        for p in problems:
            score = wu.per_problem.get(p.id)
            per_problem_scores[p.id] = score
            per_problem_comments[p.id] = wu.per_problem_comments.get(p.id)
        score_summary = _score_summary_for_work_unit(session, wu)
        total = score_summary["total_score"]

        sid = wu.student_id
        existing = rows_by_student.get(sid)
        if existing is None or total > existing["total_score"]:
            rows_by_student[sid] = {
                "student_id": sid,
                "student_name": roster_map.get(sid, sid),
                "per_problem_scores": per_problem_scores,
                "per_problem_comments": per_problem_comments,
                "regular_score": score_summary["regular_score"],
                "extra_credit_score": score_summary["extra_credit_score"],
                "raw_total_score": score_summary["raw_total_score"],
                "total_score": total,
            }

    rows = sorted(rows_by_student.values(), key=lambda r: r["student_name"])

    return templates.TemplateResponse(
        "finish.html",
        {
            "request": request,
            "problems": problems,
            "total_max": total_max,
            "scale_to": scale_to,
            "rows": rows,
        },
    )


@router.get("/export_renamed_scans")
def export_renamed_scans():
    """
    Export a ZIP of scan files renamed based on assigned student.

    For each WorkUnit with a student_id:
      - Look up the student's name in the roster.
      - Create filenames like: <slug>_<n>.<ext>
        where n increments per student across all their pages.
    Unassigned WorkUnits are skipped by default.
    """
    session = get_session()
    id_to_slug = roster_lookup(session)

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        # per-student page counters
        counters: Dict[str, int] = {}

        for wu in session.work_units:
            sid = wu.student_id
            if not sid:
                continue  # skip unassigned

            slug = id_to_slug.get(sid, f"student_{sid}")
            counters.setdefault(sid, 0)

            for page_filename in wu.page_filenames:
                counters[sid] += 1
                # original extension (normalized to lower-case)
                ext = Path(page_filename).suffix.lower() or ".jpg"
                new_name = f"{slug}_{counters[sid]}{ext}"

                src_path = RAW_SCANS_DIR / page_filename
                if not src_path.exists():
                    # skip missing files rather than crash
                    continue

                with src_path.open("rb") as f:
                    data = f.read()

                # Put into ZIP under new_name
                zf.writestr(new_name, data)

    buf.seek(0)
    filename = f"{session.grading_key.quiz_id}_renamed_scans.zip"
    return Response(
        content=buf.getvalue(),
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="{quote(filename)}"',
        },
    )


@router.get("/export")
def export_scores(scale_to: Optional[float] = None):
    """
    Export a simple CSV of student_id,total_score for completed WorkUnits.

    - Skips WorkUnits with no student_id.
    - If multiple WorkUnits are assigned to the same student_id,
      we export the *maximum* total score among them (to avoid double-grading
      when something was scanned/created twice).
    """
    session = get_session()
    scale_to = _parse_scale_to(scale_to)
    totals = _build_total_scores_by_student(session, scale_to=scale_to)

    # Build CSV in-memory
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["student_id", f"{session.grading_key.quiz_id}_score"])
    for sid, total in totals.items():
        writer.writerow([sid, total])

    csv_data = buf.getvalue()
    filename = f"{session.grading_key.quiz_id}_scores.csv"
    if scale_to is not None:
        scale_label = str(int(scale_to)) if float(scale_to).is_integer() else str(scale_to)
        filename = f"{session.grading_key.quiz_id}_scores_out_of_{scale_label}.csv"

    return Response(
        content=csv_data,
        media_type="text/csv",
        headers={
            "Content-Disposition": (f'attachment; filename="{filename}"')
        },
    )

@router.post("/finalize", response_class=HTMLResponse)
def finalize_assignment(label: str = Form(...)):
    """
    Archive the current data/ directory (including scans, session.json,
    rubric, roster, etc.) into data_archive/<timestamp>_<label>/,
    then clear raw_scans/ and session.json so the app is ready for
    a new assignment.

    The label is provided by the user via a form on the grading page.
    """
    with _session_lock:
        snapshot_id = archive_current_data(label)

    # Very simple HTML response; you can style this later.
    # We intentionally don't redirect to a grading page because after
    # reset there may be no session.json yet.
    html = f"""
    <html>
      <head>
        <title>Assignment finalized</title>
      </head>
      <body style="font-family: sans-serif; padding: 1.5rem;">
        <h1>Assignment finalized & archived</h1>
        <p>Archived current data under: <code>{snapshot_id}</code></p>
        <p>Archive path: <code>data_archive/{snapshot_id}/</code></p>
        <p>
          Next steps:
        </p>
        <ol>
          <li>Place new scans into <code>data/raw_scans/</code>.</li>
          <li>Update <code>data/rubric/...</code> if needed.</li>
          <li>Run <code>scripts/init_session.py</code> to create a new session.</li>
          <li>Reload the grading UI.</li>
        </ol>
        <p><a href="/docs">Back to API docs</a></p>
      </body>
    </html>
    """
    return HTMLResponse(content=html)


@router.get("/export_feedback_zip")
def export_feedback_zip():
    """
    Export per-student feedback files (rubric, per-problem scores, comments) as a ZIP.
    Filenames include student name (slug) and student_id for easy Canvas drag/drop.
    """
    session = get_session()
    grading_key = session.grading_key
    problems = grading_key.problems
    total_max = _required_total_max(grading_key) if problems else 0.0

    roster_map = {r.student_id: r.name for r in session.roster}
    id_to_slug = roster_lookup(session)

    # Choose the max-total WorkUnit per student_id (matches export_scores behavior)
    rows_by_student: Dict[str, Dict] = {}
    for wu in session.work_units:
        if not wu.student_id:
            continue
        per_problem_scores = {}
        per_problem_comments = {}
        for p in problems:
            score = wu.per_problem.get(p.id)
            per_problem_scores[p.id] = score
            per_problem_comments[p.id] = wu.per_problem_comments.get(p.id)
        score_summary = _score_summary_for_work_unit(session, wu)
        total = score_summary["total_score"]

        sid = wu.student_id
        existing = rows_by_student.get(sid)
        if existing is None or total > existing["total_score"]:
            rows_by_student[sid] = {
                "student_id": sid,
                "student_name": roster_map.get(sid, sid),
                "per_problem_scores": per_problem_scores,
                "per_problem_comments": per_problem_comments,
                "regular_score": score_summary["regular_score"],
                "extra_credit_score": score_summary["extra_credit_score"],
                "total_score": total,
            }

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        manifest = io.StringIO()
        writer = csv.writer(manifest)
        writer.writerow(["student_id", "student_name", "filename", "total_score"])

        for row in sorted(rows_by_student.values(), key=lambda r: r["student_name"]):
            sid = row["student_id"]
            name = row["student_name"]
            slug = id_to_slug.get(sid, f"student_{sid}")
            filename = f"{grading_key.quiz_id}_{slug}_{sid}_feedback.txt"

            lines = []
            lines.append(f"Quiz: {grading_key.quiz_id}")
            lines.append(f"Student: {name} ({sid})")
            lines.append(f"Total: {row['total_score']} / {total_max}")
            if row["extra_credit_score"] > 0:
                lines.append(
                    f"Extra credit applied: {row['extra_credit_score']:g} (capped at {total_max:g})"
                )
            lines.append("")
            lines.append("Rubric + Scores")
            for p in problems:
                score = row["per_problem_scores"].get(p.id)
                comment = row["per_problem_comments"].get(p.id)
                score_display = "" if score is None else str(score)
                points_label = f"{p.points} pts"
                if p.extra_credit:
                    points_label = f"EC {p.points} pts"
                lines.append(f"- {p.id} ({p.label}) [{points_label}]: {score_display}")
                if comment:
                    lines.append(f"  Comment: {comment}")
            lines.append("")

            zf.writestr(filename, "\n".join(lines))
            writer.writerow([sid, name, filename, row["total_score"]])

        zf.writestr(
            f"{grading_key.quiz_id}_feedback_manifest.csv",
            manifest.getvalue(),
        )

    buf.seek(0)
    filename = f"{grading_key.quiz_id}_feedback.zip"
    return Response(
        content=buf.getvalue(),
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="{quote(filename)}"',
        },
    )


@router.get("/export_canvas_reupload_feedback_zip")
def export_canvas_reupload_feedback_zip(source_zip: Optional[str] = None):
    """
    Build a Canvas "re-upload submissions" ZIP using original Canvas filenames.
    Content is replaced with per-student feedback rendered as PDF/image, preserving
    each file's original name and extension.
    """
    session = get_session()
    grading_key = session.grading_key
    problems = grading_key.problems
    total_max = _required_total_max(grading_key) if problems else 0.0
    rows_by_student = _build_best_rows_by_student(session)

    roster_index, collisions = _build_roster_prefix_index(session.roster)
    if collisions:
        raise HTTPException(
            status_code=400,
            detail="Roster collisions prevent Canvas filename matching.",
        )

    source_zip_path = _find_canvas_original_zip_for_quiz(grading_key.quiz_id, source_zip)
    out_buf = io.BytesIO()
    matched = 0
    total = 0

    with zipfile.ZipFile(source_zip_path, "r") as src, zipfile.ZipFile(
        out_buf, "w", compression=zipfile.ZIP_DEFLATED
    ) as dst:
        for info in src.infolist():
            if info.is_dir():
                continue
            total += 1
            original_name = Path(info.filename).name
            if not original_name:
                continue

            ext = Path(original_name).suffix.lower()
            prefix_key = _filename_prefix_key(original_name)
            roster_entry = roster_index.get(prefix_key)
            sid = roster_entry.student_id if roster_entry else None
            row = rows_by_student.get(sid) if sid else None
            text = _feedback_text_from_row(row, problems, grading_key.quiz_id, total_max)

            try:
                if ext == ".pdf":
                    payload = _render_feedback_pdf_bytes(text)
                elif ext in {".jpg", ".jpeg", ".png"}:
                    payload = _render_feedback_image_bytes(text, ext)
                else:
                    payload = src.read(info)
                dst.writestr(original_name, payload)
                if sid:
                    matched += 1
            except Exception as e:
                print(f"[CANVAS REUPLOAD] Failed to render {original_name}: {e}")
                dst.writestr(original_name, src.read(info))

    filename = f"{grading_key.quiz_id}_canvas_reupload_feedback.zip"
    print(
        f"[CANVAS REUPLOAD] Source={source_zip_path.name}, files={total}, "
        f"matched={matched}, output={filename}"
    )
    return Response(
        content=out_buf.getvalue(),
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="{quote(filename)}"',
        },
    )

