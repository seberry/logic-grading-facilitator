from __future__ import annotations

from typing import Any, Dict, List, Optional
import csv
import io
import json
import math
import os
import re
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
from ..storage import GRADES_DIR, RAW_SCANS_DIR, RUBRIC_ASSETS_DIR, archive_current_data
from ..classwork_core import _render_feedback_image_bytes, _render_feedback_pdf_bytes
from ..quiz_variants import available_quiz_versions, normalize_quiz_version
from ..rubric_assets import existing_question_assets

router = APIRouter()
templates = Jinja2Templates(directory=str(Path(__file__).resolve().parents[1] / 'templates'))


def _safe_bundle_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "quiz").strip())
    return slug.strip("._") or "quiz"


def _valid_normalized_bbox(value: Any) -> bool:
    if not isinstance(value, list) or len(value) != 4:
        return False
    try:
        x1, y1, x2, y2 = (float(item) for item in value)
    except (TypeError, ValueError):
        return False
    return (
        all(math.isfinite(item) for item in (x1, y1, x2, y2))
        and 0.0 <= x1 < x2 <= 1.0
        and 0.0 <= y1 < y2 <= 1.0
    )


def _validate_tf_bundle(session) -> tuple[List[str], List[Path]]:
    """Check that the handoff can grade every problem without a local model."""
    errors: List[str] = []
    cache_paths: List[Path] = []
    units = {wu.unit_id: wu for wu in session.work_units}

    if not units:
        errors.append("The session has no student work units.")

    for wu in session.work_units:
        if not wu.page_filenames:
            errors.append(f"Unit {wu.unit_id} has no scan files.")
        for filename in wu.page_filenames:
            if not (RAW_SCANS_DIR / filename).is_file():
                errors.append(f"Unit {wu.unit_id} is missing scan file {filename!r}.")

    configured_versions = available_quiz_versions(session.grading_key)
    if configured_versions:
        allowed = set(configured_versions)
        for wu in session.work_units:
            version = normalize_quiz_version(wu.quiz_version)
            if not version:
                errors.append(f"Unit {wu.unit_id} has no confirmed quiz version.")
            elif version not in allowed:
                errors.append(
                    f"Unit {wu.unit_id} has version {version!r}, which is not in the rubric "
                    f"({', '.join(configured_versions)})."
                )

    expected_ids = set(units)
    for problem in session.grading_key.problems:
        cache_path = GRADES_DIR / f"segments_{problem.id}.json"
        if not cache_path.is_file():
            errors.append(
                f"{problem.id} has no saved question-location guesses. Open its strip page first."
            )
            continue
        try:
            rows = json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception as exc:
            errors.append(f"{problem.id} location cache cannot be read: {exc}")
            continue
        if not isinstance(rows, list):
            errors.append(f"{problem.id} location cache is not a list.")
            continue

        seen: set[int] = set()
        for row in rows:
            if not isinstance(row, dict) or row.get("unit_id") not in units:
                errors.append(f"{problem.id} location cache contains an unknown unit.")
                continue
            unit_id = row["unit_id"]
            if unit_id in seen:
                errors.append(f"{problem.id} has duplicate locations for unit {unit_id}.")
                continue
            seen.add(unit_id)
            if row.get("present"):
                filename = row.get("page_filename")
                if filename not in units[unit_id].page_filenames:
                    errors.append(f"{problem.id}, unit {unit_id} points to an unknown scan.")
                if not _valid_normalized_bbox(row.get("bbox")):
                    errors.append(f"{problem.id}, unit {unit_id} has an invalid crop box.")
        missing_ids = expected_ids - seen
        if missing_ids:
            errors.append(
                f"{problem.id} location cache is missing unit(s): "
                + ", ".join(str(value) for value in sorted(missing_ids))
                + "."
            )
        cache_paths.append(cache_path)

    return errors, cache_paths


def _tf_launcher_text() -> str:
    return r'''param(
    [string]$CodeDir = "",
    [int]$Port = 5000
)
$ErrorActionPreference = "Stop"
$bundleRoot = $PSScriptRoot
if (-not $CodeDir) {
    $CodeDir = Read-Host "Full path to the logic-grading-facilitator code folder"
}
$resolvedCode = (Resolve-Path -LiteralPath $CodeDir).Path
if (-not (Test-Path -LiteralPath (Join-Path $resolvedCode "app\server.py"))) {
    throw "That folder does not contain app\server.py."
}
$env:DATA_DIR = (Resolve-Path -LiteralPath (Join-Path $bundleRoot "data")).Path
$env:DATA_ARCHIVE_DIR = Join-Path $bundleRoot "archive"
$env:ENV_MODE = "PRODUCTION"
$env:TF_GRADER_MODE = "1"
Set-Location -LiteralPath $resolvedCode
$pythonCommand = if (Get-Command py -ErrorAction SilentlyContinue) {
    "py"
} elseif (Get-Command python -ErrorAction SilentlyContinue) {
    "python"
} else {
    throw "Python was not found. Install Python, reopen PowerShell, and try again."
}
Write-Host "TF grader: http://127.0.0.1:$Port/problem/Q1/strips"
Write-Host "Keep this window open while grading. Press Ctrl+C to stop."
& $pythonCommand -m uvicorn app.server:app --port $Port
'''


def _tf_bundle_readme(quiz_id: str) -> str:
    return f"""TF GRADING BUNDLE: {quiz_id}

This bundle already contains the roster-derived names, scans, quiz versions,
and question-location guesses. You do not need LM Studio and should not run
segmentation refresh.

You need TWO separate items from the instructor:
1. The grader CODE, either as a GitHub download or a code ZIP. It contains no
   student work. Extract it to its own folder.
2. This private TF BUNDLE. It contains student work. Extract it to a different
   folder on an approved device; never upload it to GitHub.

One-time setup:
1. Install Python if needed.
2. Open PowerShell in the extracted code folder.
3. Run: py -m pip install -r requirements.txt
   If `py` is not recognized, run: python -m pip install -r requirements.txt

Start grading:
1. In this bundle folder, right-click START_TF_GRADER.ps1 and choose Run with
   PowerShell.
2. Enter the full path to the extracted CODE folder when prompted.
3. Keep the PowerShell window open and open the URL it prints.
4. If port 5000 is busy, run this from PowerShell instead:
   powershell -ExecutionPolicy Bypass -File .\\START_TF_GRADER.ps1 -Port 5001

Use the A/B/C tabs to grade one problem/version across its students. The blank
printed question stays at the top. Check a response's version badge if it looks
out of place, optionally pin a response you judge correct as that version's
visual reference, enter any partial-credit score you judge appropriate, and use
Save & Next. No supplied answer key is included. If the intended grading is
unclear, stop and contact the instructor rather than asking a model to decide.
If a crop is wrong, use Show Full Page or Fix Crop. Do not click the global
Segmentation refresh link because this handoff intentionally has no model.

When finished, download the grades CSV and, if comments were used, the feedback
ZIP from the Finish page. Return those files plus data\\grades\\session.json to
the instructor. Do not upload this bundle to GitHub or personal cloud storage;
it contains protected student work.
"""

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
        score_summary = _score_summary_for_work_unit(session, wu, scale_to=scale_to)
        total = score_summary["total_score"]

        sid = wu.student_id
        existing = rows_by_student.get(sid)
        if existing is None or total > existing["total_score"]:
            rows_by_student[sid] = {
                "student_id": sid,
                "student_name": roster_map.get(sid, sid),
                "quiz_version": normalize_quiz_version(wu.quiz_version),
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
            "tf_grader_mode": os.getenv("TF_GRADER_MODE", "").strip().lower()
            in {"1", "true", "yes", "on"},
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
                version = normalize_quiz_version(wu.quiz_version)
                version_part = f"_v{version}" if version else "_vUNKNOWN"
                new_name = f"{slug}{version_part}_{counters[sid]}{ext}"

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


@router.get("/export_tf_bundle")
def export_tf_bundle():
    """Export the minimum private dataset needed for model-free strip grading."""
    session = get_session()
    errors, cache_paths = _validate_tf_bundle(session)
    if errors:
        detail = "TF bundle is not ready:\n- " + "\n- ".join(errors[:40])
        if len(errors) > 40:
            detail += f"\n- ...and {len(errors) - 40} more problem(s)."
        raise HTTPException(status_code=400, detail=detail)

    quiz_slug = _safe_bundle_slug(session.grading_key.quiz_id)
    root = f"{quiz_slug}_tf_bundle"
    # A legacy/instructor rubric may contain answer_key fields. They are not part
    # of the TF trust boundary: the grader gets prompts, neutral rubric notes,
    # and human-selected visual references, never a model-authored solution key.
    rubric_data = session.grading_key.model_dump(exclude_none=True)
    for problem in rubric_data.get("problems", []):
        problem.pop("answer_key", None)
        for variant in (problem.get("variants") or {}).values():
            variant.pop("answer_key", None)

    session_data = session.model_dump()
    session_data["grading_key"] = rubric_data
    session_data["pinned_note"] = None
    session_json = json.dumps(session_data, indent=2, sort_keys=True, ensure_ascii=False)

    try:
        import yaml

        rubric_text = yaml.safe_dump(
            rubric_data,
            sort_keys=False,
            allow_unicode=True,
        )
    except Exception:
        rubric_text = json.dumps(
            rubric_data,
            indent=2,
            ensure_ascii=False,
        )

    version_counts: Dict[str, int] = {}
    for wu in session.work_units:
        version = normalize_quiz_version(wu.quiz_version) or "UNKNOWN"
        version_counts[version] = version_counts.get(version, 0) + 1
    question_assets = existing_question_assets(
        session.grading_key,
        root=RUBRIC_ASSETS_DIR,
    )
    manifest = {
        "quiz_id": session.grading_key.quiz_id,
        "work_unit_count": len(session.work_units),
        "problem_ids": [problem.id for problem in session.grading_key.problems],
        "version_counts": version_counts,
        "contains_protected_student_data": True,
        "local_model_required": False,
        "answer_keys_included": False,
        "blank_question_image_count": len(question_assets),
    }

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"{root}/data/grades/session.json", session_json)
        zf.writestr(f"{root}/data/rubric/{quiz_slug}.yaml", rubric_text)
        zf.writestr(
            f"{root}/bundle_manifest.json",
            json.dumps(manifest, indent=2, sort_keys=True),
        )
        zf.writestr(f"{root}/START_TF_GRADER.ps1", _tf_launcher_text())
        zf.writestr(f"{root}/READ_ME_FIRST.txt", _tf_bundle_readme(session.grading_key.quiz_id))

        for cache_path in cache_paths:
            zf.write(cache_path, f"{root}/data/grades/{cache_path.name}")
        for reference, asset_path in question_assets.items():
            zf.write(asset_path, f"{root}/data/rubric_assets/{reference}")
        for wu in session.work_units:
            for filename in wu.page_filenames:
                zf.write(RAW_SCANS_DIR / filename, f"{root}/data/raw_scans/{filename}")

    buf.seek(0)
    filename = f"{quiz_slug}_tf_grading_bundle.zip"
    return Response(
        content=buf.getvalue(),
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{quote(filename)}"'},
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
                "quiz_version": normalize_quiz_version(wu.quiz_version),
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
            version = normalize_quiz_version(row.get("quiz_version"))
            version_part = f"_v{version}" if version else ""
            filename = f"{grading_key.quiz_id}_{slug}_{sid}{version_part}_feedback.txt"

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

