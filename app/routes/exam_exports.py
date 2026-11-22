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
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from ..exam_exports_core import (
    _build_best_rows_by_student,
    _feedback_text_from_row,
    _find_canvas_original_zip_for_quiz,
    _scans_with_feedback_pdf_bytes,
    roster_lookup,
)
from ..exam_shared import (
    _build_roster_prefix_index,
    _build_total_scores_by_student,
    _parse_scale_to,
    _filename_prefix_key,
    _required_total_max,
    _score_summary_for_work_unit,
    _selected_work_units_by_student,
    _work_units_by_student,
    _incomplete_regular_scores,
    _require_complete_regular_scores,
)
from ..session_state import _session_lock, get_session, persist_session
from ..storage import GRADES_DIR, RAW_SCANS_DIR, RUBRIC_ASSETS_DIR, archive_current_data
from ..classwork_core import _render_feedback_image_bytes, _render_feedback_pdf_bytes
from ..quiz_variants import available_quiz_versions, normalize_quiz_version
from ..rubric_assets import existing_question_assets
from ..fol_truth_assistance import (enabled as fol_truth_enabled, assistance_schema,
    assert_supported as assert_fol_truth_supported, question_context as fol_truth_question_context)
from ..fol_truth_preparation import definitions_approved
from ..ta_releases import explainer_revision, tested_releases

router = APIRouter()
templates = Jinja2Templates(directory=str(Path(__file__).resolve().parents[1] / 'templates'))


def _tf_grader_mode_enabled() -> bool:
    return os.getenv("TF_GRADER_MODE", "").strip().lower() in {"1", "true", "yes", "on"}


def _tf_code_requirement(grading_key) -> tuple[int | None, str | None]:
    if fol_truth_enabled(grading_key):
        return assistance_schema(grading_key), explainer_revision()
    return None, None


def _matching_code_release(schema, revision):
    return next((release for release in tested_releases() if release.supports(schema, revision)), None)


@router.get("/tf-grader-code/{filename}")
def download_tf_grader_code(filename: str) -> Response:
    if _tf_grader_mode_enabled():
        raise HTTPException(403, "Code release downloads are available in instructor mode.")
    release = next((item for item in tested_releases() if item.archive.name == filename), None)
    if release is None:
        raise HTTPException(404, "That tested grader code ZIP is unavailable.")
    return FileResponse(release.archive, filename=release.archive.name, media_type="application/zip")


def _duplicate_submission_rows(
    session,
    scale_to: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Build a human-readable comparison of duplicate student submissions."""
    roster_map = {entry.student_id: entry.name for entry in session.roster}
    selected = _selected_work_units_by_student(session, scale_to=scale_to)
    rows: List[Dict[str, Any]] = []
    for student_id, work_units in _work_units_by_student(session).items():
        if len(work_units) < 2:
            continue
        submissions: List[Dict[str, Any]] = []
        for wu in work_units:
            summary = _score_summary_for_work_unit(session, wu, scale_to=scale_to)
            submissions.append(
                {
                    "unit_id": wu.unit_id,
                    "quiz_version": normalize_quiz_version(wu.quiz_version),
                    "per_problem_scores": {
                        problem.id: wu.per_problem.get(problem.id)
                        for problem in session.grading_key.problems
                    },
                    "regular_score": summary["regular_score"],
                    "extra_credit_score": summary["extra_credit_score"],
                    "raw_total_score": summary["raw_total_score"],
                    "total_score": summary["total_score"],
                    "is_selected": selected[student_id].unit_id == wu.unit_id,
                    "pages": [
                        {
                            "filename": Path(filename).name,
                            "url": f"/scans/{quote(Path(filename).name)}",
                        }
                        for filename in wu.page_filenames
                    ],
                }
            )
        rows.append(
            {
                "student_id": student_id,
                "student_name": roster_map.get(student_id, student_id),
                "submissions": submissions,
            }
        )
    return sorted(rows, key=lambda row: row["student_name"].lower())


def _prune_unit_from_segment_caches(unit_id: int) -> None:
    """Keep prepared location caches valid after an instructor drops a duplicate."""
    if not GRADES_DIR.exists():
        return
    for cache_path in GRADES_DIR.glob("segments_*.json"):
        try:
            rows = json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(rows, list):
            continue
        kept = [
            row
            for row in rows
            if not isinstance(row, dict) or row.get("unit_id") != unit_id
        ]
        if len(kept) != len(rows):
            cache_path.write_text(
                json.dumps(kept, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )


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


def _problem_location_cache_ready(session, problem_id: str) -> bool:
    """Return whether one problem has a complete, bundle-safe location cache."""
    units = {wu.unit_id: wu for wu in session.work_units}
    if not units:
        return False

    cache_path = GRADES_DIR / f"segments_{problem_id}.json"
    if not cache_path.is_file():
        return False
    try:
        rows = json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if not isinstance(rows, list):
        return False

    seen: set[int] = set()
    for row in rows:
        if not isinstance(row, dict) or row.get("unit_id") not in units:
            return False
        unit_id = row["unit_id"]
        if unit_id in seen:
            return False
        seen.add(unit_id)
        if row.get("present"):
            if row.get("page_filename") not in units[unit_id].page_filenames:
                return False
            if not _valid_normalized_bbox(row.get("bbox")):
                return False

    return seen == set(units)


def _validate_tf_bundle(session) -> tuple[List[str], List[Path]]:
    """Check that the handoff can grade every problem without a local model."""
    errors: List[str] = []
    cache_paths: List[Path] = []
    if not definitions_approved(session):
        errors.append('Review and approve the models and sentences on the bundle preparation page before exporting FOL feedback.')
    try:
        assert_fol_truth_supported(session.grading_key)
    except ValueError as exc:
        errors.append(str(exc))
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
$manifest = Get-Content -LiteralPath (Join-Path $bundleRoot "bundle_manifest.json") -Raw | ConvertFrom-Json
if ($manifest.fol_truth_assistance) {
    if (-not (Test-Path -LiteralPath (Join-Path $resolvedCode "app\fol_truth_assistance.py")) -or
        -not (Test-Path -LiteralPath (Join-Path $resolvedCode "app\assets\fol_truth_review.mjs")) -or
        -not (Test-Path -LiteralPath (Join-Path $resolvedCode "app\assets\fol_truth_scoring.mjs")) -or
        -not (Test-Path -LiteralPath (Join-Path $resolvedCode "app\assets\vendor\fol-explanations\index.mjs")) -or
        -not (Test-Path -LiteralPath (Join-Path $resolvedCode "app\assets\vendor\fol-explanations\SOURCE.json"))) {
        throw "This bundle needs matching grader code with FOL truth assistance and its shared browser module. Extract the updated code ZIP."
    }
    if ($manifest.fol_truth_assistance.schema -ge 2) {
        $capabilitiesPath = Join-Path $resolvedCode "app\\assets\\fol_truth_capabilities.json"
        if (-not (Test-Path -LiteralPath $capabilitiesPath)) {
            throw "This bundle needs updated rubric-driven FOL truth assistance. Extract the matching code ZIP."
        }
        $capabilities = Get-Content -LiteralPath $capabilitiesPath -Raw | ConvertFrom-Json
        if ($manifest.fol_truth_assistance.schema -notin $capabilities.schemas) {
            throw "This grader does not support the bundle's FOL truth assistance schema."
        }
    }
    $sharedSource = Get-Content -LiteralPath (Join-Path $resolvedCode "app\assets\vendor\fol-explanations\SOURCE.json") -Raw | ConvertFrom-Json
    if ($sharedSource.commit -ne $manifest.fol_truth_assistance.shared_module_commit) {
        throw "The grader code has a different FOL explanation module revision. Extract the matching updated code ZIP."
    }
}
$env:DATA_DIR = (Resolve-Path -LiteralPath (Join-Path $bundleRoot "data")).Path
$env:DATA_ARCHIVE_DIR = Join-Path $bundleRoot "archive"
$env:ENV_MODE = "PRODUCTION"
$env:TF_GRADER_MODE = "1"
Set-Location -LiteralPath $resolvedCode
$venvPython = Join-Path $resolvedCode ".venv\Scripts\python.exe"
$pythonCommand = if (Test-Path -LiteralPath $venvPython) {
    $venvPython
} elseif (Get-Command py -ErrorAction SilentlyContinue) {
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


def _tf_mac_launcher_text() -> str:
    return '''#!/bin/bash
set -euo pipefail

BUNDLE_ROOT="$(cd "$(dirname "$0")" && pwd)"
CODE_DIR="${1:-}"
PORT="${PORT:-5001}"

if [ -z "$CODE_DIR" ]; then
  read -r -p "Full path to the logic-grading-facilitator code folder: " CODE_DIR
fi
if [ ! -f "$CODE_DIR/app/server.py" ]; then
  echo "That folder does not contain app/server.py."
  exit 1
fi
if grep -q '"fol_truth_assistance"' "$BUNDLE_ROOT/bundle_manifest.json"; then
  if [ ! -f "$CODE_DIR/app/fol_truth_assistance.py" ] ||
     [ ! -f "$CODE_DIR/app/assets/fol_truth_review.mjs" ] ||
     [ ! -f "$CODE_DIR/app/assets/fol_truth_scoring.mjs" ] ||
     [ ! -f "$CODE_DIR/app/assets/vendor/fol-explanations/index.mjs" ] ||
     [ ! -f "$CODE_DIR/app/assets/vendor/fol-explanations/SOURCE.json" ]; then
    echo "This bundle needs matching grader code with FOL truth assistance and its shared browser module. Extract the updated code ZIP."
    exit 1
  fi
  REQUIRED_COMMIT="$(grep -o '"shared_module_commit": "[^"]*"' "$BUNDLE_ROOT/bundle_manifest.json" | head -1 | cut -d '"' -f 4)"
  if [ -z "$REQUIRED_COMMIT" ] || ! grep -Fq "$REQUIRED_COMMIT" "$CODE_DIR/app/assets/vendor/fol-explanations/SOURCE.json"; then
    echo "The grader code has a different FOL explanation module revision. Extract the matching updated code ZIP."
    exit 1
  fi
fi

export DATA_DIR="$BUNDLE_ROOT/data"
export DATA_ARCHIVE_DIR="$BUNDLE_ROOT/archive"
export ENV_MODE="PRODUCTION"
export TF_GRADER_MODE="1"

VENV_PYTHON="$CODE_DIR/.venv/bin/python"
if [ -x "$VENV_PYTHON" ]; then
  PYTHON_COMMAND="$VENV_PYTHON"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_COMMAND="python3"
elif command -v python >/dev/null 2>&1; then
  PYTHON_COMMAND="python"
else
  echo "Python was not found. Install Python 3, reopen Terminal, and try again."
  exit 1
fi

cd "$CODE_DIR"
"$PYTHON_COMMAND" -c 'import json,sys,pathlib; m=json.load(open(sys.argv[1])); s=m.get("fol_truth_assistance",{}).get("schema"); p=pathlib.Path("app/assets/fol_truth_capabilities.json"); c=json.loads(p.read_text()) if p.is_file() else {"schemas":[1]}; sys.exit("This bundle needs matching updated FOL truth grader code.") if s and s not in c["schemas"] else None' "$BUNDLE_ROOT/bundle_manifest.json"
echo "TF grader: http://127.0.0.1:$PORT/problem/Q1/strips"
echo "Keep this window open while grading. Press Control-C to stop."
"$PYTHON_COMMAND" -m uvicorn app.server:app --port "$PORT"
'''


def _write_executable_zip_text(zf: zipfile.ZipFile, name: str, text: str) -> None:
    info = zipfile.ZipInfo(name)
    info.create_system = 3
    info.external_attr = (0o100755 & 0xFFFF) << 16
    info.compress_type = zipfile.ZIP_DEFLATED
    zf.writestr(info, text)


def _tf_bundle_readme(quiz_id: str, fol_truth: bool = False) -> str:
    key_guidance = (
        "Deterministic FOL truth assistance is enabled for configured questions. Confirm all proposed readings "
        "against the scan and enter human-approved scores/comments. If grading is unclear, "
        "contact the instructor."
        if fol_truth else
        "No supplied answer key is included. If the intended grading is unclear, "
        "stop and contact the instructor rather than asking a model to decide."
    )
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
1. Install Python 3 if needed.
2. Open PowerShell (Windows) or Terminal (Mac) in the extracted code folder.
3. On Windows run:
   python -m venv .venv
   .\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt
   If `python` is not recognized for the first command, use `py` there.
4. On a Mac run:
   python3 -m venv .venv
   ./.venv/bin/python -m pip install -r requirements.txt

Start grading on Windows:
1. In this bundle folder, right-click START_TF_GRADER.ps1 and choose Run with
   PowerShell.
2. Enter the full path to the extracted CODE folder when prompted.
3. Keep the PowerShell window open and open the URL it prints.
4. If port 5000 is busy, run this from PowerShell instead:
   powershell -ExecutionPolicy Bypass -File .\\START_TF_GRADER.ps1 -Port 5001

Start grading on a Mac:
1. Open Terminal in this bundle folder.
2. Run: bash ./START_TF_GRADER.command
3. Enter the full path to the extracted CODE folder when prompted.
4. Keep Terminal open and open the URL it prints (normally port 5001).

Use the A/B/C tabs to grade one problem/version across its students. The blank
printed question stays at the top. Check a response's version badge if it looks
out of place, optionally pin a response you judge correct as that version's
visual reference, enter any partial-credit score you judge appropriate, and use
Save & Next. {key_guidance}
If a crop is wrong, use Show Full Page or Fix Crop. Do not click the global
Segmentation refresh link because this handoff intentionally has no model.

When finished, download the grades CSV and the scans + feedback PDFs ZIP from
the Finish page. The ZIP has one named PDF per student: original scan pages
followed by typed scores and comments. Return those files plus
data\\grades\\session.json to the instructor. Do not upload this bundle to
GitHub or personal cloud storage; it contains protected student work.
"""

@router.get("/finish", response_class=HTMLResponse)
def finish_page(
    request: Request,
    scale_to: Optional[float] = None,
    mode: Optional[str] = None,
) -> Response:
    session = get_session()
    grading_key = session.grading_key
    problems = grading_key.problems if grading_key else []
    total_max = _required_total_max(grading_key) if problems else 0.0
    scale_to = _parse_scale_to(scale_to)

    rows_by_student = _build_best_rows_by_student(session, scale_to=scale_to)
    rows = sorted(rows_by_student.values(), key=lambda r: r["student_name"])
    duplicate_rows = _duplicate_submission_rows(session, scale_to=scale_to)
    tf_grader_mode = _tf_grader_mode_enabled()
    schema, revision = _tf_code_requirement(grading_key) if not tf_grader_mode else (None, None)
    code_release = _matching_code_release(schema, revision) if not tf_grader_mode else None
    location_ready_ids = [
        problem.id
        for problem in problems
        if _problem_location_cache_ready(session, problem.id)
    ]
    location_missing_ids = [
        problem.id for problem in problems if problem.id not in location_ready_ids
    ]

    return templates.TemplateResponse(
        "finish.html",
        {
            "request": request,
            "quiz_id": grading_key.quiz_id,
            "problems": problems,
            "total_max": total_max,
            "scale_to": scale_to,
            "rows": rows,
            "duplicate_rows": duplicate_rows,
            "has_extra_credit": any(problem.extra_credit for problem in problems),
            "tf_grader_mode": tf_grader_mode,
            "tf_prep_mode": mode == "tf-prep",
            "location_ready_count": len(location_ready_ids),
            "location_total_count": len(problems),
            "location_missing_ids": location_missing_ids,
            "location_unit_ids": [wu.unit_id for wu in session.work_units],
            "locations_all_ready": bool(problems) and not location_missing_ids,
            "tf_code_schema": schema,
            "tf_code_release": code_release.archive.name if code_release else None,
            "fol_truth_enabled": fol_truth_enabled(grading_key),
            "fol_truth_definitions_approved": definitions_approved(session),
            "fol_truth_tasks": [
                {"problem_id": p.id, "unit_id": wu.unit_id}
                for p in problems for wu in session.work_units
                if fol_truth_question_context(grading_key, p.id, normalize_quiz_version(wu.quiz_version), None) is not None
            ],
            "message": request.query_params.get("message"),
            "incomplete_grades": _incomplete_regular_scores(session),
        },
    )


@router.post("/duplicate-submissions/{unit_id}/exclude")
def exclude_duplicate_submission(unit_id: int) -> Response:
    """Remove one duplicate WorkUnit while retaining its scan files on disk."""
    if _tf_grader_mode_enabled():
        raise HTTPException(
            status_code=403,
            detail="Only the instructor can exclude a duplicate submission.",
        )

    with _session_lock:
        session = get_session()
        target = next((wu for wu in session.work_units if wu.unit_id == unit_id), None)
        if target is None or not target.student_id:
            raise HTTPException(status_code=404, detail=f"Unknown submission unit {unit_id}.")
        matching = [wu for wu in session.work_units if wu.student_id == target.student_id]
        if len(matching) < 2:
            raise HTTPException(
                status_code=400,
                detail="That student no longer has duplicate submissions.",
            )

        session.work_units = [wu for wu in session.work_units if wu.unit_id != unit_id]
        for problem_id, pins in list(session.reference_pins.items()):
            kept_pins = {
                version: pinned_unit_id
                for version, pinned_unit_id in pins.items()
                if pinned_unit_id != unit_id
            }
            if kept_pins:
                session.reference_pins[problem_id] = kept_pins
            else:
                session.reference_pins.pop(problem_id, None)
        _prune_unit_from_segment_caches(unit_id)
        persist_session()

    message = quote(
        f"Excluded duplicate unit {unit_id} from grading. Its scan files remain in local storage."
    )
    return RedirectResponse(url=f"/finish?message={message}", status_code=303)


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
def export_tf_bundle(allow_duplicates: bool = False):
    """Export the minimum private dataset needed for model-free strip grading."""
    session = get_session()
    duplicate_rows = _duplicate_submission_rows(session)
    if duplicate_rows and not allow_duplicates:
        duplicate_details = []
        for row in duplicate_rows:
            scores = ", ".join(
                f"unit {submission['unit_id']} = {submission['total_score']:g}"
                for submission in row["submissions"]
            )
            duplicate_details.append(
                f"{row['student_name']} ({row['student_id']}): {scores}"
            )
        raise HTTPException(
            status_code=409,
            detail=(
                "TF bundle has duplicate submissions. Review them on the Finish page, "
                "exclude the duplicate unit, or explicitly choose Keep both and prepare bundle. "
                + " | ".join(duplicate_details)
            ),
        )
    errors, cache_paths = _validate_tf_bundle(session)
    if errors:
        detail = "TF bundle is not ready:\n- " + "\n- ".join(errors[:40])
        if len(errors) > 40:
            detail += f"\n- ...and {len(errors) - 40} more problem(s)."
        raise HTTPException(status_code=400, detail=detail)

    quiz_slug = _safe_bundle_slug(session.grading_key.quiz_id)
    root = f"{quiz_slug}_tf_bundle"
    # A legacy/instructor rubric may contain answer_key fields. They are not part
    # of the TF trust boundary: the grader gets prompts, points, and
    # human-selected visual references, never generated grading advice.
    rubric_data = session.grading_key.model_dump(exclude_none=True)
    for problem in rubric_data.get("problems", []):
        problem.pop("answer_key", None)
        problem.pop("rubric_notes", None)
        for variant in (problem.get("variants") or {}).values():
            variant.pop("answer_key", None)
            variant.pop("rubric_notes", None)

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
        "deterministic_grading_assistance_included": fol_truth_enabled(session.grading_key),
        "blank_question_image_count": len(question_assets),
    }
    schema, revision = _tf_code_requirement(session.grading_key)
    code_release = _matching_code_release(schema, revision)
    if code_release:
        manifest["grader_code_release"] = {"archive": code_release.archive.name, "sha256": code_release.sha256}
    if schema is not None:
        manifest["fol_truth_assistance"] = {
            "schema": schema,
            "assessment": ((session.grading_key.fol_truth_assistance or {}).get("assessment")
                           if schema == 1 else session.grading_key.quiz_id),
            "required_grader_feature": ("quiz9_fol_truth_assistance_v1" if schema == 1
                                        else f"fol_truth_assistance_v{schema}"),
            "shared_module_commit": revision,
            "logical_inputs_reviewed": definitions_approved(session),
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
        _write_executable_zip_text(
            zf,
            f"{root}/START_TF_GRADER.command",
            _tf_mac_launcher_text(),
        )
        readme = _tf_bundle_readme(session.grading_key.quiz_id,
                                  fol_truth=fol_truth_enabled(session.grading_key))
        if code_release:
            readme += f"\nMATCHING GRADER CODE\nUse {code_release.archive.name} with this private bundle.\n"
        if fol_truth_enabled(session.grading_key):
            readme += (
                "\nFOL TRUTH ASSISTANCE\n"
                "The updated code ZIP must support the manifest's assistance schema and the\n"
                "pinned explanation module. The strip view shows proposed vision readings\n"
                "when available. Confirm each reading against the scan, choose whether\n"
                "to use or edit suggested feedback and scores. Do not\n"
                "run vision preparation or refresh crops in TF mode.\n"
            )
        zf.writestr(f"{root}/READ_ME_FIRST.txt", readme)

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
    Export one PDF per student as a ZIP.

    Each named PDF contains the student's original scan page(s), followed by a
    typed feedback page containing scores and comments. Filenames include the
    student name slug and student_id for straightforward manual Canvas upload.
    """
    session = get_session()
    _require_complete_regular_scores(session)
    grading_key = session.grading_key
    problems = grading_key.problems
    total_max = _required_total_max(grading_key) if problems else 0.0
    id_to_slug = roster_lookup(session)
    rows_by_student = _build_best_rows_by_student(session)

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        manifest = io.StringIO()
        writer = csv.writer(manifest)
        writer.writerow(
            [
                "student_id",
                "student_name",
                "filename",
                "total_score",
                "scan_pages",
                "feedback_pages",
                "scan_warnings",
            ]
        )

        for row in sorted(rows_by_student.values(), key=lambda r: r["student_name"]):
            sid = row["student_id"]
            slug = id_to_slug.get(sid, f"student_{sid}")
            version = normalize_quiz_version(row.get("quiz_version"))
            version_part = f"_v{version}" if version else ""
            filename = f"{grading_key.quiz_id}_{slug}_{sid}{version_part}_graded.pdf"
            feedback_text = _feedback_text_from_row(
                row,
                problems,
                grading_key.quiz_id,
                total_max,
            )
            scan_paths = [
                RAW_SCANS_DIR / Path(page_filename).name
                for page_filename in row.get("page_filenames", [])
            ]
            pdf_bytes, scan_pages, feedback_pages, warnings = (
                _scans_with_feedback_pdf_bytes(scan_paths, feedback_text)
            )

            zf.writestr(filename, pdf_bytes)
            writer.writerow(
                [
                    sid,
                    row["student_name"],
                    filename,
                    row["total_score"],
                    scan_pages,
                    feedback_pages,
                    " | ".join(warnings),
                ]
            )

        zf.writestr(
            f"{grading_key.quiz_id}_feedback_manifest.csv",
            manifest.getvalue(),
        )

    buf.seek(0)
    filename = f"{grading_key.quiz_id}_scans_with_feedback.zip"
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
    _require_complete_regular_scores(session)
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

