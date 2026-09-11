from __future__ import annotations

from typing import Any, Dict, List, Optional
import io
import csv
import hashlib
import json
import textwrap
from pathlib import Path
import zipfile
from datetime import datetime
from uuid import uuid4
import re
import math
from urllib.parse import quote

from fastapi import (
    FastAPI,
    HTTPException,
    Response,
    Request,
    Form,
    UploadFile,
    File,
    Query,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from .storage import (
    RAW_SCANS_DIR,
    GRADES_DIR,
    archive_current_data,
    NAME_REVIEW_CACHE_DIR,
    DEBUG_FITCH_DIR,
    DEBUG_FOL_MODEL_DIR,
    DEBUG_MODAL_PROOF_DIR,
    AUTO_GROUPED_DIR,
    IS_PRODUCTION,
    FLASHCARDS_DIR,
    RUBRIC_ASSETS_DIR,
)
from .segmentation_core import segment_page_for_problem, QuestionSegment

from .models import (
    Session,
    WorkUnit,
    GradingKey,
    RosterEntry,
    ClassworkModeState,
    ClassworkBatchState,
    ClassworkEntry,
    ClassworkFileRef,
)
from .storage import (
    DEFAULT_SESSION_PATH,
    RAW_SCANS_DIR,  # we'll use this to serve images
    DATA_DIR,
    load_roster,
    load_grading_key,
    save_session,
    RUBRIC_DIR,
    DEFAULT_ROSTER_PATH,
    DEFAULT_GRADING_KEY_PATH,
    IS_PRODUCTION,
)
from .auto_group_core import (
    run_name_recognition,
    group_pages,
    build_roster_index,
    SubmissionGroup,
)
import shutil
import yaml
import fitz  # type: ignore
from PIL import Image, ImageDraw, ImageFont
from docx import Document


from .vlm_client import LMStudioQwenClient  # type: ignore
from .vlm_classifier import classify_page  # type: ignore
from .quiz_variants import normalize_quiz_version
from .canvas_client import CanvasClient, CanvasClientError, get_canvas_client
from .routes.fitch_debug import router as fitch_debug_router
from .routes.fol_model_debug import router as fol_model_debug_router
from .routes.modal_proof_debug import router as modal_proof_debug_router
from .routes.flashcards import router as flashcards_router
from .routes.problem_strips import router as problem_strips_router, _clear_segment_caches
from .routes.classwork import router as classwork_router
from .routes.exam_exports import router as exam_exports_router
from .helpers import _unique_path, _is_image_filename, _select_display_filename, _build_page_options
from .session_state import _session_lock, get_session, persist_session
from .classwork_core import (
    CLASSWORK_FILES_DIR,
    GRADEBOOK_MERGE_TMP_DIR,
    _assignment_column_stats,
    _candidate_assignment_columns,
    _parse_gradebook_csv,
    _render_feedback_image_bytes,
    _render_feedback_pdf_bytes,
)
from .exam_shared import (
    _build_roster_prefix_index,
    _build_total_scores_by_student,
    _canvas_import_manifest_path,
    _filename_prefix_key,
    _load_canvas_import_manifest,
    _normalize_canvas_key,
    _required_total_max,
    _save_canvas_import_manifest,
    _score_summary_for_work_unit,
    slugify_name,
)

# Submissions cache directory for downloaded Canvas files
CANVAS_SUBMISSIONS_CACHE_DIR = DATA_DIR / "canvas_cache" / "submissions"
CANVAS_SUBMISSIONS_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _create_submissions_zip_from_downloads(
    student_files: Dict[str, List[Path]],
) -> bytes:
    """
    Create a ZIP file from downloaded Canvas submissions.
    
    Args:
        student_files: Dict mapping student_id to list of downloaded file paths
        
    Returns:
        bytes: ZIP file content
    """
    buf = io.BytesIO()
    
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        # student_files maps student_id directly to file paths
        # No roster matching needed - we use the IDs from Canvas API
        for student_id, file_paths in student_files.items():
            for file_path in file_paths:
                if file_path.exists():
                    # Use original filename to preserve Canvas structure
                    original_name = file_path.name
                    zf.write(file_path, arcname=original_name)
    
    return buf.getvalue()

app = FastAPI(
    title="Logic Grading Facilitator v0",
    description="Minimal API + UI for manual grading based on WorkUnits and a Session file.",
)

# Allow local browser / tools to talk to API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # fine for local dev; don't expose publicly with real data
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve scan images under /scans/IMG_XXXX.jpg
app.mount(
    "/scans",
    StaticFiles(directory=str(RAW_SCANS_DIR)),
    name="scans",
)

TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

app.mount(
    "/debug_fitch_assets",
    StaticFiles(directory=str(DEBUG_FITCH_DIR)),
    name="debug_fitch_assets",
)

# Crops of blank quiz questions referenced by the version-aware rubric.
app.mount(
    "/rubric_assets",
    StaticFiles(directory=str(RUBRIC_ASSETS_DIR)),
    name="rubric_assets",
)

app.mount(
    "/debug_fol_model_assets",
    StaticFiles(directory=str(DEBUG_FOL_MODEL_DIR)),
    name="debug_fol_model_assets",
)

app.mount(
    "/debug_modal_proof_assets",
    StaticFiles(directory=str(DEBUG_MODAL_PROOF_DIR)),
    name="debug_modal_proof_assets",
)

app.mount(
    "/flashcards_assets",
    StaticFiles(directory=str(FLASHCARDS_DIR)),
    name="flashcards_assets",
)


LMSTUDIO_BASE_URL = "http://127.0.0.1:1234/v1"
LMSTUDIO_MODEL_ID = "qwen/qwen3-vl-4b"  # or whatever you’re actually using
CANVAS_REUPLOAD_ORIGINAL_ZIPS_DIR = DATA_DIR / "canvas_reupload" / "original_zips"
CANVAS_REUPLOAD_MANIFEST_DIR = DATA_DIR / "canvas_reupload" / "manifests"
app.mount(
    "/classwork_files",
    StaticFiles(directory=str(CLASSWORK_FILES_DIR)),
    name="classwork_files",
)
app.include_router(fitch_debug_router)
app.include_router(fol_model_debug_router)
app.include_router(modal_proof_debug_router)
app.include_router(flashcards_router)
app.include_router(problem_strips_router)
app.include_router(classwork_router)
app.include_router(exam_exports_router)

PROBLEM_TYPE_OPTIONS = [
    {"id": "none", "label": "None"},
    {"id": "fitch_proof", "label": "Fitch proof"},
    {"id": "modal_proof", "label": "Modal proof"},
    {"id": "kripke_model", "label": "Kripke model"},
    {"id": "fol_model", "label": "FOL model"},
]

def natural_sort_key(path: Path) -> List:
  """
  Sort filenames like '..._10.jpg' after '..._2.jpg' numerically.
  """
  import re
  s = path.name
  return [
      int(text) if text.isdigit() else text.lower()
      for text in re.split(r"(\d+)", s)
  ]


def canonical_filename(
    quiz_id: str,
    student_id: Optional[str],
    canvas_name: Optional[str],
    page_index: int,
    unknown_index: Optional[int],
    orig_suffix: str,
) -> str:
    """
    Produce filenames like:
      quiz1_berry_sharon_p01.jpg  (for known students)
      quiz1_unknown01_p01.jpg             (for unknown groups)
    """
    slug_quiz = quiz_id.strip().replace(" ", "_")
    if student_id and canvas_name:
        # e.g. "Berry, Sharon" -> "berry_sharon"
        name = canvas_name.lower()
        for ch in [",", ";"]:
            name = name.replace(ch, "")
        name = "_".join(part for part in name.split() if part)
        base = f"{slug_quiz}_{name}_p{page_index:02d}"
    else:
        idx = unknown_index or 1
        base = f"{slug_quiz}_unknown{idx:02d}_p{page_index:02d}"
    if not orig_suffix.startswith("."):
        orig_suffix = "." + orig_suffix
    return base + orig_suffix

def _build_page_rows_for_template(
    preds: List[Any],
    roster: List[RosterEntry],
    raw_scans_subdir: str,
) -> List[Dict[str, Any]]:
    """
    Convert VLM predictions into the page_rows structure expected by name_review.html.

    Each row dict has:
      - index
      - filename
      - image_url
      - model_student_id
      - model_canvas_name
      - confidence
      - raw_name
      - quiz_version
    """
    from pathlib import Path

    roster_by_id = {r.student_id: r.name for r in roster}

    # Ensure consistent URL prefix if you use subfolders under RAW_SCANS_DIR
    sub = raw_scans_subdir.strip().strip("/") if raw_scans_subdir else ""
    if sub:
        prefix = f"/scans/{sub}/"
    else:
        prefix = "/scans/"

    rows: List[Dict[str, Any]] = []
    for idx, p in enumerate(preds):
        filename = getattr(p, "filename", None) or getattr(p, "path", None)
        if isinstance(filename, Path):
            filename = filename.name
        if filename is None:
            continue

        student_id = getattr(p, "student_id", None)
        confidence = float(getattr(p, "confidence", 0.0))
        raw_name = getattr(p, "raw_name", None)
        quiz_version = normalize_quiz_version(getattr(p, "quiz_version", None))

        rows.append(
            {
                "index": idx,
                "filename": filename,
                "image_url": prefix + filename,
                "model_student_id": student_id,
                "model_canvas_name": roster_by_id.get(student_id) if student_id else None,
                "confidence": confidence,
                "raw_name": raw_name,
                "quiz_version": quiz_version,
            }
        )
    return rows

def _lookup_student_name(roster, student_id):
    if not student_id:
        return "(unknown student)"
    for r in roster:
        if r.student_id == student_id:
            return r.name
    return f"(unmatched: {student_id})"

def next_unit_id(session: Session) -> int:
    if not session.work_units:
        return 1
    return max(w.unit_id for w in session.work_units) + 1

def name_review_cache_path(quiz: str, raw_scans_subdir: str) -> Path:
    """
    Build a cache filename based on (quiz, raw_scans_subdir).
    Spaces and path separators get normalized.
    """
    import re, os
    key = f"{quiz or 'quiz'}__{raw_scans_subdir or 'root'}"
    # Replace path separators with underscores
    key = key.replace(os.sep, "_")
    # Collapse spaces & weird chars
    key = re.sub(r"\s+", "_", key)
    return NAME_REVIEW_CACHE_DIR / f"{key}.json"


def _has_existing_session_state() -> bool:
    """Return True if a previous grading session is present on disk.

    We only require an archive/reset when a prior session.json or cached
    name-review state exists. Allowing raw scans alone prevents the setup
    wizard from blocking itself immediately after uploading new files.
    """
    has_session = DEFAULT_SESSION_PATH.exists()
    has_name_review_cache = any(NAME_REVIEW_CACHE_DIR.glob("*.json"))
    return has_session or has_name_review_cache


def _remove_answer_key_fields(data: Any) -> int:
    """Remove legacy solution-key fields from an in-memory rubric mapping."""
    if not isinstance(data, dict):
        return 0
    removed = 0
    for problem in data.get("problems") or []:
        if not isinstance(problem, dict):
            continue
        if "answer_key" in problem:
            problem.pop("answer_key", None)
            removed += 1
        variants = problem.get("variants") or {}
        if not isinstance(variants, dict):
            continue
        for variant in variants.values():
            if isinstance(variant, dict) and "answer_key" in variant:
                variant.pop("answer_key", None)
                removed += 1
    return removed


def _read_default_rubric() -> tuple[str, str]:
    """Load rubric YAML text and quiz_id for pre-filling the setup form."""
    try:
        yaml_text = DEFAULT_GRADING_KEY_PATH.read_text(encoding="utf-8")
    except FileNotFoundError:
        yaml_text = """quiz_id: quiz1\nproblems:\n  - id: Q1\n    label: Example problem\n    points: 10\n"""

    try:
        parsed = yaml.safe_load(yaml_text) or {}
        removed = _remove_answer_key_fields(parsed)
        removed_max_points = parsed.pop("max_points", None) is not None
        if removed or removed_max_points:
            yaml_text = yaml.safe_dump(parsed, sort_keys=False, allow_unicode=True)
        quiz_id = str(parsed.get("quiz_id", "quiz1"))
    except Exception:
        quiz_id = "quiz1"

    return yaml_text, quiz_id



class GradeUpdate(BaseModel):
    student_id: Optional[str] = None
    per_problem: Optional[Dict[str, Optional[float]]] = None
    comment: Optional[str] = None
    per_problem_comments: Optional[Dict[str, Optional[str]]] = None

class WorkUnitSummary(BaseModel):
    unit_id: int
    student_id: Optional[str]
    page_count: int
    fully_scored: bool



@app.get("/")
def root():
    """Make the human-facing Setup Wizard the default landing page."""
    return RedirectResponse(url="/setup", status_code=307)


@app.get("/setup", response_class=HTMLResponse)
async def setup_wizard(request: Request) -> Response:
    """Entry point for the setup wizard UI."""
    must_archive = _has_existing_session_state()
    roster_dir = DATA_DIR / "rosters"
    roster_dir.mkdir(parents=True, exist_ok=True)
    rubric_dir = RUBRIC_DIR
    rubric_dir.mkdir(parents=True, exist_ok=True)

    available_rosters = sorted(
        [p for p in roster_dir.glob("*.csv") if p.is_file()], key=lambda p: p.name.lower()
    )

    current_roster = DEFAULT_ROSTER_PATH if DEFAULT_ROSTER_PATH.exists() else None
    rubric_files = [*rubric_dir.glob("*.yaml"), *rubric_dir.glob("*.yml")]
    available_rubrics = sorted(
        [p for p in rubric_files if p.is_file()], key=lambda p: p.name.lower()
    )
    rubric_text, quiz_id = _read_default_rubric()

    message = request.query_params.get("message")
    status = request.query_params.get("status", "")

    return templates.TemplateResponse(
        "setup.html",
        {
            "request": request,
            "must_archive": must_archive,
            "available_rosters": available_rosters,
            "current_roster": current_roster,
            "available_rubrics": available_rubrics,
            "message": message,
            "status": status,
            "rubric_text": rubric_text,
            "quiz_id": quiz_id,
            "is_production": IS_PRODUCTION,
            "data_dir": str(DATA_DIR),
            "rosters_dir": str(DATA_DIR / "rosters"),
            "rubrics_dir": str(rubric_dir),
        },
    )


@app.post("/setup/archive_reset")
async def setup_archive_reset(label: str = Form("Current session")) -> Response:
    snapshot_id = archive_current_data(label)
    return RedirectResponse(
        url=f"/setup?status=success&message={quote(f'Archived as {snapshot_id}. Ready to start fresh.')}",
        status_code=303,
    )


def _validate_roster_csv(content: str) -> None:
    reader = csv.DictReader(io.StringIO(content))
    if not reader.fieldnames or "student_id" not in reader.fieldnames or "name" not in reader.fieldnames:
        raise HTTPException(
            status_code=400,
            detail="Roster CSV must include 'student_id' and 'name' columns.",
        )


def _normalize_headers(headers: List[str]) -> Dict[str, str]:
    normalized: Dict[str, str] = {}
    for h in headers:
        key = h.strip().lower()
        normalized[key] = h
    return normalized


def _extract_canvas_roster(content: str) -> str:
    reader = csv.DictReader(io.StringIO(content))
    if not reader.fieldnames:
        raise HTTPException(status_code=400, detail="Gradebook CSV has no headers.")

    header_map = _normalize_headers(reader.fieldnames)
    name_col = header_map.get("student")
    id_col = header_map.get("sis login id")
    if not name_col or not id_col:
        raise HTTPException(
            status_code=400,
            detail="Gradebook CSV must include 'Student' and 'SIS Login ID' columns.",
        )

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=["student_id", "name"])
    writer.writeheader()

    rows = 0
    for row in reader:
        name = (row.get(name_col) or "").strip()
        sid = (row.get(id_col) or "").strip()
        if not name or not sid:
            continue
        writer.writerow({"student_id": sid, "name": name})
        rows += 1

    if rows == 0:
        raise HTTPException(
            status_code=400,
            detail="No roster rows found in gradebook CSV.",
        )

    return output.getvalue()


@app.post("/setup/roster")
async def setup_roster(
    existing_roster: Optional[str] = Form(None),
    new_roster: Optional[UploadFile] = File(None),
) -> Response:
    roster_dir = DATA_DIR / "rosters"
    roster_dir.mkdir(parents=True, exist_ok=True)

    selected_path: Optional[Path] = None

    if new_roster and new_roster.filename:
        content_bytes = await new_roster.read()
        try:
            decoded = content_bytes.decode("utf-8-sig")
        except UnicodeDecodeError:
            raise HTTPException(status_code=400, detail="Roster must be UTF-8 encoded.")

        _validate_roster_csv(decoded)

        filename = Path(new_roster.filename).name
        selected_path = roster_dir / filename
        selected_path.write_text(decoded, encoding="utf-8")
    elif existing_roster:
        candidate = roster_dir / existing_roster
        if not candidate.exists():
            raise HTTPException(status_code=404, detail="Selected roster not found.")
        selected_path = candidate

    if selected_path is None:
        raise HTTPException(status_code=400, detail="Please select or upload a roster CSV.")

    DEFAULT_ROSTER_PATH.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(selected_path, DEFAULT_ROSTER_PATH)

    return RedirectResponse(
        url="/setup?status=success&message="
        + quote(f"Roster set from {selected_path.name}."),
        status_code=303,
    )


@app.post("/setup/roster_gradebook")
async def setup_roster_gradebook(
    gradebook_csv: UploadFile = File(...),
    roster_name: Optional[str] = Form(None),
) -> Response:
    if not (roster_name or "").strip():
        raise HTTPException(status_code=400, detail="Please provide a roster name.")
    if not gradebook_csv.filename:
        raise HTTPException(status_code=400, detail="Please upload a gradebook CSV.")

    content_bytes = await gradebook_csv.read()
    try:
        decoded = content_bytes.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="Gradebook CSV must be UTF-8 encoded.")

    cleaned_csv = _extract_canvas_roster(decoded)

    roster_dir = DATA_DIR / "rosters"
    roster_dir.mkdir(parents=True, exist_ok=True)
    desired = (roster_name or "").strip()
    if not desired.lower().endswith(".csv"):
        desired = f"{desired}.csv"
    target_path = roster_dir / Path(desired).name

    target_path.write_text(cleaned_csv, encoding="utf-8")
    DEFAULT_ROSTER_PATH.parent.mkdir(parents=True, exist_ok=True)
    if target_path != DEFAULT_ROSTER_PATH:
        shutil.copy(target_path, DEFAULT_ROSTER_PATH)

    return RedirectResponse(
        url="/setup?status=success&message="
        + quote("Roster created from Canvas gradebook."),
        status_code=303,
    )


@app.get("/gradebook_merge", response_class=HTMLResponse)
def gradebook_merge_page(request: Request) -> Response:
    session = get_session()
    return templates.TemplateResponse(
        "gradebook_merge.html",
        {
            "request": request,
            "quiz_id": session.grading_key.quiz_id,
            "analysis": None,
            "token": "",
            "error_message": "",
        },
    )


@app.post("/gradebook_merge/analyze", response_class=HTMLResponse)
async def gradebook_merge_analyze(
    request: Request,
    gradebook_csv: UploadFile = File(...),
) -> Response:
    if not gradebook_csv.filename:
        raise HTTPException(status_code=400, detail="Please upload a gradebook CSV.")

    content_bytes = await gradebook_csv.read()
    try:
        decoded = content_bytes.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="Gradebook CSV must be UTF-8 encoded.")

    fieldnames, rows, id_col = _parse_gradebook_csv(decoded)
    session = get_session()
    score_map = _build_total_scores_by_student(session)
    assignment_cols = _candidate_assignment_columns(fieldnames, id_col)
    stats = _assignment_column_stats(rows, assignment_cols, id_col, score_map)

    GRADEBOOK_MERGE_TMP_DIR.mkdir(parents=True, exist_ok=True)
    token = uuid4().hex
    tmp_path = GRADEBOOK_MERGE_TMP_DIR / f"{token}.csv"
    tmp_path.write_text(decoded, encoding="utf-8")

    return templates.TemplateResponse(
        "gradebook_merge.html",
        {
            "request": request,
            "quiz_id": session.grading_key.quiz_id,
            "analysis": {
                "id_col": id_col,
                "row_count": len(rows),
                "matched_students": len(score_map),
                "columns": stats,
            },
            "token": token,
            "error_message": "",
        },
    )


@app.post("/gradebook_merge/merge")
async def gradebook_merge_merge(
    request: Request,
    token: str = Form(...),
    assignment_column: str = Form(...),
    overwrite_non_empty: Optional[str] = Form(None),
) -> Response:
    tmp_path = GRADEBOOK_MERGE_TMP_DIR / f"{token}.csv"
    if not tmp_path.exists():
        raise HTTPException(
            status_code=400,
            detail="Merge token expired or missing. Re-upload gradebook CSV.",
        )

    decoded = tmp_path.read_text(encoding="utf-8")
    fieldnames, rows, id_col = _parse_gradebook_csv(decoded)
    if assignment_column not in fieldnames:
        raise HTTPException(status_code=400, detail="Selected assignment column not found.")

    session = get_session()
    score_map = _build_total_scores_by_student(session)
    overwrite = bool(overwrite_non_empty)

    conflicts: List[Dict[str, str]] = []
    matched_rows = 0
    changed_rows = 0
    skipped_rows = 0

    for row in rows:
        sid = (row.get(id_col) or "").strip()
        if sid not in score_map:
            continue
        matched_rows += 1
        old_val = (row.get(assignment_column) or "").strip()
        if old_val != "" and not overwrite:
            conflicts.append(
                {
                    "student_id": sid,
                    "student": (row.get("Student") or row.get("student") or ""),
                    "existing": old_val,
                    "new": f"{score_map[sid]:g}",
                }
            )
            skipped_rows += 1
            continue
        row[assignment_column] = f"{score_map[sid]:g}"
        changed_rows += 1

    assignment_cols = _candidate_assignment_columns(fieldnames, id_col)
    stats = _assignment_column_stats(rows, assignment_cols, id_col, score_map)

    if conflicts and not overwrite:
        return templates.TemplateResponse(
            "gradebook_merge.html",
            {
                "request": request,
                "quiz_id": session.grading_key.quiz_id,
                "analysis": {
                    "id_col": id_col,
                    "row_count": len(rows),
                    "matched_students": len(score_map),
                    "columns": stats,
                    "conflicts": conflicts[:30],
                },
                "token": token,
                "error_message": (
                    f"Selected column has {len(conflicts)} existing grade(s) among matched students. "
                    "Choose a blank column or enable overwrite."
                ),
            },
        )

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(row)

    print(
        f"[GRADEBOOK MERGE] col={assignment_column!r}, matched={matched_rows}, "
        f"changed={changed_rows}, skipped={skipped_rows}, overwrite={overwrite}"
    )

    filename = f"{session.grading_key.quiz_id}_gradebook_merged.csv"
    return Response(
        content=output.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{quote(filename)}"'},
    )


def _unique_pdf_stem(base_stem: str, output_dir: Path) -> str:
    stem = base_stem
    counter = 2
    while (output_dir / f"{stem}_p01.jpg").exists():
        stem = f"{base_stem}_{counter}"
        counter += 1
    return stem


def _scan_digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _existing_scan_digests(output_dir: Path) -> Dict[str, Path]:
    """Index exact scan content already present in an intake directory."""
    digests: Dict[str, Path] = {}
    if not output_dir.exists():
        return digests
    for path in sorted(output_dir.iterdir(), key=natural_sort_key):
        if path.is_file() and _is_image_filename(path.name):
            digests.setdefault(_scan_digest(path.read_bytes()), path)
    return digests


def _collapse_duplicate_scan_predictions(
    paths: List[Path],
    predictions: List[Dict[str, Any]],
) -> List[tuple[Path, Dict[str, Any]]]:
    """Show one row per exact scan while preserving the best cached metadata."""
    predictions_by_filename = {
        item.get("filename"): item
        for item in predictions
        if isinstance(item, dict) and item.get("filename")
    }
    groups: Dict[str, List[Path]] = {}
    for path in paths:
        groups.setdefault(_scan_digest(path.read_bytes()), []).append(path)

    collapsed: List[tuple[Path, Dict[str, Any]]] = []
    for group in groups.values():
        representative = group[0]
        candidates = [
            predictions_by_filename[path.name]
            for path in group
            if path.name in predictions_by_filename
        ]
        ranked = sorted(
            candidates,
            key=lambda item: (
                not bool(item.get("name_reviewed")),
                not bool(item.get("student_id")),
                -float(item.get("confidence", 0.0) or 0.0),
            ),
        )
        merged: Dict[str, Any] = dict(ranked[0]) if ranked else {}
        merged["filename"] = representative.name

        # Name and version may come from different copies: a canonical filename
        # can recover the reviewed name while the original scan gets the fresh
        # printed-version prediction.
        name_source = next((item for item in ranked if item.get("student_id")), None)
        if name_source:
            for field in ("student_id", "raw_name", "name_visible"):
                merged[field] = name_source.get(field)
        version_source = next(
            (item for item in ranked if normalize_quiz_version(item.get("quiz_version"))),
            None,
        )
        if version_source:
            merged["quiz_version"] = normalize_quiz_version(
                version_source.get("quiz_version")
            )
        merged["name_reviewed"] = any(
            bool(item.get("name_reviewed")) for item in candidates
        )
        collapsed.append((representative, merged))

    return collapsed


def _save_scan_bytes(
    data: bytes,
    destination: Path,
    known_digests: Dict[str, Path],
) -> tuple[Path, bool]:
    """Save novel content, preserving same-name/different-content scans."""
    digest = _scan_digest(data)
    existing = known_digests.get(digest)
    if existing is not None:
        return existing, False

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination = _unique_path(destination)
    destination.write_bytes(data)
    known_digests[digest] = destination
    return destination, True


def _save_image_upload(
    upload: UploadFile,
    destination: Path,
    known_digests: Dict[str, Path],
) -> bool:
    _, saved = _save_scan_bytes(upload.file.read(), destination, known_digests)
    return saved


def _quarantine_exact_duplicate_scans(output_dir: Path) -> List[Path]:
    """Move redundant scan files aside, keeping the first natural-sorted copy."""
    if not output_dir.exists():
        return []

    seen: Dict[str, Path] = {}
    duplicates: List[Path] = []
    for path in sorted(output_dir.iterdir(), key=natural_sort_key):
        if not path.is_file() or not _is_image_filename(path.name):
            continue
        digest = _scan_digest(path.read_bytes())
        if digest in seen:
            duplicates.append(path)
        else:
            seen[digest] = path

    if not duplicates:
        return []

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    quarantine_dir = DATA_DIR / "duplicate_scans_quarantine" / stamp
    quarantine_dir.mkdir(parents=True, exist_ok=True)
    moved: List[Path] = []
    for source in duplicates:
        destination = _unique_path(quarantine_dir / source.name)
        shutil.move(str(source), str(destination))
        moved.append(destination)
    return moved


def _save_pdf_pages(pdf_bytes: bytes, stem: str, output_dir: Path) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = _unique_pdf_stem(stem, output_dir)
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    count = 0
    for idx, page in enumerate(doc):
        matrix = fitz.Matrix(2, 2)
        pix = page.get_pixmap(matrix=matrix)
        filename = f"{stem}_p{idx + 1}.jpg"
        target = output_dir / filename
        target = _unique_path(target)
        pix.save(str(target))
        count += 1
    doc.close()
    return count


def _save_pdf_pages_with_paths(pdf_bytes: bytes, stem: str, output_dir: Path) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = _unique_pdf_stem(stem, output_dir)
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    targets: List[Path] = []
    for idx, page in enumerate(doc):
        matrix = fitz.Matrix(2, 2)
        pix = page.get_pixmap(matrix=matrix)
        filename = f"{stem}_p{idx + 1}.jpg"
        target = output_dir / filename
        target = _unique_path(target)
        pix.save(str(target))
        targets.append(target)
    doc.close()
    return targets


def _ingest_zip_bytes(
    zip_bytes: bytes,
    output_dir: Path,
    known_digests: Optional[Dict[str, Path]] = None,
) -> tuple[int, int]:
    saved = 0
    duplicates = 0
    known_digests = known_digests if known_digests is not None else _existing_scan_digests(output_dir)
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        entries = sorted(zf.infolist(), key=lambda info: info.filename.lower())
        for entry in entries:
            if entry.is_dir():
                continue
            entry_name = entry.filename
            if entry_name.startswith("__MACOSX/"):
                continue
            filename = Path(entry_name).name
            if not filename:
                continue
            suffix = Path(filename).suffix.lower()
            if suffix in {".jpg", ".jpeg", ".png"}:
                data = zf.read(entry)
                _, was_saved = _save_scan_bytes(data, output_dir / filename, known_digests)
                saved += int(was_saved)
                duplicates += int(not was_saved)
            elif suffix == ".pdf":
                pdf_bytes = zf.read(entry)
                if not pdf_bytes:
                    continue
                stem = Path(filename).stem or "scans"
                saved += _save_pdf_pages(pdf_bytes, stem, output_dir)
    return saved, duplicates


@app.post("/setup/upload_scans")
async def setup_upload_scans(
    images: Optional[List[UploadFile]] = File(None),
    pdf_file: Optional[UploadFile] = File(None),
    zip_file: Optional[UploadFile] = File(None),
) -> Response:
    if (not images or len(images) == 0) and not pdf_file and not zip_file:
        raise HTTPException(status_code=400, detail="Upload images, a PDF, or a ZIP of scans.")

    RAW_SCANS_DIR.mkdir(parents=True, exist_ok=True)
    saved = 0
    duplicates = 0
    known_digests = _existing_scan_digests(RAW_SCANS_DIR)

    if images:
        allowed_exts = {".jpg", ".jpeg", ".png"}
        for img in images:
            filename = Path(img.filename or "").name
            if not filename:
                continue
            suffix = Path(filename).suffix.lower()
            if suffix in allowed_exts:
                dest = RAW_SCANS_DIR / filename
                was_saved = _save_image_upload(img, dest, known_digests)
                saved += int(was_saved)
                duplicates += int(not was_saved)
                continue
            if suffix == ".pdf":
                pdf_bytes = await img.read()
                if not pdf_bytes:
                    continue
                stem = Path(filename).stem or "scans"
                saved += _save_pdf_pages(pdf_bytes, stem, RAW_SCANS_DIR)
                continue
            if suffix == ".zip":
                zip_bytes = await img.read()
                if not zip_bytes:
                    continue
                try:
                    new_saved, new_duplicates = _ingest_zip_bytes(
                        zip_bytes, RAW_SCANS_DIR, known_digests
                    )
                    saved += new_saved
                    duplicates += new_duplicates
                except zipfile.BadZipFile:
                    raise HTTPException(
                        status_code=400,
                        detail="Uploaded ZIP could not be read.",
                    )
                continue
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {filename}. Upload JPG/PNG, PDFs for splitting, or ZIP files.",
            )

    if pdf_file and pdf_file.filename:
        pdf_bytes = await pdf_file.read()
        if not pdf_bytes:
            raise HTTPException(status_code=400, detail="Uploaded PDF was empty.")

        stem = Path(pdf_file.filename or "scans").stem or "scans"
        try:
            saved += _save_pdf_pages(pdf_bytes, stem, RAW_SCANS_DIR)
        except fitz.EmptyFileError:
            raise HTTPException(
                status_code=400,
                detail="Uploaded PDF could not be read (file was empty or corrupted).",
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Failed to process PDF: {exc}")

    if zip_file and zip_file.filename:
        zip_bytes = await zip_file.read()
        if not zip_bytes:
            raise HTTPException(status_code=400, detail="Uploaded ZIP was empty.")

        try:
            new_saved, new_duplicates = _ingest_zip_bytes(
                zip_bytes, RAW_SCANS_DIR, known_digests
            )
            saved += new_saved
            duplicates += new_duplicates
        except zipfile.BadZipFile:
            raise HTTPException(status_code=400, detail="Uploaded ZIP could not be read.")

    message = f"Uploaded {saved} new scan files."
    if duplicates:
        message += f" Skipped {duplicates} byte-identical duplicate(s)."
    return RedirectResponse(
        url="/setup?status=success&message=" + quote(message),
        status_code=303,
    )


@app.post("/setup/import_canvas_zip")
async def setup_import_canvas_zip(
    zip_file: UploadFile = File(...),
    quiz_id: str = Form("quiz1"),
) -> Response:
    if _has_existing_session_state():
        raise HTTPException(
            status_code=400,
            detail="Existing session or name-review state detected. Archive & reset before importing.",
        )
    if RAW_SCANS_DIR.exists() and any(RAW_SCANS_DIR.iterdir()):
        raise HTTPException(
            status_code=400,
            detail="data/raw_scans is not empty. Archive & reset before importing.",
        )
    if not zip_file.filename:
        raise HTTPException(status_code=400, detail="Please upload a ZIP file.")
    original_zip_name = Path(zip_file.filename).name or "submissions.zip"

    zip_bytes = await zip_file.read()
    if not zip_bytes:
        raise HTTPException(status_code=400, detail="Uploaded ZIP was empty.")

    # A fresh import must not inherit stale per-problem segment caches.
    _clear_segment_caches()

    anonymous_roster_from_zip = False
    anonymous_by_prefix: Dict[str, RosterEntry] = {}
    try:
        roster = load_roster()
    except FileNotFoundError:
        roster = []
        anonymous_roster_from_zip = True

    try:
        grading_key = load_grading_key()
    except Exception:
        raise HTTPException(status_code=400, detail="Rubric not found. Save a rubric first.")

    roster_index: Dict[str, RosterEntry] = {}
    if not anonymous_roster_from_zip:
        if not roster:
            raise HTTPException(status_code=400, detail="Roster not found. Upload a roster first.")
        roster_index, collisions = _build_roster_prefix_index(roster)
        if collisions:
            detail_lines = ["Duplicate roster name keys detected:"]
            for key, entries in collisions.items():
                names = ", ".join(e.name for e in entries)
                detail_lines.append(f"{key}: {names}")
            raise HTTPException(status_code=400, detail="; ".join(detail_lines))

    staging_dir = DATA_DIR / "canvas_import" / uuid4().hex
    staging_dir.mkdir(parents=True, exist_ok=True)

    supported_exts = {".jpg", ".jpeg", ".png", ".pdf"}
    records: List[Dict[str, Any]] = []
    unmatched: List[str] = []
    unsupported: List[str] = []

    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            entries = sorted(zf.infolist(), key=lambda info: info.filename.lower())
            for entry in entries:
                if entry.is_dir():
                    continue
                entry_name = entry.filename
                if entry_name.startswith("__MACOSX/"):
                    continue
                filename = Path(entry_name).name
                if not filename:
                    continue
                suffix = Path(filename).suffix.lower()
                if suffix not in supported_exts:
                    unsupported.append(filename)
                    continue
                prefix_key = _filename_prefix_key(filename)
                if anonymous_roster_from_zip:
                    if not prefix_key:
                        unmatched.append(filename)
                        continue
                    roster_entry = anonymous_by_prefix.get(prefix_key)
                    if roster_entry is None:
                        anon_index = len(anonymous_by_prefix) + 1
                        roster_entry = RosterEntry(
                            student_id=f"anon_{anon_index:04d}",
                            name=f"Anonymous {anon_index:04d}",
                        )
                        anonymous_by_prefix[prefix_key] = roster_entry
                else:
                    roster_entry = roster_index.get(prefix_key)
                if roster_entry is None:
                    unmatched.append(filename)
                    continue
                records.append(
                    {
                        "entry": entry,
                        "filename": filename,
                        "suffix": suffix,
                        "roster_entry": roster_entry,
                    }
                )
    except zipfile.BadZipFile:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise HTTPException(status_code=400, detail="Uploaded ZIP could not be read.")

    if unsupported:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise HTTPException(
            status_code=400,
            detail="Unsupported file types in ZIP: " + ", ".join(unsupported[:10]),
        )
    if unmatched:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise HTTPException(
            status_code=400,
            detail="No roster match for files: " + ", ".join(unmatched[:10]),
        )
    if not records:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise HTTPException(status_code=400, detail="No valid scans found in ZIP.")
    if anonymous_roster_from_zip:
        roster = list(anonymous_by_prefix.values())

    per_student_paths: Dict[str, List[Path]] = {}
    per_student_originals: Dict[str, List[str]] = {}
    student_order: List[str] = []

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        for record in records:
            entry = record["entry"]
            filename = record["filename"]
            suffix = record["suffix"]
            roster_entry = record["roster_entry"]
            sid = roster_entry.student_id
            if sid not in per_student_paths:
                per_student_paths[sid] = []
                per_student_originals[sid] = []
                student_order.append(sid)
            per_student_originals[sid].append(filename)

            if suffix == ".pdf":
                pdf_bytes = zf.read(entry)
                if not pdf_bytes:
                    continue
                stem = Path(filename).stem or "scans"
                paths = _save_pdf_pages_with_paths(pdf_bytes, stem, staging_dir)
                per_student_paths[sid].extend(paths)
            else:
                dest = staging_dir / filename
                dest = _unique_path(dest)
                dest.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(entry) as src, dest.open("wb") as f:
                    shutil.copyfileobj(src, f)
                per_student_paths[sid].append(dest)

    RAW_SCANS_DIR.mkdir(parents=True, exist_ok=True)
    roster_by_id = {r.student_id: r.name for r in roster}
    work_units: List[WorkUnit] = []

    unit_counter = 1
    for sid in student_order:
        canvas_name = roster_by_id.get(sid, "")
        page_index = 0
        new_filenames: List[str] = []
        for src in per_student_paths.get(sid, []):
            page_index += 1
            new_name = canonical_filename(
                quiz_id=quiz_id,
                student_id=sid,
                canvas_name=canvas_name,
                page_index=page_index,
                unknown_index=None,
                orig_suffix=src.suffix,
            )
            dst = RAW_SCANS_DIR / new_name
            dst.write_bytes(src.read_bytes())
            new_filenames.append(new_name)

        per_problem = {p.id: None for p in grading_key.problems}
        per_problem_comments = {p.id: None for p in grading_key.problems}

        work_units.append(
            WorkUnit(
                unit_id=unit_counter,
                student_id=sid,
                page_filenames=new_filenames,
                per_problem=per_problem,
                comment=None,
                per_problem_comments=per_problem_comments,
            )
        )
        unit_counter += 1

    session = Session(
        roster=roster,
        grading_key=grading_key,
        work_units=work_units,
        unclaimed_pages=[],
        pinned_note=None,
    )

    CANVAS_REUPLOAD_ORIGINAL_ZIPS_DIR.mkdir(parents=True, exist_ok=True)
    source_zip_path = CANVAS_REUPLOAD_ORIGINAL_ZIPS_DIR / f"{quiz_id}__{original_zip_name}"
    source_zip_path.write_bytes(zip_bytes)
    _save_canvas_import_manifest(
        quiz_id=quiz_id,
        source_zip_filename=source_zip_path.name,
        original_zip_name=original_zip_name,
        work_units=work_units,
        per_student_originals=per_student_originals,
    )

    global _session
    _session = session
    save_session(session, DEFAULT_SESSION_PATH)

    shutil.rmtree(staging_dir, ignore_errors=True)

    first_problem = grading_key.problems[0].id if grading_key.problems else "Q1"
    return RedirectResponse(
        url=f"/problem/{first_problem}/strips",
        status_code=303,
    )


@app.post("/setup/save_rubric")
async def setup_save_rubric(yaml_text: str = Form(...)) -> Response:
    try:
        data = yaml.safe_load(yaml_text) or {}
        removed_answer_keys = _remove_answer_key_fields(data)
        grading_key = GradingKey(**data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid rubric YAML: {e}")

    RUBRIC_DIR.mkdir(parents=True, exist_ok=True)
    target = RUBRIC_DIR / f"{grading_key.quiz_id}.yaml"
    cleaned_yaml = yaml.safe_dump(
        grading_key.model_dump(exclude_none=True),
        sort_keys=False,
        allow_unicode=True,
    )
    target.write_text(cleaned_yaml, encoding="utf-8")

    message = f"Rubric saved to {target.name}."
    if removed_answer_keys:
        message += f" Removed {removed_answer_keys} answer-key field(s)."
    return RedirectResponse(
        url="/setup?status=success&message=" + quote(message),
        status_code=303,
    )


@app.post("/setup/select_rubric")
async def setup_select_rubric(rubric_file: str = Form(...)) -> Response:
    if not (rubric_file or "").strip():
        raise HTTPException(status_code=400, detail="Please select a rubric.")

    candidate = RUBRIC_DIR / Path(rubric_file).name
    if not candidate.exists():
        raise HTTPException(status_code=404, detail="Selected rubric not found.")

    try:
        data = yaml.safe_load(candidate.read_text(encoding="utf-8")) or {}
        removed_answer_keys = _remove_answer_key_fields(data)
        grading_key = GradingKey(**data)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid selected rubric YAML: {exc}")

    DEFAULT_GRADING_KEY_PATH.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_GRADING_KEY_PATH.write_text(
        yaml.safe_dump(
            grading_key.model_dump(exclude_none=True),
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )

    message = f"Rubric set from {candidate.name}."
    if removed_answer_keys:
        message += f" Removed {removed_answer_keys} answer-key field(s)."
    return RedirectResponse(
        url="/setup?status=success&message="
        + quote(message),
        status_code=303,
    )


@app.post("/setup/run_auto_grouping")
async def setup_run_auto_grouping(quiz_id: str = Form("quiz1")) -> Response:
    roster = load_roster()
    if not RAW_SCANS_DIR.exists() or not any(RAW_SCANS_DIR.iterdir()):
        raise HTTPException(status_code=400, detail="Upload scans before running auto-grouping.")

    quarantined = _quarantine_exact_duplicate_scans(RAW_SCANS_DIR)
    if quarantined:
        print(
            f"[scan intake] Quarantined {len(quarantined)} exact duplicate scan(s) "
            f"under {quarantined[0].parent}"
        )
    predictions = run_name_recognition(RAW_SCANS_DIR, roster)
    cache_path = name_review_cache_path(quiz_id, "")
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    serializable = []
    for pred in predictions:
        filename = getattr(pred, "filename", None) or getattr(pred, "path", None)
        if isinstance(filename, Path):
            filename = filename.name
        serializable.append(
            {
                "filename": filename,
                "student_id": getattr(pred, "student_id", None),
                "raw_name": getattr(pred, "raw_name", None),
                "confidence": float(getattr(pred, "confidence", 0.0)),
                "reasoning": getattr(pred, "reasoning", ""),
                "name_visible": bool(getattr(pred, "name_visible", False)),
                "quiz_version": normalize_quiz_version(
                    getattr(pred, "quiz_version", None)
                ),
            }
        )

    import json as _json

    cache_path.write_text(_json.dumps(serializable, indent=2), encoding="utf-8")

    return RedirectResponse(
        url=f"/name_review?quiz={quote(quiz_id)}",
        status_code=303,
    )


@app.get("/session/summary")
def session_summary():
    """Lightweight summary for debugging."""
    session = get_session()
    graded = 0
    fully_scored = 0

    for wu in session.work_units:
        if wu.student_id:
            graded += 1
        # "Fully scored" if every problem id in the grading key has a non-None score
        if all(
            wu.per_problem.get(p.id) is not None
            for p in session.grading_key.problems
        ):
            fully_scored += 1

    return {
        "roster_size": len(session.roster),
        "num_work_units": len(session.work_units),
        "num_unclaimed_pages": len(session.unclaimed_pages),
        "num_with_student_id": graded,
        "num_fully_scored": fully_scored,
    }


def find_work_unit(session: Session, unit_id: int) -> WorkUnit:
    for wu in session.work_units:
        if wu.unit_id == unit_id:
            return wu
    raise HTTPException(status_code=404, detail=f"WorkUnit {unit_id} not found.")


@app.get("/work_units", response_model=List[WorkUnitSummary])
def list_work_units():
    """List work units in a lightweight form (helpful for debugging / navigation)."""
    session = get_session()
    summaries: List[WorkUnitSummary] = []
    for wu in session.work_units:
        fully = all(
            wu.per_problem.get(p.id) is not None
            for p in session.grading_key.problems
        )
        summaries.append(
            WorkUnitSummary(
                unit_id=wu.unit_id,
                student_id=wu.student_id,
                page_count=len(wu.page_filenames),
                fully_scored=fully,
            )
        )
    return summaries


@app.post("/work_units/create", response_model=WorkUnit)
def create_work_unit():
    """
    Create a new WorkUnit by assigning the next unclaimed page.

    - Pops the first filename from unclaimed_pages.
    - Creates a WorkUnit with that page and empty scores.
    """
    with _session_lock:
        session = get_session()
        if not session.unclaimed_pages:
            raise HTTPException(status_code=400, detail="No unclaimed pages remaining.")

        filename = session.unclaimed_pages.pop(0)
        unit_id = next_unit_id(session)

        # initialize per_problem with None for each problem id
        per_problem: Dict[str, Optional[float]] = {
            p.id: None for p in session.grading_key.problems
        }
        per_problem_comments: Dict[str, Optional[str]] = {
            p.id: None for p in session.grading_key.problems
        }

        wu = WorkUnit(
            unit_id=unit_id,
            page_filenames=[filename],
            student_id=None,
            per_problem=per_problem,
            comment=None,
            per_problem_comments=per_problem_comments,
        )
        session.work_units.append(wu)
        persist_session()
        return wu


@app.get("/_legacy/name_review", response_class=HTMLResponse, include_in_schema=False)
async def legacy_name_review_page(
    request: Request,
    raw_scans_subdir: str = "",
    quiz: str = "quiz1",
    refresh: bool = False,
):
    """
    Display one row per scanned page.
    Priority: 1. Refresh -> 2. CLI CSV -> 3. JSON Cache -> 4. Run VLM
    """
    # 1. Resolve scans directory (Standard checks)
    base_dir = RAW_SCANS_DIR / raw_scans_subdir if raw_scans_subdir else RAW_SCANS_DIR
    if not base_dir.exists():
        raise HTTPException(404, detail=f"Not found: {base_dir}")

    exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    paths = sorted([p for p in base_dir.iterdir() if p.is_file() and p.suffix in exts], key=natural_sort_key)
    
    # 2. Load Roster
    roster = load_roster()
    roster_by_id = {r.student_id: r.name for r in roster}

    # 3. Load Predictions
    cache_path = name_review_cache_path(quiz, raw_scans_subdir)
    csv_path = DATA_DIR / "auto_grouped" / quiz / f"{quiz}_page_predictions.csv"
    
    predictions = []
    
    # --- LOGIC BRANCHES ---
    
    # Branch A: Load from CLI CSV (Fastest & Most Accurate to CLI)
    if (not refresh) and csv_path.exists():
        print(f"[name_review] Loading from CLI CSV: {csv_path}")
        import csv
        with csv_path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                predictions.append({
                    "filename": row["filename"],
                    "student_id": row["predicted_student_id"] or None,
                    "raw_name": row.get("raw_name"),
                    "confidence": float(row.get("confidence", 0.0)),
                    "reasoning": row.get("reasoning", ""),
                    "name_visible": row.get("name_visible") == "True",
                })
        
        # CRITICAL FIX: Save to JSON cache so POST /name_review can find it later
        import json as _json
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(_json.dumps(predictions, indent=2), encoding="utf-8")

    # Branch B: Load from JSON Cache (Fast)
    elif (not refresh) and cache_path.exists():
        print(f"[name_review] Loading from JSON cache: {cache_path}")
        import json as _json
        predictions = _json.loads(cache_path.read_text(encoding="utf-8"))

    # Branch C: Run VLM (Slow Fallback)
    else:
        print("[name_review] No cache found. Running VLM...")
        client = LMStudioQwenClient(base_url=LMSTUDIO_BASE_URL, model=LMSTUDIO_MODEL_ID)
        for path in paths:
            pred = classify_page(path, roster, client)
            predictions.append({
                "filename": path.name,
                "student_id": pred.student_id,
                "raw_name": pred.raw_name,
                "confidence": pred.confidence,
                "reasoning": pred.reasoning,
                "name_visible": pred.name_visible,
            })
        
        # Save results for next time
        import json as _json
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(_json.dumps(predictions, indent=2), encoding="utf-8")

    # 4. Build View Data
    preds_map = {p["filename"]: p for p in predictions}
    page_rows = []
    
    for idx, path in enumerate(paths):
        info = preds_map.get(path.name, {})
        sid = info.get("student_id")
        
        page_rows.append({
            "index": idx,
            "filename": path.name,
            "image_url": f"/scans/{raw_scans_subdir}/{path.name}" if raw_scans_subdir else f"/scans/{path.name}",
            "model_student_id": sid,
            "model_canvas_name": roster_by_id.get(sid) if sid else None,
            "confidence": float(info.get("confidence", 0.0)),
            "raw_name": info.get("raw_name", ""),
            "reasoning": info.get("reasoning", ""),
            "name_visible": bool(info.get("name_visible", False)),
        })

    return templates.TemplateResponse("name_review.html", {
        "request": request,
        "page_rows": page_rows,
        "roster": roster,
        "quiz_slug": quiz,
        "raw_scans_subdir": raw_scans_subdir,
    })


@app.post("/grade/{unit_id}", response_model=WorkUnit)
def update_grade(unit_id: int, update: GradeUpdate):
    """
    Update grading info for a WorkUnit via JSON:

    - student_id (must exist in roster if provided)
    - per_problem scores
    - comment
    """
    with _session_lock:
        session = get_session()
        wu = find_work_unit(session, unit_id)

        if update.student_id is not None:
            # sanity-check that student_id exists in roster
            if not any(r.student_id == update.student_id for r in session.roster):
                raise HTTPException(
                    status_code=400,
                    detail=f"student_id {update.student_id} not found in roster.",
                )
            wu.student_id = update.student_id

        if update.per_problem is not None:
            valid_ids = {p.id for p in session.grading_key.problems}
            for pid, score in update.per_problem.items():
                if pid not in valid_ids:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Problem id {pid} is not in grading key.",
                    )
                wu.per_problem[pid] = score

        if update.per_problem_comments is not None:
            valid_ids = {p.id for p in session.grading_key.problems}
            for pid, text in update.per_problem_comments.items():
                if pid not in valid_ids:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Problem id {pid} is not in grading key.",
                    )
                wu.per_problem_comments[pid] = text

        if update.comment is not None:
            wu.comment = update.comment

        persist_session()
        return wu


@app.post("/work_units/{unit_id}/append_next_scan", response_model=WorkUnit)
def append_next_scan(unit_id: int):
    """
    Append the next unclaimed page to this WorkUnit.

    Pops the first filename from unclaimed_pages and appends to page_filenames.
    """
    with _session_lock:
        session = get_session()
        if not session.unclaimed_pages:
            raise HTTPException(status_code=400, detail="No unclaimed pages remaining.")

        wu = find_work_unit(session, unit_id)
        filename = session.unclaimed_pages.pop(0)
        wu.page_filenames.append(filename)
        persist_session()
        return wu


@app.post("/work_units/{unit_id}/remove_last_scan", response_model=WorkUnit)
def remove_last_scan(unit_id: int):
    """
    Remove the last page from this WorkUnit and return it to the front of unclaimed_pages.
    """
    with _session_lock:
        session = get_session()
        wu = find_work_unit(session, unit_id)

        if not wu.page_filenames:
            raise HTTPException(
                status_code=400,
                detail="WorkUnit has no pages to remove.",
            )

        last = wu.page_filenames.pop()
        # Put the removed page back at the front so it will be assigned next time.
        session.unclaimed_pages.insert(0, last)
        persist_session()
        return wu

import re

@app.post("/grade/{unit_id}/page", response_class=HTMLResponse)
async def grade_page_submit(
    request: Request,
    unit_id: int,
    student_id: Optional[str] = Form(None),
    comment: Optional[str] = Form(None),
    # per-problem scores + action will be read from the raw form data
):
    """
    Handle form submission from the HTML grading page.

    - Updates student_id
    - Updates per-problem scores
    - Updates comment
    - Optionally appends/removes scans
    - Optionally jumps to the next work unit (creating one if needed)

    Then redirects to the appropriate page.
    """
    form = await request.form()
    action = form.get("action", "save")

    with _session_lock:
        session = get_session()

        # --- NEW: handle pinned-note updates separately ---
        if action == "update_pinned_note":
            # Read text from the pinned-note form
            new_note = (form.get("pinned_note") or "").strip()
            # Store None for empty note so older sessions behave nicely
            session.pinned_note = new_note or None

            # Stay on the same work unit after saving the note
            target_unit_id = unit_id

        else:
            # --- EXISTING grading logic moved under this else ---

            wu = find_work_unit(session, unit_id)

            # student_id (optional)
            if student_id:
                if not any(r.student_id == student_id for r in session.roster):
                    raise HTTPException(
                        status_code=400,
                        detail=f"student_id {student_id} not found in roster.",
                    )
                wu.student_id = student_id

            # per-problem scores: expect fields like score_Q1, score_Q2, ...
            for p in session.grading_key.problems:
                field_name = f"score_{p.id}"
                raw = form.get(field_name)
                if raw in (None, "", " "):
                    wu.per_problem[p.id] = None
                else:
                    try:
                        wu.per_problem[p.id] = float(raw)
                    except ValueError:
                        raise HTTPException(
                            status_code=400,
                            detail=f"Invalid score for {p.id}: {raw!r}",
                        )
            # per-problem comments: fields like comment_Q1, comment_Q2, ...
            # (keep keys in sync with grading_key)
            for p in session.grading_key.problems:
                field_name = f"comment_{p.id}"
                raw = form.get(field_name)

                # Normalize empty to None so JSON stays clean
                if raw is None or raw.strip() == "":
                    wu.per_problem_comments[p.id] = None
                else:
                    wu.per_problem_comments[p.id] = raw

            # comment (optional)
            if comment is not None:
                wu.comment = comment or ""

            # Handle scan-append/remove actions. If we can't do the action
            # (e.g. no unclaimed pages to append), we silently no-op and
            # just treat it as a normal save so the UI stays simple.
            if action == "append_scan":
                # Build the global sorted list of all scan files on disk
                all_files = sorted(
                    [p.name for p in RAW_SCANS_DIR.iterdir() if p.is_file()]
                )

                if not wu.page_filenames:
                # If this unit somehow has no pages yet, fall back to old behavior:
                    if session.unclaimed_pages:
                        next_filename = session.unclaimed_pages.pop(0)
                        wu.page_filenames.append(next_filename)
                else:
                    last_page = wu.page_filenames[-1]
                    try:
                        idx = all_files.index(last_page)
                    except ValueError:
                        idx = -1

                    if idx >= 0 and idx + 1 < len(all_files):
                        candidate = all_files[idx + 1]

                        # Detach candidate from wherever it currently lives
                        if candidate in session.unclaimed_pages:
                            session.unclaimed_pages.remove(candidate)
                        else:
                            for other in session.work_units:
                                if candidate in other.page_filenames:
                                    other.page_filenames.remove(candidate)
                                    break

                    wu.page_filenames.append(candidate)
                # else: no later scan exists; do nothing

            elif action == "remove_scan":
                if wu.page_filenames:
                    last_filename = wu.page_filenames.pop()
                    # Put it back at the front of the unclaimed queue
                    session.unclaimed_pages.insert(0, last_filename)

            # Determine redirect target for save_prev / save_next
            target_unit_id = unit_id

            # Stable ordering of unit IDs for navigation
            unit_ids = sorted(w.unit_id for w in session.work_units)
            try:
                    idx = unit_ids.index(unit_id)
            except ValueError:
                idx = -1

            if action == "save_prev":
            # Go to previous unit if there is one; otherwise stay put
                if idx > 0:
                    target_unit_id = unit_ids[idx - 1]

            elif action == "save_next":
                if idx >= 0 and idx + 1 < len(unit_ids):
            # There is an existing later unit
                    target_unit_id = unit_ids[idx + 1]
                else:
            # We're at the end; create a new unit if we have unclaimed pages
                    if session.unclaimed_pages:
                        next_filename = session.unclaimed_pages.pop(0)
                        new_unit_id = (max(unit_ids) + 1) if unit_ids else 1
                        per_problem = {p.id: None for p in session.grading_key.problems}
                        new_wu = WorkUnit(
                            unit_id=new_unit_id,
                            page_filenames=[next_filename],
                            student_id=None,
                            per_problem=per_problem,
                            comment=None,
                        )
                        session.work_units.append(new_wu)
                        target_unit_id = new_unit_id
                    else:
                    # No more scans; stay on the current unit
                        target_unit_id = unit_id

        # Save session after either kind of update
        persist_session()

    # Redirect to GET so refresh doesn't re-POST
    return RedirectResponse(
        url=f"/grade/{target_unit_id}/page",
        status_code=303,
    )


@app.get("/name_review", response_class=HTMLResponse)
async def name_review_page(
    request: Request,
    raw_scans_subdir: str = "",
    quiz: str = "quiz1",
    refresh: bool = False,  # 👈 new flag
    saved: bool = False,
):
    """
    Display one row per scanned page.
    Uses cached LM predictions by default; only re-runs the model if
    refresh=true or the cache is missing.
    """
    # 1. Resolve scans directory and files
    base_dir = RAW_SCANS_DIR / raw_scans_subdir if raw_scans_subdir else RAW_SCANS_DIR
    if not base_dir.exists() or not base_dir.is_dir():
        raise HTTPException(
            status_code=404,
            detail=f"Raw scans directory not found: {base_dir}",
        )

    exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    paths = [p for p in base_dir.iterdir() if p.is_file() and p.suffix in exts]
    if not paths:
        raise HTTPException(
            status_code=400,
            detail=f"No image files found in {base_dir}",
        )
    paths.sort(key=natural_sort_key)

    # 2. Load roster
    roster = load_roster()
    roster_by_id: Dict[str, str] = {r.student_id: r.name for r in roster}

    # 3. Either load cached predictions or run LM Studio and cache them
    cache_path = name_review_cache_path(quiz, raw_scans_subdir)
    predictions: List[Dict]
    cached_predictions: List[Dict] = []
    if cache_path.exists():
        try:
            import json as _json
            cached_predictions = _json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            cached_predictions = []

    if (not refresh) and cache_path.exists():
        # ✅ Use cached predictions
        predictions = cached_predictions
    else:
        # 🔁 Run LM Studio and cache predictions
        print(f"[name_review] Cache miss. inspecting {len(paths)} files...")

        client = LMStudioQwenClient(
            base_url=LMSTUDIO_BASE_URL,
            model=LMSTUDIO_MODEL_ID,
        )
        predictions = []
        cached_by_filename = {
            item.get("filename"): item
            for item in cached_predictions
            if item.get("filename")
        }
        for path in paths:
            # --- START NEW LOGIC ---
            # If the file is ALREADY renamed (e.g. "quiz1_ballard_sharon_p01.jpg"), 
            # don't waste time running the VLM. Parse the name from the file.
            
            # Simple heuristic: Does it end in "_pXX.jpg"?
            import re
            match = re.search(r"(.+)_p\d+\.(jpg|png|jpeg)$", path.name, re.IGNORECASE)
            
            if match and "unknown" not in path.name.lower():
                # Extract the likely name part: "quiz1_ballard_sharon"
                # We try to match this against the roster to get the ID.
                # This is a 'best effort' to restore state without VLM.
                print(f"Skipping VLM for pre-labeled file: {path.name}")
                
                # Mock a prediction object
                found_student_id = None
                found_name = None
                version_match = re.search(r"(?:^|_)v(?:ersion)?[_-]?([A-Za-z0-9]+)(?:_|\.)", path.name, re.I)
                quiz_version = normalize_quiz_version(version_match.group(1)) if version_match else None
                
                # Try to find a roster match based on the filename parts
                for r in roster:
                    # simplistic check: if student's last name is in the filename
                    slug_last = slugify_name(r.name.split(",")[0])
                    if slug_last in path.name.lower():
                        found_student_id = r.student_id
                        found_name = r.name
                        break
                
                predictions.append({
                    "filename": path.name,
                    "student_id": found_student_id,
                    "raw_name": found_name or "(Recovered from filename)",
                    "confidence": 1.0,
                    "reasoning": "Filename indicates previous labeling.",
                    "name_visible": True,
                    "quiz_version": quiz_version,
                })
                continue
            # --- END NEW LOGIC ---

            # Standard VLM run for truly new/raw files
            pred = classify_page(path, roster, client)
            if getattr(pred, "filename", None) is None:
                setattr(pred, "filename", path.name)

            student_id = getattr(pred, "student_id", None)
            raw_name = getattr(pred, "raw_name", None)
            confidence = float(getattr(pred, "confidence", 0.0))
            reasoning = getattr(pred, "reasoning", "")
            name_visible = bool(getattr(pred, "name_visible", False))
            quiz_version = normalize_quiz_version(getattr(pred, "quiz_version", None))
            previous = cached_by_filename.get(path.name) or {}
            if previous.get("name_reviewed"):
                student_id = previous.get("student_id")

            predictions.append(
                {
                    "filename": path.name,
                    "student_id": student_id,
                    "raw_name": raw_name,
                    "confidence": confidence,
                    "reasoning": reasoning,
                    "name_visible": name_visible,
                    "quiz_version": quiz_version,
                    "name_reviewed": bool(previous.get("name_reviewed")),
                    "starts_new_submission": bool(
                        previous.get("starts_new_submission", False)
                    ),
                    "excluded": bool(previous.get("excluded", False)),
                }
            )

        import json as _json
        cache_path.write_text(
            _json.dumps(predictions, indent=2),
            encoding="utf-8",
        )

    # 4. Align predictions with the sorted file list and build page_rows
    page_rows: List[Dict] = []
    collapsed = _collapse_duplicate_scan_predictions(paths, predictions)
    if len(collapsed) != len(paths):
        print(
            f"[name_review] Collapsed {len(paths) - len(collapsed)} exact duplicate "
            "scan copy/copies in the review view."
        )
    for idx, (path, info) in enumerate(collapsed):
        student_id = info.get("student_id")
        canvas_name = roster_by_id.get(student_id) if student_id else None

        if raw_scans_subdir:
            image_url = f"/scans/{raw_scans_subdir}/{path.name}"
        else:
            image_url = f"/scans/{path.name}"

        page_rows.append(
            {
                "index": idx,
                "filename": path.name,
                "image_url": image_url,
                "model_student_id": student_id,
                "model_canvas_name": canvas_name or "",
                "confidence": float(info.get("confidence", 0.0)),
                "raw_name": info.get("raw_name") or "",
                "reasoning": info.get("reasoning") or "",
                "name_visible": bool(info.get("name_visible", False)),
                "quiz_version": normalize_quiz_version(info.get("quiz_version")),
                "starts_new_submission": bool(info.get("starts_new_submission", False)),
                "excluded": bool(info.get("excluded", False)),
            }
        )

    # 5. Render template as before
    return templates.TemplateResponse(
        "name_review.html",
        {
            "request": request,
            "page_rows": page_rows,
            "roster": roster,
            "quiz_slug": quiz,
            "raw_scans_subdir": raw_scans_subdir,
            "saved": saved,
        },
    )


@app.post("/name_review")
async def name_review_submit(request: Request) -> Response:
    """
    1. Reads user approvals from the form.
    2. Groups pages into WorkUnits.
    3. Renames scans to 'quiz1_student_p01.jpg' and saves them in raw_scans.
    4. Initializes session.json with these new WorkUnits.
    5. Redirects to the Strip View grader.
    """
    print("[name_review] POST: starting Hand-off...")
    form = await request.form()

    # NEW: Check which button started this request
    next_action = form.get("next_action", "grade")

    quiz_id = (form.get("quiz_id") or "quiz").strip()
    raw_scans_subdir = form.get("raw_scans_subdir") or ""
    page_count = int(form.get("page_count") or 0)

    # 1. Validation
    if page_count <= 0:
        raise HTTPException(status_code=400, detail="No pages submitted.")

    base_dir = RAW_SCANS_DIR / raw_scans_subdir if raw_scans_subdir else RAW_SCANS_DIR
    if not base_dir.exists():
        raise HTTPException(404, detail=f"Scan dir not found: {base_dir}")

    # 2. Parse Form Data (Assignments)
    # List of (filename, student, version, explicit group boundary, excluded)
    assignments: List[tuple[str, Optional[str], Optional[str], bool, bool]] = []
    for i in range(page_count):
        fn = form.get(f"filename_{i}")
        sid = form.get(f"student_{i}") or None
        quiz_version = normalize_quiz_version(form.get(f"quiz_version_{i}"))
        starts_new_submission = bool(form.get(f"starts_new_submission_{i}"))
        excluded = bool(form.get(f"exclude_{i}"))
        if fn:
            assignments.append(
                (fn, sid, quiz_version, starts_new_submission, excluded)
            )

    # Draft-save paths update the cache without renaming scans or creating a
    # grading session. Auto-fill is opt-in; ordinary Save preserves Unknowns.
    if next_action in {"save", "save_fill"}:
        cache_path = name_review_cache_path(quiz_id, raw_scans_subdir)
        import json as _json

        if cache_path.exists():
            try:
                predictions = _json.loads(cache_path.read_text(encoding="utf-8"))
            except Exception:
                predictions = []
        else:
            predictions = []

        preds_by_filename: Dict[str, Dict] = {p.get("filename"): p for p in predictions if p.get("filename")}
        updated_predictions: List[Dict] = []
        seen: set[str] = set()

        last_sid: Optional[str] = None
        for filename, sid, quiz_version, starts_new_submission, excluded in assignments:
            applied_sid = sid
            if next_action == "save_fill" and not excluded:
                applied_sid = sid or last_sid
            if applied_sid and not excluded:
                last_sid = applied_sid

            entry = dict(preds_by_filename.get(filename, {"filename": filename}))
            entry["student_id"] = applied_sid
            entry["quiz_version"] = quiz_version
            entry["name_reviewed"] = True
            entry["starts_new_submission"] = starts_new_submission
            entry["excluded"] = excluded
            preds_by_filename[filename] = entry
            seen.add(filename)
            updated_predictions.append(entry)

        # Preserve any extra prediction entries that were not part of this form
        for pred in predictions:
            fname = pred.get("filename")
            if fname and fname not in seen:
                updated_predictions.append(pred)

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(_json.dumps(updated_predictions, indent=2), encoding="utf-8")
        mode = " and auto-filled unknown names" if next_action == "save_fill" else ""
        print(f"[name_review] Saved {len(updated_predictions)} corrected cache entries{mode}.")

        redirect_url = f"/name_review?quiz={quote(quiz_id)}&saved=1"
        if raw_scans_subdir:
            redirect_url += f"&raw_scans_subdir={quote(raw_scans_subdir)}"

        return RedirectResponse(url=redirect_url, status_code=303)

    # 3. Load Roster & Grading Key
    roster = load_roster()
    roster_by_id = {r.student_id: r.name for r in roster}
    try:
        grading_key = load_grading_key()
    except Exception:
        # Fallback if no key exists yet
        grading_key = None

    # 4. Group Pages (Simple Logic)
    # We group consecutive pages that share the same student_id
    groups = []
    current_group = None

    for filename, sid, quiz_version, starts_new_submission, excluded in assignments:
        if excluded:
            if current_group:
                groups.append(current_group)
                current_group = None
            continue
        if current_group is None:
            current_group = {
                "student_id": sid,
                "pages": [filename],
                "quiz_version": quiz_version,
            }
            continue

        # If same student (and not None), append to group
        # Note: We break groups on 'None' (unknowns) so they don't clump together?
        # Actually, let's clump unknowns if they are consecutive? 
        # For safety, let's break if student_id changes.
        versions_conflict = bool(
            quiz_version
            and current_group.get("quiz_version")
            and quiz_version != current_group["quiz_version"]
        )
        if (
            sid == current_group["student_id"]
            and not starts_new_submission
            and not versions_conflict
        ):
            current_group["pages"].append(filename)
            if not current_group.get("quiz_version") and quiz_version:
                current_group["quiz_version"] = quiz_version
        else:
            groups.append(current_group)
            current_group = {
                "student_id": sid,
                "pages": [filename],
                "quiz_version": quiz_version,
            }
    if current_group:
        groups.append(current_group)

    if not groups:
        raise HTTPException(
            status_code=400,
            detail="Every page is excluded. Keep at least one submission before continuing.",
        )

    # 5. Build WorkUnits and Rename Files
    work_units: List[WorkUnit] = []
    
    # We need a path map to find the source files
    # (Assuming filenames in form are relative to base_dir)
    
    unit_counter = 1
    
    # Helper to track page numbers per student for naming
    student_page_counts = {} 
    
    print(f"[name_review] Processing {len(groups)} groups...")

    for g in groups:
        sid = g["student_id"]
        quiz_version = normalize_quiz_version(g.get("quiz_version"))
        
        # Determine canonical name base
        if sid:
            cname = roster_by_id.get(sid, "student")
            student_page_counts.setdefault(sid, 0)
        else:
            cname = "unknown"
            # We track unknowns globally or per group? Let's just use unit_id
        
        new_filenames = []
        
        for raw_name in g["pages"]:
            src = base_dir / raw_name
            if not src.exists():
                print(f"[WARN] Source file missing: {src}")
                continue
            
            # Generate new name
            # Pattern: quiz1_student_p01.jpg
            if sid:
                student_page_counts[sid] += 1
                p_num = student_page_counts[sid]
                # Use the canonical_filename helper from server.py
                new_name = canonical_filename(
                    quiz_id=quiz_id, 
                    student_id=sid, 
                    canvas_name=cname, 
                    page_index=p_num, 
                    unknown_index=None, 
                    orig_suffix=src.suffix
                )
            else:
                # For unknowns, use unit_id as the unique index
                new_name = canonical_filename(
                    quiz_id=quiz_id, 
                    student_id=None, 
                    canvas_name=None, 
                    page_index=len(new_filenames)+1, 
                    unknown_index=unit_counter, 
                    orig_suffix=src.suffix
                )

            # COPY to RAW_SCANS_DIR (root) so the server can serve it at /scans/new_name
            # We copy instead of rename to be safe? 
            # Let's write bytes to ensure it works across file systems.
            dst = RAW_SCANS_DIR / new_name
            dst.write_bytes(src.read_bytes())
            
            new_filenames.append(new_name)

        # Create WorkUnit
        # Initialize empty scores
        per_problem = {}
        per_problem_comments = {}
        if grading_key:
            for p in grading_key.problems:
                per_problem[p.id] = None
                per_problem_comments[p.id] = None

        wu = WorkUnit(
            unit_id=unit_counter,
            student_id=sid,
            quiz_version=quiz_version,
            page_filenames=new_filenames,
            per_problem=per_problem,
            comment=None,
            per_problem_comments=per_problem_comments
        )
        work_units.append(wu)
        unit_counter += 1

    # 6. Initialize Session
    # Use existing grading key if loaded, else we need one.
    if not grading_key:
        # Fallback: try loading default or error out
        # For now, let's assume one exists or creating session will fail
        grading_key = load_grading_key() 

    session = Session(
        roster=roster,
        grading_key=grading_key,
        work_units=work_units,
        unclaimed_pages=[], # We consumed them all?
        pinned_note=None,
        problem_types={}
    )

    # 7. Save Session to Disk
    # This creates data/grades/session.json
    global _session
    _session = session
    save_session(session, DEFAULT_SESSION_PATH)
    print(f"[name_review] Session initialized with {len(work_units)} units.")

    # 8. Check Next Action
    if next_action == "download":
        # The user clicked "Save & Download"
        print("[name_review] User requested immediate ZIP download.")
        return export_renamed_scans()
    else:
        # The user clicked "Save & Start Grading" (Default)
        first_problem = grading_key.problems[0].id if grading_key.problems else "Q1"
        return RedirectResponse(
            url=f"/problem/{first_problem}/strips",
            status_code=303
        )

