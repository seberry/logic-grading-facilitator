from __future__ import annotations

from typing import Any, Dict, List, Optional
import io
import csv
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
    AUTO_GROUPED_DIR,
    IS_PRODUCTION,
    FLASHCARDS_DIR,
)
from .segmentation_core import segment_page_for_problem, QuestionSegment

from .models import (
    Session,
    WorkUnit,
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
from .canvas_client import CanvasClient, CanvasClientError, get_canvas_client
from .routes.fitch_debug import router as fitch_debug_router
from .routes.flashcards import router as flashcards_router
from .routes.problem_strips import router as problem_strips_router, _clear_segment_caches
from .helpers import _unique_path, _is_image_filename, _select_display_filename, _build_page_options
from .session_state import _session_lock, get_session, persist_session

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

app.mount(
    "/flashcards_assets",
    StaticFiles(directory=str(FLASHCARDS_DIR)),
    name="flashcards_assets",
)


LMSTUDIO_BASE_URL = "http://127.0.0.1:1234/v1"
LMSTUDIO_MODEL_ID = "qwen/qwen3-vl-4b"  # or whatever you’re actually using
CANVAS_REUPLOAD_ORIGINAL_ZIPS_DIR = DATA_DIR / "canvas_reupload" / "original_zips"
CANVAS_REUPLOAD_MANIFEST_DIR = DATA_DIR / "canvas_reupload" / "manifests"
GRADEBOOK_MERGE_TMP_DIR = DATA_DIR / "gradebook_merge" / "tmp"
CLASSWORK_FILES_DIR = DATA_DIR / "classwork" / "files"
CLASSWORK_EXPORTS_DIR = DATA_DIR / "classwork" / "exports"
CLASSWORK_FILES_DIR.mkdir(parents=True, exist_ok=True)
CLASSWORK_EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
app.mount(
    "/classwork_files",
    StaticFiles(directory=str(CLASSWORK_FILES_DIR)),
    name="classwork_files",
)
app.include_router(fitch_debug_router)
app.include_router(flashcards_router)
app.include_router(problem_strips_router)

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

        rows.append(
            {
                "index": idx,
                "filename": filename,
                "image_url": prefix + filename,
                "model_student_id": student_id,
                "model_canvas_name": roster_by_id.get(student_id) if student_id else None,
                "confidence": confidence,
                "raw_name": raw_name,
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


def _read_default_rubric() -> tuple[str, str]:
    """Load rubric YAML text and quiz_id for pre-filling the setup form."""
    try:
        yaml_text = DEFAULT_GRADING_KEY_PATH.read_text(encoding="utf-8")
    except FileNotFoundError:
        yaml_text = """quiz_id: quiz1\nmax_points: 10\nproblems:\n  - id: Q1\n    label: Example problem\n    points: 10\n"""

    try:
        parsed = yaml.safe_load(yaml_text) or {}
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
    """Tiny message at / so you don't just see 'Not Found'."""
    return {"message": "Logic Grading Facilitator is running. See /docs for API, or /grade/1/page (etc.) to grade."}


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


def _build_total_scores_by_student(session: Session) -> Dict[str, float]:
    totals: Dict[str, float] = {}
    for wu in session.work_units:
        if not wu.student_id:
            continue
        total = _score_summary_for_work_unit(session, wu)["total_score"]
        sid = wu.student_id
        if sid in totals:
            totals[sid] = max(totals[sid], total)
        else:
            totals[sid] = total
    return totals


def _required_total_max(grading_key: GradingKey) -> float:
    return float(grading_key.max_points)


def _score_summary_for_work_unit(session: Session, wu: WorkUnit) -> Dict[str, float]:
    regular_total = 0.0
    extra_credit_total = 0.0

    for problem in session.grading_key.problems:
        score = wu.per_problem.get(problem.id)
        if score is None:
            continue
        if problem.extra_credit:
            extra_credit_total += float(score)
        else:
            regular_total += float(score)

    total_max = _required_total_max(session.grading_key)
    total_score = min(total_max, regular_total + extra_credit_total)
    return {
        "regular_score": regular_total,
        "extra_credit_score": extra_credit_total,
        "total_score": total_score,
        "total_max": total_max,
    }


def _parse_gradebook_csv(content: str) -> tuple[List[str], List[Dict[str, str]], str]:
    reader = csv.DictReader(io.StringIO(content))
    if not reader.fieldnames:
        raise HTTPException(status_code=400, detail="Gradebook CSV has no headers.")

    fieldnames = list(reader.fieldnames)
    rows = list(reader)
    header_map = _normalize_headers(fieldnames)

    id_col = (
        header_map.get("sis login id")
        or header_map.get("sis user id")
        or header_map.get("login id")
        or header_map.get("id")
    )
    if not id_col:
        raise HTTPException(
            status_code=400,
            detail=(
                "Could not find a student ID column. Expected one of: "
                "'SIS Login ID', 'SIS User ID', 'Login ID', 'ID'."
            ),
        )
    return fieldnames, rows, id_col


def _candidate_assignment_columns(fieldnames: List[str], id_col: str) -> List[str]:
    excluded = {
        "student",
        "id",
        "sis user id",
        "sis login id",
        "login id",
        "section",
        "current score",
        "unposted current score",
        "final score",
        "unposted final score",
        "current grade",
        "unposted current grade",
        "final grade",
        "unposted final grade",
    }
    cols: List[str] = []
    for name in fieldnames:
        key = name.strip().lower()
        if key in excluded:
            continue
        if name == id_col:
            continue
        cols.append(name)
    return cols


def _assignment_column_stats(
    rows: List[Dict[str, str]],
    assignment_cols: List[str],
    id_col: str,
    score_map: Dict[str, float],
) -> List[Dict[str, Any]]:
    stats: List[Dict[str, Any]] = []
    for col in assignment_cols:
        non_empty = 0
        matched = 0
        matched_non_empty = 0
        for row in rows:
            sid = (row.get(id_col) or "").strip()
            if sid in score_map:
                matched += 1
            val = (row.get(col) or "").strip()
            if val != "":
                non_empty += 1
                if sid in score_map:
                    matched_non_empty += 1
        stats.append(
            {
                "name": col,
                "non_empty": non_empty,
                "matched": matched,
                "matched_non_empty": matched_non_empty,
            }
        )
    return stats


def _classwork_gradebook_tmp_path(token: str) -> Path:
    return GRADEBOOK_MERGE_TMP_DIR / f"classwork_{token}.csv"


def _sanitize_assignment_id(raw: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "_", (raw or "").strip())
    cleaned = cleaned.strip("_")
    return cleaned or "classwork_assignment"


def _classwork_make_assignment_id_from_column(column_name: str) -> str:
    return _sanitize_assignment_id((column_name or "").strip()).lower()


def _classwork_unique_assignment_id(base_id: str, existing_ids: set[str]) -> str:
    base = _sanitize_assignment_id(base_id).lower()
    if base not in existing_ids:
        return base
    counter = 2
    while True:
        candidate = f"{base}_{counter}"
        if candidate not in existing_ids:
            return candidate
        counter += 1


def _normalize_match_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (value or "").lower())


def _classwork_resolve_roster(
    match_roster_source: str,
    gradebook_rows: List[Dict[str, str]],
    gradebook_id_col: str,
) -> List[RosterEntry]:
    source = (match_roster_source or "").strip().lower()
    if source == "session":
        session = get_session()
        roster = session.roster or []
        if not roster:
            raise HTTPException(
                status_code=400,
                detail="Current session roster is empty. Choose gradebook matching or load a roster first.",
            )
        return roster

    roster = _classwork_build_gradebook_roster(gradebook_rows, gradebook_id_col)
    if not roster:
        raise HTTPException(status_code=400, detail="No roster rows found in gradebook CSV.")
    return roster


def _classwork_match_file_to_roster(
    zip_entry_name: str,
    roster_index: Dict[str, RosterEntry],
    sid_index: Dict[str, RosterEntry],
) -> Optional[RosterEntry]:
    # 1) Fast path: existing Canvas-style prefix match.
    base_name = Path(zip_entry_name).name
    prefix_key = _filename_prefix_key(base_name)
    entry = roster_index.get(prefix_key)
    if entry is not None:
        return entry

    # 2) Match by student-id token anywhere in the ZIP entry path.
    # Canvas exports sometimes keep identifying metadata in parent folders.
    search_text = (zip_entry_name or "").replace("\\", "/").lower()
    stem = Path(search_text).stem
    tokens = [t for t in re.split(r"[^a-z0-9]+", search_text) if t]
    for token in tokens:
        sid_entry = sid_index.get(token)
        if sid_entry is not None:
            return sid_entry

    # 3) Fallback: normalized full path/stem contains normalized roster name key.
    normalized_stem = _normalize_match_key(stem)
    normalized_path = _normalize_match_key(search_text)
    for key, roster_entry in sorted(roster_index.items(), key=lambda kv: len(kv[0]), reverse=True):
        if key and (key in normalized_stem or key in normalized_path):
            return roster_entry

    return None


def _classwork_build_gradebook_roster(
    rows: List[Dict[str, str]],
    id_col: str,
) -> List[RosterEntry]:
    roster: List[RosterEntry] = []
    seen_ids: set[str] = set()
    for row in rows:
        sid = (row.get(id_col) or "").strip()
        if not sid or sid in seen_ids:
            continue
        name = (row.get("Student") or row.get("student") or sid).strip() or sid
        roster.append(RosterEntry(student_id=sid, name=name))
        seen_ids.add(sid)
    return roster


def _classwork_entry_default_score(entry: ClassworkEntry, max_points: float) -> float:
    return float(max_points) if entry.has_submission else 0.0


def _build_classwork_score_map(mode: ClassworkModeState) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for entry in mode.entries:
        out[entry.student_id] = float(entry.score)
    return out


def _classwork_zip_unique_name(used: set[str], preferred: str) -> str:
    base = Path(preferred).name or "feedback.txt"
    if base not in used:
        used.add(base)
        return base
    stem = Path(base).stem
    suffix = Path(base).suffix
    counter = 2
    while True:
        candidate = f"{stem}_{counter}{suffix}"
        if candidate not in used:
            used.add(candidate)
            return candidate
        counter += 1


def _classwork_comment_text(mode: ClassworkModeState, entry: ClassworkEntry) -> str:
    comment = (entry.comment or "").strip()
    return (
        f"Assignment: {mode.assignment_id}\n"
        f"Student: {entry.student_name} ({entry.student_id})\n"
        f"Score: {entry.score:g} / {mode.max_points:g}\n\n"
        f"Comment:\n{comment}\n"
    )


def _classwork_apply_form_updates(
    form: Any,
    mode: ClassworkModeState,
    action: str,
) -> None:
    for entry in mode.entries:
        sid = entry.student_id
        score_raw = form.get(f"score_{sid}")
        if score_raw is not None:
            score_text = str(score_raw).strip()
            if score_text == "":
                entry.score = _classwork_entry_default_score(entry, mode.max_points)
            else:
                try:
                    entry.score = float(score_text)
                except ValueError:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Invalid score for {sid}: {score_text!r}",
                    )

        comment_raw = form.get(f"comment_{sid}")
        if comment_raw is not None:
            text = str(comment_raw).strip()
            entry.comment = text or None

    if action == "credit_submitted":
        for entry in mode.entries:
            if entry.has_submission:
                entry.score = float(mode.max_points)
    elif action == "zero_missing":
        for entry in mode.entries:
            if not entry.has_submission:
                entry.score = 0.0
    elif action == "reset_defaults":
        for entry in mode.entries:
            entry.score = _classwork_entry_default_score(entry, mode.max_points)


def _classwork_analysis(mode: ClassworkModeState) -> Dict[str, Any]:
    tmp_path = _classwork_gradebook_tmp_path(mode.gradebook_token)
    if not tmp_path.exists():
        raise HTTPException(
            status_code=400,
            detail="Stored classwork gradebook token is missing. Rebuild classwork setup.",
        )
    decoded = tmp_path.read_text(encoding="utf-8")
    fieldnames, rows, id_col = _parse_gradebook_csv(decoded)
    score_map = _build_classwork_score_map(mode)
    assignment_cols = _candidate_assignment_columns(fieldnames, id_col)
    stats = _assignment_column_stats(rows, assignment_cols, id_col, score_map)
    return {
        "token": mode.gradebook_token,
        "id_col": id_col,
        "columns": stats,
        "row_count": len(rows),
    }


def _normalize_for_matching(name: str) -> str:
    """
    Normalize a string for comparison between Canvas assignment names and gradebook columns.
    
    Args:
        name: The string to normalize (assignment name or column name)
    
    Returns:
        Normalized string suitable for comparison
    """
    if not name:
        return ""
    
    # Convert to lowercase
    normalized = name.lower().strip()
    
    # Remove parenthetical numbers (Canvas assignment IDs in gradebook)
    # Pattern: "(12345678)" or similar - removes the ID suffix
    # Example: "Wed Attendance (18715846)" -> "Wed Attendance"
    import re
    normalized = re.sub(r"\s*\(\d+\)\s*$", "", normalized)
    
    # Remove special characters (keep only alphanumeric, spaces, hyphens, underscores)
    # This handles variations like "Worksheet - 1" vs "Worksheet_1"
    normalized = re.sub(r"[^a-z0-9\s\-_]", "", normalized)
    
    # Collapse multiple spaces into single space
    normalized = re.sub(r"\s+", " ", normalized)
    
    # Strip leading/trailing whitespace again after cleanup
    return normalized.strip()


def _find_matching_column(
    canvas_assignment_name: str,
    gradebook_fieldnames: List[str],
    id_col: str,
) -> Optional[str]:
    """
    Find a matching gradebook column for a Canvas assignment name.
    
    Args:
        canvas_assignment_name: Name of assignment from Canvas API
        gradebook_fieldnames: List of column names from gradebook CSV
        id_col: The ID column name to exclude from matching
    
    Returns:
        Matching column name if found, None otherwise
    """
    # Normalize the Canvas assignment name
    normalized_canvas = _normalize_for_matching(canvas_assignment_name)
    
    # Build list of candidate columns (exclude ID column and common system columns)
    excluded = {id_col.lower(), "student", "current score", "final score", 
                 "current grade", "final grade"}
    
    candidates = []
    for col in gradebook_fieldnames:
        if col.lower() not in excluded:
            candidates.append(col)
    
    if not candidates:
        return None
    
    # Try to find exact match after normalization
    for col in candidates:
        if _normalize_for_matching(col) == normalized_canvas:
            return col
    
    # No match found
    return None


def _classwork_build_mode_from_downloads(
    *,
    student_files: Dict[str, List[Path]],
    assignment_id: str,
    assignment_name: str,  # Added: original Canvas assignment name
    max_points: float,
    active_roster: List[RosterEntry],
    gradebook_token: str,
    id_col: str,
    gradebook_column: Optional[str] = None,
    gradebook_fieldnames: Optional[List[str]] = None,  # Added: for column matching
) -> ClassworkModeState:
    """
    Build ClassworkModeState directly from Canvas-downloaded files.
    
    This bypasses ZIP filename matching since we already have student_id-to-files mapping
    from the Canvas API.
    """
    setup_id = uuid4().hex
    student_files_ref: Dict[str, List[ClassworkFileRef]] = {}
    
    # Map roster to dict for lookup
    roster_by_id = {r.student_id: r.name for r in active_roster}
    
    # Auto-match gradebook column if not provided and we have fieldnames
    if gradebook_column is None and gradebook_fieldnames:
        matched_column = _find_matching_column(
            canvas_assignment_name=assignment_name,
            gradebook_fieldnames=gradebook_fieldnames,
            id_col=id_col,
        )
        
        if matched_column:
            print(f"[COLUMN MATCH] Auto-matched '{assignment_name}' to gradebook column: '{matched_column}'")
            gradebook_column = matched_column
        else:
            print(f"[COLUMN MATCH WARNING] No match found for assignment '{assignment_name}'. "
                  f"Manual column selection required.")
            # Leave gradebook_column=None to force manual selection
    
    # Process files for each student
    for student_id, file_paths in student_files.items():
        student_name = roster_by_id.get(student_id, "")
        if not file_paths:
            continue
        
        # Store files in CLASSWORK_FILES_DIR
        refs = []
        for file_path in file_paths:
            if not file_path.exists():
                continue
            
            original_name = file_path.name
            ext = Path(original_name).suffix.lower()
            
            # Create stored filename
            stored_name = f"{setup_id}_{original_name}"
            stored_path = CLASSWORK_FILES_DIR / stored_name
            stored_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Copy file to storage
            shutil.copy2(file_path, stored_path)
            
            refs.append(
                ClassworkFileRef(
                    stored_filename=stored_path.name,
                    original_filename=original_name,
                    ext=ext,
                )
            )
        
        student_files_ref[student_id] = refs
    
    # Build entries
    assignment_clean = _sanitize_assignment_id(assignment_id)
    entries: List[ClassworkEntry] = []
    for r in active_roster:
        files = student_files_ref.get(r.student_id, [])
        has_submission = bool(files)
        score = float(max_points) if has_submission else 0.0
        entries.append(
            ClassworkEntry(
                student_id=r.student_id,
                student_name=r.name,
                has_submission=has_submission,
                files=files,
                score=score,
                comment=None,
            )
        )
    
    return ClassworkModeState(
        assignment_id=assignment_clean,
        max_points=float(max_points),
        gradebook_token=gradebook_token,
        id_column=id_col,
        gradebook_column=(gradebook_column or "").strip() or None,
        entries=entries,
        unmatched_files=[],  # No unmatched files since we use direct student_id mapping
    )


def _classwork_sync_mode_into_batch(session: Session) -> None:
    mode = session.classwork_mode
    batch = session.classwork_batch
    if mode is None or batch is None:
        return
    for idx, item in enumerate(batch.assignments):
        if item.assignment_id == mode.assignment_id:
            batch.assignments[idx] = mode
            return


def _same_numeric_score(old_val: str, new_score: float) -> bool:
    try:
        return math.isclose(float(old_val), float(new_score), rel_tol=0.0, abs_tol=1e-9)
    except ValueError:
        return False


def _classwork_open_next_batch_assignment(session: Session, *, include_current: bool = False) -> bool:
    batch = session.classwork_batch
    if batch is None or not batch.assignments:
        return False
    current_id = session.classwork_mode.assignment_id if session.classwork_mode else None
    if include_current or not current_id:
        session.classwork_mode = batch.assignments[0]
        return True
    for idx, item in enumerate(batch.assignments):
        if item.assignment_id == current_id:
            if idx + 1 < len(batch.assignments):
                session.classwork_mode = batch.assignments[idx + 1]
                return True
            return False
    session.classwork_mode = batch.assignments[0]
    return True


def _classwork_batch_assignment_stats(
    batch: ClassworkBatchState,
    rows: List[Dict[str, str]],
    id_col: str,
) -> List[Dict[str, Any]]:
    assignment_stats: List[Dict[str, Any]] = []
    for mode in batch.assignments:
        score_map = _build_classwork_score_map(mode)
        col = (mode.gradebook_column or "").strip()
        non_empty = 0
        matched = 0
        matched_non_empty = 0
        if col:
            for row in rows:
                sid = (row.get(id_col) or "").strip()
                if sid in score_map:
                    matched += 1
                val = (row.get(col) or "").strip()
                if val != "":
                    non_empty += 1
                    if sid in score_map:
                        matched_non_empty += 1
        assignment_stats.append(
            {
                "assignment_id": mode.assignment_id,
                "max_points": mode.max_points,
                "submitted_count": sum(1 for e in mode.entries if e.has_submission),
                "commented_count": sum(1 for e in mode.entries if (e.comment or "").strip()),
                "gradebook_column": col or "",
                "column_non_empty": non_empty,
                "column_matched": matched,
                "column_matched_non_empty": matched_non_empty,
            }
        )
    return assignment_stats


def _classwork_resolve_stored_file(stored_filename: str) -> Path:
    safe_name = Path(stored_filename or "").name
    if not safe_name or safe_name != stored_filename:
        raise HTTPException(status_code=400, detail="Invalid classwork file path.")
    path = CLASSWORK_FILES_DIR / safe_name
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="Classwork file not found.")
    return path


def _classwork_extract_docx_text(path: Path) -> str:
    try:
        doc = Document(str(path))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read DOCX preview: {exc}")

    lines = [(p.text or "").strip() for p in doc.paragraphs]
    text = "\n\n".join(line for line in lines if line)
    return text.strip()


def _classwork_extract_text_preview(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".docx":
        return _classwork_extract_docx_text(path)

    raw = path.read_bytes()
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            text = raw.decode(encoding)
            # Clean HTML content if present (e.g., from Canvas text submissions)
            from .utils import clean_html_text
            return clean_html_text(text)
        except UnicodeDecodeError:
            continue
    raise HTTPException(status_code=400, detail="Could not decode text preview for this file.")


@app.get("/classwork/canvas_courses")
def get_canvas_courses():
    """Fetch list of Canvas courses for the user."""
    try:
        client = get_canvas_client()
        courses = client.get_courses()
        return {"courses": courses}
    except CanvasClientError as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/classwork/canvas_assignments")
def get_canvas_assignments(course_id: int = Query(...)):
    """Fetch assignments for a Canvas course, sorted by due date."""
    try:
        client = get_canvas_client()
        assignments = client.get_assignments(course_id)
        return {"course_id": course_id, "assignments": assignments}
    except CanvasClientError as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/classwork/canvas_submission_counts")
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


@app.get("/classwork", response_class=HTMLResponse)
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


@app.post("/classwork/setup", response_class=HTMLResponse)
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


@app.post("/classwork/batch/start", response_class=HTMLResponse)
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


@app.post("/classwork/batch/add", response_class=HTMLResponse)
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


@app.post("/classwork/batch/open", response_class=HTMLResponse)
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


@app.post("/classwork/batch/open_next", response_class=HTMLResponse)
async def classwork_batch_open_next() -> Response:
    with _session_lock:
        session = get_session()
        if not _classwork_open_next_batch_assignment(session, include_current=True):
            raise HTTPException(status_code=400, detail="No queued assignments to review.")
        persist_session()
    return RedirectResponse(url="/classwork/review", status_code=303)


@app.post("/classwork/batch/remove", response_class=HTMLResponse)
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


@app.post("/classwork/batch/set_column", response_class=HTMLResponse)
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


@app.post("/classwork/batch/clear", response_class=HTMLResponse)
async def classwork_batch_clear() -> Response:
    with _session_lock:
        session = get_session()
        session.classwork_batch = None
        session.classwork_mode = None
        persist_session()
    return RedirectResponse(url="/classwork?status=success&message=" + quote("Queue cleared."), status_code=303)


@app.get("/classwork/batch/finalize", response_class=HTMLResponse)
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


@app.post("/classwork/batch/finalize", response_class=HTMLResponse)
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


@app.post("/classwork/batch/close_session", response_class=HTMLResponse)
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


@app.get("/classwork/review", response_class=HTMLResponse)
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


@app.post("/classwork/review", response_class=HTMLResponse)
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


@app.post("/classwork/finalize", response_class=HTMLResponse)
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


@app.get("/classwork/preview_text/{stored_filename:path}", response_class=PlainTextResponse)
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


@app.get("/classwork/export/{export_id}/{kind}")
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


@app.get("/classwork/batch/export/{export_id}/gradebook")
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


@app.get("/classwork/batch/export/{export_id}/comments/{assignment_id}")
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


@app.get("/classwork/canvas_setup", response_class=HTMLResponse)
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


@app.post("/classwork/canvas_setup", response_class=HTMLResponse)
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


@app.post("/classwork/canvas_post_comments", response_class=HTMLResponse)
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


def _save_image_upload(upload: UploadFile, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination = _unique_path(destination)
    with destination.open("wb") as f:
        shutil.copyfileobj(upload.file, f)


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


def _ingest_zip_bytes(zip_bytes: bytes, output_dir: Path) -> int:
    saved = 0
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
                dest = output_dir / filename
                dest = _unique_path(dest)
                dest.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(entry) as src, dest.open("wb") as f:
                    shutil.copyfileobj(src, f)
                saved += 1
            elif suffix == ".pdf":
                pdf_bytes = zf.read(entry)
                if not pdf_bytes:
                    continue
                stem = Path(filename).stem or "scans"
                saved += _save_pdf_pages(pdf_bytes, stem, output_dir)
    return saved


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

    if images:
        allowed_exts = {".jpg", ".jpeg", ".png"}
        for img in images:
            filename = Path(img.filename or "").name
            if not filename:
                continue
            suffix = Path(filename).suffix.lower()
            if suffix in allowed_exts:
                dest = RAW_SCANS_DIR / filename
                _save_image_upload(img, dest)
                saved += 1
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
                    saved += _ingest_zip_bytes(zip_bytes, RAW_SCANS_DIR)
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
            saved += _ingest_zip_bytes(zip_bytes, RAW_SCANS_DIR)
        except zipfile.BadZipFile:
            raise HTTPException(status_code=400, detail="Uploaded ZIP could not be read.")

    return RedirectResponse(
        url="/setup?status=success&message=" + quote(f"Uploaded {saved} scan files."),
        status_code=303,
    )


def _normalize_canvas_key(value: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]", "", (value or "").lower())
    return cleaned


def _filename_prefix_key(filename: str) -> str:
    base = Path(filename).name
    stem = Path(base).stem
    prefix = stem.split("_", 1)[0]
    return _normalize_canvas_key(prefix)


def _build_roster_prefix_index(
    roster: List[RosterEntry],
) -> tuple[Dict[str, RosterEntry], Dict[str, List[RosterEntry]]]:
    index: Dict[str, RosterEntry] = {}
    collisions: Dict[str, List[RosterEntry]] = {}
    for entry in roster:
        key = _normalize_canvas_key(entry.name)
        if not key:
            continue
        if key in index:
            collisions.setdefault(key, [index[key]]).append(entry)
        else:
            index[key] = entry
    return index, collisions


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

    roster = load_roster()
    if not roster:
        raise HTTPException(status_code=400, detail="Roster not found. Upload a roster first.")

    try:
        grading_key = load_grading_key()
    except Exception:
        raise HTTPException(status_code=400, detail="Rubric not found. Save a rubric first.")

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
        grading_key = GradingKey(**data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid rubric YAML: {e}")

    RUBRIC_DIR.mkdir(parents=True, exist_ok=True)
    target = RUBRIC_DIR / f"{grading_key.quiz_id}.yaml"
    cleaned_yaml = yaml.safe_dump(grading_key.model_dump(), sort_keys=False)
    target.write_text(cleaned_yaml, encoding="utf-8")

    return RedirectResponse(
        url="/setup?status=success&message=" + quote(f"Rubric saved to {target.name}."),
        status_code=303,
    )


@app.post("/setup/select_rubric")
async def setup_select_rubric(rubric_file: str = Form(...)) -> Response:
    if not (rubric_file or "").strip():
        raise HTTPException(status_code=400, detail="Please select a rubric.")

    candidate = RUBRIC_DIR / Path(rubric_file).name
    if not candidate.exists():
        raise HTTPException(status_code=404, detail="Selected rubric not found.")

    DEFAULT_GRADING_KEY_PATH.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(candidate, DEFAULT_GRADING_KEY_PATH)

    return RedirectResponse(
        url="/setup?status=success&message="
        + quote(f"Rubric set from {candidate.name}."),
        status_code=303,
    )


@app.post("/setup/run_auto_grouping")
async def setup_run_auto_grouping(quiz_id: str = Form("quiz1")) -> Response:
    roster = load_roster()
    if not RAW_SCANS_DIR.exists() or not any(RAW_SCANS_DIR.iterdir()):
        raise HTTPException(status_code=400, detail="Upload scans before running auto-grouping.")

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


@app.get("/name_review", response_class=HTMLResponse)
async def name_review_page(
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

def slugify_name(name: str) -> str:
    """
    Turn 'Berry, Sharon A.' into 'sharon_berry_a'.
    Very basic; tweak to taste.
    """
    s = name.strip().lower()
    s = s.replace(",", " ")
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9_]", "", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "student"

def roster_lookup(session: Session) -> Dict[str, str]:
    """Map student_id -> slugified name."""
    return {r.student_id: slugify_name(r.name) for r in session.roster}


@app.get("/finish", response_class=HTMLResponse)
def finish_page(request: Request) -> Response:
    session = get_session()
    grading_key = session.grading_key
    problems = grading_key.problems if grading_key else []
    total_max = _required_total_max(grading_key) if problems else 0.0

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
                "total_score": total,
            }

    rows = sorted(rows_by_student.values(), key=lambda r: r["student_name"])

    return templates.TemplateResponse(
        "finish.html",
        {
            "request": request,
            "problems": problems,
            "total_max": total_max,
            "rows": rows,
        },
    )


@app.get("/export_renamed_scans")
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


@app.get("/export")
def export_scores():
    """
    Export a simple CSV of student_id,total_score for completed WorkUnits.

    - Skips WorkUnits with no student_id.
    - If multiple WorkUnits are assigned to the same student_id,
      we export the *maximum* total score among them (to avoid double-grading
      when something was scanned/created twice).
    """
    session = get_session()
    totals = _build_total_scores_by_student(session)

    # Build CSV in-memory
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["student_id", f"{session.grading_key.quiz_id}_score"])
    for sid, total in totals.items():
        writer.writerow([sid, total])

    csv_data = buf.getvalue()
    return Response(
        content=csv_data,
        media_type="text/csv",
        headers={
            "Content-Disposition": (
                f'attachment; filename="{session.grading_key.quiz_id}_scores.csv"'
            )
        },
    )

@app.post("/finalize", response_class=HTMLResponse)
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


# === HTML Grading UI ===

@app.get("/grade/{unit_id}/page", response_class=HTMLResponse)
def grade_page(request: Request, unit_id: int):
    """
    HTML grading UI for a WorkUnit.

    Shows:
    - scan images
    - roster dropdown
    - per-problem score fields
    - comment box
    """
    session = get_session()
    wu = find_work_unit(session, unit_id)

    # Build URLs for images (served via /scans mount)
    image_urls = [f"/scans/{fn}" for fn in wu.page_filenames]

    return templates.TemplateResponse(
        "grade.html",
        {
            "request": request,
            "work_unit": wu,
            "grading_key": session.grading_key,
            "roster": session.roster,
            "image_urls": image_urls,
            "pinned_note": session.pinned_note or "",
        },
      )


@app.get("/export_feedback_zip")
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


def _list_canvas_original_zips() -> List[Path]:
    if not CANVAS_REUPLOAD_ORIGINAL_ZIPS_DIR.exists():
        raise HTTPException(
            status_code=400,
            detail=(
                "Canvas original ZIP directory not found. "
                f"Create: {CANVAS_REUPLOAD_ORIGINAL_ZIPS_DIR}"
            ),
        )
    zips = sorted(
        [p for p in CANVAS_REUPLOAD_ORIGINAL_ZIPS_DIR.glob("*.zip") if p.is_file()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not zips:
        raise HTTPException(
            status_code=400,
            detail=(
                "No ZIP files found for Canvas reupload source. "
                f"Put original Canvas ZIP(s) under {CANVAS_REUPLOAD_ORIGINAL_ZIPS_DIR}"
            ),
    )
    return zips


def _canvas_import_manifest_path(quiz_id: str) -> Path:
    safe_quiz_id = re.sub(r"[^A-Za-z0-9_.-]", "_", (quiz_id or "quiz").strip()) or "quiz"
    return CANVAS_REUPLOAD_MANIFEST_DIR / f"{safe_quiz_id}.json"


def _load_canvas_import_manifest(quiz_id: str) -> Optional[Dict[str, Any]]:
    path = _canvas_import_manifest_path(quiz_id)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _save_canvas_import_manifest(
    quiz_id: str,
    source_zip_filename: str,
    original_zip_name: str,
    work_units: List[WorkUnit],
    per_student_originals: Dict[str, List[str]],
) -> None:
    CANVAS_REUPLOAD_MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

    manifest_work_units: List[Dict[str, Any]] = []
    for wu in work_units:
        manifest_work_units.append(
            {
                "unit_id": wu.unit_id,
                "student_id": wu.student_id,
                "original_filenames": per_student_originals.get(wu.student_id or "", []),
                "canonical_page_filenames": list(wu.page_filenames),
            }
        )

    manifest = {
        "quiz_id": quiz_id,
        "source_zip_filename": source_zip_filename,
        "original_zip_name": original_zip_name,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "work_units": manifest_work_units,
    }
    _canvas_import_manifest_path(quiz_id).write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )


def _find_canvas_original_zip_for_quiz(
    quiz_id: str,
    requested_name: Optional[str] = None,
) -> Path:
    zips = _list_canvas_original_zips()

    if requested_name:
        requested = Path(requested_name).name
        for zip_path in zips:
            if zip_path.name == requested:
                return zip_path
        available = ", ".join(p.name for p in zips[:10])
        raise HTTPException(
            status_code=404,
            detail=(
                f"Requested Canvas source ZIP not found: {requested}. "
                f"Available ZIPs: {available}"
            ),
        )

    manifest = _load_canvas_import_manifest(quiz_id)
    if manifest:
        manifest_zip_name = Path(str(manifest.get("source_zip_filename") or "")).name
        if manifest_zip_name:
            for zip_path in zips:
                if zip_path.name == manifest_zip_name:
                    return zip_path

    quiz_key = _normalize_canvas_key(quiz_id)
    exact_prefix_matches = [
        p for p in zips if _normalize_canvas_key(p.stem.split("__", 1)[0]) == quiz_key
    ]
    if len(exact_prefix_matches) == 1:
        return exact_prefix_matches[0]
    if len(exact_prefix_matches) > 1:
        available = ", ".join(p.name for p in exact_prefix_matches[:10])
        raise HTTPException(
            status_code=400,
            detail=(
                f"Multiple Canvas source ZIPs match quiz_id '{quiz_id}'. "
                f"Choose one explicitly with ?source_zip=... Available matches: {available}"
            ),
        )

    loose_matches = [p for p in zips if quiz_key and quiz_key in _normalize_canvas_key(p.stem)]
    if len(loose_matches) == 1:
        return loose_matches[0]
    if len(loose_matches) > 1:
        available = ", ".join(p.name for p in loose_matches[:10])
        raise HTTPException(
            status_code=400,
            detail=(
                f"Multiple loosely matching Canvas source ZIPs found for quiz_id '{quiz_id}'. "
                f"Choose one explicitly with ?source_zip=... Available matches: {available}"
            ),
        )

    available = ", ".join(p.name for p in zips[:10])
    raise HTTPException(
        status_code=400,
        detail=(
            f"No Canvas source ZIP found for active quiz_id '{quiz_id}'. "
            "The exporter will not guess from an unrelated ZIP anymore. "
            f"Put the original Canvas submissions ZIP in {CANVAS_REUPLOAD_ORIGINAL_ZIPS_DIR} "
            f"with a name like '{quiz_id}__submissions.zip', or pass ?source_zip=... "
            f"Current ZIPs: {available}"
        ),
    )


def _build_best_rows_by_student(session: Session) -> Dict[str, Dict[str, Any]]:
    grading_key = session.grading_key
    problems = grading_key.problems
    total_max = _required_total_max(grading_key) if problems else 0.0
    roster_map = {r.student_id: r.name for r in session.roster}

    rows_by_student: Dict[str, Dict[str, Any]] = {}
    for wu in session.work_units:
        if not wu.student_id:
            continue
        per_problem_scores: Dict[str, Optional[float]] = {}
        per_problem_comments: Dict[str, Optional[str]] = {}
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
                "total_max": total_max,
                "quiz_id": grading_key.quiz_id,
            }
    return rows_by_student


def _feedback_text_from_row(
    row: Optional[Dict[str, Any]],
    problems: List[Any],
    quiz_id: str,
    total_max: float,
) -> str:
    if row is None:
        return (
            f"Quiz: {quiz_id}\n"
            "No grading record found for this submission.\n"
            "Please contact the instructor if you believe this is an error.\n"
        )

    lines: List[str] = []
    lines.append(f"Quiz: {quiz_id}")
    lines.append(f"Student: {row['student_name']} ({row['student_id']})")
    lines.append(f"Total: {row['total_score']} / {total_max}")
    if row.get("extra_credit_score", 0) > 0:
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
    return "\n".join(lines)


def _render_feedback_image_bytes(text: str, ext: str) -> bytes:
    width, height = 1500, 2100
    img = Image.new("RGB", (width, height), color="white")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    wrapped: List[str] = []
    for para in text.splitlines():
        chunks = textwrap.wrap(para, width=95) if para else [""]
        wrapped.extend(chunks)
    rendered_text = "\n".join(wrapped[:220])
    draw.multiline_text((40, 40), rendered_text, fill="black", font=font, spacing=4)

    buf = io.BytesIO()
    if ext in {".jpg", ".jpeg"}:
        img.save(buf, format="JPEG", quality=92)
    else:
        img.save(buf, format="PNG")
    return buf.getvalue()


def _render_feedback_pdf_bytes(text: str) -> bytes:
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    rect = fitz.Rect(40, 40, 572, 752)
    wrapped: List[str] = []
    for para in text.splitlines():
        chunks = textwrap.wrap(para, width=95) if para else [""]
        wrapped.extend(chunks)
    rendered_text = "\n".join(wrapped[:320])
    page.insert_textbox(rect, rendered_text, fontsize=11, fontname="helv")
    pdf_bytes = doc.tobytes(deflate=True)
    doc.close()
    return pdf_bytes


@app.get("/export_canvas_reupload_feedback_zip")
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

    if (not refresh) and cache_path.exists():
        # ✅ Use cached predictions
        import json as _json
        predictions = _json.loads(cache_path.read_text(encoding="utf-8"))
    else:
        # 🔁 Run LM Studio and cache predictions
        print(f"[name_review] Cache miss. inspecting {len(paths)} files...")

        client = LMStudioQwenClient(
            base_url=LMSTUDIO_BASE_URL,
            model=LMSTUDIO_MODEL_ID,
        )
        predictions = []
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

            predictions.append(
                {
                    "filename": path.name,
                    "student_id": student_id,
                    "raw_name": raw_name,
                    "confidence": confidence,
                    "reasoning": reasoning,
                    "name_visible": name_visible,
                }
            )

        import json as _json
        cache_path.write_text(
            _json.dumps(predictions, indent=2),
            encoding="utf-8",
        )

    # 4. Align predictions with the sorted file list and build page_rows
    preds_by_filename: Dict[str, Dict] = {
        p["filename"]: p for p in predictions
    }

    page_rows: List[Dict] = []
    for idx, path in enumerate(paths):
        info = preds_by_filename.get(path.name, {})
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
    # List of (original_filename, student_id)
    assignments: List[tuple[str, Optional[str]]] = []
    for i in range(page_count):
        fn = form.get(f"filename_{i}")
        sid = form.get(f"student_{i}") or None
        if fn:
            assignments.append((fn, sid))

    # Fast path: only update the cache and auto-fill unknowns in order
    if next_action == "save_fill":
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
        for filename, sid in assignments:
            applied_sid = sid or last_sid
            if applied_sid:
                last_sid = applied_sid

            entry = dict(preds_by_filename.get(filename, {"filename": filename}))
            entry["student_id"] = applied_sid
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
        print(f"[name_review] Saved {len(updated_predictions)} cache entries (auto-filled unknowns).")

        redirect_url = f"/name_review?quiz={quote(quiz_id)}"
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

    for filename, sid in assignments:
        if current_group is None:
            current_group = {
                "student_id": sid,
                "pages": [filename]
            }
            continue

        # If same student (and not None), append to group
        # Note: We break groups on 'None' (unknowns) so they don't clump together?
        # Actually, let's clump unknowns if they are consecutive? 
        # For safety, let's break if student_id changes.
        if sid == current_group["student_id"]:
            current_group["pages"].append(filename)
        else:
            groups.append(current_group)
            current_group = {
                "student_id": sid,
                "pages": [filename]
            }
    if current_group:
        groups.append(current_group)

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

