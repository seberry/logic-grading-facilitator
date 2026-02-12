from __future__ import annotations

from typing import Any, Dict, List, Optional
import asyncio
import threading
import io
import csv
import json
import textwrap
from pathlib import Path
import zipfile
import sys
from datetime import datetime
from uuid import uuid4
import re
import math
from urllib.parse import quote
from app.models import GradingKey

from fastapi import (
    FastAPI,
    HTTPException,
    Response,
    Request,
    Form,
    UploadFile,
    File,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
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

from .models import Session, WorkUnit, RosterEntry
from .storage import (
    load_session,
    save_session,
    DEFAULT_SESSION_PATH,
    RAW_SCANS_DIR,  # we’ll use this to serve images
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
from .verifier import ProofLine, check_proof, flatten_proof
import shutil
import yaml
import fitz  # type: ignore
from PIL import Image, ImageDraw, ImageFont
from docx import Document


from .vlm_client import LMStudioQwenClient  # type: ignore
from .vlm_classifier import classify_page  # type: ignore

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
GRADEBOOK_MERGE_TMP_DIR = DATA_DIR / "gradebook_merge" / "tmp"

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

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]


def _is_image_filename(filename: str) -> bool:
    return Path(filename).suffix in IMAGE_EXTENSIONS


def _select_display_filename(
    page_filenames: List[str],
    seg_filename: Optional[str],
) -> Optional[str]:
    if seg_filename and _is_image_filename(seg_filename):
        return seg_filename
    for fname in page_filenames:
        if _is_image_filename(fname):
            return fname
    return None


def _build_page_options(page_filenames: List[str]) -> List[Dict[str, str]]:
    options: List[Dict[str, str]] = []
    for fname in page_filenames:
        if not _is_image_filename(fname):
            continue
        options.append(
            {
                "filename": fname,
                "url": f"/scans/{quote(fname)}",
            }
        )
    return options


def list_fitch_debug_scans() -> List[str]:
    if not DEBUG_FITCH_DIR.exists():
        return []
    return sorted([p.name for p in DEBUG_FITCH_DIR.iterdir() if p.is_dir()], key=str.lower)


def final_structure_path(scan_slug: str) -> Path:
    return DEBUG_FITCH_DIR / scan_slug / "final_structure.json"


def load_structure_text(scan_slug: str) -> str:
    path = final_structure_path(scan_slug)
    if path.exists():
        return path.read_text(encoding="utf-8")
    return ""


def find_scan_image_path(scan_slug: str) -> Optional[Path]:
    for ext in IMAGE_EXTENSIONS:
        candidate = RAW_SCANS_DIR / f"{scan_slug}{ext}"
        if candidate.exists():
            return candidate

    debug_dir = DEBUG_FITCH_DIR / scan_slug
    if not debug_dir.exists():
        return None

    for ext in IMAGE_EXTENSIONS:
        candidate = debug_dir / f"{scan_slug}{ext}"
        if candidate.exists():
            return candidate

    for child in sorted(debug_dir.iterdir()):
        if child.is_file() and child.suffix in IMAGE_EXTENSIONS:
            return child
    return None


def scan_image_url(scan_slug: str) -> Optional[str]:
    path = find_scan_image_path(scan_slug)
    if path is None:
        return None

    if RAW_SCANS_DIR in path.parents:
        return f"/scans/{path.name}"

    if DEBUG_FITCH_DIR in path.parents:
        rel = path.relative_to(DEBUG_FITCH_DIR)
        return f"/debug_fitch_assets/{rel.as_posix()}"

    return None


def _clean_meta(meta: Any) -> Dict[str, Any]:
    if not isinstance(meta, dict):
        return {"num_premises": None, "conclusion": ""}

    raw_num = meta.get("num_premises")
    num_premises: Optional[int] = None
    if raw_num is not None and raw_num != "":
        try:
            num_premises = max(0, int(raw_num))
        except (TypeError, ValueError):
            num_premises = None

    conclusion = meta.get("conclusion")
    conclusion_str = str(conclusion) if conclusion is not None else ""

    return {
        "num_premises": num_premises,
        "conclusion": conclusion_str,
    }


def normalize_structure_payload(payload: Any) -> Dict[str, Any]:
    """Accept legacy list payloads or new dicts, and return a normalized object.

    Normalized shape:
      {"lines": <list>, "meta": {"num_premises": Optional[int], "conclusion": str}}
    """

    lines: Any = []
    meta: Dict[str, Any] = {}

    if isinstance(payload, list):
        lines = payload
        meta = {}
    elif isinstance(payload, dict):
        candidate_lines = payload.get("lines")
        lines = candidate_lines if isinstance(candidate_lines, list) else payload.get("structure", [])
        meta = payload.get("meta", {})
    else:
        raise HTTPException(status_code=400, detail="final_structure.json must be a list or object containing lines/meta.")

    if not isinstance(lines, list):
        raise HTTPException(status_code=400, detail="'lines' must be a list (allowing nested subproofs).")

    return {
        "lines": lines,
        "meta": _clean_meta(meta),
    }


def load_structure_json(scan_slug: str) -> Dict[str, Any]:
    text = load_structure_text(scan_slug)
    if not text:
        return {"lines": [], "meta": {"num_premises": None, "conclusion": ""}}

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON in final_structure.json: {e}")

    return normalize_structure_payload(parsed)


def ensure_raw_scan_available(scan_slug: str) -> Path:
    existing = find_scan_image_path(scan_slug)
    if existing and RAW_SCANS_DIR in existing.parents:
        return existing

    if existing and DEBUG_FITCH_DIR in existing.parents:
        RAW_SCANS_DIR.mkdir(parents=True, exist_ok=True)
        destination = RAW_SCANS_DIR / existing.name
        if not destination.exists():
            shutil.copy(existing, destination)
        return destination

    raise HTTPException(
        status_code=404,
        detail=f"Could not locate a scan image for '{scan_slug}' in {RAW_SCANS_DIR} or {DEBUG_FITCH_DIR}.",
    )


async def run_subprocess_with_logs(command: List[str]) -> tuple[int, str]:
    proc = await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd=Path(__file__).resolve().parents[1],
    )
    stdout, _ = await proc.communicate()
    output = stdout.decode("utf-8", errors="replace")
    return proc.returncode, output


def structure_to_proof_lines(structure: List[Any]) -> List[Any]:
    lines: List[Any] = []
    for item in structure:
        if isinstance(item, list):
            lines.append(structure_to_proof_lines(item))
        elif isinstance(item, dict):
            lines.append(
                ProofLine(
                    wffstr=str(item.get("wffstr", "")).strip(),
                    jstr=str(item.get("jstr", "")).strip(),
                    location=[],
                )
            )
    return lines


def label_and_count_premises(structure: List[Any]) -> int:
    count = 0
    for item in structure:
        if isinstance(item, list):
            break
        if isinstance(item, dict):
            jstr = str(item.get("jstr", "")).strip()
            if not jstr:
                item["jstr"] = "Pr"
                count += 1
            elif jstr.lower() != "pr":
                break
    return count


_session_lock = threading.Lock()
_session: Optional[Session] = None
_segmentation_lock = threading.Lock()


def get_session() -> Session:
    """Load the session from disk on first use, then cache in memory."""
    global _session
    if _session is None:
        try:
            _session = load_session(DEFAULT_SESSION_PATH)
        except FileNotFoundError as e:
            raise HTTPException(status_code=500, detail=f"Session file not found: {e}")
    return _session


def persist_session() -> None:
    """Persist current in-memory session to disk."""
    global _session
    if _session is None:
        return
    save_session(_session, DEFAULT_SESSION_PATH)

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


@app.get("/debug/fitch", response_class=HTMLResponse)
async def fitch_debug_page(request: Request, scan: Optional[str] = None) -> Response:
    scans = list_fitch_debug_scans()
    selected = scan if scan in scans else (scans[0] if scans else None)

    structure_text = load_structure_text(selected) if selected else ""
    final_path = final_structure_path(selected) if selected else None

    return templates.TemplateResponse(
        "debug_fitch.html",
        {
            "request": request,
            "available_scans": scans,
            "selected_scan": selected,
            "structure_text": structure_text,
            "structure_json": load_structure_json(selected) if selected else {"lines": [], "meta": {"num_premises": None, "conclusion": ""}},
            "image_url": scan_image_url(selected) if selected else None,
            "final_structure_exists": bool(final_path and final_path.exists()),
        },
    )


@app.post("/debug/fitch/save")
async def save_fitch_structure(scan: str = Form(...), structure: str = Form(...)) -> Response:
    target_dir = DEBUG_FITCH_DIR / scan
    if not target_dir.exists():
        raise HTTPException(status_code=404, detail=f"Debug folder not found for '{scan}'.")

    try:
        parsed_raw = json.loads(structure)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")

    normalized = normalize_structure_payload(parsed_raw)

    path = final_structure_path(scan)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(normalized, indent=2, ensure_ascii=False), encoding="utf-8")

    return JSONResponse({"status": "ok", "message": f"Saved {path.name}."})


@app.post("/debug/fitch/rerun")
async def rerun_fitch_pipeline(scan: str = Form(...)) -> Response:
    if not (DEBUG_FITCH_DIR / scan).exists():
        raise HTTPException(status_code=404, detail=f"Debug folder not found for '{scan}'.")

    ensure_raw_scan_available(scan)
    command = [sys.executable, "scripts/debug_fitch_ocr.py", f"{scan}.jpg"]

    # If the raw scan existed with a different extension, prefer that filename.
    image_path = find_scan_image_path(scan)
    if image_path:
        command[-1] = image_path.name

    code, output = await run_subprocess_with_logs(command)
    structure_text = load_structure_text(scan)
    normalized = load_structure_json(scan)

    status = "ok" if code == 0 else "error"
    return JSONResponse(
        {
            "status": status,
            "log": output,
            "structure": structure_text,
            "normalized": normalized,
        }
    )


@app.post("/debug/fitch/run_grader")
async def run_fitch_grader(scan: str = Form(...)) -> Response:
    normalized = load_structure_json(scan)

    structure_copy = json.loads(json.dumps(normalized["lines"]))
    meta = normalized.get("meta", {})

    num_premises = meta.get("num_premises")
    if num_premises is None:
        num_premises = label_and_count_premises(structure_copy)

    proof_lines = structure_to_proof_lines(structure_copy)
    flattened = flatten_proof(proof_lines, [])
    conclusion_guess = meta.get("conclusion") or (flattened[-1].wffstr if flattened else "")

    result = check_proof(proof_lines, numprems=num_premises, conc=conclusion_guess)

    log_lines = [
        f"Scan: {scan}",
        f"Lines: {len(flattened)}",
        f"Premises (used): {num_premises}",
        f"Target conclusion: {conclusion_guess or '(empty)'}",
        f"Conclusion reached: {'yes' if result.conc_reached else 'no'}",
    ]

    if result.issues:
        log_lines.append("Issues:")
        log_lines.extend(result.issues)
    else:
        log_lines.append("No rule issues detected.")

    return JSONResponse(
        {
            "status": "ok",
            "log": "\n".join(log_lines),
        }
    )

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
        total = 0.0
        for p in session.grading_key.problems:
            score = wu.per_problem.get(p.id)
            if score is None:
                continue
            total += float(score)
        sid = wu.student_id
        if sid in totals:
            totals[sid] = max(totals[sid], total)
        else:
            totals[sid] = total
    return totals


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


def _unique_path(dest: Path) -> Path:
    if not dest.exists():
        return dest
    stem = dest.stem
    suffix = dest.suffix
    parent = dest.parent
    counter = 2
    while True:
        candidate = parent / f"{stem}_{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


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
                student_order.append(sid)

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


FLASHCARDS_STATE_FILENAME = "state.json"
FLASHCARDS_IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def _flashcards_session_dir(session_id: str) -> Path:
    return FLASHCARDS_DIR / session_id


def _flashcards_state_path(session_id: str) -> Path:
    return _flashcards_session_dir(session_id) / FLASHCARDS_STATE_FILENAME


def _load_flashcards_state(session_id: str) -> Dict[str, Any]:
    path = _flashcards_state_path(session_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Flashcards session not found.")
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _save_flashcards_state(session_id: str, state: Dict[str, Any]) -> None:
    path = _flashcards_state_path(session_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def _normalize_for_match(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"[^a-z0-9]", "", value)
    return value


def _guess_name_from_filename(stem: str, roster_index: Dict[str, str]) -> str:
    cleaned = re.sub(r"\d+", "", stem).replace("_", "")
    key = _normalize_for_match(cleaned)
    return roster_index.get(key, "")


def _safe_extract_zip(zf: zipfile.ZipFile, dest_dir: Path) -> List[Path]:
    extracted: List[Path] = []
    for entry in zf.infolist():
        if entry.is_dir():
            continue
        name = entry.filename.replace("\\", "/")
        if name.startswith("__MACOSX/"):
            continue
        target = Path(name)
        if target.is_absolute() or ".." in target.parts:
            continue
        out_path = dest_dir / target
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with zf.open(entry) as src, out_path.open("wb") as f:
            shutil.copyfileobj(src, f)
        extracted.append(out_path)
    return extracted


def _render_pdf_to_images(pdf_path: Path, output_dir: Path) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    images: List[Path] = []
    doc = fitz.open(str(pdf_path))
    for idx, page in enumerate(doc):
        matrix = fitz.Matrix(2, 2)
        pix = page.get_pixmap(matrix=matrix)
        filename = f"{pdf_path.stem}_p{idx + 1}.jpg"
        target = _unique_path(output_dir / filename)
        pix.save(str(target))
        images.append(target)
    doc.close()
    return images


def _extract_docx_assets(docx_path: Path, output_dir: Path) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    doc = Document(str(docx_path))
    text = "\n".join(p.text for p in doc.paragraphs if p.text).strip()
    images: List[Path] = []
    for rel in doc.part.rels.values():
        rel_type = getattr(rel, "reltype", "")
        if "image" not in rel_type:
            continue
        blob = rel.target_part.blob
        suffix = Path(rel.target_ref).suffix or ".jpg"
        filename = f"{docx_path.stem}_img_{len(images) + 1}{suffix}"
        target = _unique_path(output_dir / filename)
        target.write_bytes(blob)
        images.append(target)
    return {"text": text, "images": images}


def _slugify_name(name: str) -> str:
    slug = name.strip().lower()
    slug = re.sub(r"[^a-z0-9]+", "_", slug)
    slug = slug.strip("_")
    return slug or "student"


def _parse_bbox_value(value: str) -> Optional[List[float]]:
    if not value:
        return None
    parts = [p.strip() for p in value.split(",")]
    if len(parts) != 4:
        return None
    try:
        coords = [float(p) for p in parts]
    except ValueError:
        return None
    clamped = [max(0.0, min(1.0, c)) for c in coords]
    return clamped


@app.get("/flashcards", response_class=HTMLResponse)
async def flashcards_setup(request: Request, status: str = "", message: str = "") -> Response:
    roster_dir = DATA_DIR / "rosters"
    roster_dir.mkdir(parents=True, exist_ok=True)
    available_rosters = sorted(
        [p.name for p in roster_dir.iterdir() if p.is_file() and p.suffix.lower() == ".csv"],
        key=str.lower,
    )
    return templates.TemplateResponse(
        "flashcards_setup.html",
        {
            "request": request,
            "available_rosters": available_rosters,
            "rosters_dir": str(roster_dir),
            "status": status,
            "message": message,
        },
    )


@app.post("/flashcards/upload")
async def flashcards_upload(
    roster_choice: str = Form(""),
    roster_upload: Optional[UploadFile] = File(None),
    dataset_zip: UploadFile = File(...),
) -> Response:
    if not dataset_zip or not dataset_zip.filename:
        raise HTTPException(status_code=400, detail="Please upload a ZIP file.")

    roster_dir = DATA_DIR / "rosters"
    roster_dir.mkdir(parents=True, exist_ok=True)

    roster_path: Optional[Path] = None
    if roster_upload and roster_upload.filename:
        filename = Path(roster_upload.filename).name
        if Path(filename).suffix.lower() != ".csv":
            raise HTTPException(status_code=400, detail="Roster upload must be a CSV.")
        dest = _unique_path(roster_dir / filename)
        with dest.open("wb") as f:
            shutil.copyfileobj(roster_upload.file, f)
        roster_path = dest
    elif roster_choice:
        candidate = roster_dir / roster_choice
        if not candidate.exists():
            raise HTTPException(status_code=400, detail="Selected roster not found.")
        roster_path = candidate

    roster_entries: List[RosterEntry] = []
    if roster_path:
        roster_entries = load_roster(roster_path)

    roster_index = {
        _normalize_for_match(entry.name): entry.name for entry in roster_entries
    }

    session_id = f"{datetime.now().strftime('%Y-%m-%d_%H%M%S')}_{uuid4().hex[:6]}"
    session_dir = _flashcards_session_dir(session_id)
    sources_dir = session_dir / "sources"
    raw_dir = sources_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    zip_bytes = await dataset_zip.read()
    if not zip_bytes:
        raise HTTPException(status_code=400, detail="Uploaded ZIP was empty.")

    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            _safe_extract_zip(zf, raw_dir)
    except zipfile.BadZipFile:
        raise HTTPException(status_code=400, detail="Uploaded ZIP could not be read.")

    items: List[Dict[str, Any]] = []
    for file_path in sorted(raw_dir.rglob("*")):
        if not file_path.is_file():
            continue
        suffix = file_path.suffix.lower()
        if suffix == ".pdf":
            name_guess = _guess_name_from_filename(file_path.stem, roster_index)
            try:
                images = _render_pdf_to_images(file_path, sources_dir)
            except Exception:
                continue
            for img_path in images:
                items.append(
                    {
                        "id": len(items) + 1,
                        "source_image": img_path.relative_to(session_dir).as_posix(),
                        "source_text": "",
                        "name": name_guess,
                        "facts": "",
                        "bbox": None,
                        "skip": False,
                    }
                )
            continue
        if suffix == ".docx":
            name_guess = _guess_name_from_filename(file_path.stem, roster_index)
            try:
                assets = _extract_docx_assets(file_path, sources_dir)
            except Exception:
                continue
            text = assets.get("text", "")
            images = assets.get("images", [])
            for img_path in images:
                items.append(
                    {
                        "id": len(items) + 1,
                        "source_image": img_path.relative_to(session_dir).as_posix(),
                        "source_text": text,
                        "name": name_guess,
                        "facts": "",
                        "bbox": None,
                        "skip": False,
                    }
                )
            continue
        if suffix in FLASHCARDS_IMAGE_EXTS:
            name_guess = _guess_name_from_filename(file_path.stem, roster_index)
            target = _unique_path(sources_dir / file_path.name)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file_path, target)
            items.append(
                {
                    "id": len(items) + 1,
                    "source_image": target.relative_to(session_dir).as_posix(),
                    "source_text": "",
                    "name": name_guess,
                    "facts": "",
                    "bbox": None,
                    "skip": False,
                }
            )

    if not items:
        return RedirectResponse(
            url="/flashcards?status=error&message="
            + quote("No usable files found in the ZIP."),
            status_code=303,
        )

    state = {
        "session_id": session_id,
        "created_at": datetime.now().isoformat(),
        "roster_path": str(roster_path) if roster_path else None,
        "items": items,
    }
    _save_flashcards_state(session_id, state)

    return RedirectResponse(
        url=f"/flashcards/review?session={quote(session_id)}&index=0",
        status_code=303,
    )


@app.get("/flashcards/review", response_class=HTMLResponse)
async def flashcards_review(
    request: Request,
    session: str,
    index: int = 0,
) -> Response:
    state = _load_flashcards_state(session)
    items = state.get("items", [])
    session_dir = _flashcards_session_dir(session)
    dataset_path = session_dir / "dataset.json"
    dataset_url = None
    if dataset_path.exists():
        dataset_url = f"/flashcards_assets/{session}/dataset.json"

    if not items or index >= len(items):
        return templates.TemplateResponse(
            "flashcards_done.html",
            {
                "request": request,
                "session_id": session,
                "dataset_url": dataset_url,
                "dataset_count": 0,
            },
        )

    index = max(0, index)
    item = items[index]

    roster_names: List[str] = []
    roster_path = state.get("roster_path")
    if roster_path:
        roster_names = [entry.name for entry in load_roster(Path(roster_path))]

    source_image = item.get("source_image", "")
    source_url = (
        f"/flashcards_assets/{session}/{source_image}"
        if source_image
        else ""
    )
    bbox = item.get("bbox")
    bbox_value = ",".join(str(v) for v in bbox) if bbox else ""

    return templates.TemplateResponse(
        "flashcards_review.html",
        {
            "request": request,
            "session_id": session,
            "index": index,
            "total": len(items),
            "item": item,
            "source_url": source_url,
            "bbox_value": bbox_value,
            "roster_names": roster_names,
            "has_prev": index > 0,
            "has_next": index + 1 < len(items),
        },
    )


@app.post("/flashcards/review")
async def flashcards_review_submit(
    request: Request,
    session_id: str = Form(...),
    index: int = Form(...),
    name: str = Form(""),
    facts: str = Form(""),
    bbox: str = Form(""),
    action: str = Form("next"),
) -> Response:
    state = _load_flashcards_state(session_id)
    items = state.get("items", [])
    if index < 0 or index >= len(items):
        raise HTTPException(status_code=400, detail="Invalid flashcards index.")

    item = items[index]
    item["name"] = name.strip()
    item["facts"] = facts.strip()
    item["bbox"] = _parse_bbox_value(bbox)
    item["skip"] = action == "skip"

    _save_flashcards_state(session_id, state)

    if action == "export":
        return await flashcards_export(request, session_id)

    if action == "prev":
        target_index = max(0, index - 1)
    else:
        target_index = index + 1

    return RedirectResponse(
        url=f"/flashcards/review?session={quote(session_id)}&index={target_index}",
        status_code=303,
    )


@app.post("/flashcards/export", response_class=HTMLResponse)
async def flashcards_export(request: Request, session_id: str = Form(...)) -> Response:
    state = _load_flashcards_state(session_id)
    items = state.get("items", [])
    session_dir = _flashcards_session_dir(session_id)
    images_dir = session_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    name_counts: Dict[str, int] = {}
    dataset: List[Dict[str, str]] = []

    for item in items:
        if item.get("skip"):
            continue
        name = (item.get("name") or "").strip()
        bbox = item.get("bbox")
        if not name or not bbox:
            continue
        source_rel = item.get("source_image") or ""
        if not source_rel:
            continue
        source_path = session_dir / source_rel
        if not source_path.exists():
            continue

        with Image.open(source_path) as img:
            width, height = img.size
            x1 = int(round(bbox[0] * width))
            y1 = int(round(bbox[1] * height))
            x2 = int(round(bbox[2] * width))
            y2 = int(round(bbox[3] * height))
            x1 = max(0, min(width, x1))
            y1 = max(0, min(height, y1))
            x2 = max(0, min(width, x2))
            y2 = max(0, min(height, y2))
            if x2 <= x1 or y2 <= y1:
                continue
            crop = img.crop((x1, y1, x2, y2))

            base = _slugify_name(name)
            count = name_counts.get(base, 0) + 1
            name_counts[base] = count

            suffix = source_path.suffix.lower()
            if suffix == ".jpeg":
                suffix = ".jpg"
            if suffix not in {".jpg", ".png"}:
                suffix = ".jpg"

            filename = f"{base}{suffix}" if count == 1 else f"{base}_{count:02d}{suffix}"
            target = images_dir / filename
            crop.save(target)

        dataset.append(
            {
                "name": name,
                "photo_path": f"images/{filename}",
                "facts": (item.get("facts") or "").strip(),
            }
        )

    dataset_path = session_dir / "dataset.json"
    dataset_path.write_text(json.dumps(dataset, indent=2), encoding="utf-8")
    dataset_url = f"/flashcards_assets/{session_id}/dataset.json"

    return templates.TemplateResponse(
        "flashcards_done.html",
        {
            "request": request,
            "session_id": session_id,
            "dataset_url": dataset_url,
            "dataset_count": len(dataset),
        },
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
    total_max = sum(p.points for p in problems) if problems else 0.0

    roster_map = {r.student_id: r.name for r in session.roster}
    rows_by_student: Dict[str, Dict] = {}

    for wu in session.work_units:
        if not wu.student_id:
            continue
        per_problem_scores = {}
        per_problem_comments = {}
        total = 0.0
        for p in problems:
            score = wu.per_problem.get(p.id)
            per_problem_scores[p.id] = score
            per_problem_comments[p.id] = wu.per_problem_comments.get(p.id)
            if score is None:
                continue
            total += float(score)

        sid = wu.student_id
        existing = rows_by_student.get(sid)
        if existing is None or total > existing["total_score"]:
            rows_by_student[sid] = {
                "student_id": sid,
                "student_name": roster_map.get(sid, sid),
                "per_problem_scores": per_problem_scores,
                "per_problem_comments": per_problem_comments,
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

    # Aggregate scores per student_id, taking the max over all their WorkUnits
    totals: Dict[str, float] = {}
    for wu in session.work_units:
        if not wu.student_id:
            continue

        # Compute this WorkUnit's total
        total = 0.0
        for p in session.grading_key.problems:
            score = wu.per_problem.get(p.id)
            if score is None:
                # For now: treat missing as 0 (you can tighten this later)
                continue
            total += float(score)

        # Take the max across all WorkUnits for this student
        sid = wu.student_id
        if sid in totals:
            totals[sid] = max(totals[sid], total)
        else:
            totals[sid] = total

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
    total_max = sum(p.points for p in problems) if problems else 0.0

    roster_map = {r.student_id: r.name for r in session.roster}
    id_to_slug = roster_lookup(session)

    # Choose the max-total WorkUnit per student_id (matches export_scores behavior)
    rows_by_student: Dict[str, Dict] = {}
    for wu in session.work_units:
        if not wu.student_id:
            continue
        per_problem_scores = {}
        per_problem_comments = {}
        total = 0.0
        for p in problems:
            score = wu.per_problem.get(p.id)
            per_problem_scores[p.id] = score
            per_problem_comments[p.id] = wu.per_problem_comments.get(p.id)
            if score is None:
                continue
            total += float(score)

        sid = wu.student_id
        existing = rows_by_student.get(sid)
        if existing is None or total > existing["total_score"]:
            rows_by_student[sid] = {
                "student_id": sid,
                "student_name": roster_map.get(sid, sid),
                "per_problem_scores": per_problem_scores,
                "per_problem_comments": per_problem_comments,
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
            lines.append("")
            lines.append("Rubric + Scores")
            for p in problems:
                score = row["per_problem_scores"].get(p.id)
                comment = row["per_problem_comments"].get(p.id)
                score_display = "" if score is None else str(score)
                lines.append(f"- {p.id} ({p.label}) [{p.points} pts]: {score_display}")
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


def _latest_canvas_original_zip() -> Path:
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
    return zips[0]


def _build_best_rows_by_student(session: Session) -> Dict[str, Dict[str, Any]]:
    grading_key = session.grading_key
    problems = grading_key.problems
    total_max = sum(p.points for p in problems) if problems else 0.0
    roster_map = {r.student_id: r.name for r in session.roster}

    rows_by_student: Dict[str, Dict[str, Any]] = {}
    for wu in session.work_units:
        if not wu.student_id:
            continue
        per_problem_scores: Dict[str, Optional[float]] = {}
        per_problem_comments: Dict[str, Optional[str]] = {}
        total = 0.0
        for p in problems:
            score = wu.per_problem.get(p.id)
            per_problem_scores[p.id] = score
            per_problem_comments[p.id] = wu.per_problem_comments.get(p.id)
            if score is not None:
                total += float(score)

        sid = wu.student_id
        existing = rows_by_student.get(sid)
        if existing is None or total > existing["total_score"]:
            rows_by_student[sid] = {
                "student_id": sid,
                "student_name": roster_map.get(sid, sid),
                "per_problem_scores": per_problem_scores,
                "per_problem_comments": per_problem_comments,
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
    lines.append("")
    lines.append("Rubric + Scores")
    for p in problems:
        score = row["per_problem_scores"].get(p.id)
        comment = row["per_problem_comments"].get(p.id)
        score_display = "" if score is None else str(score)
        lines.append(f"- {p.id} ({p.label}) [{p.points} pts]: {score_display}")
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
def export_canvas_reupload_feedback_zip():
    """
    Build a Canvas "re-upload submissions" ZIP using original Canvas filenames.
    Content is replaced with per-student feedback rendered as PDF/image, preserving
    each file's original name and extension.
    """
    session = get_session()
    grading_key = session.grading_key
    problems = grading_key.problems
    total_max = sum(p.points for p in problems) if problems else 0.0
    rows_by_student = _build_best_rows_by_student(session)

    roster_index, collisions = _build_roster_prefix_index(session.roster)
    if collisions:
        raise HTTPException(
            status_code=400,
            detail="Roster collisions prevent Canvas filename matching.",
        )

    source_zip = _latest_canvas_original_zip()
    out_buf = io.BytesIO()
    matched = 0
    total = 0

    with zipfile.ZipFile(source_zip, "r") as src, zipfile.ZipFile(
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
        f"[CANVAS REUPLOAD] Source={source_zip.name}, files={total}, "
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


from fastapi import Query

def _compute_segments_data(session: Session, spec, cache_path: Path) -> List[Dict]:
    problem_id = spec.id
    print(f"[CACHE MISS] Computing fresh segmentation for {problem_id}...")
    client = LMStudioQwenClient()
    segments_data: List[Dict] = []

    for i, wu in enumerate(session.work_units):
        print(f"  > Processing Unit {wu.unit_id} ({i+1}/{len(session.work_units)})...")
        best_seg_dict = None

        for idx, fname in enumerate(wu.page_filenames):
            if not _is_image_filename(fname):
                continue
            image_path = RAW_SCANS_DIR / fname

            # --- THE SLOW VLM CALL ---
            seg = segment_page_for_problem(
                image_path=image_path,
                problem_id=problem_id,
                problem_description=spec.label,
                client=client,
            )

            if seg.present and seg.bbox:
                # Found it! Save as dict for JSON serialization
                best_seg_dict = {
                    "unit_id": wu.unit_id,
                    "page_filename": seg.page_filename,
                    "page_index": idx,
                    "bbox": seg.bbox,
                    "present": seg.present,
                    "comment": seg.comment,
                    "confidence": seg.confidence,
                }
                break  # Stop checking other pages for this student

        # Add result (or placeholder if not found)
        segments_data.append(
            best_seg_dict
            or {
                "unit_id": wu.unit_id,
                "present": False,
                "bbox": None,
            }
        )

    print(f"[CACHE SAVE] Writing to {cache_path}")
    GRADES_DIR.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as f:
        json.dump(segments_data, f, indent=2)

    return segments_data


def _is_normalized_bbox(value: Any) -> bool:
    if not isinstance(value, list) or len(value) != 4:
        return False
    try:
        x1, y1, x2, y2 = (float(v) for v in value)
    except (TypeError, ValueError):
        return False
    if any(not math.isfinite(v) for v in (x1, y1, x2, y2)):
        return False
    return 0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0


def _segments_cache_matches_session(session: Session, segments_data: Any) -> bool:
    if not isinstance(segments_data, list):
        return False

    unit_to_pages: Dict[int, set[str]] = {
        wu.unit_id: set(wu.page_filenames) for wu in session.work_units
    }
    expected_unit_ids = set(unit_to_pages.keys())
    seen_unit_ids: set[int] = set()

    for row in segments_data:
        if not isinstance(row, dict):
            return False
        uid = row.get("unit_id")
        if uid not in expected_unit_ids or uid in seen_unit_ids:
            return False
        seen_unit_ids.add(uid)

        present = bool(row.get("present", False))
        bbox = row.get("bbox")
        page_filename = row.get("page_filename")

        if not present:
            continue

        if not _is_normalized_bbox(bbox):
            return False
        if not isinstance(page_filename, str):
            return False
        if page_filename not in unit_to_pages[uid]:
            return False

    return seen_unit_ids == expected_unit_ids


def _clear_segment_caches() -> None:
    if not GRADES_DIR.exists():
        return
    for cache_path in GRADES_DIR.glob("segments_*.json"):
        try:
            cache_path.unlink()
        except OSError:
            pass


def ensure_segments_cached(
    session: Session,
    spec,
    refresh: bool = False,
    blocking: bool = True,
) -> Optional[List[Dict]]:
    problem_id = spec.id
    cache_path = GRADES_DIR / f"segments_{problem_id}.json"
    segments_data: Optional[List[Dict]] = None

    if cache_path.exists() and not refresh:
        print(f"[CACHE HIT] Loading segmentation from {cache_path}")
        try:
            with cache_path.open("r", encoding="utf-8") as f:
                segments_data = json.load(f)
            if not _segments_cache_matches_session(session, segments_data):
                print(f"[CACHE INVALID] {cache_path} does not match current session; recomputing.")
                segments_data = None
        except Exception as e:
            print(f"[CACHE ERROR] Could not read cache: {e}. Recomputing.")
            segments_data = None

    if segments_data is None:
        acquired = _segmentation_lock.acquire(blocking=blocking)
        if not acquired:
            return None
        try:
            if cache_path.exists() and not refresh:
                try:
                    with cache_path.open("r", encoding="utf-8") as f:
                        segments_data = json.load(f)
                    if not _segments_cache_matches_session(session, segments_data):
                        print(f"[CACHE INVALID] {cache_path} does not match current session; recomputing.")
                        segments_data = None
                except Exception as e:
                    print(f"[CACHE ERROR] Could not read cache: {e}. Recomputing.")
                    segments_data = None

            if segments_data is None:
                segments_data = _compute_segments_data(session, spec, cache_path)
        finally:
            _segmentation_lock.release()

    return segments_data


@app.post("/problem/{problem_id}/prefetch")
def prefetch_problem(problem_id: str):
    session = get_session()
    spec = next((p for p in session.grading_key.problems if p.id == problem_id), None)
    if spec is None:
        raise HTTPException(status_code=404, detail=f"Unknown problem id {problem_id!r}")

    segments_data = ensure_segments_cached(session, spec, refresh=False, blocking=False)
    if segments_data is None:
        return Response(content="Busy", status_code=429)

    return {"status": "cached", "segments": len(segments_data)}


@app.get("/problem/{problem_id}/strips", response_class=HTMLResponse)
def get_problem_strips(
    request: Request,
    problem_id: str,
    refresh: int = 0,
):
    session = get_session()
    grading_key = session.grading_key

    # 1. Find the problem definition
    spec = next((p for p in grading_key.problems if p.id == problem_id), None)
    if spec is None:
        raise HTTPException(status_code=404, detail=f"Unknown problem id {problem_id!r}")

    segments_data = ensure_segments_cached(
        session=session,
        spec=spec,
        refresh=bool(refresh),
        blocking=True,
    )

    if segments_data is None:
        raise HTTPException(status_code=503, detail="Segmentation currently busy")

    # --- STEP B: Build UI Objects (Fast) ---
    
    # Index data by unit_id for O(1) lookup
    segments_map = {d["unit_id"]: d for d in segments_data}
    
    strips = []
    
    def lookup_student_name(student_id):
        if not student_id: return "(unknown)"
        for r in session.roster:
            if r.student_id == student_id: return r.name
        return student_id

    for wu in session.work_units:
        seg_data = segments_map.get(wu.unit_id)
        
        # Reconstruct the QuestionSegment object for the template
        seg_obj = None
        if seg_data and seg_data.get("present"):
            seg_obj = QuestionSegment(
                problem_id=problem_id,
                page_filename=seg_data["page_filename"],
                page_index=seg_data.get("page_index", 0),
                bbox=seg_data["bbox"],
                present=True,
                comment=seg_data.get("comment"),
                confidence=seg_data.get("confidence", 0.0)
            )

        page_options = _build_page_options(wu.page_filenames)
        display_filename = _select_display_filename(
            wu.page_filenames,
            seg_obj.page_filename if seg_obj else None,
        )
        display_url = f"/scans/{quote(display_filename)}" if display_filename else None

        strips.append({
            "work_unit": wu,
            "student_name": lookup_student_name(wu.student_id),
            "segment": seg_obj,
            "display_filename": display_filename,
            "display_url": display_url,
            "page_options": page_options,
        })

    # --- STEP C: Handling the URL (The Fix) ---
    # If we just forced a refresh, redirect to the clean URL so 
    # hitting "Reload" later doesn't trigger a re-compute.
    if refresh:
        return RedirectResponse(
            url=f"/problem/{problem_id}/strips", 
            status_code=303
        )

    problem_ids = [p.id for p in grading_key.problems]
    try:
        idx = problem_ids.index(problem_id)
    except ValueError:
        idx = -1

    next_problem_id = None
    if idx != -1 and idx < len(problem_ids) - 1:
        next_problem_id = problem_ids[idx + 1]

    # Standard render for normal (cached) views
    return templates.TemplateResponse(
        "problem_strips.html",
        {
            "request": request,
            "problem_id": problem_id,
            "problem_label": spec.label,
            "strips": strips,
            # NEW:
            "max_points": spec.points,
            "problem_note": session.problem_notes.get(problem_id, ""),
            "problem_type": session.problem_types.get(problem_id),
            "problem_type_options": PROBLEM_TYPE_OPTIONS,
            "next_problem_id": next_problem_id,
        },
    )

@app.post("/problem/{problem_id}/strips", response_class=HTMLResponse)
async def post_problem_strips(
    request: Request,
    problem_id: str,
):
    form = await request.form()
    action = form.get("action")
    if not action:
        action = form.get("action_mode", "save_next")

    with _session_lock:
        session = get_session()
        
        # 1. Validation (Existing)
        spec = next((p for p in session.grading_key.problems if p.id == problem_id), None)
        if spec is None:
            raise HTTPException(status_code=404, detail=f"Unknown problem id {problem_id!r}")

        # --- NEW: Handle Bounding Box Updates ---
        # We check if any bboxes were modified before processing scores
        updates_map = {} # unit_id -> [x1, y1, x2, y2]
        page_updates = {} # unit_id -> filename
        
        for wu in session.work_units:
            bbox_raw = form.get(f"bbox_update_{wu.unit_id}")
            if bbox_raw and bbox_raw.strip():
                try:
                    # Parse "[0.1, 0.2, 0.5, 0.6]"
                    new_bbox = json.loads(bbox_raw)
                    if isinstance(new_bbox, list) and len(new_bbox) == 4:
                        updates_map[wu.unit_id] = new_bbox
                except json.JSONDecodeError:
                    pass # Ignore malformed data

            page_raw = form.get(f"page_update_{wu.unit_id}")
            if page_raw and page_raw.strip():
                page_updates[wu.unit_id] = page_raw.strip()

        # If we have updates, load the segment cache, apply them, and save
        if updates_map or page_updates:
            cache_path = GRADES_DIR / f"segments_{problem_id}.json"
            if cache_path.exists():
                try:
                    # Read existing
                    with cache_path.open("r", encoding="utf-8") as f:
                        segments_data = json.load(f)

                    # Update matched records
                    # Note: segments_data is a list of dicts. 
                    # We iterate over it to find matching unit_ids.
                    updated_count = 0
                    unit_pages = {wu.unit_id: wu.page_filenames for wu in session.work_units}
                    for seg in segments_data:
                        uid = seg.get("unit_id")
                        if uid in page_updates:
                            filename = page_updates[uid]
                            page_list = unit_pages.get(uid) or []
                            if filename in page_list:
                                seg["page_filename"] = filename
                                seg["page_index"] = page_list.index(filename)
                                updated_count += 1
                        if uid in updates_map:
                            seg["bbox"] = updates_map[uid]
                            seg["present"] = True # If we manually cropped it, it's definitely present
                            # Optional: mark confidence as 1.0 since human verified
                            seg["confidence"] = 1.0 
                            updated_count += 1

                    # Write back
                    with cache_path.open("w", encoding="utf-8") as f:
                        json.dump(segments_data, f, indent=2)

                    print(f"Updated {updated_count} manual crops for problem {problem_id}")
                except Exception as e:
                    print(f"Error updating segment cache: {e}")

        # --- Existing Score Saving Logic ---
        for wu in session.work_units:
            field_name = f"score_{wu.unit_id}"
            raw = form.get(field_name)

            if raw is None:
                continue

            raw = raw.strip()
            if raw == "":
                wu.per_problem[problem_id] = None
            else:
                try:
                    wu.per_problem[problem_id] = float(raw)
                except ValueError:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Invalid score for {problem_id} on unit {wu.unit_id}: {raw!r}",
                    )
        # --- Per-Problem Comment Saving Logic ---
        for wu in session.work_units:
            field_name = f"comment_{wu.unit_id}"
            raw = form.get(field_name)
            if raw is None or raw.strip() == "":
                wu.per_problem_comments[problem_id] = None
            else:
                wu.per_problem_comments[problem_id] = raw

        # --- Existing Note Logic ---
        note_raw = form.get("problem_note")
        if note_raw is not None:
            note = note_raw.strip()
            if note:
                session.problem_notes[problem_id] = note
            else:
                session.problem_notes.pop(problem_id, None)

        type_raw = form.get("problem_type")
        if type_raw is not None:
            selected = type_raw.strip()
            if not selected or selected == "none":
                session.problem_types.pop(problem_id, None)
            else:
                session.problem_types[problem_id] = selected

        # --- Existing Navigation Logic ---
        problem_ids = [p.id for p in session.grading_key.problems]
        try:
            idx = problem_ids.index(problem_id)
        except ValueError:
            idx = 0

        target_problem_id = problem_id

        if action == "save_next":
            if idx < len(problem_ids) - 1:
                target_problem_id = problem_ids[idx + 1]
            else:
                target_problem_id = None
        elif action == "save_back":
            if idx > 0:
                target_problem_id = problem_ids[idx - 1]
        elif action == "finish":
            target_problem_id = None

        persist_session()

    if target_problem_id is None:
        return RedirectResponse(url="/finish", status_code=303)

    return RedirectResponse(
        url=f"/problem/{target_problem_id}/strips",
        status_code=303,
    )


@app.post("/problem/{problem_id}/type")
async def post_problem_type(
    request: Request,
    problem_id: str,
):
    form = await request.form()
    type_raw = form.get("problem_type")

    with _session_lock:
        session = get_session()
        spec = next((p for p in session.grading_key.problems if p.id == problem_id), None)
        if spec is None:
            raise HTTPException(status_code=404, detail=f"Unknown problem id {problem_id!r}")

        selected = type_raw.strip() if type_raw else ""
        if not selected or selected == "none":
            session.problem_types.pop(problem_id, None)
            stored = None
        else:
            session.problem_types[problem_id] = selected
            stored = selected

        persist_session()

    return {"status": "ok", "problem_type": stored}
