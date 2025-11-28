from __future__ import annotations

from typing import Dict, List, Optional
import threading
import io
import csv
import json
from pathlib import Path
import zipfile
from urllib.parse import quote

from fastapi import FastAPI, HTTPException, Response, Request, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from .storage import load_session, save_session, RAW_SCANS_DIR, GRADES_DIR, archive_current_data
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
)
from .auto_group_core import (
    run_name_recognition,
    group_pages,
    build_roster_index,
    SubmissionGroup,
)


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


AUTO_GROUPED_DIR = DATA_DIR / "auto_grouped"

LMSTUDIO_BASE_URL = "http://127.0.0.1:1234/v1"
LMSTUDIO_MODEL_ID = "qwen/qwen3-vl-4b"  # or whatever you’re actually using

NAME_REVIEW_CACHE_DIR = DATA_DIR / "name_review_cache"
NAME_REVIEW_CACHE_DIR.mkdir(parents=True, exist_ok=True)

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


_session_lock = threading.Lock()
_session: Optional[Session] = None


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
        client = LMStudioQwenClient(
            base_url=LMSTUDIO_BASE_URL,
            model=LMSTUDIO_MODEL_ID,
        )
        predictions = []
        for path in paths:
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
        pinned_note=None
    )

    # 7. Save Session to Disk
    # This creates data/grades/session.json
    global _session
    _session = session
    save_session(session, DEFAULT_SESSION_PATH)
    print(f"[name_review] Session initialized with {len(work_units)} units.")

    # 8. Redirect to Grading
    # Pick the first problem ID
    first_problem = grading_key.problems[0].id if grading_key.problems else "Q1"
    
    return RedirectResponse(
        url=f"/problem/{first_problem}/strips",
        status_code=303
    )
from fastapi import Query

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

    # 2. Setup Cache Path
    # This resolves to: data/grades/segments_Q1.json
    cache_path = GRADES_DIR / f"segments_{problem_id}.json"
    
    segments_data = None
    
    # --- STEP A: Get Segmentation Data (Cache or Compute) ---
    
    # If file exists AND we aren't forcing a refresh...
    if cache_path.exists() and not refresh:
        print(f"[CACHE HIT] Loading segmentation from {cache_path}")
        try:
            with cache_path.open("r", encoding="utf-8") as f:
                segments_data = json.load(f)
        except Exception as e:
            print(f"[CACHE ERROR] Could not read cache: {e}. Recomputing.")
            segments_data = None

    if segments_data is None:
        print(f"[CACHE MISS] Computing fresh segmentation for {problem_id}...")
        client = LMStudioQwenClient()
        segments_data = []
        
        for i, wu in enumerate(session.work_units):
            print(f"  > Processing Unit {wu.unit_id} ({i+1}/{len(session.work_units)})...")
            best_seg_dict = None

            for idx, fname in enumerate(wu.page_filenames):
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
                    break # Stop checking other pages for this student
            
            # Add result (or placeholder if not found)
            segments_data.append(best_seg_dict or {
                "unit_id": wu.unit_id,
                "present": False, 
                "bbox": None
            })

        # Save to disk
        print(f"[CACHE SAVE] Writing to {cache_path}")
        GRADES_DIR.mkdir(parents=True, exist_ok=True)
        with cache_path.open("w", encoding="utf-8") as f:
            json.dump(segments_data, f, indent=2)

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

        strips.append({
            "work_unit": wu,
            "student_name": lookup_student_name(wu.student_id),
            "segment": seg_obj
        })

    # --- STEP C: Handling the URL (The Fix) ---
    # If we just forced a refresh, redirect to the clean URL so 
    # hitting "Reload" later doesn't trigger a re-compute.
    if refresh:
        return RedirectResponse(
            url=f"/problem/{problem_id}/strips", 
            status_code=303
        )

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
        },
    )

@app.post("/problem/{problem_id}/strips", response_class=HTMLResponse)
async def post_problem_strips(
    request: Request,
    problem_id: str,
):
    """
    Handle form submission from the per-problem strip view.

    Expected form fields:
      - score_<unit_id> for each WorkUnit shown (e.g. score_1, score_2, ...)
      - problem_note (optional free-text)
      - action = "save_next" | "save_back" | "save"
    """
    form = await request.form()
    action = form.get("action", "save_next")  # default: move forward

    with _session_lock:
        session = get_session()

        # Sanity-check that the problem exists
        spec = next((p for p in session.grading_key.problems if p.id == problem_id), None)
        if spec is None:
            raise HTTPException(
                status_code=404,
                detail=f"Unknown problem id {problem_id!r}",
            )

        # --- Update per-problem scores for this problem across WorkUnits ---
        for wu in session.work_units:
            field_name = f"score_{wu.unit_id}"
            raw = form.get(field_name)

            if raw is None:
                continue  # no input for this unit

            raw = raw.strip()
            if raw == "":
                wu.per_problem[problem_id] = None   # blank = clear
            else:
                try:
                    wu.per_problem[problem_id] = float(raw)
                except ValueError:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Invalid score for {problem_id} on unit {wu.unit_id}: {raw!r}",
                    )

        # --- Update per-problem note ---
        note_raw = form.get("problem_note")
        if note_raw is not None:
            note = note_raw.strip()
            if note:
                session.problem_notes[problem_id] = note
            else:
                session.problem_notes.pop(problem_id, None)

        # --- Decide which problem to show next ---
        problem_ids = [p.id for p in session.grading_key.problems]
        try:
            idx = problem_ids.index(problem_id)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Problem {problem_id!r} not found in grading key order.",
            )

        target_problem_id = problem_id

        if action == "save_next":
            if idx < len(problem_ids) - 1:
                target_problem_id = problem_ids[idx + 1]
        elif action == "save_back":
            if idx > 0:
                target_problem_id = problem_ids[idx - 1]
        else:
            target_problem_id = problem_id  # stay on page

        # Persist session.json
        persist_session()

    # Redirect to GET so refresh doesn't re-post
    return RedirectResponse(
        url=f"/problem/{target_problem_id}/strips",
        status_code=303,
    )
