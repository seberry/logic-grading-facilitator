from __future__ import annotations

from typing import Any, Dict, List, Optional
import csv
import io
import json
import math
import re
import shutil
import textwrap
import zipfile
from pathlib import Path
from uuid import uuid4

import fitz  # type: ignore
from docx import Document
from fastapi import HTTPException
from PIL import Image, ImageDraw, ImageFont

from .helpers import _unique_path
from .models import (
    ClassworkBatchState,
    ClassworkEntry,
    ClassworkFileRef,
    ClassworkModeState,
    RosterEntry,
    Session,
)
from .session_state import get_session
from .storage import DATA_DIR
from .utils import clean_html_text


GRADEBOOK_MERGE_TMP_DIR = DATA_DIR / "gradebook_merge" / "tmp"
CLASSWORK_FILES_DIR = DATA_DIR / "classwork" / "files"
CLASSWORK_EXPORTS_DIR = DATA_DIR / "classwork" / "exports"
CLASSWORK_FILES_DIR.mkdir(parents=True, exist_ok=True)
CLASSWORK_EXPORTS_DIR.mkdir(parents=True, exist_ok=True)


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


def _normalize_headers(fieldnames: List[str]) -> Dict[str, str]:
    return {name.strip().lower(): name for name in fieldnames}


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
    base_name = Path(zip_entry_name).name
    prefix_key = _filename_prefix_key(base_name)
    entry = roster_index.get(prefix_key)
    if entry is not None:
        return entry

    search_text = (zip_entry_name or "").replace("\\", "/").lower()
    stem = Path(search_text).stem
    tokens = [t for t in re.split(r"[^a-z0-9]+", search_text) if t]
    for token in tokens:
        sid_entry = sid_index.get(token)
        if sid_entry is not None:
            return sid_entry

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


def _classwork_make_stored_name(setup_id: str, original_name: str) -> str:
    """
    Generate a filesystem-safe app-owned storage name.

    We preserve the exact original Canvas/ZIP filename separately in
    ClassworkFileRef.original_filename for display and re-export. The stored
    filename is only for local disk paths and /classwork_files URLs.
    """
    ext = Path(original_name or "").suffix.lower()
    return f"{setup_id}_{uuid4().hex}{ext}"


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
    if not name:
        return ""
    normalized = name.lower().strip()
    normalized = re.sub(r"\s*\(\d+\)\s*$", "", normalized)
    normalized = re.sub(r"[^a-z0-9\s\-_]", "", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def _find_matching_column(
    canvas_assignment_name: str,
    gradebook_fieldnames: List[str],
    id_col: str,
) -> Optional[str]:
    normalized_canvas = _normalize_for_matching(canvas_assignment_name)
    excluded = {id_col.lower(), "student", "current score", "final score", "current grade", "final grade"}

    candidates = [col for col in gradebook_fieldnames if col.lower() not in excluded]
    if not candidates:
        return None

    for col in candidates:
        if _normalize_for_matching(col) == normalized_canvas:
            return col
    return None


def _classwork_build_mode_from_zip(
    *,
    zip_bytes: bytes,
    assignment_id: str,
    max_points: float,
    active_roster: List[RosterEntry],
    gradebook_token: str,
    id_col: str,
    gradebook_column: Optional[str] = None,
) -> ClassworkModeState:
    setup_id = uuid4().hex
    student_files: Dict[str, List[ClassworkFileRef]] = {}
    unmatched_files: List[str] = []

    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            roster_index, collisions = _build_roster_prefix_index(active_roster)
            if collisions:
                detail_lines = ["Duplicate gradebook name keys detected:"]
                for key, entries in collisions.items():
                    names = ", ".join(e.name for e in entries)
                    detail_lines.append(f"{key}: {names}")
                raise HTTPException(status_code=400, detail="; ".join(detail_lines))
            sid_index = {
                _normalize_match_key(r.student_id): r
                for r in active_roster
                if _normalize_match_key(r.student_id)
            }

            for info in sorted(zf.infolist(), key=lambda item: item.filename.lower()):
                if info.is_dir() or info.filename.startswith("__MACOSX/"):
                    continue
                original_name = Path(info.filename).name
                if not original_name:
                    continue

                roster_entry = _classwork_match_file_to_roster(
                    info.filename,
                    roster_index=roster_index,
                    sid_index=sid_index,
                )
                if roster_entry is None:
                    unmatched_files.append(info.filename)
                    continue

                ext = Path(original_name).suffix.lower()
                payload = zf.read(info)
                stored_name = _classwork_make_stored_name(setup_id, original_name)
                stored_path = _unique_path(CLASSWORK_FILES_DIR / stored_name)
                stored_path.parent.mkdir(parents=True, exist_ok=True)
                stored_path.write_bytes(payload)

                refs = student_files.setdefault(roster_entry.student_id, [])
                refs.append(
                    ClassworkFileRef(
                        stored_filename=stored_path.name,
                        original_filename=original_name,
                        ext=ext,
                    )
                )
    except zipfile.BadZipFile:
        raise HTTPException(status_code=400, detail="Uploaded ZIP could not be read.")

    assignment_clean = _sanitize_assignment_id(assignment_id)
    entries: List[ClassworkEntry] = []
    for roster_entry in active_roster:
        files = student_files.get(roster_entry.student_id, [])
        has_submission = bool(files)
        score = float(max_points) if has_submission else 0.0
        entries.append(
            ClassworkEntry(
                student_id=roster_entry.student_id,
                student_name=roster_entry.name,
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
        unmatched_files=sorted(set(unmatched_files)),
    )


def _classwork_build_mode_from_downloads(
    *,
    student_files: Dict[str, List[Path]],
    assignment_id: str,
    assignment_name: str,
    max_points: float,
    active_roster: List[RosterEntry],
    gradebook_token: str,
    id_col: str,
    gradebook_column: Optional[str] = None,
    gradebook_fieldnames: Optional[List[str]] = None,
) -> ClassworkModeState:
    setup_id = uuid4().hex
    student_files_ref: Dict[str, List[ClassworkFileRef]] = {}
    roster_by_id = {r.student_id: r.name for r in active_roster}

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
            print(
                f"[COLUMN MATCH WARNING] No match found for assignment '{assignment_name}'. "
                "Manual column selection required."
            )

    for student_id, file_paths in student_files.items():
        if not file_paths:
            continue

        refs = []
        for file_path in file_paths:
            if not file_path.exists():
                continue

            original_name = file_path.name
            ext = Path(original_name).suffix.lower()
            stored_name = _classwork_make_stored_name(setup_id, original_name)
            stored_path = CLASSWORK_FILES_DIR / stored_name
            stored_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file_path, stored_path)

            refs.append(
                ClassworkFileRef(
                    stored_filename=stored_path.name,
                    original_filename=original_name,
                    ext=ext,
                )
            )

        student_files_ref[student_id] = refs

    assignment_clean = _sanitize_assignment_id(assignment_id)
    entries: List[ClassworkEntry] = []
    for roster_entry in active_roster:
        files = student_files_ref.get(roster_entry.student_id, [])
        has_submission = bool(files)
        score = float(max_points) if has_submission else 0.0
        entries.append(
            ClassworkEntry(
                student_id=roster_entry.student_id,
                student_name=roster_entry.name,
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
        unmatched_files=[],
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
            return clean_html_text(raw.decode(encoding))
        except UnicodeDecodeError:
            continue
    raise HTTPException(status_code=400, detail="Could not decode text preview for this file.")


def _normalize_canvas_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (value or "").lower())


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
