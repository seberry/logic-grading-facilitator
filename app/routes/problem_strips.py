from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import quote

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from app.helpers import _build_page_options, _is_image_filename, _select_display_filename
from app.models import Session
from app.segmentation_core import QuestionSegment, segment_page_for_problem
from app.session_state import _segmentation_lock, _session_lock, get_session, persist_session
from app.storage import GRADES_DIR, RAW_SCANS_DIR
from app.vlm_client import LMStudioQwenClient

router = APIRouter()

TEMPLATES_DIR = Path(__file__).resolve().parents[1] / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

PROBLEM_TYPE_OPTIONS = [
    {"id": "none", "label": "None"},
    {"id": "fitch_proof", "label": "Fitch proof"},
    {"id": "modal_proof", "label": "Modal proof"},
    {"id": "kripke_model", "label": "Kripke model"},
    {"id": "fol_model", "label": "FOL model"},
]


def _effective_problem_type(session: Session, problem_id: str, spec) -> Optional[str]:
    if getattr(spec, "problem_type", None):
        return spec.problem_type
    return session.problem_types.get(problem_id)


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

            seg = segment_page_for_problem(
                image_path=image_path,
                problem_id=problem_id,
                problem_description=spec.label,
                client=client,
            )

            if seg.present and seg.bbox:
                best_seg_dict = {
                    "unit_id": wu.unit_id,
                    "page_filename": seg.page_filename,
                    "page_index": idx,
                    "bbox": seg.bbox,
                    "present": seg.present,
                    "comment": seg.comment,
                    "confidence": seg.confidence,
                }
                break

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
        except Exception as exc:
            print(f"[CACHE ERROR] Could not read cache: {exc}. Recomputing.")
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
                except Exception as exc:
                    print(f"[CACHE ERROR] Could not read cache: {exc}. Recomputing.")
                    segments_data = None

            if segments_data is None:
                segments_data = _compute_segments_data(session, spec, cache_path)
        finally:
            _segmentation_lock.release()

    return segments_data


@router.post("/problem/{problem_id}/prefetch")
def prefetch_problem(problem_id: str):
    session = get_session()
    spec = next((p for p in session.grading_key.problems if p.id == problem_id), None)
    if spec is None:
        raise HTTPException(status_code=404, detail=f"Unknown problem id {problem_id!r}")

    segments_data = ensure_segments_cached(session, spec, refresh=False, blocking=False)
    if segments_data is None:
        return Response(content="Busy", status_code=429)

    return {"status": "cached", "segments": len(segments_data)}


@router.get("/problem/{problem_id}/strips", response_class=HTMLResponse)
def get_problem_strips(
    request: Request,
    problem_id: str,
    refresh: int = 0,
):
    session = get_session()
    grading_key = session.grading_key

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

    segments_map = {d["unit_id"]: d for d in segments_data}
    strips = []

    def lookup_student_name(student_id):
        if not student_id:
            return "(unknown)"
        for r in session.roster:
            if r.student_id == student_id:
                return r.name
        return student_id

    for wu in session.work_units:
        seg_data = segments_map.get(wu.unit_id)
        seg_obj = None
        if seg_data and seg_data.get("present"):
            seg_obj = QuestionSegment(
                problem_id=problem_id,
                page_filename=seg_data["page_filename"],
                page_index=seg_data.get("page_index", 0),
                bbox=seg_data["bbox"],
                present=True,
                comment=seg_data.get("comment"),
                confidence=seg_data.get("confidence", 0.0),
            )

        page_options = _build_page_options(wu.page_filenames)
        display_filename = _select_display_filename(
            wu.page_filenames,
            seg_obj.page_filename if seg_obj else None,
        )
        display_url = f"/scans/{quote(display_filename)}" if display_filename else None

        strips.append(
            {
                "work_unit": wu,
                "student_name": lookup_student_name(wu.student_id),
                "segment": seg_obj,
                "display_filename": display_filename,
                "display_url": display_url,
                "page_options": page_options,
            }
        )

    if refresh:
        return RedirectResponse(
            url=f"/problem/{problem_id}/strips",
            status_code=303,
        )

    problem_ids = [p.id for p in grading_key.problems]
    try:
        idx = problem_ids.index(problem_id)
    except ValueError:
        idx = -1

    next_problem_id = None
    if idx != -1 and idx < len(problem_ids) - 1:
        next_problem_id = problem_ids[idx + 1]

    return templates.TemplateResponse(
        "problem_strips.html",
        {
            "request": request,
            "problem_id": problem_id,
            "problem_label": spec.label,
            "strips": strips,
            "max_points": spec.points,
            "problem_note": session.problem_notes.get(problem_id, ""),
            "problem_type": _effective_problem_type(session, problem_id, spec),
            "problem_type_options": PROBLEM_TYPE_OPTIONS,
            "next_problem_id": next_problem_id,
        },
    )


@router.post("/problem/{problem_id}/strips", response_class=HTMLResponse)
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
        spec = next((p for p in session.grading_key.problems if p.id == problem_id), None)
        if spec is None:
            raise HTTPException(status_code=404, detail=f"Unknown problem id {problem_id!r}")

        updates_map = {}
        page_updates = {}

        for wu in session.work_units:
            bbox_raw = form.get(f"bbox_update_{wu.unit_id}")
            if bbox_raw and bbox_raw.strip():
                try:
                    new_bbox = json.loads(bbox_raw)
                    if isinstance(new_bbox, list) and len(new_bbox) == 4:
                        updates_map[wu.unit_id] = new_bbox
                except json.JSONDecodeError:
                    pass

            page_raw = form.get(f"page_update_{wu.unit_id}")
            if page_raw and page_raw.strip():
                page_updates[wu.unit_id] = page_raw.strip()

        if updates_map or page_updates:
            cache_path = GRADES_DIR / f"segments_{problem_id}.json"
            if cache_path.exists():
                try:
                    with cache_path.open("r", encoding="utf-8") as f:
                        segments_data = json.load(f)

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
                            seg["present"] = True
                            seg["confidence"] = 1.0
                            updated_count += 1

                    with cache_path.open("w", encoding="utf-8") as f:
                        json.dump(segments_data, f, indent=2)

                    print(f"Updated {updated_count} manual crops for problem {problem_id}")
                except Exception as exc:
                    print(f"Error updating segment cache: {exc}")

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

        for wu in session.work_units:
            field_name = f"comment_{wu.unit_id}"
            raw = form.get(field_name)
            if raw is None or raw.strip() == "":
                wu.per_problem_comments[problem_id] = None
            else:
                wu.per_problem_comments[problem_id] = raw

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


@router.post("/problem/{problem_id}/type")
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
