from __future__ import annotations

import json
import math
import asyncio
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import quote

from fastapi import APIRouter, Form, HTTPException, Request, Response
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from PIL import Image

from app.helpers import _build_page_options, _is_image_filename, _select_display_filename
from app.models import Session
from app.segmentation_core import QuestionSegment, segment_page_for_problem
from app.routes import modal_proof_debug as modal_proof_debug_routes
from app.session_state import _segmentation_lock, _session_lock, get_session, persist_session
from app.storage import DEBUG_FITCH_DIR, DEBUG_MODAL_PROOF_DIR, GRADES_DIR, RAW_SCANS_DIR
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


def _build_problem_locator_text(spec) -> str:
    parts: List[str] = []
    if getattr(spec, "section", None):
        parts.append(f'section "{spec.section}"')
    if getattr(spec, "problem_number", None):
        parts.append(f'printed problem number "{spec.problem_number}"')
    return ", ".join(parts)


def _build_segmentation_description(spec) -> str:
    details: List[str] = []
    locator = _build_problem_locator_text(spec)
    if locator:
        details.append(f"Locator: {locator}.")
    if getattr(spec, "label", None):
        details.append(f"Rubric label: {spec.label}.")
    if getattr(spec, "prompt", None):
        details.append(f"Problem prompt: {spec.prompt}.")
    if getattr(spec, "segmentation_hint", None):
        details.append(f"Segmentation hint: {spec.segmentation_hint}.")
    return " ".join(details) or spec.id


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
                problem_description=_build_segmentation_description(spec),
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


def _clean_fitch_debug_slug(raw: str) -> str:
    text = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(raw or "").strip())
    while "__" in text:
        text = text.replace("__", "_")
    return text.strip("_") or "fitch_debug"


def _clean_modal_debug_slug(raw: str) -> str:
    text = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(raw or "").strip())
    while "__" in text:
        text = text.replace("__", "_")
    return text.strip("_") or "modal_debug"


def _modal_debug_slug_for_strip(problem_id: str, unit_id: int, page_filename: str) -> str:
    return _clean_modal_debug_slug(f"{problem_id}_{unit_id}_{Path(page_filename).stem}")


def _modal_debug_url(slug: str) -> str:
    return f"/debug/modal-proof?example={quote(slug)}"


def _has_meaningful_modal_preview(state: Dict[str, Any], unary_results: List[Dict[str, Any]]) -> bool:
    if unary_results:
        return True
    if state.get("checker_response"):
        return True
    proof_json_text = str(state.get("proof_json_text") or "")
    original_proof_json_text = str(state.get("original_proof_json_text") or "")
    if proof_json_text != modal_proof_debug_routes.DEFAULT_PROOF_JSON_TEXT:
        return True
    if original_proof_json_text != modal_proof_debug_routes.DEFAULT_PROOF_JSON_TEXT:
        return True
    return False


def _load_modal_strip_preview(
    *,
    problem_id: str,
    unit_id: int,
    page_filename: Optional[str],
) -> Optional[Dict[str, Any]]:
    if not page_filename:
        return None

    slug = _modal_debug_slug_for_strip(problem_id, unit_id, page_filename)
    state_path = DEBUG_MODAL_PROOF_DIR / slug / modal_proof_debug_routes.STATE_FILENAME
    preview: Dict[str, Any] = {
        "slug": slug,
        "debug_url": _modal_debug_url(slug),
        "has_state": state_path.exists(),
        "has_preview": False,
        "lines": [],
        "flagged_count": 0,
        "unresolved_count": 0,
        "disagreement_count": 0,
        "checker_issue_count": 0,
        "message": "No saved modal proof state yet. Use Open Proof Editor to create one.",
    }
    if not state_path.exists():
        return preview

    state = modal_proof_debug_routes._load_example_state(slug)
    unary_results = modal_proof_debug_routes._load_unary_verifier_results(slug)
    if not _has_meaningful_modal_preview(state, unary_results):
        preview["message"] = "A modal debug example exists, but no OCR/checker output has been saved yet."
        return preview

    review_panel = modal_proof_debug_routes._build_review_panel(
        state=state,
        unary_verifier_results=unary_results,
    )
    review_by_line: Dict[int, Dict[str, Any]] = {}
    for item in review_panel.get("items") or []:
        if not isinstance(item, dict):
            continue
        line_number = item.get("line_number")
        if isinstance(line_number, int):
            review_by_line[line_number] = item

    proof_lines = modal_proof_debug_routes._proof_lines_from_text(str(state.get("proof_json_text") or ""))
    checker_issues_by_line = modal_proof_debug_routes._checker_issues_by_line(state.get("checker_response"))
    line_numbers = sorted(set(proof_lines.keys()) | set(review_by_line.keys()) | set(checker_issues_by_line.keys()))

    lines: List[Dict[str, Any]] = []
    disagreement_count = 0
    checker_issue_count = 0
    for line_number in line_numbers:
        line = proof_lines.get(line_number, {})
        review_item = review_by_line.get(line_number, {})
        flag_reasons = [
            str(reason).strip()
            for reason in (review_item.get("flag_reasons") or [])
            if str(reason).strip()
        ]
        has_checker_issue = any(reason == "checker issue" for reason in flag_reasons) or bool(
            checker_issues_by_line.get(line_number)
        )
        has_disagreement = any(reason != "checker issue" for reason in flag_reasons)

        if has_disagreement:
            disagreement_count += 1
        if has_checker_issue:
            checker_issue_count += 1

        if has_checker_issue and has_disagreement:
            status = "both"
        elif has_checker_issue:
            status = "checker"
        elif has_disagreement:
            status = "disagreement"
        else:
            status = "ok"

        lines.append(
            {
                "line_number": line_number,
                "formula": str(line.get("formula") or review_item.get("current_formula") or "").strip(),
                "citation": str(line.get("citation") or "").strip(),
                "depth": max(0, int(line.get("depth") or 0)),
                "status": status,
                "ocr_reasons": [reason for reason in flag_reasons if reason != "checker issue"],
                "checker_issues": checker_issues_by_line.get(line_number, []),
            }
        )

    for index, line in enumerate(lines):
        next_depth = 0
        if index + 1 < len(lines):
            next_depth = max(0, int(lines[index + 1].get("depth") or 0))
        current_depth = max(0, int(line.get("depth") or 0))
        line["closes_subproof"] = current_depth > next_depth

    preview.update(
        {
            "has_preview": bool(lines),
            "lines": lines,
            "ascii_records": modal_proof_debug_routes._build_ascii_proof_records(
                [
                    {
                        "line_number": line["line_number"],
                        "depth": line["depth"],
                        "sentence": line["formula"],
                        "citation": line["citation"],
                        "note_text": " | ".join(
                            ([f"checker: {'; '.join(str(issue.get('message') or '').strip() for issue in line['checker_issues'] if str(issue.get('message') or '').strip())}"] if line["checker_issues"] else [])
                            + ([f"ocr: {', '.join(line['ocr_reasons'])}"] if line["ocr_reasons"] else [])
                        ),
                        "status": line["status"],
                    }
                    for line in lines
                ]
            ),
            "flagged_count": int(review_panel.get("flagged_count") or 0),
            "unresolved_count": int(review_panel.get("unresolved_count") or 0),
            "disagreement_count": disagreement_count,
            "checker_issue_count": checker_issue_count,
            "message": "",
        }
    )
    if not lines:
        preview["message"] = "Modal OCR state exists, but the saved proof JSON did not produce any preview lines."
    return preview


def _bbox_for_strip(seg_data: Optional[Dict[str, Any]], bbox_override_raw: str) -> Optional[List[float]]:
    if bbox_override_raw and bbox_override_raw.strip():
        try:
            candidate = json.loads(bbox_override_raw)
        except json.JSONDecodeError:
            candidate = None
        if _is_normalized_bbox(candidate):
            return [float(v) for v in candidate]

    if seg_data and _is_normalized_bbox(seg_data.get("bbox")):
        return [float(v) for v in seg_data["bbox"]]

    return None


async def _run_subprocess_with_logs(command: List[str]) -> tuple[int, str]:
    proc = await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd=Path(__file__).resolve().parents[2],
    )
    stdout, _ = await proc.communicate()
    return proc.returncode, stdout.decode("utf-8", errors="replace")


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
    effective_problem_type = _effective_problem_type(session, problem_id, spec)

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

        modal_preview = None
        if effective_problem_type == "modal_proof":
            modal_preview = _load_modal_strip_preview(
                problem_id=problem_id,
                unit_id=wu.unit_id,
                page_filename=display_filename,
            )

        strips.append(
            {
                "work_unit": wu,
                "student_name": lookup_student_name(wu.student_id),
                "segment": seg_obj,
                "display_filename": display_filename,
                "display_url": display_url,
                "page_options": page_options,
                "modal_preview": modal_preview,
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
            "problem_spec": spec,
            "problem_locator_text": _build_problem_locator_text(spec),
            "strips": strips,
            "max_points": spec.points,
            "problem_note": session.problem_notes.get(problem_id, ""),
            "problem_type": effective_problem_type,
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


@router.post("/problem/{problem_id}/fitch_debug")
async def open_fitch_debug_from_strip(
    problem_id: str,
    unit_id: int = Form(...),
    page_filename: str = Form(...),
    bbox_json: str = Form(""),
) -> Response:
    session = get_session()
    spec = next((p for p in session.grading_key.problems if p.id == problem_id), None)
    if spec is None:
        raise HTTPException(status_code=404, detail=f"Unknown problem id {problem_id!r}")

    effective_type = _effective_problem_type(session, problem_id, spec)
    if effective_type != "fitch_proof":
        raise HTTPException(status_code=400, detail=f"Problem {problem_id!r} is not tagged as fitch_proof.")

    work_unit = next((wu for wu in session.work_units if wu.unit_id == unit_id), None)
    if work_unit is None:
        raise HTTPException(status_code=404, detail=f"Unknown work unit {unit_id}.")
    if page_filename not in work_unit.page_filenames:
        raise HTTPException(status_code=400, detail=f"Page {page_filename!r} does not belong to unit {unit_id}.")

    image_path = RAW_SCANS_DIR / page_filename
    if not image_path.exists():
        raise HTTPException(status_code=404, detail=f"Scan image not found: {image_path}")

    cache_path = GRADES_DIR / f"segments_{problem_id}.json"
    seg_data = None
    if cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if isinstance(cached, list):
                seg_data = next((row for row in cached if isinstance(row, dict) and row.get("unit_id") == unit_id), None)
        except Exception:
            seg_data = None

    bbox = _bbox_for_strip(seg_data, bbox_json)

    slug = _clean_fitch_debug_slug(f"{problem_id}_{unit_id}_{Path(page_filename).stem}")
    target_dir = DEBUG_FITCH_DIR / slug
    target_dir.mkdir(parents=True, exist_ok=True)
    target_image = target_dir / f"{slug}{image_path.suffix.lower()}"

    with Image.open(image_path) as im:
        working = im.convert("RGB")
        if bbox:
            width, height = working.size
            x1 = max(0, min(width - 1, int(round(bbox[0] * width))))
            y1 = max(0, min(height - 1, int(round(bbox[1] * height))))
            x2 = max(x1 + 1, min(width, int(round(bbox[2] * width))))
            y2 = max(y1 + 1, min(height, int(round(bbox[3] * height))))
            working = working.crop((x1, y1, x2, y2))
        working.save(target_image)

    command = [sys.executable, "scripts/debug_fitch_ocr.py", target_image.name]
    code, output = await _run_subprocess_with_logs(command)
    if code != 0:
        raise HTTPException(status_code=500, detail=f"Fitch OCR rerun failed for {slug}:\n{output}")

    return RedirectResponse(url=f"/debug/fitch?scan={quote(slug)}", status_code=303)


@router.post("/problem/{problem_id}/modal_proof_debug")
async def open_modal_proof_debug_from_strip(
    problem_id: str,
    unit_id: int = Form(...),
    page_filename: str = Form(...),
    bbox_json: str = Form(""),
) -> Response:
    session = get_session()
    spec = next((p for p in session.grading_key.problems if p.id == problem_id), None)
    if spec is None:
        raise HTTPException(status_code=404, detail=f"Unknown problem id {problem_id!r}")

    effective_type = _effective_problem_type(session, problem_id, spec)
    if effective_type != "modal_proof":
        raise HTTPException(status_code=400, detail=f"Problem {problem_id!r} is not tagged as modal_proof.")

    work_unit = next((wu for wu in session.work_units if wu.unit_id == unit_id), None)
    if work_unit is None:
        raise HTTPException(status_code=404, detail=f"Unknown work unit {unit_id}.")
    if page_filename not in work_unit.page_filenames:
        raise HTTPException(status_code=400, detail=f"Page {page_filename!r} does not belong to unit {unit_id}.")

    image_path = RAW_SCANS_DIR / page_filename
    if not image_path.exists():
        raise HTTPException(status_code=404, detail=f"Scan image not found: {image_path}")

    cache_path = GRADES_DIR / f"segments_{problem_id}.json"
    seg_data = None
    if cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if isinstance(cached, list):
                seg_data = next((row for row in cached if isinstance(row, dict) and row.get("unit_id") == unit_id), None)
        except Exception:
            seg_data = None

    bbox = _bbox_for_strip(seg_data, bbox_json)

    slug = _clean_modal_debug_slug(f"{problem_id}_{unit_id}_{Path(page_filename).stem}")
    target_dir = DEBUG_MODAL_PROOF_DIR / slug
    target_dir.mkdir(parents=True, exist_ok=True)
    target_image = target_dir / f"{slug}{image_path.suffix.lower()}"

    with Image.open(image_path) as im:
        working = im.convert("RGB")
        if bbox:
            width, height = working.size
            x1 = max(0, min(width - 1, int(round(bbox[0] * width))))
            y1 = max(0, min(height - 1, int(round(bbox[1] * height))))
            x2 = max(x1 + 1, min(width, int(round(bbox[2] * width))))
            y2 = max(y1 + 1, min(height, int(round(bbox[3] * height))))
            working = working.crop((x1, y1, x2, y2))
        working.save(target_image)

    state_path = target_dir / "state.json"
    state = {
        "system": "K",
        "include_derived_rules": False,
        "conclusion": "",
        "vision_base_url": "",
        "vision_model": "",
        "source_image_name": target_image.name,
        "proof_json_text": json.dumps({"lines": [{"wffstr": "A", "jstr": "PR"}], "meta": {}}, indent=2),
        "draft_comment": "",
        "draft_grade": "",
        "checker_request": None,
        "checker_response": None,
        "run_summary": "Cropped proof imported from strip view. Review the image, then run OCR.",
        "status": "success",
        "message": "Cropped proof imported from strip view.",
        "checker_error": "",
    }
    state_path.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")

    return RedirectResponse(url=f"/debug/modal-proof?example={quote(slug)}", status_code=303)
