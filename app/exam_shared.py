from __future__ import annotations

from typing import Any, Dict, List, Optional
import re
import json
import math
from pathlib import Path
from datetime import datetime

from fastapi import HTTPException

from .models import GradingKey, RosterEntry, Session, WorkUnit
from .storage import DATA_DIR


CANVAS_REUPLOAD_ORIGINAL_ZIPS_DIR = DATA_DIR / "canvas_reupload" / "original_zips"
CANVAS_REUPLOAD_MANIFEST_DIR = DATA_DIR / "canvas_reupload" / "manifests"


def _build_total_scores_by_student(
    session: Session,
    scale_to: Optional[float] = None,
) -> Dict[str, float]:
    _require_complete_regular_scores(session)
    return {
        student_id: _score_summary_for_work_unit(session, wu, scale_to=scale_to)[
            "total_score"
        ]
        for student_id, wu in _selected_work_units_by_student(
            session, scale_to=scale_to
        ).items()
    }


def _incomplete_regular_scores(session: Session) -> List[str]:
    """Missing required grades in identified submissions, including duplicates."""
    return [
        f"unit {wu.unit_id}: {problem.id}"
        for wu in session.work_units
        if wu.student_id
        for problem in session.grading_key.problems
        if not problem.extra_credit and wu.per_problem.get(problem.id) is None
    ]


def _require_complete_regular_scores(session: Session) -> None:
    missing = _incomplete_regular_scores(session)
    if missing:
        raise HTTPException(
            status_code=409,
            detail=(
                "Final grade export is blocked: regular questions are still ungraded. "
                "Enter an explicit 0 for zero-credit answers. Blank optional extra credit "
                "counts as zero. Missing grades: " + ", ".join(missing[:30])
                + (f" (and {len(missing) - 30} more)" if len(missing) > 30 else "")
            ),
        )


def _work_units_by_student(session: Session) -> Dict[str, List[WorkUnit]]:
    """Return every identified submission, preserving session order."""
    out: Dict[str, List[WorkUnit]] = {}
    for wu in session.work_units:
        if not wu.student_id:
            continue
        out.setdefault(wu.student_id, []).append(wu)
    return out


def _selected_work_units_by_student(
    session: Session,
    scale_to: Optional[float] = None,
) -> Dict[str, WorkUnit]:
    """Choose the highest-total submission for each student.

    The first WorkUnit in session order wins an exact tie. Callers that expose
    this fallback must also surface duplicates to the human grader.
    """
    selected: Dict[str, WorkUnit] = {}
    selected_totals: Dict[str, float] = {}
    for student_id, work_units in _work_units_by_student(session).items():
        for wu in work_units:
            total = _score_summary_for_work_unit(session, wu, scale_to=scale_to)[
                "total_score"
            ]
            if student_id not in selected or total > selected_totals[student_id]:
                selected[student_id] = wu
                selected_totals[student_id] = total
    return selected


def _required_total_max(grading_key: GradingKey) -> float:
    return sum(float(p.points) for p in grading_key.problems if not p.extra_credit)


def _scaled_total_score(raw_total: float, raw_total_max: float, scale_to: Optional[float]) -> float:
    if scale_to is None:
        return raw_total
    if raw_total_max <= 0:
        return 0.0
    return round((raw_total / raw_total_max) * scale_to, 4)


def _parse_scale_to(scale_to: Optional[float]) -> Optional[float]:
    if scale_to is None:
        return None
    value = float(scale_to)
    if not math.isfinite(value) or value <= 0:
        raise HTTPException(status_code=400, detail="scale_to must be a positive number.")
    return value


def _score_summary_for_work_unit(
    session: Session,
    wu: WorkUnit,
    scale_to: Optional[float] = None,
) -> Dict[str, float]:
    regular_total = 0.0
    extra_credit_total = 0.0
    problems_by_id = {p.id: p for p in session.grading_key.problems}
    for problem_id, score in wu.per_problem.items():
        if score is None:
            continue
        spec = problems_by_id.get(problem_id)
        if spec and spec.extra_credit:
            extra_credit_total += float(score)
        else:
            regular_total += float(score)

    total_max = _required_total_max(session.grading_key)
    raw_total_score = min(total_max, regular_total + extra_credit_total)
    scaled_to = _parse_scale_to(scale_to)
    total_score = _scaled_total_score(raw_total_score, total_max, scaled_to)
    return {
        "regular_score": regular_total,
        "extra_credit_score": extra_credit_total,
        "raw_total_score": raw_total_score,
        "total_score": total_score,
        "total_max": total_max,
        "scale_to": scaled_to or total_max,
    }


def slugify_name(name: str) -> str:
    s = name.strip().lower()
    s = s.replace(",", " ")
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9_]", "", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "student"


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
