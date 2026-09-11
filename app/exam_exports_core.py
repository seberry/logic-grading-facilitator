from __future__ import annotations

from typing import Any, Dict, List, Optional
import html
import io
from pathlib import Path

import fitz  # type: ignore
from fastapi import HTTPException

from .exam_shared import (
    _list_canvas_original_zips,
    _load_canvas_import_manifest,
    _normalize_canvas_key,
    _required_total_max,
    _score_summary_for_work_unit,
    _selected_work_units_by_student,
    CANVAS_REUPLOAD_ORIGINAL_ZIPS_DIR,
    slugify_name,
)
from .models import Session


def roster_lookup(session: Session) -> Dict[str, str]:
    return {r.student_id: slugify_name(r.name) for r in session.roster}


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


def _build_best_rows_by_student(
    session: Session,
    scale_to: Optional[float] = None,
) -> Dict[str, Dict[str, Any]]:
    grading_key = session.grading_key
    problems = grading_key.problems
    total_max = _required_total_max(grading_key) if problems else 0.0
    roster_map = {r.student_id: r.name for r in session.roster}

    rows_by_student: Dict[str, Dict[str, Any]] = {}
    for sid, wu in _selected_work_units_by_student(session, scale_to=scale_to).items():
        per_problem_scores: Dict[str, Optional[float]] = {}
        per_problem_comments: Dict[str, Optional[str]] = {}
        for p in problems:
            score = wu.per_problem.get(p.id)
            per_problem_scores[p.id] = score
            per_problem_comments[p.id] = wu.per_problem_comments.get(p.id)
        score_summary = _score_summary_for_work_unit(session, wu, scale_to=scale_to)
        total = score_summary["total_score"]

        rows_by_student[sid] = {
            "unit_id": wu.unit_id,
            "student_id": sid,
            "student_name": roster_map.get(sid, sid),
            "per_problem_scores": per_problem_scores,
            "per_problem_comments": per_problem_comments,
            "regular_score": score_summary["regular_score"],
            "extra_credit_score": score_summary["extra_credit_score"],
            "raw_total_score": score_summary["raw_total_score"],
            "total_score": total,
            "total_max": total_max,
            "quiz_id": grading_key.quiz_id,
            "quiz_version": wu.quiz_version,
            "page_filenames": list(wu.page_filenames),
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


def _render_feedback_text_pdf_bytes(text: str) -> bytes:
    """Render all feedback text into one or more searchable letter-size pages."""
    escaped = html.escape(text).replace("\n", "<br>")
    story = fitz.Story(
        html=f'<div class="feedback">{escaped}</div>',
        user_css=(
            "body { font-family: sans-serif; font-size: 11pt; line-height: 1.35; } "
            ".feedback { white-space: normal; overflow-wrap: anywhere; }"
        ),
    )
    output = io.BytesIO()
    writer = fitz.DocumentWriter(output)
    page_rect = fitz.paper_rect("letter")
    content_rect = page_rect + (40, 45, -40, -45)
    more = 1
    while more:
        device = writer.begin_page(page_rect)
        more, _ = story.place(content_rect)
        story.draw(device)
        writer.end_page()
    writer.close()
    return output.getvalue()


def _scans_with_feedback_pdf_bytes(
    scan_paths: List[Path],
    feedback_text: str,
) -> tuple[bytes, int, int, List[str]]:
    """Combine original scan pages and final typed feedback pages into one PDF."""
    output = fitz.open()
    scan_page_count = 0
    warnings: List[str] = []

    for scan_path in scan_paths:
        if not scan_path.is_file():
            warnings.append(f"Missing scan: {scan_path.name}")
            continue
        try:
            with fitz.open(scan_path) as source:
                if source.is_pdf:
                    output.insert_pdf(source)
                    scan_page_count += source.page_count
                else:
                    image_pdf_bytes = source.convert_to_pdf()
                    with fitz.open(stream=image_pdf_bytes, filetype="pdf") as image_pdf:
                        output.insert_pdf(image_pdf)
                        scan_page_count += image_pdf.page_count
        except Exception as exc:
            warnings.append(f"Could not include scan {scan_path.name}: {exc}")

    if warnings:
        feedback_text += "\n\nScan attachment warnings\n" + "\n".join(
            f"- {warning}" for warning in warnings
        )

    feedback_bytes = _render_feedback_text_pdf_bytes(feedback_text)
    with fitz.open(stream=feedback_bytes, filetype="pdf") as feedback_pdf:
        feedback_page_count = feedback_pdf.page_count
        output.insert_pdf(feedback_pdf)

    output.set_metadata(
        {
            "title": "Graded quiz scans and feedback",
            "subject": "Original scan pages followed by instructor feedback",
        }
    )
    pdf_bytes = output.tobytes(deflate=True)
    output.close()
    return pdf_bytes, scan_page_count, feedback_page_count, warnings
