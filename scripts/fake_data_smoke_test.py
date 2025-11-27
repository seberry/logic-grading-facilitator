"""
Smoke-test the bundled fake data without external dependencies.

The script uses the existing fake rubric and scan images to:
1. Ensure a minimal roster exists (creates one if missing).
2. Initialize a grading session in `data/grades/session.json`.
3. Create a WorkUnit from the first unclaimed scan.
4. Render the grading template for that WorkUnit.
5. Assign a full score to the WorkUnit's problems and persist the session.

Run from the repo root:
    python -m scripts.fake_data_smoke_test
"""
from __future__ import annotations

from pathlib import Path
from typing import List

from starlette.requests import Request
from starlette.templating import Jinja2Templates

from app.models import Session, WorkUnit
from app.storage import (
    DEFAULT_GRADING_KEY_PATH,
    DEFAULT_ROSTER_PATH,
    DEFAULT_SESSION_PATH,
    RAW_SCANS_DIR,
    init_session,
    save_session,
)


def _ensure_fake_roster() -> List[str]:
    """Create a small fake roster if one does not already exist."""
    if DEFAULT_ROSTER_PATH.exists():
        return [line.strip() for line in DEFAULT_ROSTER_PATH.read_text(encoding="utf-8").splitlines()]

    DEFAULT_ROSTER_PATH.parent.mkdir(parents=True, exist_ok=True)
    fake_lines = [
        "student_id,name",
        "u0001,Fake Student",
        "u0002,Second Student",
    ]
    DEFAULT_ROSTER_PATH.write_text("\n".join(fake_lines) + "\n", encoding="utf-8")
    return fake_lines


def _render_grade_page(session: Session, work_unit: WorkUnit) -> int:
    """Render the grading template and return the length of the resulting HTML body."""
    templates = Jinja2Templates(directory=Path(__file__).resolve().parents[1] / "app" / "templates")
    scope = {"type": "http", "method": "GET", "path": f"/grade/{work_unit.unit_id}/page", "headers": []}
    request = Request(scope)
    response = templates.TemplateResponse(
        "grade.html",
        {
            "request": request,
            "work_unit": work_unit,
            "grading_key": session.grading_key,
            "roster": session.roster,
            "image_urls": [f"/scans/{fn}" for fn in work_unit.page_filenames],
            "pinned_note": session.pinned_note or "",
        },
    )
    rendered_html = response.template.render(response.context)
    return len(rendered_html)


def main() -> None:
    if not RAW_SCANS_DIR.exists():
        raise SystemExit(f"Expected fake raw scans under {RAW_SCANS_DIR}.")
    if not DEFAULT_GRADING_KEY_PATH.exists():
        raise SystemExit(f"Expected grading key at {DEFAULT_GRADING_KEY_PATH}.")

    roster_preview = _ensure_fake_roster()
    print("Roster ready (first lines):")
    for line in roster_preview:
        print(f"  {line}")

    session: Session = init_session(
        roster_path=DEFAULT_ROSTER_PATH,
        grading_key_path=DEFAULT_GRADING_KEY_PATH,
        session_path=DEFAULT_SESSION_PATH,
        raw_scans_dir=RAW_SCANS_DIR,
    )
    print(
        f"Initialized session with {len(session.roster)} roster entries, "
        f"{len(session.unclaimed_pages)} unclaimed pages, and "
        f"grading key '{session.grading_key.quiz_id}'."
    )

    if not session.unclaimed_pages:
        raise SystemExit("No unclaimed pages available to create a WorkUnit.")

    next_page = session.unclaimed_pages.pop(0)
    per_problem = {p.id: None for p in session.grading_key.problems}
    per_problem_comments = {p.id: None for p in session.grading_key.problems}
    work_unit = WorkUnit(
        unit_id=1,
        page_filenames=[next_page],
        student_id=None,
        per_problem=per_problem,
        comment=None,
        per_problem_comments=per_problem_comments,
    )
    session.work_units.append(work_unit)
    save_session(session, DEFAULT_SESSION_PATH)
    print(f"Created WorkUnit {work_unit.unit_id} using page {next_page}")

    html_length = _render_grade_page(session, work_unit)
    print(f"Grade page HTML length: {html_length}")

    first_student = session.roster[0].student_id
    for problem in session.grading_key.problems:
        work_unit.per_problem[problem.id] = problem.points
    work_unit.student_id = first_student
    save_session(session, DEFAULT_SESSION_PATH)
    print(
        f"Updated WorkUnit {work_unit.unit_id}: student_id={work_unit.student_id}, "
        f"scores={work_unit.per_problem}"
    )


if __name__ == "__main__":
    main()
