"""
Initialize a new grading session.

Usage (from repo root):

    python -m scripts.init_session

Assumes:
- DATA_DIR/roster.csv exists with headers: student_id,name
- DATA_DIR/rubric/quiz1.yaml exists
- DATA_DIR/raw_scans/ contains JPG/PNG files
"""

from pathlib import Path

from app.storage import (
    init_session,
    DEFAULT_SESSION_PATH,
    DEFAULT_ROSTER_PATH,
    DEFAULT_GRADING_KEY_PATH,
)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    print(f"Repo root: {root}")
    print(f"Using roster: {DEFAULT_ROSTER_PATH}")
    print(f"Using grading key: {DEFAULT_GRADING_KEY_PATH}")
    print(f"Session will be written to: {DEFAULT_SESSION_PATH}")

    session = init_session()
    print(
        f"Initialized session with {len(session.roster)} roster entries, "
        f"{len(session.unclaimed_pages)} unclaimed pages, "
        f"{len(session.work_units)} existing work units."
    )


if __name__ == "__main__":
    main()
