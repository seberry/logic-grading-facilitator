import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.classwork_core import _find_matching_column, _normalize_for_matching


def test_normalize_for_matching_handles_canvas_gradebook_ids() -> None:
    cases = [
        ("Worksheet 1", "worksheet 1"),
        ("Quiz-#1", "quiz-1"),
        ("Proof Assignment", "proof assignment"),
        ("Assignment #3", "assignment 3"),
        ("Worksheet - Proofs", "worksheet - proofs"),
        ("Wed Attendance (18715846)", "wed attendance"),
        ("Exam 1 (18697401)", "exam 1"),
        ("", ""),
    ]

    for original, expected in cases:
        assert _normalize_for_matching(original) == expected


def test_find_matching_column_ignores_system_columns() -> None:
    gradebook_columns = [
        "Student",
        "SIS Login ID",
        "Worksheet 1",
        "Quiz-#1",
        "Proof Assignment",
        "Assignment #3",
        "Worksheet - Proofs",
        "Midterm Exam",
        "Current Score",
        "Final Score",
    ]

    cases = [
        ("Worksheet 1", "Worksheet 1"),
        ("Quiz-#1", "Quiz-#1"),
        ("Proof Assignment", "Proof Assignment"),
        ("Assignment #3", "Assignment #3"),
        ("Worksheet - Proofs", "Worksheet - Proofs"),
        ("Midterm Exam", "Midterm Exam"),
        ("Nonexistent Assignment", None),
        ("Current Score", None),
    ]

    for canvas_name, expected in cases:
        assert (
            _find_matching_column(
                canvas_assignment_name=canvas_name,
                gradebook_fieldnames=gradebook_columns,
                id_col="SIS Login ID",
            )
            == expected
        )
