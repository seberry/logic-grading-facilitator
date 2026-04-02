import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from app import server
from app.models import GradingKey, ProblemSpec, RosterEntry, Session, WorkUnit
from app.routes import exam_exports as exam_exports_routes


def _build_session() -> Session:
    return Session(
        roster=[RosterEntry(student_id="s1", name="Berry, Sharon")],
        grading_key=GradingKey(
            quiz_id="quiz1",
            max_points=10,
            problems=[
                ProblemSpec(id="Q1", label="FOL model", points=4),
                ProblemSpec(id="Q2", label="Proof", points=6),
            ],
        ),
        work_units=[
            WorkUnit(
                unit_id=1,
                student_id="s1",
                page_filenames=["quiz1_berry_sharon_p01.jpg"],
                per_problem={"Q1": 4.0, "Q2": 5.0},
                comment="Good work",
                per_problem_comments={"Q1": "Correct", "Q2": "One minor issue"},
            )
        ],
        unclaimed_pages=[],
    )


class ExamExportsRoutesTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(server.app)

    def test_finish_page_renders_summary(self) -> None:
        session = _build_session()

        with patch.object(exam_exports_routes, "get_session", return_value=session):
            response = self.client.get("/finish")

        self.assertEqual(response.status_code, 200)
        self.assertIn("Berry, Sharon", response.text)
        self.assertIn("Q1", response.text)
        self.assertIn("Q2", response.text)

    def test_export_scores_returns_csv(self) -> None:
        session = _build_session()

        with patch.object(exam_exports_routes, "get_session", return_value=session):
            response = self.client.get("/export")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "text/csv; charset=utf-8")
        self.assertIn('filename="quiz1_scores.csv"', response.headers["content-disposition"])
        self.assertIn("student_id,quiz1_score", response.text)
        self.assertIn("s1,9.0", response.text)

    def test_finalize_assignment_archives_and_renders_confirmation(self) -> None:
        with patch.object(
            exam_exports_routes,
            "archive_current_data",
            return_value="20260402_exam1_archive",
        ):
            response = self.client.post("/finalize", data={"label": "exam1"})

        self.assertEqual(response.status_code, 200)
        self.assertIn("Assignment finalized", response.text)
        self.assertIn("20260402_exam1_archive", response.text)


if __name__ == "__main__":
    unittest.main()
