import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from app import server
from app.models import (
    ClassworkEntry,
    ClassworkModeState,
    GradingKey,
    RosterEntry,
    Session,
)
from app.routes import classwork as classwork_routes


def _build_session_with_classwork() -> Session:
    return Session(
        roster=[RosterEntry(student_id="s1", name="Berry, Sharon")],
        grading_key=GradingKey(quiz_id="quiz1", max_points=10, problems=[]),
        work_units=[],
        unclaimed_pages=[],
        classwork_mode=ClassworkModeState(
            assignment_id="cw_week_01",
            max_points=10,
            gradebook_token="test_gradebook_token",
            id_column="SIS User ID",
            entries=[
                ClassworkEntry(
                    student_id="s1",
                    student_name="Berry, Sharon",
                    has_submission=True,
                    files=[],
                    score=8,
                    comment=None,
                ),
                ClassworkEntry(
                    student_id="s2",
                    student_name="Jones, Eli",
                    has_submission=False,
                    files=[],
                    score=0,
                    comment=None,
                ),
            ],
        ),
    )


class ClassworkRoutesTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(server.app)

    def test_classwork_setup_page_renders(self) -> None:
        session = Session(
            roster=[],
            grading_key=GradingKey(quiz_id="quiz1", max_points=10, problems=[]),
            work_units=[],
            unclaimed_pages=[],
        )

        with patch.object(classwork_routes, "get_session", return_value=session):
            response = self.client.get("/classwork")

        self.assertEqual(response.status_code, 200)
        self.assertIn("Classwork Queue Setup", response.text)

    def test_classwork_review_page_renders_existing_mode(self) -> None:
        session = _build_session_with_classwork()

        with (
            patch.object(classwork_routes, "get_session", return_value=session),
            patch.object(
                classwork_routes,
                "_classwork_analysis",
                return_value={"columns": [], "conflicts": [], "row_count": 2},
            ),
        ):
            response = self.client.get("/classwork/review")

        self.assertEqual(response.status_code, 200)
        self.assertIn("Classwork Review", response.text)
        self.assertIn("Berry, Sharon", response.text)

    def test_classwork_review_submit_updates_state_and_redirects(self) -> None:
        session = _build_session_with_classwork()

        with (
            patch.object(classwork_routes, "get_session", return_value=session),
            patch.object(classwork_routes, "persist_session"),
            patch.object(classwork_routes, "_classwork_sync_mode_into_batch"),
        ):
            response = self.client.post(
                "/classwork/review",
                data={
                    "score_s1": "9.5",
                    "comment_s1": "Strong work",
                    "score_s2": "",
                    "comment_s2": "",
                    "action": "save",
                },
                follow_redirects=False,
            )

        self.assertEqual(response.status_code, 303)
        self.assertEqual(response.headers["location"], "/classwork/review")
        self.assertEqual(session.classwork_mode.entries[0].score, 9.5)
        self.assertEqual(session.classwork_mode.entries[0].comment, "Strong work")
        self.assertEqual(session.classwork_mode.entries[1].score, 0.0)
        self.assertIsNone(session.classwork_mode.entries[1].comment)


if __name__ == "__main__":
    unittest.main()
