import unittest
from pathlib import Path
import shutil
from unittest.mock import patch

from fastapi.testclient import TestClient

from app import server
from app.models import (
    ClassworkBatchState,
    ClassworkDownloadWarning,
    ClassworkEntry,
    ClassworkFileRef,
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
            download_warnings=[
                ClassworkDownloadWarning(
                    student_id="s1",
                    student_name="Berry, Sharon",
                    filename="submission.pdf",
                    error="connection reset",
                )
            ],
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
        self.assertIn("Canvas download warning", response.text)
        self.assertIn("submission.pdf", response.text)

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

    def test_classwork_reset_clears_session_and_temp_files(self) -> None:
        test_root = Path("working") / "test_classwork_reset"
        test_root.mkdir(parents=True, exist_ok=True)
        stored_name = "reset_test_submission.pdf"
        stored_path = test_root / stored_name
        stored_path.write_bytes(b"pdf")

        token = "reset_test_token"
        tmp_path = test_root / f"{token}.csv"
        tmp_path.write_text("Student,SIS Login ID\nBerry, Sharon,s1\n", encoding="utf-8")

        mode = ClassworkModeState(
            assignment_id="cw_week_01",
            max_points=10,
            gradebook_token=token,
            id_column="SIS Login ID",
            entries=[
                ClassworkEntry(
                    student_id="s1",
                    student_name="Berry, Sharon",
                    has_submission=True,
                    files=[
                        ClassworkFileRef(
                            stored_filename=stored_name,
                            original_filename="submission.pdf",
                            ext=".pdf",
                        )
                    ],
                    score=10,
                    comment=None,
                )
            ],
        )
        session = Session(
            roster=[RosterEntry(student_id="s1", name="Berry, Sharon")],
            grading_key=GradingKey(quiz_id="quiz1", max_points=10, problems=[]),
            work_units=[],
            unclaimed_pages=[],
            classwork_mode=mode,
            classwork_batch=ClassworkBatchState(
                gradebook_token=token,
                id_column="SIS Login ID",
                assignments=[mode],
            ),
        )

        try:
            with (
                patch.object(classwork_routes, "get_session", return_value=session),
                patch.object(classwork_routes, "persist_session"),
                patch.object(classwork_routes, "_classwork_resolve_stored_file", return_value=stored_path),
                patch.object(classwork_routes, "_classwork_gradebook_tmp_path", return_value=tmp_path),
            ):
                response = self.client.post("/classwork/reset", follow_redirects=False)

            self.assertEqual(response.status_code, 303)
            self.assertIn("/classwork?status=success", response.headers["location"])
            self.assertIsNone(session.classwork_mode)
            self.assertIsNone(session.classwork_batch)
            self.assertFalse(stored_path.exists())
            self.assertFalse(tmp_path.exists())
        finally:
            shutil.rmtree(test_root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
