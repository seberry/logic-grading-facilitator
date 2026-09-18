from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from app import server
from app.models import GradingKey, ProblemSpec, RosterEntry


class NameReviewHandoffTests(unittest.TestCase):
    def test_save_and_prepare_redirects_to_assignment_overview(self) -> None:
        client = TestClient(server.app)
        grading_key = GradingKey(
            quiz_id="quiz3",
            problems=[ProblemSpec(id="Q1", label="Question 1", points=10)],
        )
        roster = [RosterEntry(student_id="s1", name="Example, Ada")]

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            raw_dir = root / "raw_scans"
            grades_dir = root / "grades"
            raw_dir.mkdir(parents=True)
            grades_dir.mkdir(parents=True)
            (raw_dir / "scan.jpg").write_bytes(b"fake scan bytes")
            session_path = grades_dir / "session.json"

            had_server_session = hasattr(server, "_session")
            previous_session = getattr(server, "_session", None)
            try:
                with (
                    patch.object(server, "RAW_SCANS_DIR", raw_dir),
                    patch.object(server, "DEFAULT_SESSION_PATH", session_path),
                    patch.object(server, "load_roster", return_value=roster),
                    patch.object(server, "load_grading_key", return_value=grading_key),
                ):
                    response = client.post(
                        "/name_review",
                        data={
                            "quiz_id": "quiz3",
                            "raw_scans_subdir": "",
                            "page_count": "1",
                            "filename_0": "scan.jpg",
                            "student_0": "s1",
                            "quiz_version_0": "A",
                            "next_action": "prepare",
                        },
                        follow_redirects=False,
                    )
            finally:
                if had_server_session:
                    server._session = previous_session
                elif hasattr(server, "_session"):
                    delattr(server, "_session")

            self.assertEqual(response.status_code, 303)
            self.assertEqual(response.headers["location"], "/finish?mode=tf-prep")
            self.assertTrue(session_path.is_file())


if __name__ == "__main__":
    unittest.main()
