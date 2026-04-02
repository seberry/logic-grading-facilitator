import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from app import server
from app.models import GradingKey, ProblemSpec, RosterEntry, Session, WorkUnit
from app.routes import problem_strips as problem_strips_routes


def _build_session() -> Session:
    return Session(
        roster=[RosterEntry(student_id="s1", name="Berry, Sharon")],
        grading_key=GradingKey(
            quiz_id="quiz1",
            max_points=10,
            problems=[
                ProblemSpec(id="Q1", label="FOL model", points=5),
                ProblemSpec(id="Q2", label="Proof", points=5),
            ],
        ),
        work_units=[
            WorkUnit(
                unit_id=1,
                student_id="s1",
                page_filenames=["quiz1_berry_sharon_p01.jpg"],
                per_problem={"Q1": None, "Q2": None},
                comment=None,
                per_problem_comments={"Q1": None, "Q2": None},
            )
        ],
        unclaimed_pages=[],
    )


class ProblemStripsRoutesTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(server.app)

    def test_problem_strips_page_renders(self) -> None:
        session = _build_session()
        segments = [
            {
                "unit_id": 1,
                "page_filename": "quiz1_berry_sharon_p01.jpg",
                "page_index": 0,
                "bbox": [0.1, 0.2, 0.8, 0.9],
                "present": True,
                "comment": "found",
                "confidence": 0.9,
            }
        ]

        with (
            patch.object(problem_strips_routes, "get_session", return_value=session),
            patch.object(problem_strips_routes, "ensure_segments_cached", return_value=segments),
        ):
            response = self.client.get("/problem/Q1/strips")

        self.assertEqual(response.status_code, 200)
        self.assertIn("Problem Q1 Grading", response.text)
        self.assertIn("Berry, Sharon", response.text)

    def test_problem_type_post_updates_session(self) -> None:
        session = _build_session()

        with (
            patch.object(problem_strips_routes, "get_session", return_value=session),
            patch.object(problem_strips_routes, "persist_session"),
        ):
            response = self.client.post(
                "/problem/Q1/type",
                data={"problem_type": "fol_model"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok", "problem_type": "fol_model"})
        self.assertEqual(session.problem_types["Q1"], "fol_model")


if __name__ == "__main__":
    unittest.main()
