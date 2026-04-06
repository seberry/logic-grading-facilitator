import unittest
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4
import shutil

from fastapi.testclient import TestClient

from app import server
from app.models import GradingKey, ProblemSpec, RosterEntry, Session, WorkUnit
from app.routes import problem_strips as problem_strips_routes
from app.storage import load_grading_key


def _build_session() -> Session:
    return Session(
        roster=[RosterEntry(student_id="s1", name="Berry, Sharon")],
        grading_key=GradingKey(
            quiz_id="quiz1",
            max_points=10,
            problems=[
                ProblemSpec(
                    id="Q1",
                    label="FOL model",
                    points=5,
                    section="Models",
                    problem_number="1",
                    prompt="Create a model making the target sentence false.",
                    segmentation_hint="Look in the Models section, problem 1.",
                    rubric_notes="Full credit for a model with the required evaluation.",
                ),
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
        self.assertIn("What the problem asked", response.text)
        self.assertIn("Models", response.text)

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

    def test_problem_strips_prefers_rubric_problem_type(self) -> None:
        session = _build_session()
        session.grading_key.problems[0].problem_type = "fol_model"
        session.problem_types["Q1"] = "fitch_proof"
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
        self.assertIn("fol model", response.text.lower())
        self.assertIn('data-type="fol_model"', response.text)


class RubricSchemaTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = Path("working") / "test_tmp" / f"rubric_schema_{uuid4().hex}"
        self.tmp_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_load_grading_key_preserves_problem_type_and_autograde(self) -> None:
        yaml_text = """
quiz_id: quiz1
max_points: 5
problems:
  - id: Q1
    label: FOL model
    points: 5
    problem_type: fol_model
    autograde:
      task_kind: non_validity
      prompt_data:
        premises:
          - "Pa"
        conclusion: "Qa"
"""
        rubric_path = self.tmp_dir / "quiz1.yaml"
        rubric_path.write_text(yaml_text, encoding="utf-8")
        grading_key = load_grading_key(rubric_path)

        self.assertEqual(grading_key.problems[0].problem_type, "fol_model")
        self.assertEqual(grading_key.problems[0].autograde["task_kind"], "non_validity")
        self.assertEqual(
            grading_key.problems[0].autograde["prompt_data"]["conclusion"],
            "Qa",
        )

    def test_load_grading_key_preserves_problem_context_fields(self) -> None:
        yaml_text = """
quiz_id: quiz1
max_points: 5
problems:
  - id: Q1
    label: Modal model
    points: 5
    section: Models
    problem_number: "3"
    prompt: Show that two sentences are not equivalent.
    segmentation_hint: Look under the Models heading for item 3.
    rubric_notes: Full credit needs a countermodel at w0.
"""
        rubric_path = self.tmp_dir / "quiz1_context.yaml"
        rubric_path.write_text(yaml_text, encoding="utf-8")
        grading_key = load_grading_key(rubric_path)

        problem = grading_key.problems[0]
        self.assertEqual(problem.section, "Models")
        self.assertEqual(problem.problem_number, "3")
        self.assertEqual(problem.prompt, "Show that two sentences are not equivalent.")
        self.assertEqual(problem.segmentation_hint, "Look under the Models heading for item 3.")
        self.assertEqual(problem.rubric_notes, "Full credit needs a countermodel at w0.")


if __name__ == "__main__":
    unittest.main()
