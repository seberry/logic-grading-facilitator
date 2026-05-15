import json
import shutil
import unittest
from pathlib import Path
from uuid import uuid4

from app.autograde import autograde_submission
from scripts.generate_fol_model_exam_demo import STUDENTS, generate


class FolModelExamDemoTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = Path("working") / "test_tmp" / f"fol_model_exam_demo_{uuid4().hex}"

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_generator_creates_loadable_demo_assignment(self) -> None:
        generate(self.tmp_dir, force=True)

        session_path = self.tmp_dir / "grades" / "session.json"
        rubric_path = self.tmp_dir / "rubric" / "fol_model_exam_demo.yaml"

        self.assertTrue(session_path.exists())
        self.assertTrue(rubric_path.exists())
        self.assertEqual(len(list((self.tmp_dir / "raw_scans").glob("*.png"))), len(STUDENTS))

        session = json.loads(session_path.read_text(encoding="utf-8"))
        self.assertEqual(session["grading_key"]["quiz_id"], "fol_model_exam_demo")
        self.assertEqual(len(session["work_units"]), len(STUDENTS))
        self.assertEqual(session["grading_key"]["problems"][0]["problem_type"], "fol_model")

        for problem_id in ("Q1", "Q2", "Q3"):
            segments = json.loads((self.tmp_dir / "grades" / f"segments_{problem_id}.json").read_text(encoding="utf-8"))
            self.assertEqual(len(segments), len(STUDENTS))
            self.assertTrue(all(row["present"] for row in segments))

    def test_demo_answers_include_passing_and_failing_models(self) -> None:
        generate(self.tmp_dir, force=True)
        session = json.loads((self.tmp_dir / "grades" / "session.json").read_text(encoding="utf-8"))
        problems = {p["id"]: p for p in session["grading_key"]["problems"]}

        expected = {
            "Q1": [True, False, True, False],
            "Q2": [True, False, True, True],
            "Q3": [True, False, True, True],
        }
        student_answers = {student.student_id: student.answers for student in STUDENTS}

        for problem_id, expected_results in expected.items():
            config = problems[problem_id]["autograde"]
            actual_results = []
            for work_unit in session["work_units"]:
                answer_text = student_answers[work_unit["student_id"]][problem_id]
                result = autograde_submission(
                    problem_type="fol_model",
                    answer_text=answer_text,
                    autograde_config=config,
                    runtime_options={"trace_mode": "terse"},
                )
                actual_results.append(result.passed)

            self.assertEqual(actual_results, expected_results)


if __name__ == "__main__":
    unittest.main()
