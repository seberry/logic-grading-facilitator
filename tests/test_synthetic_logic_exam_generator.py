import json
import shutil
import unittest
from pathlib import Path
from uuid import uuid4

from app.autograde import autograde_submission
from scripts.generate_synthetic_logic_exam import PROBLEMS, generate


class SyntheticLogicExamGeneratorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = Path("working") / "test_tmp" / f"synthetic_logic_exam_{uuid4().hex}"

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_generate_printed_fol_fixture_shape(self) -> None:
        generate(self.tmp_dir, family="fol_model", style="printed", count=5, seed=7, force=True)

        self.assertTrue((self.tmp_dir / "roster.csv").exists())
        self.assertTrue((self.tmp_dir / "rubric" / "synthetic_fol_models.yaml").exists())
        self.assertTrue((self.tmp_dir / "grades" / "session.json").exists())
        self.assertTrue((self.tmp_dir / "gold" / "fol_model_gold.json").exists())
        self.assertEqual(len(list((self.tmp_dir / "raw_scans").glob("*.png"))), 5)

        session = json.loads((self.tmp_dir / "grades" / "session.json").read_text(encoding="utf-8"))
        self.assertEqual(session["grading_key"]["quiz_id"], "synthetic_fol_models")
        self.assertEqual(len(session["work_units"]), 5)
        self.assertEqual(len(session["grading_key"]["problems"]), len(PROBLEMS))
        self.assertTrue(all(problem["problem_type"] == "fol_model" for problem in session["grading_key"]["problems"]))

        for problem in PROBLEMS:
            segments = json.loads(
                (self.tmp_dir / "grades" / f"segments_{problem.problem_id}.json").read_text(encoding="utf-8")
            )
            self.assertEqual(len(segments), 5)
            self.assertTrue(all(row["present"] for row in segments))

    def test_gold_answers_match_checker_verdicts(self) -> None:
        generate(self.tmp_dir, family="fol_model", style="handwriting_font", count=6, seed=11, force=True)

        session = json.loads((self.tmp_dir / "grades" / "session.json").read_text(encoding="utf-8"))
        gold = json.loads((self.tmp_dir / "gold" / "fol_model_gold.json").read_text(encoding="utf-8"))
        problems = {problem["id"]: problem for problem in session["grading_key"]["problems"]}

        saw_pass = False
        saw_fail = False
        for work_unit in session["work_units"]:
            sid = work_unit["student_id"]
            for problem_id, answer in gold["answers"][sid].items():
                result = autograde_submission(
                    problem_type="fol_model",
                    answer_text=answer["model_text"],
                    autograde_config=problems[problem_id]["autograde"],
                    runtime_options={"trace_mode": "terse"},
                )
                self.assertEqual(result.status, "scored")
                self.assertEqual(result.passed, answer["expected_passed"])
                saw_pass = saw_pass or bool(result.passed)
                saw_fail = saw_fail or not bool(result.passed)

        self.assertTrue(saw_pass)
        self.assertTrue(saw_fail)

    def test_noisy_styles_generate_images(self) -> None:
        for style in ("phone_scan_light", "phone_scan_medium", "faint_pencil", "crowded_answer", "messy_layout"):
            style_dir = self.tmp_dir / style
            generate(style_dir, family="fol_model", style=style, count=2, seed=13, force=True)

            images = sorted((style_dir / "raw_scans").glob("*.png"))
            self.assertEqual(len(images), 2)
            self.assertTrue(all(path.stat().st_size > 10_000 for path in images))


if __name__ == "__main__":
    unittest.main()
