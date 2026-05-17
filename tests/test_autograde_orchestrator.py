import unittest

from app.autograde import autograde_submission


class AutoGradeOrchestratorTests(unittest.TestCase):
    def test_skips_when_problem_type_missing(self) -> None:
        result = autograde_submission(problem_type=None, answer_text="", autograde_config={})

        self.assertEqual(result.status, "skipped")
        self.assertIsNone(result.passed)

    def test_returns_unsupported_for_unknown_problem_type(self) -> None:
        result = autograde_submission(
            problem_type="mystery_logic",
            answer_text="",
            autograde_config={},
        )

        self.assertEqual(result.status, "unsupported")
        self.assertIn("mystery_logic", result.summary)

    def test_fol_model_engine_scores_from_config(self) -> None:
        result = autograde_submission(
            problem_type="fol_model",
            answer_text="DOMAIN: {1,2}\n\nCONSTANTS:\n\nPREDICATES:\nP: {1}\nQ: {1}\n",
            autograde_config={
                "task_kind": "non_validity",
                "default_score_correct": 3,
                "default_score_incorrect": 0,
                "prompt_data": {
                    "premises": ["∀x (Px → Qx)", "∃x Px"],
                    "conclusion": "∀x Qx",
                },
            },
        )

        self.assertEqual(result.status, "scored")
        self.assertTrue(result.passed)
        self.assertEqual(result.score, 3.0)
        self.assertEqual(result.problem_type, "fol_model")
        self.assertTrue(result.trace_blocks)

    def test_kripke_model_engine_scores_from_config(self) -> None:
        result = autograde_submission(
            problem_type="kripke_model",
            answer_text="W: {0,1}\nR: <0,1>\nP: {1}\nQ: {}\n",
            autograde_config={
                "task_kind": "non_entailment",
                "default_score_correct": 4,
                "default_score_incorrect": 0,
                "prompt_data": {
                    "world": 0,
                    "premises": ["<>P"],
                    "conclusion": "[]Q",
                },
            },
        )

        self.assertEqual(result.status, "scored")
        self.assertTrue(result.passed)
        self.assertEqual(result.score, 4.0)
        self.assertIn("countermodel", result.summary)


if __name__ == "__main__":
    unittest.main()
