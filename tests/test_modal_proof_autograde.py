import json
import unittest
from unittest.mock import patch

from app.autograde import autograde_submission
from app.modal_proof_deduct import ModalProofCheckerUnavailable


class ModalProofAutogradeTests(unittest.TestCase):
    def test_modal_proof_scores_with_mocked_checker(self) -> None:
        answer_text = json.dumps([{"wffstr": "A", "jstr": "PR"}])

        class ResultObj:
            ok = True
            parse_errors = []
            check_errors = []
            conclusion_reached = True
            normalized_lines = []
            metadata = {"system": "K"}

        with patch("app.autograde.modal_proof.run_modal_proof_checker", return_value=ResultObj()):
            result = autograde_submission(
                problem_type="modal_proof",
                answer_text=answer_text,
                autograde_config={
                    "system": "K",
                    "conclusion": "A",
                    "default_score_correct": 2,
                    "default_score_incorrect": 0,
                },
            )

        self.assertEqual(result.status, "scored")
        self.assertTrue(result.passed)
        self.assertEqual(result.score, 2.0)
        self.assertIn("conclusion reached", result.summary.lower())

    def test_modal_proof_invalid_system_returns_invalid_input(self) -> None:
        answer_text = json.dumps([{"wffstr": "A", "jstr": "PR"}])
        result = autograde_submission(
            problem_type="modal_proof",
            answer_text=answer_text,
            autograde_config={"system": "XYZ"},
        )

        self.assertEqual(result.status, "invalid_input")
        self.assertIn("modal_proof requires autograde.system", result.summary)

    def test_modal_proof_unavailable_binary_returns_not_implemented(self) -> None:
        answer_text = json.dumps([{"wffstr": "A", "jstr": "PR"}])

        with patch(
            "app.autograde.modal_proof.run_modal_proof_checker",
            side_effect=ModalProofCheckerUnavailable("checker binary missing"),
        ):
            result = autograde_submission(
                problem_type="modal_proof",
                answer_text=answer_text,
                autograde_config={"system": "K"},
            )

        self.assertEqual(result.status, "not_implemented")
        self.assertIn("checker binary missing", result.summary)
