import json
import subprocess
import unittest
from unittest.mock import patch

from app.modal_proof_deduct import (
    DEDUCT_CHECKER_BIN_ENV,
    ModalProofCheckerProtocolError,
    ModalProofCheckerUnavailable,
    build_modal_proof_request,
    run_modal_proof_checker,
)


class ModalProofDeductTests(unittest.TestCase):
    def test_build_request_flattens_nested_structure(self) -> None:
        answer_text = json.dumps(
            [
                {"wffstr": "A", "jstr": "PR"},
                [
                    {"wffstr": "□", "jstr": "PR"},
                    {"wffstr": "□A", "jstr": "□I 2-2"},
                ],
            ]
        )

        request = build_modal_proof_request(
            answer_text,
            {"system": "K", "include_derived_rules": True, "conclusion": "□A"},
        )

        self.assertEqual(request["system"], "K")
        self.assertTrue(request["include_derived_rules"])
        self.assertEqual(request["conclusion"], "□A")
        self.assertEqual([line["depth"] for line in request["lines"]], [0, 1, 1])
        self.assertEqual(request["lines"][0]["sentence"], "A")
        self.assertEqual(request["lines"][2]["citation"], "□I 2-2")

    def test_build_request_uses_location_depth_when_present(self) -> None:
        answer_text = json.dumps(
            {
                "lines": [
                    {"wffstr": "A", "jstr": "PR", "location": [0]},
                    {"wffstr": "□A", "jstr": "R 1", "location": [0, 1]},
                ]
            }
        )

        request = build_modal_proof_request(answer_text, {"system": "T"})

        self.assertEqual([line["depth"] for line in request["lines"]], [0, 1])

    def test_run_checker_requires_binary(self) -> None:
        answer_text = json.dumps([{"wffstr": "A", "jstr": "PR"}])
        with patch.dict("os.environ", {DEDUCT_CHECKER_BIN_ENV: ""}, clear=False):
            with self.assertRaises(ModalProofCheckerUnavailable):
                run_modal_proof_checker(answer_text, {"system": "K"})

    def test_run_checker_parses_success_response(self) -> None:
        answer_text = json.dumps([{"wffstr": "A", "jstr": "PR"}])
        completed = subprocess.CompletedProcess(
            args=["fake"],
            returncode=0,
            stdout=json.dumps(
                {
                    "ok": True,
                    "parse_errors": [],
                    "check_errors": [],
                    "conclusion_reached": True,
                    "normalized_lines": [{"depth": 0, "sentence": "A", "citation": "PR"}],
                    "metadata": {"system": "K", "checker_version": "test"},
                }
            ),
            stderr="",
        )

        with (
            patch.dict("os.environ", {DEDUCT_CHECKER_BIN_ENV: "C:\\fake\\deduct_checker_cli.exe"}, clear=False),
            patch("app.modal_proof_deduct.Path.exists", return_value=True),
            patch("app.modal_proof_deduct.subprocess.run", return_value=completed),
        ):
            result = run_modal_proof_checker(answer_text, {"system": "K"})

        self.assertTrue(result.ok)
        self.assertEqual(result.conclusion_reached, True)
        self.assertEqual(result.normalized_lines[0].sentence, "A")

    def test_run_checker_rejects_bad_json_response(self) -> None:
        answer_text = json.dumps([{"wffstr": "A", "jstr": "PR"}])
        completed = subprocess.CompletedProcess(
            args=["fake"],
            returncode=0,
            stdout="not-json",
            stderr="",
        )

        with (
            patch.dict("os.environ", {DEDUCT_CHECKER_BIN_ENV: "C:\\fake\\deduct_checker_cli.exe"}, clear=False),
            patch("app.modal_proof_deduct.Path.exists", return_value=True),
            patch("app.modal_proof_deduct.subprocess.run", return_value=completed),
        ):
            with self.assertRaises(ModalProofCheckerProtocolError):
                run_modal_proof_checker(answer_text, {"system": "K"})
