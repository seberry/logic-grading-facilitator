import json
import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch

from app.modal_proof_deduct import (
    DEFAULT_DEDUCT_BINARY_CANDIDATES,
    DEDUCT_CHECKER_BIN_ENV,
    ModalProofCheckerProtocolError,
    ModalProofCheckerUnavailable,
    _candidate_modal_citation_repairs,
    _candidate_modal_sentence_repairs,
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

    def test_build_request_normalizes_fitch_premise_and_assumption_labels(self) -> None:
        answer_text = json.dumps(
            {
                "lines": [
                    {"wffstr": "A & B", "jstr": "Pr"},
                    [
                        {"wffstr": "A", "jstr": "Hyp"},
                        {"wffstr": "B", "jstr": "∧E 1"},
                    ],
                ]
            }
        )

        request = build_modal_proof_request(answer_text, {"system": "K"})

        self.assertEqual(request["lines"][0]["sentence"], "A ∧ B")
        self.assertEqual(request["lines"][0]["citation"], "PR")
        self.assertEqual(request["lines"][1]["citation"], "PR")

    def test_build_request_normalizes_common_fitch_aliases(self) -> None:
        answer_text = json.dumps(
            [
                {"wffstr": "~A", "jstr": "premise"},
                [
                    {"wffstr": "B", "jstr": "assumption"},
                    {"wffstr": "A", "jstr": "X 1"},
                ],
            ]
        )

        request = build_modal_proof_request(answer_text, {"system": "K"})

        self.assertEqual(request["lines"][0]["sentence"], "¬A")
        self.assertEqual(request["lines"][0]["citation"], "PR")
        self.assertEqual(request["lines"][1]["citation"], "PR")

    def test_run_checker_requires_binary(self) -> None:
        answer_text = json.dumps([{"wffstr": "A", "jstr": "PR"}])
        with (
            patch.dict("os.environ", {DEDUCT_CHECKER_BIN_ENV: ""}, clear=False),
            patch("app.modal_proof_deduct.DEFAULT_DEDUCT_BINARY_CANDIDATES", [Path("missing_deduct_checker_cli.exe")]),
        ):
            with self.assertRaises(ModalProofCheckerUnavailable):
                run_modal_proof_checker(answer_text, {"system": "K"})

    def test_run_checker_uses_default_built_binary_when_env_missing(self) -> None:
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
                    "metadata": {"system": "K"},
                }
            ),
            stderr="",
        )

        with (
            patch.dict("os.environ", {DEDUCT_CHECKER_BIN_ENV: ""}, clear=False),
            patch("app.modal_proof_deduct.DEFAULT_DEDUCT_BINARY_CANDIDATES", [Path("tools/deduct_checker_cli/target/release/deduct_checker_cli.exe")]),
            patch("app.modal_proof_deduct.Path.exists", return_value=True),
            patch("app.modal_proof_deduct.subprocess.run", return_value=completed) as mock_run,
        ):
            result = run_modal_proof_checker(answer_text, {"system": "K"})

        self.assertTrue(result.ok)
        self.assertTrue(str(mock_run.call_args.args[0][0]).endswith("deduct_checker_cli.exe"))

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

    def test_run_checker_uses_utf8_for_unicode_input(self) -> None:
        answer_text = json.dumps([{"wffstr": "A ∧ B", "jstr": "PR"}], ensure_ascii=False)
        completed = subprocess.CompletedProcess(
            args=["fake"],
            returncode=0,
            stdout=json.dumps(
                {
                    "ok": True,
                    "parse_errors": [],
                    "check_errors": [],
                    "conclusion_reached": None,
                    "normalized_lines": [{"depth": 0, "sentence": "A ∧ B", "citation": "PR"}],
                    "metadata": {"system": "K", "checker_version": "test"},
                },
                ensure_ascii=False,
            ),
            stderr="",
        )

        with (
            patch.dict("os.environ", {DEDUCT_CHECKER_BIN_ENV: "C:\\fake\\deduct_checker_cli.exe"}, clear=False),
            patch("app.modal_proof_deduct.Path.exists", return_value=True),
            patch("app.modal_proof_deduct.subprocess.run", return_value=completed) as mock_run,
        ):
            run_modal_proof_checker(answer_text, {"system": "K"})

        self.assertEqual(mock_run.call_args.kwargs.get("encoding"), "utf-8")

    def test_candidate_sentence_repairs_convert_box_lookalike_d(self) -> None:
        self.assertEqual(_candidate_modal_sentence_repairs("D(P→Q)"), ["□(P→Q)"])
        self.assertEqual(_candidate_modal_sentence_repairs("DP"), ["□P"])
        self.assertEqual(_candidate_modal_sentence_repairs("OP"), ["□P"])
        self.assertEqual(_candidate_modal_sentence_repairs("AD"), [])

    def test_candidate_citation_repairs_convert_box_rule_lookalikes(self) -> None:
        self.assertEqual(_candidate_modal_citation_repairs("OE 1"), ["□E 1"])
        self.assertEqual(_candidate_modal_citation_repairs("OI 3-6"), ["□I 3-6"])
        self.assertEqual(_candidate_modal_citation_repairs("DE1"), ["□E 1"])
        self.assertEqual(_candidate_modal_citation_repairs("OE1"), ["□E 1"])
        self.assertEqual(_candidate_modal_citation_repairs("OI3-6"), ["□I 3-6"])
        self.assertEqual(_candidate_modal_citation_repairs("PR"), [])

    def test_run_checker_retries_with_box_lookalike_repair_when_parse_improves(self) -> None:
        answer_text = json.dumps(
            [
                {"wffstr": "□P", "jstr": "PR"},
                {"wffstr": "D(P→Q)", "jstr": "PR"},
                {"wffstr": "□Q", "jstr": "□I 3-6"},
            ],
            ensure_ascii=False,
        )
        first = subprocess.CompletedProcess(
            args=["fake"],
            returncode=0,
            stdout=json.dumps(
                {
                    "ok": False,
                    "parse_errors": [{"line": 2, "code": "parse", "message": "could not parse sentence"}],
                    "check_errors": [],
                    "conclusion_reached": None,
                    "normalized_lines": [],
                    "metadata": {"system": "K"},
                }
            ),
            stderr="",
        )
        second = subprocess.CompletedProcess(
            args=["fake"],
            returncode=0,
            stdout=json.dumps(
                {
                    "ok": True,
                    "parse_errors": [],
                    "check_errors": [],
                    "conclusion_reached": True,
                    "normalized_lines": [
                        {"depth": 0, "sentence": "□P", "citation": "PR"},
                        {"depth": 0, "sentence": "□(P→Q)", "citation": "PR"},
                        {"depth": 0, "sentence": "□Q", "citation": "□I 3-6"},
                    ],
                    "metadata": {"system": "K"},
                },
                ensure_ascii=False,
            ),
            stderr="",
        )

        with (
            patch.dict("os.environ", {DEDUCT_CHECKER_BIN_ENV: "C:\\fake\\deduct_checker_cli.exe"}, clear=False),
            patch("app.modal_proof_deduct.Path.exists", return_value=True),
            patch("app.modal_proof_deduct.subprocess.run", side_effect=[first, second]) as mock_run,
        ):
            result = run_modal_proof_checker(answer_text, {"system": "K"})

        self.assertTrue(result.ok)
        self.assertEqual(result.request_payload["lines"][1]["sentence"], "□(P→Q)")
        self.assertEqual(result.metadata["auto_repairs"][0]["line"], 2)
        self.assertEqual(result.metadata["auto_repairs"][0]["field"], "sentence")
        self.assertEqual(mock_run.call_count, 2)

    def test_run_checker_retries_with_box_rule_repair_when_no_such_rule_improves(self) -> None:
        answer_text = json.dumps(
            [
                {"wffstr": "□P", "jstr": "PR"},
                {"wffstr": "P", "jstr": "OE 1"},
            ],
            ensure_ascii=False,
        )
        first = subprocess.CompletedProcess(
            args=["fake"],
            returncode=0,
            stdout=json.dumps(
                {
                    "ok": False,
                    "parse_errors": [],
                    "check_errors": [{"line": 2, "code": "NoSuchRule", "message": "no such rule"}],
                    "conclusion_reached": None,
                    "normalized_lines": [],
                    "metadata": {"system": "K"},
                }
            ),
            stderr="",
        )
        second = subprocess.CompletedProcess(
            args=["fake"],
            returncode=0,
            stdout=json.dumps(
                {
                    "ok": True,
                    "parse_errors": [],
                    "check_errors": [],
                    "conclusion_reached": None,
                    "normalized_lines": [
                        {"depth": 0, "sentence": "□P", "citation": "PR"},
                        {"depth": 0, "sentence": "P", "citation": "□E 1"},
                    ],
                    "metadata": {"system": "K"},
                },
                ensure_ascii=False,
            ),
            stderr="",
        )

        with (
            patch.dict("os.environ", {DEDUCT_CHECKER_BIN_ENV: "C:\\fake\\deduct_checker_cli.exe"}, clear=False),
            patch("app.modal_proof_deduct.Path.exists", return_value=True),
            patch("app.modal_proof_deduct.subprocess.run", side_effect=[first, second]) as mock_run,
        ):
            result = run_modal_proof_checker(answer_text, {"system": "K"})

        self.assertTrue(result.ok)
        self.assertEqual(result.request_payload["lines"][1]["citation"], "□E 1")
        self.assertEqual(result.metadata["auto_repairs"][0]["field"], "citation")
        self.assertEqual(mock_run.call_count, 2)

    def test_run_checker_can_chain_sentence_and_citation_repairs(self) -> None:
        answer_text = json.dumps(
            [
                {"wffstr": "□P", "jstr": "PR"},
                {"wffstr": "D(P→Q)", "jstr": "PR"},
                [
                    {"wffstr": "□", "jstr": "PR"},
                    {"wffstr": "P", "jstr": "OE1"},
                    {"wffstr": "P→Q", "jstr": "OE2"},
                    {"wffstr": "Q", "jstr": "→E 4 5"},
                ],
                {"wffstr": "□Q", "jstr": "□I 3-6"},
            ],
            ensure_ascii=False,
        )
        first = subprocess.CompletedProcess(
            args=["fake"],
            returncode=0,
            stdout=json.dumps(
                {
                    "ok": False,
                    "parse_errors": [{"line": 2, "code": "parse", "message": "could not parse sentence"}],
                    "check_errors": [],
                    "conclusion_reached": None,
                    "normalized_lines": [],
                    "metadata": {"system": "K"},
                }
            ),
            stderr="",
        )
        second = subprocess.CompletedProcess(
            args=["fake"],
            returncode=0,
            stdout=json.dumps(
                {
                    "ok": False,
                    "parse_errors": [],
                    "check_errors": [
                        {"line": 4, "code": "NoSuchRule", "message": "cited a rule that does not exist or is badly formed"},
                        {"line": 5, "code": "NoSuchRule", "message": "cited a rule that does not exist or is badly formed"},
                    ],
                    "conclusion_reached": None,
                    "normalized_lines": [],
                    "metadata": {"system": "K"},
                }
            ),
            stderr="",
        )
        third = subprocess.CompletedProcess(
            args=["fake"],
            returncode=0,
            stdout=json.dumps(
                {
                    "ok": True,
                    "parse_errors": [],
                    "check_errors": [
                        {"line": 5, "code": "NoSuchRule", "message": "cited a rule that does not exist or is badly formed"},
                    ],
                    "conclusion_reached": None,
                    "normalized_lines": [],
                    "metadata": {"system": "K"},
                }
            ),
            stderr="",
        )
        fourth = subprocess.CompletedProcess(
            args=["fake"],
            returncode=0,
            stdout=json.dumps(
                {
                    "ok": True,
                    "parse_errors": [],
                    "check_errors": [],
                    "conclusion_reached": None,
                    "normalized_lines": [
                        {"depth": 0, "sentence": "□P", "citation": "PR"},
                        {"depth": 0, "sentence": "□(P→Q)", "citation": "PR"},
                        {"depth": 1, "sentence": "□", "citation": "PR"},
                        {"depth": 1, "sentence": "P", "citation": "□E 1"},
                        {"depth": 1, "sentence": "P→Q", "citation": "□E 2"},
                        {"depth": 1, "sentence": "Q", "citation": "→E 4 5"},
                        {"depth": 0, "sentence": "□Q", "citation": "□I 3-6"},
                    ],
                    "metadata": {"system": "K"},
                },
                ensure_ascii=False,
            ),
            stderr="",
        )

        with (
            patch.dict("os.environ", {DEDUCT_CHECKER_BIN_ENV: "C:\\fake\\deduct_checker_cli.exe"}, clear=False),
            patch("app.modal_proof_deduct.Path.exists", return_value=True),
            patch("app.modal_proof_deduct.subprocess.run", side_effect=[first, second, third, fourth]) as mock_run,
        ):
            result = run_modal_proof_checker(answer_text, {"system": "K"})

        self.assertTrue(result.ok)
        self.assertEqual(result.request_payload["lines"][1]["sentence"], "□(P→Q)")
        self.assertEqual(result.request_payload["lines"][3]["citation"], "□E 1")
        self.assertEqual(result.request_payload["lines"][4]["citation"], "□E 2")
        self.assertEqual(len(result.metadata["auto_repairs"]), 3)
        self.assertEqual(mock_run.call_count, 4)

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
