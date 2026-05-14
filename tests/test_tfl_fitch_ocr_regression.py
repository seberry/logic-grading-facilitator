"""
TFL Fitch OCR regression tests.

Two layers:
  - **CI-safe tests** (always run): use a fake transcribe function to verify the
    comparison/reporting logic, the TFL domain canonicalization, and the runner
    integration without requiring a live VLM.
  - **Live OCR tests** (opt-in only): run the real Fitch OCR pipeline against
    the handwritten TFL fixtures.  Enable with ``RUN_LIVE_TFL_OCR=1``.

Run CI-safe tests only::

    pytest tests/test_tfl_fitch_ocr_regression.py -v

Run live OCR regressions (requires a running VLM endpoint)::

    RUN_LIVE_TFL_OCR=1 pytest tests/test_tfl_fitch_ocr_regression.py -v -k Live

Important:
    The CI-safe tests below do not prove that the real handwritten OCR is good.
    They validate the TFL comparison/reporting harness using fake transcribers.
    The live tests are intentionally strict and should fail loudly if the real
    OCR times out, fails to produce a final structure, or does not match gold.
"""

import json
import os
import shutil
import subprocess
import sys
import unittest
from pathlib import Path
from uuid import uuid4

from proofs_ocr.domains import tfl as tfl_domain
from proofs_ocr.regression import tfl_runner
from proofs_ocr.regression.runner import GoldRegressionResult, load_manifest, get_repo_root


# ── TFL Domain Canonicalization Tests ────────────────────────────────


class TFLCanonicalizationTests(unittest.TestCase):
    """Unit tests for TFL-specific formula and justification canonicalization."""

    def test_formula_normalizes_ascii_connectives(self) -> None:
        self.assertEqual(
            tfl_domain.canonicalize_formula_for_comparison("(P&Q)->R"),
            "(P∧Q)→R",
        )

    def test_formula_normalizes_tilde_to_not(self) -> None:
        self.assertEqual(
            tfl_domain.canonicalize_formula_for_comparison("~P"),
            "¬P",
        )

    def test_formula_normalizes_v_to_or(self) -> None:
        self.assertEqual(
            tfl_domain.canonicalize_formula_for_comparison("PvQ"),
            "P∨Q",
        )

    def test_formula_normalizes_biconditional(self) -> None:
        self.assertEqual(
            tfl_domain.canonicalize_formula_for_comparison("P<->Q"),
            "P↔Q",
        )

    def test_formula_strips_whitespace(self) -> None:
        self.assertEqual(
            tfl_domain.canonicalize_formula_for_comparison("  P ∧ Q  "),
            "P∧Q",
        )

    def test_formula_empty_string(self) -> None:
        self.assertEqual(tfl_domain.canonicalize_formula_for_comparison(""), "")
        self.assertEqual(tfl_domain.canonicalize_formula_for_comparison(None), "")

    def test_justification_normalizes_premise_alias(self) -> None:
        self.assertEqual(
            tfl_domain.canonicalize_justification_for_comparison("Premise"),
            "Pr",
        )

    def test_justification_normalizes_and_elim(self) -> None:
        self.assertEqual(
            tfl_domain.canonicalize_justification_for_comparison("&E 3"),
            "∧E 3",
        )

    def test_justification_normalizes_arrow_intro(self) -> None:
        self.assertEqual(
            tfl_domain.canonicalize_justification_for_comparison("->I 3-7"),
            "→I 3-7",
        )

    def test_justification_normalizes_tilde_elim(self) -> None:
        self.assertEqual(
            tfl_domain.canonicalize_justification_for_comparison("~E 1,6"),
            "¬E 1,6",
        )

    def test_justification_normalizes_hyp_alias(self) -> None:
        self.assertEqual(
            tfl_domain.canonicalize_justification_for_comparison("assumption"),
            "Hyp",
        )

    def test_justification_empty_string(self) -> None:
        self.assertEqual(tfl_domain.canonicalize_justification_for_comparison(""), "")
        self.assertEqual(tfl_domain.canonicalize_justification_for_comparison(None), "")

    def test_justification_preserves_citation_dashes(self) -> None:
        self.assertEqual(
            tfl_domain.canonicalize_justification_for_comparison("→I 3-7"),
            "→I 3-7",
        )

    def test_justification_normalizes_em_dash(self) -> None:
        self.assertEqual(
            tfl_domain.canonicalize_justification_for_comparison("→I 3–7"),
            "→I 3-7",
        )


# ── TFL Comparison Logic Tests ───────────────────────────────────────


class TFLComparisonTests(unittest.TestCase):
    """Tests for the TFL-specific proof structure comparison functions."""

    def test_compare_identical_structures_passes(self) -> None:
        structure = [
            {"wffstr": "(P∧Q)→R", "jstr": "Pr"},
            {"wffstr": "P→Q", "jstr": "Pr"},
        ]
        passed, diff_text = tfl_runner.compare_proof_structures(structure, structure)
        self.assertTrue(passed)
        self.assertEqual(diff_text, "")

    def test_compare_ignores_ascii_vs_unicode(self) -> None:
        expected = [
            {"wffstr": "(P∧Q)→R", "jstr": "Pr"},
        ]
        actual = [
            {"wffstr": "(P&Q)->R", "jstr": "Premise"},
        ]
        passed, diff_text = tfl_runner.compare_proof_structures(actual, expected)
        self.assertTrue(passed)
        self.assertEqual(diff_text, "")

    def test_compare_detects_formula_mismatch(self) -> None:
        expected = [{"wffstr": "P∧Q", "jstr": "Pr"}]
        actual = [{"wffstr": "P∨Q", "jstr": "Pr"}]
        passed, diff_text = tfl_runner.compare_proof_structures(actual, expected)
        self.assertFalse(passed)
        self.assertIn("P∨Q", diff_text)

    def test_compare_detects_justification_mismatch(self) -> None:
        expected = [{"wffstr": "P", "jstr": "∧E 3"}]
        actual = [{"wffstr": "P", "jstr": "∨E 3"}]
        passed, diff_text = tfl_runner.compare_proof_structures(actual, expected)
        self.assertFalse(passed)

    def test_granular_exact_match(self) -> None:
        structure = [
            {"wffstr": "P∧Q", "jstr": "Pr"},
            [{"wffstr": "P", "jstr": "∧E 3"}],
        ]
        metrics = tfl_runner.compare_proof_structures_granular(structure, structure)
        self.assertEqual(metrics["expected_line_count"], 2)
        self.assertEqual(metrics["actual_line_count"], 2)
        self.assertEqual(metrics["matched_line_count"], 2)
        self.assertEqual(metrics["formula_correct_line_count"], 2)
        self.assertEqual(metrics["justification_correct_line_count"], 2)
        self.assertEqual(metrics["both_correct_line_count"], 2)
        self.assertEqual(metrics["missing_line_count"], 0)
        self.assertEqual(metrics["extra_line_count"], 0)
        self.assertEqual(metrics["structure_mismatch_count"], 0)

    def test_granular_formula_only_mismatch(self) -> None:
        expected = [{"wffstr": "P∧Q", "jstr": "Pr"}]
        actual = [{"wffstr": "P∨Q", "jstr": "Pr"}]
        metrics = tfl_runner.compare_proof_structures_granular(actual, expected)
        self.assertEqual(metrics["formula_correct_line_count"], 0)
        self.assertEqual(metrics["justification_correct_line_count"], 1)
        self.assertEqual(metrics["both_correct_line_count"], 0)

    def test_granular_missing_lines(self) -> None:
        expected = [
            {"wffstr": "P", "jstr": "Pr"},
            {"wffstr": "Q", "jstr": "Pr"},
        ]
        actual = [{"wffstr": "P", "jstr": "Pr"}]
        metrics = tfl_runner.compare_proof_structures_granular(actual, expected)
        self.assertEqual(metrics["missing_line_count"], 1)
        self.assertEqual(metrics["extra_line_count"], 0)

    def test_granular_extra_lines(self) -> None:
        expected = [{"wffstr": "P", "jstr": "Pr"}]
        actual = [
            {"wffstr": "P", "jstr": "Pr"},
            {"wffstr": "Q", "jstr": "Pr"},
        ]
        metrics = tfl_runner.compare_proof_structures_granular(actual, expected)
        self.assertEqual(metrics["missing_line_count"], 0)
        self.assertEqual(metrics["extra_line_count"], 1)

    def test_granular_structure_mismatch(self) -> None:
        expected = [{"wffstr": "A", "jstr": "Pr"}, [{"wffstr": "B", "jstr": "R 1"}]]
        actual = [[{"wffstr": "A", "jstr": "Pr"}], {"wffstr": "B", "jstr": "R 1"}]
        metrics = tfl_runner.compare_proof_structures_granular(actual, expected)
        self.assertEqual(metrics["formula_correct_line_count"], 2)
        self.assertEqual(metrics["justification_correct_line_count"], 2)
        self.assertEqual(metrics["both_correct_line_count"], 2)
        self.assertEqual(metrics["structure_mismatch_count"], 2)

    def test_granular_reports_missing_and_extra(self) -> None:
        expected = [
            {"wffstr": "A", "jstr": "Pr"},
            {"wffstr": "B", "jstr": "R 1"},
        ]
        actual = [{"wffstr": "A", "jstr": "Pr"}]
        missing_metrics = tfl_runner.compare_proof_structures_granular(actual, expected)
        extra_metrics = tfl_runner.compare_proof_structures_granular(expected, actual)
        self.assertEqual(missing_metrics["missing_line_count"], 1)
        self.assertEqual(missing_metrics["extra_line_count"], 0)
        self.assertEqual(extra_metrics["missing_line_count"], 0)
        self.assertEqual(extra_metrics["extra_line_count"], 1)


# ── TFL Runner Integration Tests (CI-safe, fake transcribe) ──────────


class TFLRunnerIntegrationTests(unittest.TestCase):
    """Integration tests for the TFL regression runner using a fake transcribe function."""

    def setUp(self) -> None:
        self.tmp_dir = Path("working") / "test_tmp" / f"tfl_regression_{uuid4().hex}"
        self.tmp_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_select_tfl_cases_returns_only_tfl_handwritten(self) -> None:
        manifest = {
            "cases": [
                {"id": "tfl_a", "domain": "tfl", "kind": "handwritten", "oracle_kind": "gold_curated_from_image"},
                {"id": "modal_b", "domain": "modal", "kind": "handwritten", "oracle_kind": "gold_curated_from_image"},
                {"id": "tfl_c", "domain": "tfl", "kind": "typeset", "oracle_kind": "gold_from_source_tex"},
            ]
        }
        selected = tfl_runner.select_tfl_cases(manifest)
        self.assertEqual([c["id"] for c in selected], ["tfl_a"])

    def test_select_tfl_cases_filters_by_id(self) -> None:
        manifest = {
            "cases": [
                {"id": "tfl_a", "domain": "tfl", "kind": "handwritten", "oracle_kind": "gold_curated_from_image"},
                {"id": "tfl_b", "domain": "tfl", "kind": "handwritten", "oracle_kind": "gold_curated_from_image"},
            ]
        }
        selected = tfl_runner.select_tfl_cases(manifest, case_ids=["tfl_b"])
        self.assertEqual([c["id"] for c in selected], ["tfl_b"])

    def test_run_tfl_gold_regressions_pass_with_matching_structure(self) -> None:
        fixtures_dir = self.tmp_dir / "fixtures"
        fixtures_dir.mkdir(parents=True, exist_ok=True)
        image_path = fixtures_dir / "proof.png"
        image_path.write_bytes(b"fake")
        expected_path = fixtures_dir / "expected.json"
        expected = [
            {"line": 1, "wffstr": "(P∧Q)→R", "jstr": "Pr"},
            {"line": 2, "wffstr": "P→Q", "jstr": "Pr"},
        ]
        expected_path.write_text(json.dumps(expected), encoding="utf-8")

        manifest = {
            "cases": [
                {
                    "id": "tfl_pass_test",
                    "domain": "tfl",
                    "kind": "handwritten",
                    "oracle_kind": "gold_curated_from_image",
                    "image_path": str(image_path.relative_to(self.tmp_dir)).replace("\\", "/"),
                    "expected_structure_path": str(expected_path.relative_to(self.tmp_dir)).replace("\\", "/"),
                },
            ]
        }

        def fake_transcribe(*, image_path, output_dir):
            (output_dir / "final_structure.json").write_text(
                json.dumps(expected, ensure_ascii=False),
                encoding="utf-8",
            )
            return {"run_summary": "fake pass"}

        output_root = self.tmp_dir / "out"
        results = tfl_runner.run_tfl_gold_regressions(
            manifest,
            repo_root=self.tmp_dir,
            output_root=output_root,
            transcribe_fn=fake_transcribe,
        )

        self.assertEqual(len(results), 1)
        self.assertTrue(results[0].passed)
        self.assertEqual(results[0].case_id, "tfl_pass_test")
        self.assertTrue((output_root / "tfl_pass_test" / "gold_regression_result.json").exists())
        self.assertTrue((output_root / "tfl_pass_test" / "gold_regression_metrics.json").exists())

    def test_run_tfl_gold_regressions_fail_with_wrong_structure(self) -> None:
        fixtures_dir = self.tmp_dir / "fixtures_fail"
        fixtures_dir.mkdir(parents=True, exist_ok=True)
        image_path = fixtures_dir / "proof.png"
        image_path.write_bytes(b"fake")
        expected_path = fixtures_dir / "expected.json"
        expected = [
            {"line": 1, "wffstr": "(P∧Q)→R", "jstr": "Pr"},
            {"line": 2, "wffstr": "P→Q", "jstr": "Pr"},
        ]
        expected_path.write_text(json.dumps(expected), encoding="utf-8")

        manifest = {
            "cases": [
                {
                    "id": "tfl_fail_test",
                    "domain": "tfl",
                    "kind": "handwritten",
                    "oracle_kind": "gold_curated_from_image",
                    "image_path": str(image_path.relative_to(self.tmp_dir)).replace("\\", "/"),
                    "expected_structure_path": str(expected_path.relative_to(self.tmp_dir)).replace("\\", "/"),
                },
            ]
        }

        actual_wrong = [
            {"line": 1, "wffstr": "P∨Q", "jstr": "Pr"},
            {"line": 2, "wffstr": "P∧R", "jstr": "∧E 1"},
        ]

        def fake_transcribe(*, image_path, output_dir):
            (output_dir / "final_structure.json").write_text(
                json.dumps(actual_wrong, ensure_ascii=False),
                encoding="utf-8",
            )
            return {"run_summary": "fake fail"}

        output_root = self.tmp_dir / "out_fail"
        results = tfl_runner.run_tfl_gold_regressions(
            manifest,
            repo_root=self.tmp_dir,
            output_root=output_root,
            transcribe_fn=fake_transcribe,
        )

        self.assertEqual(len(results), 1)
        self.assertFalse(results[0].passed)
        metrics = results[0].granular_metrics or {}
        self.assertEqual(metrics.get("formula_correct_line_count"), 0)
        # Line 1 justification "Pr" matches in both expected and actual
        self.assertEqual(metrics.get("justification_correct_line_count"), 1)
        self.assertTrue((output_root / "tfl_fail_test" / "gold_regression_diff.txt").exists())

    def test_run_tfl_gold_regressions_reports_error(self) -> None:
        fixtures_dir = self.tmp_dir / "fixtures_err"
        fixtures_dir.mkdir(parents=True, exist_ok=True)
        image_path = fixtures_dir / "proof.png"
        image_path.write_bytes(b"fake")
        expected_path = fixtures_dir / "expected.json"
        expected_path.write_text(json.dumps([{"wffstr": "A", "jstr": "Pr"}]), encoding="utf-8")

        manifest = {
            "cases": [
                {
                    "id": "tfl_err_test",
                    "domain": "tfl",
                    "kind": "handwritten",
                    "oracle_kind": "gold_curated_from_image",
                    "image_path": str(image_path.relative_to(self.tmp_dir)).replace("\\", "/"),
                    "expected_structure_path": str(expected_path.relative_to(self.tmp_dir)).replace("\\", "/"),
                },
            ]
        }

        def broken_transcribe(*, image_path, output_dir):
            raise RuntimeError("VLM is offline")

        output_root = self.tmp_dir / "out_err"
        results = tfl_runner.run_tfl_gold_regressions(
            manifest,
            repo_root=self.tmp_dir,
            output_root=output_root,
            transcribe_fn=broken_transcribe,
        )

        self.assertEqual(len(results), 1)
        self.assertFalse(results[0].passed)
        self.assertIn("VLM is offline", results[0].error)

    def test_run_tfl_gold_regressions_writes_granular_metrics(self) -> None:
        fixtures_dir = self.tmp_dir / "fixtures_metrics"
        fixtures_dir.mkdir(parents=True, exist_ok=True)
        image_path = fixtures_dir / "proof.png"
        image_path.write_bytes(b"fake")
        expected_path = fixtures_dir / "expected.json"
        expected = [
            {"line": 1, "wffstr": "(P∧Q)→R", "jstr": "Pr"},
            {"line": 2, "wffstr": "P→Q", "jstr": "Pr"},
            [
                {"line": 3, "wffstr": "P∧S", "jstr": "Hyp"},
                {"line": 4, "wffstr": "P", "jstr": "∧E 3"},
                {"line": 5, "wffstr": "Q", "jstr": "→E 2,4"},
            ],
            {"line": 6, "wffstr": "(P∧S)→R", "jstr": "→I 3-5"},
        ]
        expected_path.write_text(json.dumps(expected), encoding="utf-8")

        manifest = {
            "cases": [
                {
                    "id": "tfl_metrics_test",
                    "domain": "tfl",
                    "kind": "handwritten",
                    "oracle_kind": "gold_curated_from_image",
                    "image_path": str(image_path.relative_to(self.tmp_dir)).replace("\\", "/"),
                    "expected_structure_path": str(expected_path.relative_to(self.tmp_dir)).replace("\\", "/"),
                },
            ]
        }

        # Same structure (formula match, justification match, structure match)
        def fake_transcribe(*, image_path, output_dir):
            (output_dir / "final_structure.json").write_text(
                json.dumps(expected, ensure_ascii=False),
                encoding="utf-8",
            )
            return {"run_summary": "ok"}

        output_root = self.tmp_dir / "out_metrics"
        results = tfl_runner.run_tfl_gold_regressions(
            manifest,
            repo_root=self.tmp_dir,
            output_root=output_root,
            transcribe_fn=fake_transcribe,
        )

        self.assertEqual(len(results), 1)
        self.assertTrue(results[0].passed)
        metrics = results[0].granular_metrics or {}
        # 3 top-level lines + 3 subproof lines = 6 total
        self.assertEqual(metrics["expected_line_count"], 6)
        self.assertEqual(metrics["actual_line_count"], 6)
        self.assertEqual(metrics["formula_correct_line_count"], 6)
        self.assertEqual(metrics["justification_correct_line_count"], 6)
        self.assertEqual(metrics["both_correct_line_count"], 6)

    def test_run_tfl_gold_regressions_enriches_verifier_reread_summary_with_gold(self) -> None:
        fixtures_dir = self.tmp_dir / "fixtures_reread_summary"
        fixtures_dir.mkdir(parents=True, exist_ok=True)
        image_path = fixtures_dir / "proof.png"
        image_path.write_bytes(b"fake")
        expected_path = fixtures_dir / "expected.json"
        expected = [
            {"line": 1, "wffstr": "A", "jstr": "Pr"},
            {"line": 2, "wffstr": "A→B", "jstr": "Pr"},
            {"line": 3, "wffstr": "B", "jstr": "→E 1,2"},
        ]
        expected_path.write_text(json.dumps(expected), encoding="utf-8")

        manifest = {
            "cases": [
                {
                    "id": "tfl_reread_summary_test",
                    "domain": "tfl",
                    "kind": "handwritten",
                    "oracle_kind": "gold_curated_from_image",
                    "image_path": str(image_path.relative_to(self.tmp_dir)).replace("\\", "/"),
                    "expected_structure_path": str(expected_path.relative_to(self.tmp_dir)).replace("\\", "/"),
                },
            ]
        }

        actual = [
            {"line": 1, "wffstr": "A", "jstr": "Pr"},
            {"line": 2, "wffstr": "A→B", "jstr": "Pr"},
            {"line": 3, "wffstr": "B", "jstr": "→E 1,3"},
        ]

        def fake_transcribe(*, image_path, output_dir):
            (output_dir / "final_structure.json").write_text(
                json.dumps(actual, ensure_ascii=False),
                encoding="utf-8",
            )
            candidate_records = [
                {
                    "line": 3,
                    "target_kind": "rejected",
                    "source_rejected_line": None,
                    "status": "candidate",
                    "suspicion": {"line": 3, "wffstr": "B", "jstr": "→E 1,3"},
                    "parsed": {"line": 3, "wffstr": "B", "jstr": "→E 1,2"},
                    "parsed_line": 3,
                    "line_matches_target": True,
                    "material_change": True,
                }
            ]
            (output_dir / "verifier_line_reread_candidates.json").write_text(
                json.dumps(candidate_records, ensure_ascii=False),
                encoding="utf-8",
            )
            return {"run_summary": "fake reread summary"}

        output_root = self.tmp_dir / "out_reread_summary"
        tfl_runner.run_tfl_gold_regressions(
            manifest,
            repo_root=self.tmp_dir,
            output_root=output_root,
            transcribe_fn=fake_transcribe,
        )

        summary_path = output_root / "tfl_reread_summary_test" / "verifier_line_reread_summary.json"
        self.assertTrue(summary_path.exists())
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        self.assertTrue(summary["gold_metrics_available"])
        self.assertEqual(summary["records"][0]["candidate_class"], "would_match_gold")
        self.assertTrue(summary["records"][0]["gold"]["candidate_matches_gold"]["both"])
        self.assertEqual(
            json.loads((output_root / "tfl_reread_summary_test" / "final_structure.json").read_text(encoding="utf-8")),
            actual,
        )

    def test_render_tfl_gold_regression_summary(self) -> None:
        fixtures_dir = self.tmp_dir / "fixtures_summary"
        fixtures_dir.mkdir(parents=True, exist_ok=True)
        image_a = fixtures_dir / "a.png"
        image_b = fixtures_dir / "b.png"
        image_a.write_bytes(b"fake")
        image_b.write_bytes(b"fake")
        expected_a = fixtures_dir / "a_expected.json"
        expected_b = fixtures_dir / "b_expected.json"
        expected_a.write_text(json.dumps([{"wffstr": "A", "jstr": "Pr", "line": 1}]), encoding="utf-8")
        expected_b.write_text(json.dumps([{"wffstr": "B", "jstr": "Pr", "line": 1}]), encoding="utf-8")

        manifest = {
            "cases": [
                {
                    "id": "case_a",
                    "domain": "tfl",
                    "kind": "handwritten",
                    "oracle_kind": "gold_curated_from_image",
                    "image_path": str(image_a.relative_to(self.tmp_dir)).replace("\\", "/"),
                    "expected_structure_path": str(expected_a.relative_to(self.tmp_dir)).replace("\\", "/"),
                },
                {
                    "id": "case_b",
                    "domain": "tfl",
                    "kind": "handwritten",
                    "oracle_kind": "gold_curated_from_image",
                    "image_path": str(image_b.relative_to(self.tmp_dir)).replace("\\", "/"),
                    "expected_structure_path": str(expected_b.relative_to(self.tmp_dir)).replace("\\", "/"),
                },
            ]
        }

        def fake_transcribe(*, image_path, output_dir):
            if image_path.name == "a.png":
                payload = [{"wffstr": "A", "jstr": "Pr", "line": 99}]
            else:
                payload = [{"wffstr": "C", "jstr": "Pr", "line": 1}]
            (output_dir / "final_structure.json").write_text(
                json.dumps(payload, ensure_ascii=False),
                encoding="utf-8",
            )
            return {"run_summary": f"ran {image_path.name}"}

        output_root = self.tmp_dir / "out_summary"
        results = tfl_runner.run_tfl_gold_regressions(
            manifest,
            repo_root=self.tmp_dir,
            output_root=output_root,
            transcribe_fn=fake_transcribe,
        )

        summary = tfl_runner.render_tfl_gold_regression_summary(results)
        self.assertIn("TFL Gold cases: 2", summary)
        self.assertIn("Passed: 1", summary)
        self.assertIn("Failed: 1", summary)
        self.assertIn("PASS case_a", summary)
        self.assertIn("FAIL case_b", summary)

    def test_core_tfl_handwritten_cases_in_manifest(self) -> None:
        """Verify the real manifest contains the original core TFL fixtures."""
        manifest = load_manifest()
        selected = tfl_runner.select_tfl_cases(manifest)
        case_ids = {c["id"] for c in selected}
        expected_ids = {
            "tfl_handwritten_q3_P_and_Q_imp_R",
            "tfl_handwritten_q4_N_or_K_and_W",
            "tfl_handwritten_q5_P_or_K_biconditional",
            "tfl_handwritten_q6_S_iff_W",
        }
        self.assertTrue(expected_ids.issubset(case_ids))
        for case in selected:
            self.assertEqual(case["domain"], "tfl")
            self.assertEqual(case["oracle_kind"], "gold_curated_from_image")


# ── Live OCR Regression Tests (opt-in) ───────────────────────────────


_LIVE_TFL_OCR = bool(os.environ.get("RUN_LIVE_TFL_OCR", "").strip())

TFL_HANDWRITTEN_CASE_IDS = [
    "tfl_handwritten_q3_P_and_Q_imp_R",
    "tfl_handwritten_q4_N_or_K_and_W",
    "tfl_handwritten_q5_P_or_K_biconditional",
    "tfl_handwritten_q6_S_iff_W",
    "tfl_fake_implication_intro",
    "tfl_fake_neg_intro_error",
    "tfl_fake_nested_disjunction_scratch_completed",
]


class TFLLiveCaseListTests(unittest.TestCase):
    """CI-safe checks for the opt-in live OCR case registry."""

    def test_live_case_ids_match_manifest_tfl_handwritten_gold(self) -> None:
        manifest = load_manifest()
        selected = tfl_runner.select_tfl_cases(manifest)
        self.assertEqual(TFL_HANDWRITTEN_CASE_IDS, [str(case["id"]) for case in selected])


@unittest.skipUnless(_LIVE_TFL_OCR, "Set RUN_LIVE_TFL_OCR=1 to run live OCR regressions")
class TFLFitchLiveOCRRegressionTests(unittest.TestCase):
    """Live end-to-end OCR regression tests for TFL handwritten fixtures.

    These tests run the real Fitch OCR pipeline (which requires a VLM endpoint)
    and compare the output against curated expected structures.

    Enable with::

        RUN_LIVE_TFL_OCR=1 pytest tests/test_tfl_fitch_ocr_regression.py -v -k Live
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.repo_root = get_repo_root()
        cls.manifest = load_manifest()
        cls.pythonpath = str(cls.repo_root)

    def _run_live_case_subprocess(self, case_id: str, output_root: Path, timeout_seconds: int = 180) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        existing_pythonpath = str(env.get("PYTHONPATH") or "").strip()
        env["PYTHONPATH"] = self.pythonpath if not existing_pythonpath else f"{self.pythonpath}{os.pathsep}{existing_pythonpath}"
        command = [
            sys.executable,
            "-m",
            "proofs_ocr.regression.tfl_runner",
            "run-gold",
            "--case",
            case_id,
            "--output-root",
            str(output_root),
        ]
        return subprocess.run(
            command,
            cwd=self.repo_root,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )

    def _load_case_result_payload(self, output_root: Path, case_id: str) -> dict:
        result_path = output_root / case_id / "gold_regression_result.json"
        self.assertTrue(result_path.exists(), f"Missing gold_regression_result.json for {case_id}")
        return json.loads(result_path.read_text(encoding="utf-8"))

    def _load_case_metrics_payload(self, output_root: Path, case_id: str) -> dict:
        metrics_path = output_root / case_id / "gold_regression_metrics.json"
        self.assertTrue(metrics_path.exists(), f"Missing gold_regression_metrics.json for {case_id}")
        return json.loads(metrics_path.read_text(encoding="utf-8"))

    def test_live_ocr_all_tfl_handwritten_fixtures(self) -> None:
        """Run every handwritten TFL fixture as a strict live OCR gold regression."""
        output_root = self.repo_root / "working" / "tfl_live_ocr_test"
        output_root.mkdir(parents=True, exist_ok=True)

        failures: list[str] = []
        for case_id in TFL_HANDWRITTEN_CASE_IDS:
            try:
                completed = self._run_live_case_subprocess(case_id, output_root)
            except subprocess.TimeoutExpired as exc:
                failures.append(f"{case_id}: timed out after {exc.timeout}s")
                continue

            result_payload = self._load_case_result_payload(output_root, case_id)
            metrics_payload = self._load_case_metrics_payload(output_root, case_id)
            passed = bool(result_payload.get("passed"))
            error_text = str(result_payload.get("error") or "").strip()
            metrics = metrics_payload.get("metrics") or {}
            matched = int(metrics.get("matched_line_count", 0))
            both = int(metrics.get("both_correct_line_count", 0))

            if completed.returncode != 0 or error_text or not passed:
                failures.append(
                    f"{case_id}: returncode={completed.returncode} passed={passed} error={error_text!r} both={both}/{matched}"
                )

        self.assertEqual(
            failures,
            [],
            "Live OCR regressions did not cleanly match gold:\n" + "\n".join(failures),
        )

    def test_live_ocr_q3_matches_gold_without_timeout(self) -> None:
        """Q3 is the canary case: it should finish and match gold exactly."""
        case_id = "tfl_handwritten_q3_P_and_Q_imp_R"
        output_root = self.repo_root / "working" / "tfl_live_q3"
        output_root.mkdir(parents=True, exist_ok=True)

        try:
            completed = self._run_live_case_subprocess(case_id, output_root)
        except subprocess.TimeoutExpired as exc:
            self.fail(f"{case_id} timed out after {exc.timeout}s before producing a gold comparison")

        result_payload = self._load_case_result_payload(output_root, case_id)
        metrics_payload = self._load_case_metrics_payload(output_root, case_id)
        metrics = metrics_payload.get("metrics") or {}

        final_structure_path = output_root / case_id / "final_structure.json"
        diff_path = output_root / case_id / "gold_regression_diff.txt"

        self.assertEqual(
            completed.returncode,
            0,
            f"{case_id} subprocess failed.\nSTDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}",
        )
        self.assertTrue(final_structure_path.exists(), f"{case_id} never produced final_structure.json")
        self.assertEqual(str(result_payload.get("error") or "").strip(), "", f"{case_id} reported a runner error")
        self.assertTrue(bool(result_payload.get("passed")), f"{case_id} did not match gold. Diff path: {diff_path}")
        self.assertEqual(int(metrics.get("missing_line_count", 0)), 0)
        self.assertEqual(int(metrics.get("extra_line_count", 0)), 0)
        self.assertEqual(
            int(metrics.get("both_correct_line_count", 0)),
            int(metrics.get("expected_line_count", 0)),
            f"{case_id} matched only {metrics.get('both_correct_line_count', 0)}/{metrics.get('expected_line_count', 0)} lines",
        )


if __name__ == "__main__":
    unittest.main()
