import json
import shutil
import unittest
from pathlib import Path
from uuid import uuid4

from proofs_ocr.regression import runner


class ProofOcrRegressionRunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = Path("working") / "test_tmp" / f"proofs_ocr_regression_{uuid4().hex}"
        self.tmp_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_select_cases_defaults_to_gold_oracles(self) -> None:
        manifest = {
            "cases": [
                {"id": "gold_a", "oracle_kind": "gold_from_source_tex", "kind": "typeset"},
                {"id": "gold_b", "oracle_kind": "gold_curated_from_image", "kind": "handwritten"},
                {"id": "smoke", "oracle_kind": "smoke_only", "kind": "typeset"},
            ]
        }

        selected = runner.select_cases(manifest)

        self.assertEqual([case["id"] for case in selected], ["gold_a", "gold_b"])

    def test_compare_proof_structures_ignores_line_metadata(self) -> None:
        expected = [
            {"line_label": "1", "wffstr": "□A", "jstr": "PR", "line": 1},
            {"line_label": "2", "wffstr": "□B", "jstr": "PR", "line": 2},
        ]
        actual = [
            {"wffstr": "□A", "jstr": "PR", "line": 9},
            {"wffstr": "□B", "jstr": "PR", "line": 10},
        ]

        passed, diff_text = runner.compare_proof_structures(actual, expected)

        self.assertTrue(passed)
        self.assertEqual(diff_text, "")

    def test_flatten_proof_structure_records_nesting_path(self) -> None:
        structure = [
            {"line": 1, "wffstr": "P", "jstr": "PR"},
            [
                {"line": 2, "wffstr": "A", "jstr": "PR"},
                {"line": 3, "wffstr": "B", "jstr": "R 1"},
            ],
            {"line": 4, "wffstr": "Q", "jstr": "R 2"},
        ]

        flattened = runner.flatten_proof_structure(structure)

        self.assertEqual([line.wffstr for line in flattened], ["P", "A", "B", "Q"])
        self.assertEqual([line.nesting_path for line in flattened], [(), (2,), (2,), ()])

    def test_compare_proof_structures_granular_reports_exact_match(self) -> None:
        structure = [
            {"wffstr": "A", "jstr": "PR"},
            [{"wffstr": "B", "jstr": "R 1"}],
        ]

        metrics = runner.compare_proof_structures_granular(structure, structure)

        self.assertEqual(metrics["expected_line_count"], 2)
        self.assertEqual(metrics["actual_line_count"], 2)
        self.assertEqual(metrics["matched_line_count"], 2)
        self.assertEqual(metrics["formula_correct_line_count"], 2)
        self.assertEqual(metrics["justification_correct_line_count"], 2)
        self.assertEqual(metrics["both_correct_line_count"], 2)
        self.assertEqual(metrics["missing_line_count"], 0)
        self.assertEqual(metrics["extra_line_count"], 0)
        self.assertEqual(metrics["structure_mismatch_count"], 0)

    def test_compare_proof_structures_granular_reports_formula_only_mismatch(self) -> None:
        expected = [{"wffstr": "A", "jstr": "PR"}]
        actual = [{"wffstr": "B", "jstr": "PR"}]

        metrics = runner.compare_proof_structures_granular(actual, expected)

        self.assertEqual(metrics["formula_correct_line_count"], 0)
        self.assertEqual(metrics["justification_correct_line_count"], 1)
        self.assertEqual(metrics["both_correct_line_count"], 0)

    def test_compare_proof_structures_granular_reports_justification_only_mismatch(self) -> None:
        expected = [{"wffstr": "A", "jstr": "PR"}]
        actual = [{"wffstr": "A", "jstr": "R 1"}]

        metrics = runner.compare_proof_structures_granular(actual, expected)

        self.assertEqual(metrics["formula_correct_line_count"], 1)
        self.assertEqual(metrics["justification_correct_line_count"], 0)
        self.assertEqual(metrics["both_correct_line_count"], 0)

    def test_compare_proof_structures_granular_reports_missing_and_extra_lines(self) -> None:
        expected = [{"wffstr": "A", "jstr": "PR"}, {"wffstr": "B", "jstr": "R 1"}]
        actual = [{"wffstr": "A", "jstr": "PR"}]

        missing_metrics = runner.compare_proof_structures_granular(actual, expected)
        extra_metrics = runner.compare_proof_structures_granular(expected, actual)

        self.assertEqual(missing_metrics["missing_line_count"], 1)
        self.assertEqual(missing_metrics["extra_line_count"], 0)
        self.assertEqual(extra_metrics["missing_line_count"], 0)
        self.assertEqual(extra_metrics["extra_line_count"], 1)

    def test_compare_proof_structures_granular_reports_structure_mismatch_with_matching_text(self) -> None:
        expected = [{"wffstr": "A", "jstr": "PR"}, [{"wffstr": "B", "jstr": "R 1"}]]
        actual = [[{"wffstr": "A", "jstr": "PR"}], {"wffstr": "B", "jstr": "R 1"}]

        metrics = runner.compare_proof_structures_granular(actual, expected)

        self.assertEqual(metrics["formula_correct_line_count"], 2)
        self.assertEqual(metrics["justification_correct_line_count"], 2)
        self.assertEqual(metrics["both_correct_line_count"], 2)
        self.assertEqual(metrics["structure_mismatch_count"], 2)

    def test_run_gold_regressions_writes_pass_fail_reports(self) -> None:
        fixtures_dir = self.tmp_dir / "fixtures"
        fixtures_dir.mkdir(parents=True, exist_ok=True)
        image_a = fixtures_dir / "a.png"
        image_b = fixtures_dir / "b.png"
        image_a.write_bytes(b"fake")
        image_b.write_bytes(b"fake")
        expected_a = fixtures_dir / "a_expected.json"
        expected_b = fixtures_dir / "b_expected.json"
        expected_a.write_text(json.dumps([{"wffstr": "A", "jstr": "PR", "line": 1}]), encoding="utf-8")
        expected_b.write_text(json.dumps([{"wffstr": "B", "jstr": "PR", "line": 1}]), encoding="utf-8")

        manifest = {
            "cases": [
                {
                    "id": "case_a",
                    "oracle_kind": "gold_curated_from_image",
                    "image_path": str(image_a.relative_to(self.tmp_dir)).replace("\\", "/"),
                    "expected_structure_path": str(expected_a.relative_to(self.tmp_dir)).replace("\\", "/"),
                    "kind": "handwritten",
                },
                {
                    "id": "case_b",
                    "oracle_kind": "gold_curated_from_image",
                    "image_path": str(image_b.relative_to(self.tmp_dir)).replace("\\", "/"),
                    "expected_structure_path": str(expected_b.relative_to(self.tmp_dir)).replace("\\", "/"),
                    "kind": "handwritten",
                    "tags": ["s4"],
                },
            ]
        }

        def fake_transcribe(*, image_path, output_dir, modal_config, vision_base_url, vision_model):
            del vision_base_url, vision_model
            if image_path.name == "a.png":
                payload = [{"wffstr": "A", "jstr": "PR", "line": 99}]
            else:
                payload = [{"wffstr": "C", "jstr": "PR", "line": 1}]
            (output_dir / "final_structure_baseline.json").write_text(
                json.dumps(payload, ensure_ascii=False),
                encoding="utf-8",
            )
            return {"run_summary": f"ran {image_path.name} under {modal_config['system']}"}

        output_root = self.tmp_dir / "out"
        results = runner.run_gold_regressions(
            manifest,
            repo_root=self.tmp_dir,
            output_root=output_root,
            transcribe_fn=fake_transcribe,
        )

        self.assertEqual(len(results), 2)
        result_a = next(item for item in results if item.case_id == "case_a")
        result_b = next(item for item in results if item.case_id == "case_b")
        self.assertTrue(result_a.passed)
        self.assertFalse(result_b.passed)
        self.assertIn("ran a.png under K", result_a.run_summary)
        self.assertIn("ran b.png under S4", result_b.run_summary)
        self.assertTrue((output_root / "case_a" / "gold_regression_result.json").exists())
        self.assertTrue((output_root / "case_a" / "gold_regression_metrics.json").exists())
        self.assertTrue((output_root / "case_b" / "gold_regression_diff.txt").exists())
        metrics_b = json.loads((output_root / "case_b" / "gold_regression_metrics.json").read_text(encoding="utf-8"))
        self.assertEqual(metrics_b["metrics"]["formula_correct_line_count"], 0)
        self.assertEqual(metrics_b["metrics"]["justification_correct_line_count"], 1)

        summary = runner.render_gold_regression_summary(results)
        self.assertIn("Gold cases: 2", summary)
        self.assertIn("Passed: 1", summary)
        self.assertIn("Failed: 1", summary)
        self.assertIn("Aggregate lines: formula 1/2, justification 2/2, both 1/2", summary)
        self.assertIn("FAIL case_b", summary)
        self.assertIn("formula 0/1, justification 1/1, both 0/1", summary)


if __name__ == "__main__":
    unittest.main()
