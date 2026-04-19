import shutil
import unittest
from pathlib import Path
from uuid import uuid4

from PIL import Image

from proofs_ocr.regression import unary_probe_suite


class UnaryProbeSuiteTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = Path("working") / "test_tmp" / f"unary_probe_suite_{uuid4().hex}"
        self.tmp_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_default_probe_suite_cases_include_double_negation_targets(self) -> None:
        cases = unary_probe_suite.default_probe_suite_cases()
        case_ids = {case.case_id for case in cases}

        self.assertIn("box_inner_double_negation", case_ids)
        self.assertIn("negated_box_inner_double_negation", case_ids)
        self.assertIn("atom_triple_negation", case_ids)

    def test_render_formula_image_writes_nonempty_image(self) -> None:
        output_path = unary_probe_suite.render_formula_image(
            formula_text="¬□¬¬A",
            output_path=self.tmp_dir / "formula.png",
        )

        self.assertTrue(output_path.exists())
        image = Image.open(output_path)
        self.assertGreaterEqual(image.size[0], 120)
        self.assertGreaterEqual(image.size[1], 70)

    def test_count_formula_blob_components_separates_single_double_and_triple_negation(self) -> None:
        single_path = unary_probe_suite.render_formula_image(
            formula_text="¬A",
            output_path=self.tmp_dir / "single.png",
        )
        double_path = unary_probe_suite.render_formula_image(
            formula_text="¬¬A",
            output_path=self.tmp_dir / "double.png",
        )
        triple_path = unary_probe_suite.render_formula_image(
            formula_text="¬¬¬A",
            output_path=self.tmp_dir / "triple.png",
        )

        single_count = unary_probe_suite.count_formula_blob_components(single_path)
        double_count = unary_probe_suite.count_formula_blob_components(double_path)
        triple_count = unary_probe_suite.count_formula_blob_components(triple_path)

        self.assertEqual(single_count, 2)
        self.assertEqual(double_count, 3)
        self.assertEqual(triple_count, 4)

    def test_count_unary_formula_symbol_slots_matches_simple_formula_shapes(self) -> None:
        self.assertEqual(unary_probe_suite.count_unary_formula_symbol_slots("¬A"), 2)
        self.assertEqual(unary_probe_suite.count_unary_formula_symbol_slots("¬¬A"), 3)
        self.assertEqual(unary_probe_suite.count_unary_formula_symbol_slots("□¬¬A"), 4)

    def test_render_probe_suite_summary_reports_pass_fail_counts(self) -> None:
        results = [
            unary_probe_suite.UnaryProbeSuiteResult(
                case_id="a",
                expected_formula="¬¬A",
                current_ocr_formula="¬A",
                structured_formula="¬¬A",
                forced_choice_formula="¬¬A",
                structured_passed=True,
                forced_choice_passed=True,
                structured_changed=True,
                forced_choice_changed=True,
                image_blob_count=3,
                current_ocr_symbol_count=2,
                blob_count_exceeds_ocr=True,
                confidence="1.0",
                result_path="a.json",
                notes="",
            ),
            unary_probe_suite.UnaryProbeSuiteResult(
                case_id="b",
                expected_formula="□¬¬A",
                current_ocr_formula="□¬A",
                structured_formula="□¬¬A",
                forced_choice_formula="□¬A",
                structured_passed=True,
                forced_choice_passed=False,
                structured_changed=True,
                forced_choice_changed=False,
                image_blob_count=4,
                current_ocr_symbol_count=3,
                blob_count_exceeds_ocr=True,
                confidence="0.4",
                result_path="b.json",
                notes="",
            ),
        ]

        summary = unary_probe_suite.render_probe_suite_summary(results)

        self.assertIn("Unary probe cases: 2", summary)
        self.assertIn("Structured-count reconstruction passed: 2", summary)
        self.assertIn("Forced choice failed: 1", summary)
        self.assertIn("CV blob-count flags (image > OCR symbols): 2", summary)
        self.assertIn("b: expected □¬¬A, structured=□¬¬A (PASS), forced=□¬A (FAIL), blobs=4, ocr_symbols=3, blob_flag=yes", summary)


if __name__ == "__main__":
    unittest.main()
