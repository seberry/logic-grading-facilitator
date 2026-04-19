import unittest

from proofs_ocr.regression import unary_probe


class UnaryProbeTests(unittest.TestCase):
    def test_normalize_probe_formula_text_handles_ascii_modal_prefixes(self) -> None:
        normalized = unary_probe.normalize_probe_formula_text("~[]~~A")

        self.assertEqual(normalized, "¬□¬¬A")

    def test_build_unary_formula_candidates_includes_double_negation_variant(self) -> None:
        candidates = unary_probe.build_unary_formula_candidates(
            formula_text="¬□A",
            justification_text="MC 3",
        )

        self.assertIn("¬□A", candidates)
        self.assertIn("¬□¬A", candidates)
        self.assertIn("¬□¬¬A", candidates)

    def test_build_unary_formula_candidates_preserves_atom_body(self) -> None:
        candidates = unary_probe.build_unary_formula_candidates(
            formula_text="□¬A",
            justification_text="□I 5-7",
        )

        self.assertIn("□A", candidates)
        self.assertIn("□¬A", candidates)
        self.assertIn("□¬¬A", candidates)
        self.assertNotIn("□¬¬B", candidates)

    def test_parse_probe_response_accepts_fenced_json(self) -> None:
        parsed = unary_probe.parse_probe_response(
            """```json
            {"selected_formula":"¬□¬¬A","confidence":0.92}
            ```"""
        )

        self.assertEqual(parsed["selected_formula"], "¬□¬¬A")
        self.assertEqual(parsed["confidence"], 0.92)

    def test_derive_formula_from_structured_counts_rebuilds_box_double_negation(self) -> None:
        reconstructed = unary_probe.derive_formula_from_structured_counts(
            formula_text="□¬A",
            structured_parsed={
                "main_modal_operator": "box",
                "negations_before_modal": 0,
                "negations_before_final_atom": 2,
            },
        )

        self.assertEqual(reconstructed, "□¬¬A")


if __name__ == "__main__":
    unittest.main()
