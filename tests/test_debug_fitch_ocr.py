import unittest

from scripts import debug_fitch_ocr


class DebugFitchOcrNormalizationTests(unittest.TestCase):
    def test_build_ocr_prompt_mentions_formula_justification_separation(self) -> None:
        prompt = debug_fitch_ocr.build_ocr_prompt()

        self.assertIn("Use 'wffstr' only for the formula", prompt)
        self.assertIn("Do not place justification text inside 'wffstr'", prompt)
        self.assertIn("<SUBPROOF_ID_X>", prompt)

    def test_normalize_structure_splits_embedded_justification_from_formula(self) -> None:
        structure = [
            {
                "line": 3,
                "wffstr": "X -> S ->I 2-5",
                "jstr": "",
            }
        ]

        debug_fitch_ocr.normalize_proof_structure(structure)

        self.assertEqual(structure[0]["wffstr"], "X → S")
        self.assertEqual(structure[0]["jstr"], "→I 2-5")

    def test_normalize_structure_treats_digit_only_justification_as_blank(self) -> None:
        structure = [
            {"line": 1, "wffstr": "P ↔ Q", "jstr": "2"},
        ]

        debug_fitch_ocr.normalize_proof_structure(structure)

        self.assertEqual(structure[0]["wffstr"], "P ↔ Q")
        self.assertEqual(structure[0]["jstr"], "")

    def test_normalize_structure_recovers_contradiction_line_from_justification_only_formula(self) -> None:
        structure = [
            {"line": 5, "wffstr": "¬E 4,5", "jstr": ""},
        ]

        debug_fitch_ocr.normalize_proof_structure(structure)

        self.assertEqual(structure[0]["wffstr"], "⊥")
        self.assertEqual(structure[0]["jstr"], "¬E 4,5")

    def test_normalize_structure_expands_compact_rule_citations(self) -> None:
        structure = [
            {"line": 3, "wffstr": "P", "jstr": "&E1"},
            {"line": 4, "wffstr": "Q", "jstr": "<->E2,3"},
            {"line": 7, "wffstr": "A", "jstr": "X6"},
        ]

        debug_fitch_ocr.normalize_proof_structure(structure)

        self.assertEqual(structure[0]["jstr"], "∧E 1")
        self.assertEqual(structure[1]["jstr"], "↔E 2,3")
        self.assertEqual(structure[2]["jstr"], "X 6")

    def test_filter_scaffold_bars_keeps_full_height_left_spine(self) -> None:
        """Scaffold removal is disabled — the outer bar is a real subproof delimiter."""
        boxes = [(30, 0, 12, 460), (120, 80, 8, 140)]

        filtered = debug_fitch_ocr.filter_scaffold_bars(boxes, img_w=560, img_h=480)

        self.assertEqual(filtered, [(30, 0, 12, 460), (120, 80, 8, 140)])

    def test_filter_scaffold_bars_keeps_tall_outer_bar_even_at_edge(self) -> None:
        """Scaffold removal is disabled — keep both bars for the tree builder."""
        boxes = [(118, 45, 17, 538), (140, 242, 8, 138)]

        filtered = debug_fitch_ocr.filter_scaffold_bars(boxes, img_w=564, img_h=564)

        self.assertEqual(filtered, [(118, 45, 17, 538), (140, 242, 8, 138)])

    def test_filter_scaffold_bars_drops_extreme_page_edge_artifacts_before_scaffold_logic(self) -> None:
        """Narrow bars at the absolute page edges are removed; interior bars survive."""
        boxes = [(0, 0, 9, 285), (559, 99, 20, 675), (612, 328, 20, 391), (1651, 0, 5, 197)]

        filtered = debug_fitch_ocr.filter_scaffold_bars(boxes, img_w=1656, img_h=875)

        # (0,0,9,285) dropped: narrow + left edge.  (1651,0,5,197) dropped: narrow + right edge.
        # (559,99,20,675) survives: not at edge.  (612,328,20,391) survives: not at edge.
        self.assertEqual(filtered, [(559, 99, 20, 675), (612, 328, 20, 391)])

    def test_filter_scaffold_bars_keeps_lone_tall_bar_after_edge_cleanup(self) -> None:
        """After edge-artifact cleanup, a single remaining bar that is NOT at
        the left edge should be kept — it is a genuine subproof bar, not
        page scaffolding.  (Regression: the code used to drop it.)"""
        boxes = [(0, 0, 7, 201), (215, 148, 23, 517), (1581, 0, 3, 101)]

        filtered = debug_fitch_ocr.filter_scaffold_bars(boxes, img_w=1584, img_h=730)

        self.assertEqual(filtered, [(215, 148, 23, 517)])

    def test_remove_empty_lines_strips_blank_ocr_noise(self) -> None:
        structure = [
            {"wffstr": "A → ¬B", "jstr": ""},
            {"wffstr": "", "jstr": ""},
            [
                {"wffstr": "A ∧ B", "jstr": ""},
                {"wffstr": "", "jstr": ""},
            ],
            {"wffstr": "", "jstr": ""},
        ]

        debug_fitch_ocr.remove_empty_lines(structure)

        self.assertEqual(len(structure), 2)
        self.assertEqual(len(structure[1]), 1)

    def test_infer_missing_rule_heads_restores_neg_intro(self) -> None:
        structure = [
            {"wffstr": "¬(A ∧ B)", "jstr": "I 2-6"},
        ]

        debug_fitch_ocr.infer_missing_rule_heads(structure)

        self.assertEqual(structure[0]["jstr"], "¬I 2-6")


if __name__ == "__main__":
    unittest.main()
