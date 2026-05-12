import unittest
from unittest.mock import patch

from scripts import debug_fitch_ocr


class TestFuzzySnapJustification(unittest.TestCase):
    """Tests for the combinatorial OCR justification repair."""

    def test_valid_head_unchanged(self):
        from proofs_ocr.tricks.justification_snaps import fuzzy_snap_justification
        self.assertEqual(fuzzy_snap_justification("∧E 5"), "∧E 5")

    def test_tail_S_to_5_rescue(self):
        from proofs_ocr.tricks.justification_snaps import fuzzy_snap_justification
        self.assertEqual(fuzzy_snap_justification("∧E S"), "∧E 5")

    def test_tail_O_to_0_rescue(self):
        from proofs_ocr.tricks.justification_snaps import fuzzy_snap_justification
        self.assertEqual(fuzzy_snap_justification("→I O"), "→I 0")

    def test_unknown_head_no_unique_match_returns_tail_fixed(self):
        from proofs_ocr.tricks.justification_snaps import fuzzy_snap_justification
        # "XX" has no unique valid mapping, tail still gets S→5
        result = fuzzy_snap_justification("XX S")
        self.assertTrue(result.endswith("5"))

    def test_empty_string_returns_empty(self):
        from proofs_ocr.tricks.justification_snaps import fuzzy_snap_justification
        self.assertEqual(fuzzy_snap_justification(""), "")

    def test_head_BI_repairs_to_wedge_I(self):
        from proofs_ocr.tricks.justification_snaps import fuzzy_snap_justification
        self.assertEqual(fuzzy_snap_justification("BI 3,4"), "∧I 3,4")


class TestCountTextLinesInRegion(unittest.TestCase):
    """Tests for the horizontal projection line counter."""

    def test_blank_image_returns_1(self):
        import numpy as np
        from proofs_ocr.pipelines.tfl_fitch import count_text_lines_in_region
        blank = np.ones((100, 200, 3), dtype=np.uint8) * 255
        self.assertEqual(count_text_lines_in_region(blank), 1)

    def test_tiny_image_returns_1(self):
        import numpy as np
        from proofs_ocr.pipelines.tfl_fitch import count_text_lines_in_region
        tiny = np.ones((5, 5), dtype=np.uint8) * 255
        self.assertEqual(count_text_lines_in_region(tiny), 1)

    def test_two_text_bands_detected(self):
        import numpy as np
        from proofs_ocr.pipelines.tfl_fitch import count_text_lines_in_region
        img = np.ones((200, 300), dtype=np.uint8) * 255
        # Draw two horizontal bands of "text" (dark pixels)
        img[40:60, 50:250] = 0   # band 1
        img[130:150, 50:250] = 0  # band 2
        self.assertEqual(count_text_lines_in_region(img), 2)

    def test_slanted_repair_line_count_ignores_lined_paper_and_clipped_top_row(self) -> None:
        import cv2
        import numpy as np
        from proofs_ocr.pipelines.tfl_fitch import estimate_line_count

        img = np.ones((620, 900, 3), dtype=np.uint8) * 236
        for y in [20, 120, 220, 320, 420, 520, 610]:
            cv2.line(img, (0, y), (899, y), (145, 145, 145), 2)

        # Clipped carryover from the row above the child crop.
        cv2.putText(img, "A ->", (80, 28), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (25, 25, 25), 4)

        for index, y in enumerate([95, 205, 315, 425, 535], start=2):
            cv2.putText(img, f"A{index}", (120, y + 35), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (25, 25, 25), 4)
            cv2.putText(img, f"E {index}", (500, y + 38), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (25, 25, 25), 4)

        with patch.dict("os.environ", {"TFL_FITCH_SLANTED_BAR_REPAIR": "1"}):
            self.assertEqual(estimate_line_count(img), 5)

    def test_slanted_repair_line_count_handles_fake_implication_child_crop(self) -> None:
        import cv2
        from pathlib import Path
        from proofs_ocr.pipelines.tfl_fitch import estimate_line_count

        fixture_path = (
            Path(__file__).resolve().parents[1]
            / "proofs_ocr"
            / "regression"
            / "fixtures"
            / "tfl"
            / "handwritten"
            / "fake_implication_intro"
            / "image.png"
        )
        img = cv2.imread(str(fixture_path))
        self.assertIsNotNone(img)

        # Approximate the 5-line child subproof crop after slanted-bar repair.
        # The top padding intentionally includes clipped ink from line 1.
        child_crop = img[518:1588, 379:]

        with patch.dict("os.environ", {"TFL_FITCH_SLANTED_BAR_REPAIR": "1"}):
            self.assertEqual(estimate_line_count(child_crop), 5)


class TestSlantedBarRepair(unittest.TestCase):
    def test_slanted_bar_repair_extends_nested_bar_to_opener_row(self) -> None:
        import cv2
        import numpy as np
        from proofs_ocr.pipelines.tfl_fitch import detect_vertical_bars

        img = np.ones((430, 700, 3), dtype=np.uint8) * 255
        cv2.line(img, (125, 25), (110, 370), (0, 0, 0), 3)
        cv2.line(img, (165, 105), (150, 350), (0, 0, 0), 3)

        # Dark text rows to the right of the bars.  The row at y=70 stands in
        # for an assumption/opener that the slanted inner bar should include.
        for y in [70, 105, 155, 205, 255, 345]:
            cv2.rectangle(img, (180, y), (300, y + 18), (0, 0, 0), -1)

        with patch.dict("os.environ", {"TFL_FITCH_SLANTED_BAR_REPAIR": "1"}):
            boxes = detect_vertical_bars(img)

        self.assertEqual(len(boxes), 2)
        outer, inner = sorted(boxes, key=lambda box: box[0])
        self.assertLessEqual(outer[1], 30)
        self.assertLessEqual(inner[1], 75)

    def test_slanted_phone_preprocess_recovers_fake_implication_geometry(self) -> None:
        import cv2
        from pathlib import Path
        from proofs_ocr.pipelines.tfl_fitch import (
            build_proof_tree,
            detect_vertical_bars,
            filter_scaffold_bars,
            preprocess_slanted_phone_crop,
        )

        fixture_path = (
            Path(__file__).resolve().parents[1]
            / "proofs_ocr"
            / "regression"
            / "fixtures"
            / "tfl"
            / "handwritten"
            / "fake_implication_intro"
            / "image.png"
        )
        img = cv2.imread(str(fixture_path))
        self.assertIsNotNone(img)

        with patch.dict("os.environ", {"TFL_FITCH_SLANTED_BAR_REPAIR": "1"}):
            normalized = preprocess_slanted_phone_crop(img)
            bars = filter_scaffold_bars(
                detect_vertical_bars(normalized),
                img_w=normalized.shape[1],
                img_h=normalized.shape[0],
            )
            tree = build_proof_tree(bars, normalized.shape[1], normalized.shape[0])

        self.assertEqual(len(bars), 2)
        self.assertEqual(len(tree.children), 1)
        self.assertEqual(len(tree.children[0].children), 1)
        outer = tree.children[0].box
        inner = tree.children[0].children[0].box
        self.assertLess(inner[0] - outer[0], 80)


class TestBuildOcrPromptWithExpectedLines(unittest.TestCase):
    """Tests for the expected_lines parameter in build_ocr_prompt."""

    def test_leaf_prompt_includes_line_count(self):
        from proofs_ocr.pipelines.tfl_fitch import build_ocr_prompt
        prompt = build_ocr_prompt(has_children=False, expected_lines=2)
        self.assertIn("exactly 2 proof line(s)", prompt)
        self.assertIn("Output exactly 2 line objects", prompt)

    def test_parent_prompt_ignores_line_count(self):
        from proofs_ocr.pipelines.tfl_fitch import build_ocr_prompt
        prompt = build_ocr_prompt(has_children=True, expected_lines=2)
        self.assertNotIn("exactly 2 proof line(s)", prompt)


class TestSeparatedNestedLeafOcr(unittest.TestCase):
    def test_stitch_separated_nested_leaf_ocr_accepts_expected_shape(self) -> None:
        from proofs_ocr.pipelines.tfl_fitch import stitch_separated_nested_leaf_ocr

        payload = {
            "formulas": ["W∧A", "W", "S", "S∧A"],
            "justifications": ["∧E 7", "→E 2,8", "∧I 8,9"],
        }

        self.assertEqual(
            stitch_separated_nested_leaf_ocr(payload, expected_lines=4),
            [
                {"line": 1, "wffstr": "W∧A", "jstr": ""},
                {"line": 2, "wffstr": "W", "jstr": "∧E 7"},
                {"line": 3, "wffstr": "S", "jstr": "→E 2,8"},
                {"line": 4, "wffstr": "S∧A", "jstr": "∧I 8,9"},
            ],
        )

    def test_stitch_separated_nested_leaf_ocr_rejects_count_mismatch(self) -> None:
        from proofs_ocr.pipelines.tfl_fitch import stitch_separated_nested_leaf_ocr

        payload = {
            "formulas": ["W∧A", "W", "S", "S∧A"],
            "justifications": ["∧E 7", "→E 2,8"],
        }

        self.assertIsNone(stitch_separated_nested_leaf_ocr(payload, expected_lines=4))


class DebugFitchOcrNormalizationTests(unittest.TestCase):
    def test_build_ocr_prompt_mentions_formula_justification_separation(self) -> None:
        prompt = debug_fitch_ocr.build_ocr_prompt()

        self.assertIn("Use 'wffstr' only for the formula", prompt)
        self.assertIn("Do not place justification text inside 'wffstr'", prompt)
        self.assertIn("<SUBPROOF_ID_X>", prompt)

    def test_build_ocr_prompt_parent_includes_subproof_token_instruction(self) -> None:
        prompt = debug_fitch_ocr.build_ocr_prompt(has_children=True)

        self.assertIn("<SUBPROOF_ID_X>", prompt)
        self.assertIn("red box token", prompt)

    def test_build_ocr_prompt_leaf_omits_subproof_token_instruction(self) -> None:
        prompt = debug_fitch_ocr.build_ocr_prompt(has_children=False)

        self.assertNotIn("<SUBPROOF_ID_X>", prompt)
        self.assertNotIn("red box token", prompt)
        # Leaf prompt still has the core transcription instructions
        self.assertIn("Use 'wffstr' only for the formula", prompt)

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
