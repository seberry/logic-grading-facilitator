"""
Tests for TFL justification normalization improvements and re-scoring
of saved handwritten OCR probe results.

Phase 1: Space-between-digits → comma normalization.
Phase 2: Handwritten ∧→8 post-OCR repair heuristic.
"""

import json
import unittest
from pathlib import Path

from proofs_ocr.domains.tfl import (
    canonicalize_formula_for_comparison,
    canonicalize_justification_for_comparison,
)
from proofs_ocr.regression.tfl_runner import compare_proof_structures_granular


# ── Unit tests for canonicalization ──────────────────────────────────


class TestJustificationTailNormalization(unittest.TestCase):
    """Tests for space-between-digits → comma normalization in justification tails."""

    def test_space_between_digits_becomes_comma(self):
        """OCR '∨E 1, 3-4 5-6' should match gold '∨E 1,3-4,5-6'."""
        self.assertEqual(
            canonicalize_justification_for_comparison("∨E 1, 3-4 5-6"),
            canonicalize_justification_for_comparison("∨E 1,3-4,5-6"),
        )

    def test_single_digit_tail_unchanged(self):
        """Single citation like '∧E 5' should remain unchanged."""
        result = canonicalize_justification_for_comparison("∧E 5")
        self.assertIn("5", result)

    def test_dash_range_preserved(self):
        """Dash ranges like '3-7' should be preserved (not get comma inserted)."""
        result = canonicalize_justification_for_comparison("→I 3-7")
        self.assertEqual(result, "→I 3-7")

    def test_comma_already_present_unchanged(self):
        """Already-correct comma formatting should remain unchanged."""
        result = canonicalize_justification_for_comparison("→E 2,4")
        self.assertEqual(result, "→E 2,4")

    def test_mixed_comma_and_space_becomes_comma(self):
        """'∨E 1, 3-4 5-6' normalizes tail '3-4 5-6' → '3-4,5-6'."""
        result = canonicalize_justification_for_comparison("∨E 1, 3-4 5-6")
        self.assertEqual(result, "∨E 1,3-4,5-6")

    def test_premise_stays_premise(self):
        """'Pr' should normalize to 'Pr'."""
        self.assertEqual(canonicalize_justification_for_comparison("Pr"), "Pr")

    def test_hyp_stays_hyp(self):
        """'Hyp' should normalize to 'Hyp'."""
        self.assertEqual(canonicalize_justification_for_comparison("Hyp"), "Hyp")

    def test_empty_string(self):
        """Empty string should normalize to empty string."""
        self.assertEqual(canonicalize_justification_for_comparison(""), "")

    def test_biconditional_elim_with_commas(self):
        """'↔E 1,4' normalizes correctly."""
        result = canonicalize_justification_for_comparison("↔E 1,4")
        self.assertEqual(result, "↔E 1,4")

    def test_biconditional_intro_with_range(self):
        """'↔I 3-6,7-10' normalizes correctly."""
        result = canonicalize_justification_for_comparison("↔I 3-6,7-10")
        self.assertEqual(result, "↔I 3-6,7-10")

    def test_space_only_between_digits_in_tail(self):
        """'∧I 4 5' (space between 4 and 5) should become '∧I 4,5'."""
        result = canonicalize_justification_for_comparison("∧I 4 5")
        self.assertEqual(result, "∧I 4,5")


class TestFormulaNormalization(unittest.TestCase):
    """Tests for formula canonicalization (smoke tests to confirm existing behavior)."""

    def test_spaces_stripped(self):
        """Spaces around operators are stripped for comparison."""
        self.assertEqual(
            canonicalize_formula_for_comparison("P ∧ Q"),
            canonicalize_formula_for_comparison("P∧Q"),
        )

    def test_and_symbol_normalized(self):
        """Ampersand normalizes to wedge."""
        self.assertEqual(
            canonicalize_formula_for_comparison("P&Q"),
            canonicalize_formula_for_comparison("P∧Q"),
        )

    def test_arrow_normalized(self):
        """ASCII arrow normalizes to Unicode."""
        self.assertEqual(
            canonicalize_formula_for_comparison("P->Q"),
            canonicalize_formula_for_comparison("P→Q"),
        )

    def test_biconditional_normalized(self):
        """ASCII biconditional normalizes to Unicode."""
        self.assertEqual(
            canonicalize_formula_for_comparison("P<->Q"),
            canonicalize_formula_for_comparison("P↔Q"),
        )


# ── Phase 2: ∧→8 post-OCR repair tests ──────────────────────────────


class TestEightAsAndJustificationRepair(unittest.TestCase):
    """Tests for the 8→∧ repair heuristic in justification heads."""

    def test_eight_E_replaced_with_wedge_E(self):
        """OCR '8 E 3' should normalize same as gold '∧E 3'."""
        self.assertEqual(
            canonicalize_justification_for_comparison("8 E 3"),
            canonicalize_justification_for_comparison("∧E 3"),
        )

    def test_eight_I_replaced_with_wedge_I(self):
        """OCR '8 I 4,5' should normalize same as gold '∧I 4,5'."""
        self.assertEqual(
            canonicalize_justification_for_comparison("8 I 4,5"),
            canonicalize_justification_for_comparison("∧I 4,5"),
        )

    def test_eight_epsilon_chain_repairs_to_wedge_E(self):
        """OCR '8 ∈ S' chains ∈→E then 8→∧ to produce '∧E S' head."""
        result = canonicalize_justification_for_comparison("8 ∈ S")
        self.assertTrue(result.startswith("∧E"), f"Expected '∧E ...' after ∈→E and 8→∧ chain, got: {result}")

    def test_greek_epsilon_also_repaired(self):
        """OCR '8 ε 5' chains ε→E then 8→∧ to produce '∧E 5'."""
        result = canonicalize_justification_for_comparison("8 ε 5")
        self.assertEqual(result, "∧E 5")

    def test_epsilon_without_eight_still_repairs(self):
        """'∨∈ 3' → '∨E 3' (epsilon→E even without 8→∧)."""
        result = canonicalize_justification_for_comparison("∨∈ 3")
        self.assertEqual(result, "∨E 3")

    def test_eight_not_replaced_when_not_EI(self):
        """'8 X 3' should NOT get 8 replaced (X is not E or I rule letter)."""
        result = canonicalize_justification_for_comparison("8 X 3")
        # The head should still start with 8 (not ∧) since X is not E or I
        self.assertTrue(result.startswith("8"), f"Expected head to keep '8', got: {result}")

    def test_eight_E_normalizes_to_canonical_head(self):
        """'8 E 3' should produce canonical '∧E 3'."""
        result = canonicalize_justification_for_comparison("8 E 3")
        self.assertEqual(result, "∧E 3")

    def test_eight_I_normalizes_to_canonical_head(self):
        """'8 I 4,5' should produce canonical '∧I 4,5'."""
        result = canonicalize_justification_for_comparison("8 I 4,5")
        self.assertEqual(result, "∧I 4,5")

    def test_real_digit_8_not_in_head_position_unchanged(self):
        """A digit 8 in the tail (citation) should remain unchanged."""
        result = canonicalize_justification_for_comparison("→E 2,8")
        self.assertEqual(result, "→E 2,8")


class TestEightAsAndFormulaRepair(unittest.TestCase):
    """Tests for the 8→∧ repair heuristic in formulas (between caps)."""

    def test_formula_eight_between_caps(self):
        """OCR 'K8W' should normalize same as gold 'K∧W'."""
        self.assertEqual(
            canonicalize_formula_for_comparison("K8W"),
            canonicalize_formula_for_comparison("K∧W"),
        )

    def test_formula_eight_between_caps_with_spaces(self):
        """OCR 'K 8 W' should normalize same as gold 'K∧W'."""
        self.assertEqual(
            canonicalize_formula_for_comparison("K 8 W"),
            canonicalize_formula_for_comparison("K∧W"),
        )

    def test_formula_eight_not_between_caps_unchanged(self):
        """'P8' (8 at end, no cap after) should not get 8 replaced."""
        result = canonicalize_formula_for_comparison("P8")
        self.assertIn("8", result)

    def test_formula_eight_at_start_unchanged(self):
        """'8P' (8 at start, no cap before) should not get 8 replaced."""
        result = canonicalize_formula_for_comparison("8P")
        self.assertIn("8", result)

    def test_formula_multiple_eights_between_caps(self):
        """'A8B8C' should become 'A∧B∧C'."""
        self.assertEqual(
            canonicalize_formula_for_comparison("A8B8C"),
            canonicalize_formula_for_comparison("A∧B∧C"),
        )


# ── Re-scoring tests using saved OCR results ─────────────────────────


_SAVED_OCR_ROOT = Path("working/handwritten_ocr_probe/tfl")

TFL_HANDWRITTEN_CASES = [
    "tfl_handwritten_q3_P_and_Q_imp_R",
    "tfl_handwritten_q4_N_or_K_and_W",
    "tfl_handwritten_q5_P_or_K_biconditional",
    "tfl_handwritten_q6_S_iff_W",
]


def _load_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _load_gold_structure(case_id: str):
    """Load the gold expected_structure.json from regression fixtures."""
    from proofs_ocr.regression.runner import get_repo_root, load_manifest, case_expected_path

    manifest = load_manifest()
    case = None
    for c in manifest.get("cases", []):
        if c.get("id") == case_id:
            case = c
            break
    if case is None:
        return None
    expected_path = case_expected_path(case, repo_root=get_repo_root())
    if not expected_path.exists():
        return None
    return json.loads(expected_path.read_text(encoding="utf-8"))


@unittest.skipUnless(
    _SAVED_OCR_ROOT.exists(),
    "No saved OCR probe results in working/handwritten_ocr_probe/tfl. "
    "Run the handwritten OCR probe first to generate results.",
)
class TestRescoreWithUpdatedNormalization(unittest.TestCase):
    """Re-score saved OCR probe results with updated normalization.

    These tests load the *already-saved* OCR outputs from a previous probe
    run and re-compare them against gold using the current normalization
    logic. No VLM calls are needed.
    """

    def _rescore_case(self, case_id: str) -> dict:
        """Load saved OCR output and gold, compare with current normalization."""
        case_dir = _SAVED_OCR_ROOT / case_id
        actual = _load_json(case_dir / "final_structure.json")
        expected = _load_gold_structure(case_id)
        self.assertIsNotNone(actual, f"No final_structure.json for {case_id}")
        self.assertIsNotNone(expected, f"No gold expected_structure.json for {case_id}")
        return compare_proof_structures_granular(actual, expected)

    def test_q3_rescore(self):
        metrics = self._rescore_case("tfl_handwritten_q3_P_and_Q_imp_R")
        # Q3 Phase 1: 8/8 formula, 6/8 justification, 6/8 both
        # Q3 Phase 2: 8/8 formula, 8/8 justification, 8/8 both
        #   (lines 4 "8 E 3"→"∧E 3", line 6 "8 I 4,5"→"∧I 4,5" now match)
        self.assertEqual(metrics["formula_correct_line_count"], 8)
        self.assertGreaterEqual(metrics["justification_correct_line_count"], 8)
        self.assertGreaterEqual(metrics["both_correct_line_count"], 8)
        self.assertEqual(metrics["expected_line_count"], 8)

    def test_q4_rescore(self):
        metrics = self._rescore_case("tfl_handwritten_q4_N_or_K_and_W")
        # Q4 previously: 5/7 formula, 4/7 justification
        # Key improvement: line 7 "∨E 1, 3-4 5-6" should now match gold "∨E 1,3-4,5-6"
        self.assertEqual(metrics["expected_line_count"], 7)
        # Check if the justification for line 7 now matches
        line_records = metrics.get("line_records", [])
        line_7 = next((r for r in line_records if r.get("order_index") == 7), None)
        if line_7:
            # The normalization should make "∨E 1, 3-4 5-6" match "∨E 1,3-4,5-6"
            self.assertTrue(
                line_7.get("justification_matches", False),
                f"Q4 line 7 justification should now match after normalization. "
                f"Expected: {line_7.get('expected', {}).get('jstr')}, "
                f"Actual: {line_7.get('actual', {}).get('jstr')}",
            )

    def test_q5_rescore(self):
        metrics = self._rescore_case("tfl_handwritten_q5_P_or_K_biconditional")
        # Q5 previously: 4/5 formula, 4/5 justification
        self.assertEqual(metrics["expected_line_count"], 5)
        self.assertGreaterEqual(metrics["formula_correct_line_count"], 4)
        self.assertGreaterEqual(metrics["justification_correct_line_count"], 4)

    def test_q6_rescore(self):
        metrics = self._rescore_case("tfl_handwritten_q6_S_iff_W")
        # Q6 previously: 11/11 formula, 4/11 justification
        self.assertEqual(metrics["expected_line_count"], 11)
        self.assertEqual(metrics["formula_correct_line_count"], 11)
        self.assertGreaterEqual(metrics["justification_correct_line_count"], 4)

    def test_aggregate_rescore_improvement(self):
        """Aggregate metrics across all 4 cases should show improvement or parity."""
        total_formula = 0
        total_justification = 0
        total_both = 0
        total_matched = 0

        for case_id in TFL_HANDWRITTEN_CASES:
            metrics = self._rescore_case(case_id)
            total_formula += metrics["formula_correct_line_count"]
            total_justification += metrics["justification_correct_line_count"]
            total_both += metrics["both_correct_line_count"]
            total_matched += metrics["matched_line_count"]

        # Phase 1 baseline: 28/31 formula, 19/31 justification, 17/31 both
        # Phase 2 expected: 28/31 formula, 21/31 justification, 19/31 both
        #   (Q3 lines 4,6 justification now match via 8→∧ repair)
        self.assertEqual(total_matched, 31, "All 31 lines should be matched")
        self.assertEqual(total_formula, 28, "Formula accuracy should remain 28/31")
        self.assertGreaterEqual(
            total_justification,
            21,
            f"Justification accuracy should be >= 21/31, got {total_justification}/31. "
            "Improvement expected from Phase 2 8→∧ repair.",
        )
        self.assertGreaterEqual(
            total_both,
            19,
            f"Both-correct should be >= 19/31, got {total_both}/31. "
            "Improvement expected from Phase 2 8→∧ repair.",
        )

        # Print comparison for visibility
        print(f"\n=== Re-scored results with Phase 1+2 normalization ===")
        print(f"Formula:      {total_formula}/{total_matched} "
              f"(baseline 28/31 = 90%)")
        print(f"Justification: {total_justification}/{total_matched} "
              f"(baseline 19/31 = 61%, Phase 2 target 21/31 = 68%)")
        print(f"Both correct:  {total_both}/{total_matched} "
              f"(baseline 17/31 = 55%, Phase 2 target 19/31 = 61%)")


if __name__ == "__main__":
    unittest.main()