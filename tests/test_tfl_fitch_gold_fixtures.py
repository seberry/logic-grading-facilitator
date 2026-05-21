"""
Gold fixture tests for TFL (truth-functional logic) handwritten Fitch proofs.

These tests verify that the curated expected structures:
1. Load and parse correctly.
2. Produce the expected verification results when run through the TFL verifier.
3. Have semantic notes where intentional student errors are documented.

Proofs Q3 and Q4 are valid; Q5 and Q6 contain intentional errors that the
verifier should flag.
"""

import json
import unittest
from typing import List, Union

import yaml

from proofs_ocr.regression.runner import get_repo_root, load_manifest
from app.verifier import ProofLine, check_proof


def _structure_to_proof_lines(structure: list) -> List[Union[ProofLine, List]]:
    """Convert nested list/dict structure to ProofLine objects for the verifier."""
    result: List[Union[ProofLine, List]] = []
    for item in structure:
        if isinstance(item, list):
            result.append(_structure_to_proof_lines(item))
        elif isinstance(item, dict):
            wffstr = str(item.get("wffstr", "")).strip()
            jstr = str(item.get("jstr", "")).strip()
            result.append(ProofLine(wffstr=wffstr, jstr=jstr, location=[]))
    return result


def _line_depths(structure: list, depth: int = 0) -> List[int]:
    depths: List[int] = []
    for item in structure:
        if isinstance(item, list):
            depths.extend(_line_depths(item, depth + 1))
        elif isinstance(item, dict):
            depths.append(depth)
    return depths


class TFLGoldFixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo_root = get_repo_root()
        manifest = load_manifest()
        cls.case_lookup = {
            str(case.get("id") or ""): case
            for case in manifest.get("cases") or []
            if isinstance(case, dict)
        }

    def _load_expected(self, case_id: str):
        case = self.case_lookup[case_id]
        expected_path = self.repo_root / str(case["expected_structure_path"])
        return json.loads(expected_path.read_text(encoding="utf-8"))

    def _load_semantic_notes(self, case_id: str) -> dict:
        case = self.case_lookup[case_id]
        notes_path_text = str(case.get("semantic_notes_path") or "").strip()
        if not notes_path_text:
            return {}
        notes_path = self.repo_root / notes_path_text
        if not notes_path.exists():
            return {}
        parsed = yaml.safe_load(notes_path.read_text(encoding="utf-8")) or {}
        return parsed if isinstance(parsed, dict) else {}

    def _verify_proof(self, case_id: str, numprems: int, conc: str):
        expected = self._load_expected(case_id)
        proof_lines = _structure_to_proof_lines(expected)
        return check_proof(proof_lines, numprems=numprems, conc=conc)

    # -- Fixture integrity --

    def test_all_tfl_cases_present_in_manifest(self) -> None:
        tfl_ids = [
            "tfl_handwritten_q3_P_and_Q_imp_R",
            "tfl_handwritten_q4_N_or_K_and_W",
            "tfl_handwritten_q5_P_or_K_biconditional",
            "tfl_handwritten_q6_S_iff_W",
        ]
        for case_id in tfl_ids:
            self.assertIn(case_id, self.case_lookup, f"Missing TFL case: {case_id}")

    def test_tfl_cases_have_correct_domain(self) -> None:
        tfl_ids = [
            "tfl_handwritten_q3_P_and_Q_imp_R",
            "tfl_handwritten_q4_N_or_K_and_W",
            "tfl_handwritten_q5_P_or_K_biconditional",
            "tfl_handwritten_q6_S_iff_W",
        ]
        for case_id in tfl_ids:
            case = self.case_lookup[case_id]
            self.assertEqual(case["domain"], "tfl")
            self.assertEqual(case["oracle_kind"], "gold_curated_from_image")

    def test_all_tfl_images_exist(self) -> None:
        tfl_ids = [
            "tfl_handwritten_q3_P_and_Q_imp_R",
            "tfl_handwritten_q4_N_or_K_and_W",
            "tfl_handwritten_q5_P_or_K_biconditional",
            "tfl_handwritten_q6_S_iff_W",
        ]
        for case_id in tfl_ids:
            case = self.case_lookup[case_id]
            image_path = self.repo_root / str(case["image_path"])
            self.assertTrue(image_path.exists(), f"Image missing for {case_id}: {image_path}")

    def test_tfl_gold_depths_are_logical_not_visual_outer_frame(self) -> None:
        tfl_ids = [
            "tfl_handwritten_q3_P_and_Q_imp_R",
            "tfl_handwritten_q4_N_or_K_and_W",
            "tfl_handwritten_q5_P_or_K_biconditional",
            "tfl_handwritten_q6_S_iff_W",
        ]
        for case_id in tfl_ids:
            expected = self._load_expected(case_id)
            depths = _line_depths(expected)
            self.assertTrue(depths, f"{case_id} has no proof lines")
            self.assertEqual(depths[0], 0, f"{case_id} first line should be top-level")
            self.assertEqual(min(depths), 0, f"{case_id} should not count the printed outer frame")

    def test_checker_accepts_theorem_proof_with_top_level_conclusion(self) -> None:
        proof = [
            [
                {"line": 1, "wffstr": "P", "jstr": "Hyp"},
            ],
            {"line": 2, "wffstr": "P→P", "jstr": "→I 1-1"},
        ]

        result = check_proof(_structure_to_proof_lines(proof), numprems=0, conc="P→P")

        self.assertEqual(result.issues, [])
        self.assertTrue(result.conc_reached)

    # -- Q3: Valid proof --

    def test_q3_structure_loads(self) -> None:
        expected = self._load_expected("tfl_handwritten_q3_P_and_Q_imp_R")
        self.assertEqual(
            expected,
            [
                {"line": 1, "wffstr": "(P∧Q)→R", "jstr": "Pr"},
                {"line": 2, "wffstr": "P→Q", "jstr": "Pr"},
                [
                    {"line": 3, "wffstr": "P∧S", "jstr": "Hyp"},
                    {"line": 4, "wffstr": "P", "jstr": "∧E 3"},
                    {"line": 5, "wffstr": "Q", "jstr": "→E 2,4"},
                    {"line": 6, "wffstr": "P∧Q", "jstr": "∧I 4,5"},
                    {"line": 7, "wffstr": "R", "jstr": "→E 1,6"},
                ],
                {"line": 8, "wffstr": "(P∧S)→R", "jstr": "→I 3-7"},
            ],
        )

    def test_q3_verifies_clean(self) -> None:
        result = self._verify_proof("tfl_handwritten_q3_P_and_Q_imp_R", numprems=2, conc="(P\u2227S)\u2192R")
        self.assertEqual(result.issues, [], f"Q3 should have no verifier issues, got: {result.issues}")
        self.assertTrue(result.conc_reached, "Q3 should reach conclusion (P\u2227S)\u2192R")

    # -- Q4: Valid proof with two subproofs for or-elimination --

    def test_q4_structure_loads(self) -> None:
        expected = self._load_expected("tfl_handwritten_q4_N_or_K_and_W")
        self.assertEqual(
            expected,
            [
                {"line": 1, "wffstr": "N∨(K∧W)", "jstr": "Pr"},
                {"line": 2, "wffstr": "N↔W", "jstr": "Pr"},
                [
                    {"line": 3, "wffstr": "N", "jstr": "Hyp"},
                    {"line": 4, "wffstr": "W", "jstr": "↔E 2,3"},
                ],
                [
                    {"line": 5, "wffstr": "K∧W", "jstr": "Hyp"},
                    {"line": 6, "wffstr": "W", "jstr": "∧E 5"},
                ],
                {"line": 7, "wffstr": "W", "jstr": "∨E 1,3-4,5-6"},
            ],
        )

    def test_q4_verifies_clean(self) -> None:
        result = self._verify_proof("tfl_handwritten_q4_N_or_K_and_W", numprems=2, conc="W")
        self.assertEqual(result.issues, [], f"Q4 should have no verifier issues, got: {result.issues}")
        self.assertTrue(result.conc_reached, "Q4 should reach conclusion W")

    def test_q4_has_semantic_notes(self) -> None:
        notes = self._load_semantic_notes("tfl_handwritten_q4_N_or_K_and_W")
        self.assertEqual(
            notes,
            {
                "notes": [
                    {
                        "line": 4,
                        "field": "jstr",
                        "description": "Image shows F 2,3 normalized to E 2,3",
                        "as_written": "F 2,3",
                        "normalized_to": "E 2,3",
                    }
                ]
            },
        )

    # -- Q5: Intentional self-citation error --

    def test_q5_structure_loads(self) -> None:
        expected = self._load_expected("tfl_handwritten_q5_P_or_K_biconditional")
        self.assertEqual(
            expected,
            [
                {"line": 1, "wffstr": "(P∨K)↔(Z∧W)", "jstr": "Pr"},
                {"line": 2, "wffstr": "P", "jstr": "Pr"},
                {"line": 3, "wffstr": "P∨K", "jstr": "∨I 2"},
                {"line": 4, "wffstr": "Z∧W", "jstr": "↔E 1,4"},
                {"line": 5, "wffstr": "W", "jstr": "∧E 4"},
            ],
        )

    def test_q5_verifier_flags_self_citation(self) -> None:
        result = self._verify_proof("tfl_handwritten_q5_P_or_K_biconditional", numprems=2, conc="W")
        self.assertFalse(result.conc_reached, "Q5 should not reach conclusion W")
        self.assertEqual(result.issues, ["Line 4: Cites itself."])

    def test_q5_has_semantic_notes(self) -> None:
        notes = self._load_semantic_notes("tfl_handwritten_q5_P_or_K_biconditional")
        self.assertIn("notes", notes)
        self.assertEqual(len(notes["notes"]), 1)
        self.assertEqual(notes["notes"][0]["line"], 4)
        self.assertEqual(notes["notes"][0]["error_kind"], "self_citation")

    # -- Q6: Intentional invalid rule applications --

    def test_q6_structure_loads(self) -> None:
        expected = self._load_expected("tfl_handwritten_q6_S_iff_W")
        self.assertEqual(
            expected,
            [
                {"line": 1, "wffstr": "S→W", "jstr": "Pr"},
                {"line": 2, "wffstr": "W→S", "jstr": "Pr"},
                [
                    {"line": 3, "wffstr": "S∧A", "jstr": "Hyp"},
                    {"line": 4, "wffstr": "S", "jstr": "∧E 3"},
                    {"line": 5, "wffstr": "W", "jstr": "→E 1,4"},
                    {"line": 6, "wffstr": "W∧A", "jstr": "∧I 4,5"},
                ],
                [
                    {"line": 7, "wffstr": "W∧A", "jstr": "Hyp"},
                    {"line": 8, "wffstr": "W", "jstr": "∧E 7"},
                    {"line": 9, "wffstr": "S", "jstr": "→E 2,8"},
                    {"line": 10, "wffstr": "S∧A", "jstr": "∧I 8,9"},
                ],
                {"line": 11, "wffstr": "(S∧A)↔(W∧A)", "jstr": "↔I 3-6,7-10"},
            ],
        )

    def test_q6_verifier_flags_invalid_rule_applications(self) -> None:
        result = self._verify_proof("tfl_handwritten_q6_S_iff_W", numprems=2, conc="(S\u2227A)\u2194(W\u2227A)")
        self.assertFalse(result.conc_reached, "Q6 should not reach conclusion (S∧A)↔(W∧A)")
        self.assertEqual(
            result.issues,
            [
                "Line 6: Is not a proper application of the rule ∧I (for the line(s) cited).",
                "Line 10: Is not a proper application of the rule ∧I (for the line(s) cited).",
            ],
        )

    def test_q6_has_semantic_notes(self) -> None:
        notes = self._load_semantic_notes("tfl_handwritten_q6_S_iff_W")
        self.assertIn("notes", notes)
        self.assertEqual(len(notes["notes"]), 2)
        noted_lines = [n["line"] for n in notes["notes"]]
        self.assertIn(6, noted_lines)
        self.assertIn(10, noted_lines)
        for note in notes["notes"]:
            self.assertEqual(note["error_kind"], "invalid_rule_application")


if __name__ == "__main__":
    unittest.main()
