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
from pathlib import Path
from typing import List, Union

import yaml

from proofs_ocr.regression.runner import get_repo_root, load_manifest
from app.verifier import ProofLine, check_proof, flatten_proof


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

    # -- Q3: Valid proof --

    def test_q3_structure_loads(self) -> None:
        expected = self._load_expected("tfl_handwritten_q3_P_and_Q_imp_R")
        self.assertEqual(len(expected), 4)  # 2 premises + 1 subproof list + 1 conclusion
        self.assertIsInstance(expected[2], list)
        self.assertEqual(len(expected[2]), 5)

    def test_q3_verifies_clean(self) -> None:
        result = self._verify_proof("tfl_handwritten_q3_P_and_Q_imp_R", numprems=2, conc="(P\u2227S)\u2192R")
        self.assertEqual(result.issues, [], f"Q3 should have no verifier issues, got: {result.issues}")
        self.assertTrue(result.conc_reached, "Q3 should reach conclusion (P\u2227S)\u2192R")

    # -- Q4: Valid proof with two subproofs for or-elimination --

    def test_q4_structure_loads(self) -> None:
        expected = self._load_expected("tfl_handwritten_q4_N_or_K_and_W")
        self.assertEqual(len(expected), 5)
        self.assertIsInstance(expected[2], list)
        self.assertIsInstance(expected[3], list)

    def test_q4_verifies_clean(self) -> None:
        result = self._verify_proof("tfl_handwritten_q4_N_or_K_and_W", numprems=2, conc="W")
        self.assertEqual(result.issues, [], f"Q4 should have no verifier issues, got: {result.issues}")
        self.assertTrue(result.conc_reached, "Q4 should reach conclusion W")

    def test_q4_has_semantic_notes(self) -> None:
        notes = self._load_semantic_notes("tfl_handwritten_q4_N_or_K_and_W")
        self.assertIn("notes", notes)
        note_lines = [n.get("line") for n in notes["notes"]]
        self.assertIn(4, note_lines, "Q4 semantic notes should mention line 4")

    # -- Q5: Intentional self-citation error --

    def test_q5_structure_loads(self) -> None:
        expected = self._load_expected("tfl_handwritten_q5_P_or_K_biconditional")
        self.assertEqual(len(expected), 5)
        for item in expected:
            self.assertIsInstance(item, dict, "Q5 should have no subproofs")

    def test_q5_verifier_flags_self_citation(self) -> None:
        result = self._verify_proof("tfl_handwritten_q5_P_or_K_biconditional", numprems=2, conc="W")
        self.assertTrue(len(result.issues) > 0, "Q5 should have verifier issues (self-citation on line 4)")
        issue_text = " ".join(result.issues).lower()
        self.assertIn("4", " ".join(result.issues), f"Expected issue mentioning line 4, got: {result.issues}")

    def test_q5_has_semantic_notes(self) -> None:
        notes = self._load_semantic_notes("tfl_handwritten_q5_P_or_K_biconditional")
        self.assertIn("notes", notes)
        self.assertEqual(len(notes["notes"]), 1)
        self.assertEqual(notes["notes"][0]["line"], 4)
        self.assertEqual(notes["notes"][0]["error_kind"], "self_citation")

    # -- Q6: Intentional invalid rule applications --

    def test_q6_structure_loads(self) -> None:
        expected = self._load_expected("tfl_handwritten_q6_S_iff_W")
        self.assertEqual(len(expected), 5)
        self.assertIsInstance(expected[2], list)
        self.assertIsInstance(expected[3], list)

    def test_q6_verifier_flags_invalid_rule_applications(self) -> None:
        result = self._verify_proof("tfl_handwritten_q6_S_iff_W", numprems=2, conc="(S\u2227A)\u2194(W\u2227A)")
        self.assertTrue(len(result.issues) > 0, "Q6 should have verifier issues (invalid rule on lines 6 and 10)")

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