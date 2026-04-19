import json
import unittest
from pathlib import Path

import yaml

from proofs_ocr.regression.runner import get_repo_root, load_manifest
from proofs_ocr.regression.typeset_gold import load_modal_typeset_gold_from_source


class ModalGoldFixtureTests(unittest.TestCase):
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

    def test_modal_k_typeset_mixed_subproofs_gold_matches_source_tex(self) -> None:
        case = self.case_lookup["modal_k_typeset_mixed_subproofs"]
        source_path = self.repo_root / str(case["source"])
        expected = self._load_expected("modal_k_typeset_mixed_subproofs")

        parsed = load_modal_typeset_gold_from_source(source_path)

        self.assertEqual(parsed, expected)
        self.assertEqual(case["oracle_kind"], "gold_from_source_tex")

    def test_modal_s4_typeset_r4_corrected_gold_matches_source_tex(self) -> None:
        case = self.case_lookup["modal_s4_typeset_r4_corrected"]
        source_path = self.repo_root / str(case["source"])
        expected = self._load_expected("modal_s4_typeset_r4_corrected")

        parsed = load_modal_typeset_gold_from_source(source_path)

        self.assertEqual(parsed, expected)
        self.assertEqual(case["oracle_kind"], "gold_from_source_tex")

    def test_modal_s5_typeset_example_gold_matches_source_tex(self) -> None:
        case = self.case_lookup["modal_s5_typeset_example"]
        source_path = self.repo_root / str(case["source"])
        expected = self._load_expected("modal_s5_typeset_example")

        parsed = load_modal_typeset_gold_from_source(source_path)

        self.assertEqual(parsed, expected)
        self.assertEqual(case["oracle_kind"], "gold_from_source_tex")

    def test_modal_handwritten_user_box_q_gold_is_curated(self) -> None:
        case = self.case_lookup["modal_handwritten_user_box_q"]
        expected = self._load_expected("modal_handwritten_user_box_q")
        notes_path = self.repo_root / str(case["semantic_notes_path"])
        notes = yaml.safe_load(notes_path.read_text(encoding="utf-8")) or {}

        self.assertEqual(case["oracle_kind"], "gold_curated_from_image")
        self.assertEqual(notes.get("notes"), [])
        self.assertEqual(
            expected,
            [
                {"line_label": "1", "wffstr": "□P", "jstr": "PR", "line": 1},
                {"line_label": "2", "wffstr": "□(P→Q)", "jstr": "PR", "line": 2},
                [
                    {"line_label": "3", "wffstr": "□", "jstr": "PR", "line": 3},
                    {"line_label": "4", "wffstr": "P", "jstr": "□E 1", "line": 4},
                    {"line_label": "5", "wffstr": "P→Q", "jstr": "□E 2", "line": 5},
                    {"line_label": "6", "wffstr": "Q", "jstr": "→E 4,5", "line": 6},
                ],
                {"line_label": "7", "wffstr": "□Q", "jstr": "□I 3-6", "line": 7},
            ],
        )

    def test_modal_handwritten_user_negated_modal_conjunction_gold_is_curated(self) -> None:
        case = self.case_lookup["modal_handwritten_user_negated_modal_conjunction"]
        expected = self._load_expected("modal_handwritten_user_negated_modal_conjunction")
        notes_path = self.repo_root / str(case["semantic_notes_path"])
        notes = yaml.safe_load(notes_path.read_text(encoding="utf-8")) or {}

        self.assertEqual(case["oracle_kind"], "gold_curated_from_image")
        self.assertEqual(len(notes.get("notes") or []), 2)
        self.assertEqual(notes["notes"][0]["line"], 4)
        self.assertEqual(notes["notes"][1]["line"], 7)
        self.assertEqual(
            expected,
            [
                [
                    {"line_label": "1", "wffstr": "□A ∧ ◇¬A", "jstr": "Assum", "line": 1},
                    {"line_label": "2", "wffstr": "□A", "jstr": "∧E 1", "line": 2},
                    {"line_label": "3", "wffstr": "◇¬A", "jstr": "∧E 1", "line": 3},
                    {"line_label": "4", "wffstr": "¬□¬¬A", "jstr": "MC 3", "line": 4},
                    [
                        {"line_label": "5", "wffstr": "□", "jstr": "Assum", "line": 5},
                        {"line_label": "6", "wffstr": "A", "jstr": "□E 2", "line": 6},
                        {"line_label": "7", "wffstr": "¬¬A", "jstr": "DN 6", "line": 7},
                    ],
                    {"line_label": "8", "wffstr": "□¬¬A", "jstr": "□I 5-7", "line": 8},
                    {"line_label": "9", "wffstr": "⊥", "jstr": "¬E 4,8", "line": 9},
                ],
                {"line_label": "10", "wffstr": "¬(□A ∧ ◇¬A)", "jstr": "¬I 1-9", "line": 10},
            ],
        )


if __name__ == "__main__":
    unittest.main()
