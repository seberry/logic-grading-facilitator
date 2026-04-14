from __future__ import annotations

import json
import shutil
import unittest
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np

from scripts import debug_modal_proof_ocr


class FakeModalVisionClient:
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)

    def chat(self, messages) -> str:
        if not self._responses:
            raise AssertionError("No fake responses left")
        return self._responses.pop(0)


class DebugModalProofOcrTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = Path("working") / "test_tmp" / f"modal_ocr_test_{uuid4().hex}"
        self.tmp_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_load_modal_config_accepts_saved_state_shape(self) -> None:
        path = self.tmp_dir / "state.json"
        path.write_text(
            json.dumps(
                {
                    "system": "S4",
                    "include_derived_rules": True,
                    "conclusion": "□□P",
                    "proof_json_text": "{}",
                }
            ),
            encoding="utf-8",
        )

        parsed = debug_modal_proof_ocr._load_modal_config(path)
        self.assertEqual(parsed["system"], "S4")
        self.assertTrue(parsed["include_derived_rules"])
        self.assertEqual(parsed["conclusion"], "□□P")

    def test_build_prompt_mentions_modal_roles(self) -> None:
        prompt = debug_modal_proof_ocr.build_prompt(
            {"system": "S4", "include_derived_rules": False, "conclusion": "□□P"}
        )

        self.assertIn("strict_open", prompt)
        self.assertIn("<SUBPROOF_ID_X>", prompt)
        self.assertIn("Modal system: S4", prompt)
        self.assertIn("Target conclusion: □□P", prompt)

    def test_recursive_stitch_builds_nested_modal_structure(self) -> None:
        library = {
            "ROOT": [
                {"formula_text": "□P", "scope_role": "premise"},
                {"formula_text": "<SUBPROOF_ID_1>", "scope_role": "subproof_token"},
                {"formula_text": "□□P", "justification_text": "□I 2-3", "scope_role": "derived_line"},
            ],
            "1": [
                {"formula_text": "□", "scope_role": "strict_open"},
                {"formula_text": "□P", "justification_text": "R4 1", "scope_role": "derived_line"},
            ],
        }

        stitched = debug_modal_proof_ocr.recursive_stitch(library["ROOT"], library)
        debug_modal_proof_ocr.renumber_lines(stitched)

        self.assertEqual(stitched[0]["jstr"], "PR")
        self.assertEqual(stitched[1][0]["wffstr"], "□")
        self.assertEqual(stitched[1][0]["jstr"], "PR")
        self.assertEqual(stitched[1][1]["jstr"], "R4 1")
        self.assertEqual(stitched[2]["jstr"], "□I 2-3")

    def test_rewrite_strict_subproof_local_citations_remaps_internal_refs(self) -> None:
        structure = [
            {"line_label": "", "wffstr": "□A", "jstr": "PR", "line": 1},
            {"line_label": "", "wffstr": "□B", "jstr": "PR", "line": 2},
            [
                {"line_label": "1", "wffstr": "□", "jstr": "PR", "line": 3},
                {"line_label": "2", "wffstr": "A", "jstr": "□E 1", "line": 4},
                {"line_label": "3", "wffstr": "B", "jstr": "□E 2", "line": 5},
                {"line_label": "4", "wffstr": "A ∧ B", "jstr": "∧I 2 3", "line": 6},
            ],
            {"line_label": "", "wffstr": "□(A ∧ B)", "jstr": "□I 3-6", "line": 7},
        ]

        debug_modal_proof_ocr.rewrite_strict_subproof_local_citations(structure)

        self.assertEqual(structure[2][1]["jstr"], "□E 1")
        self.assertEqual(structure[2][2]["jstr"], "□E 2")
        self.assertEqual(structure[2][3]["jstr"], "∧I 4 5")

    def test_rewrite_strict_subproof_keeps_already_global_refs(self) -> None:
        structure = [
            {"line_label": "", "wffstr": "□A", "jstr": "PR", "line": 1},
            {"line_label": "", "wffstr": "□B", "jstr": "PR", "line": 2},
            [
                {"line_label": "1", "wffstr": "□", "jstr": "PR", "line": 3},
                {"line_label": "2", "wffstr": "A", "jstr": "□E 1", "line": 4},
                {"line_label": "3", "wffstr": "B", "jstr": "□E 2", "line": 5},
                {"line_label": "4", "wffstr": "A ∧ B", "jstr": "∧I 4 5", "line": 6},
            ],
            {"line_label": "", "wffstr": "□(A ∧ B)", "jstr": "□I 3-6", "line": 7},
        ]

        debug_modal_proof_ocr.rewrite_strict_subproof_local_citations(structure)

        self.assertEqual(structure[2][3]["jstr"], "∧I 4 5")

    def test_transcribe_image_to_modal_debug_state_writes_state(self) -> None:
        image_path = self.tmp_dir / "sample.png"
        cv2.imwrite(str(image_path), np.full((40, 40, 3), 255, dtype=np.uint8))
        output_dir = self.tmp_dir / "out"

        fake_client = FakeModalVisionClient(
            responses=[
                json.dumps(
                    [
                        {"formula_text": "□", "scope_role": "strict_open", "line_label": "2"},
                        {"formula_text": "□P", "justification_text": "R4 1", "scope_role": "derived_line", "line_label": "3"},
                    ],
                    ensure_ascii=False,
                ),
                json.dumps(
                    [
                        {"formula_text": "□P", "scope_role": "premise", "line_label": "1"},
                        {"formula_text": "<SUBPROOF_ID_1>", "scope_role": "subproof_token"},
                        {"formula_text": "□□P", "justification_text": "□I 2-3", "scope_role": "derived_line"},
                    ],
                    ensure_ascii=False,
                ),
            ]
        )

        original_detect = debug_modal_proof_ocr.detect_vertical_bars
        original_run_checker = debug_modal_proof_ocr.run_modal_proof_checker
        try:
            debug_modal_proof_ocr.detect_vertical_bars = lambda img: [(0, 0, 2, 100)]
            debug_modal_proof_ocr.run_modal_proof_checker = lambda answer_text, autograde_config: type(
                "Result",
                (),
                {
                    "request_payload": {"system": "S4", "lines": []},
                    "response_payload": {"ok": True, "parse_errors": [], "check_errors": []},
                },
            )()

            state = debug_modal_proof_ocr.transcribe_image_to_modal_debug_state(
                image_path=image_path,
                output_dir=output_dir,
                modal_config={"system": "S4", "conclusion": "□□P"},
                client=fake_client,
            )
        finally:
            debug_modal_proof_ocr.detect_vertical_bars = original_detect
            debug_modal_proof_ocr.run_modal_proof_checker = original_run_checker

        self.assertEqual(state["system"], "S4")
        self.assertIn("□□P", state["proof_json_text"])
        self.assertTrue((output_dir / "state.json").exists())
        self.assertTrue((output_dir / "final_structure.json").exists())


if __name__ == "__main__":
    unittest.main()
