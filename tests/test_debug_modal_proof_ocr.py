from __future__ import annotations

import json
import shutil
import unittest
from pathlib import Path
from unittest.mock import patch
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
        self.assertIn("Do not stop early just because a child subproof was masked", prompt)

    def test_fit_token_layout_keeps_stamp_within_available_width(self) -> None:
        scale, thickness, rect = debug_modal_proof_ocr._fit_token_layout(
            "<SUBPROOF_ID_2>",
            start_x=30,
            center_y=40,
            max_width=120,
        )

        left, top, right, bottom = rect
        self.assertGreaterEqual(scale, 0.4)
        self.assertEqual(left, 30)
        self.assertLessEqual(right - left, 120)
        self.assertGreater(bottom, top)

    def test_refine_vertical_bar_box_prefers_rightmost_actual_bar(self) -> None:
        img = np.full((120, 40, 3), 255, dtype=np.uint8)
        img[10:80, 18:20] = 0
        img[85:110, 8:9] = 0

        refined = debug_modal_proof_ocr._refine_vertical_bar_box(img, (5, 0, 20, 120))

        self.assertEqual(refined[0], 18)
        self.assertEqual(refined[1], 10)
        self.assertEqual(refined[2], 2)
        self.assertEqual(refined[3], 70)

    def test_crop_region_can_avoid_bottom_padding_for_child_crop(self) -> None:
        img = np.zeros((200, 50, 3), dtype=np.uint8)

        cropped = debug_modal_proof_ocr.crop_region(
            img,
            (10, 40, 2, 80),
            bottom_padding=0,
        )

        self.assertEqual(cropped.shape[0], 90)

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

    def test_normalize_formula_text_normalizes_diamond_variants(self) -> None:
        self.assertEqual(debug_modal_proof_ocr._normalize_formula_text("◇A", "derived_line"), "⋄A")
        self.assertEqual(debug_modal_proof_ocr._normalize_formula_text("◊A", "derived_line"), "⋄A")
        self.assertEqual(debug_modal_proof_ocr._normalize_formula_text("<>A", "derived_line"), "⋄A")

    def test_should_recheck_unary_formula_targets_modal_prefix_lines(self) -> None:
        self.assertTrue(
            debug_modal_proof_ocr._should_recheck_unary_formula(
                {
                    "formula_text": "¬□¬A",
                    "justification_text": "MC 3",
                    "scope_role": "derived_line",
                    "line_label": "4",
                }
            )
        )
        self.assertFalse(
            debug_modal_proof_ocr._should_recheck_unary_formula(
                {
                    "formula_text": "A ∧ B",
                    "justification_text": "∧I 1 2",
                    "scope_role": "derived_line",
                    "line_label": "5",
                }
            )
        )

    def test_build_unary_formula_candidates_preserves_skeleton(self) -> None:
        candidates = debug_modal_proof_ocr._build_unary_formula_candidates(
            {
                "formula_text": "□A",
                "justification_text": "DN 1",
                "scope_role": "derived_line",
                "line_label": "4",
            }
        )

        self.assertEqual(candidates[0], "□A")
        self.assertIn("¬¬A", candidates)
        self.assertIn("¬¬¬A", candidates)
        self.assertNotIn("□B", candidates)

    def test_build_unary_formula_candidates_ignores_non_unary_formula(self) -> None:
        candidates = debug_modal_proof_ocr._build_unary_formula_candidates(
            {
                "formula_text": "A ∧ B",
                "justification_text": "∧I 1 2",
                "scope_role": "derived_line",
                "line_label": "5",
            }
        )

        self.assertEqual(candidates, [])

    def test_select_unary_forced_choice_candidates_keeps_nearby_variants(self) -> None:
        forced = debug_modal_proof_ocr._select_unary_forced_choice_candidates(
            original_formula="¬□¬A",
            generated_candidates=["¬□¬A", "¬□¬¬A", "¬□A", "¬¬A"],
            candidate_scores={
                "¬□¬A": (0, 3),
                "¬□¬¬A": (0, 4),
                "¬□A": (0, 3),
                "¬¬A": (1, 2),
            },
        )

        self.assertEqual(forced[0], "¬□¬A")
        self.assertIn("¬□¬¬A", forced)
        self.assertIn("¬□A", forced)
        self.assertNotIn("¬¬A", forced)

    def test_recheck_suspicious_unary_formulas_keeps_original_when_visual_choice_stays_original(self) -> None:
        crop_path = self.tmp_dir / "crop_1.jpg"
        cv2.imwrite(str(crop_path), np.full((40, 40, 3), 255, dtype=np.uint8))
        structure = [
            {"line_label": "4", "wffstr": "□A", "jstr": "DN 1", "_ocr_node_id": "1"},
        ]

        def fake_run_checker(answer_text, autograde_config):
            payload = json.loads(answer_text)
            sentence = payload["lines"][0]["wffstr"]
            score = (0, 2) if sentence == "□A" else (0, 2)
            return type(
                "Result",
                (),
                {
                    "parse_errors": [object()] * score[0],
                    "check_errors": [object()] * score[1],
                },
            )()

        fake_client = FakeModalVisionClient(
            [json.dumps({"selected_formula": "□A", "confidence": "medium"}, ensure_ascii=False)]
        )
        with patch.object(debug_modal_proof_ocr, "run_modal_proof_checker", side_effect=fake_run_checker):
            records = debug_modal_proof_ocr._recheck_suspicious_unary_formulas(
                proof_structure=structure,
                crop_path=crop_path,
                client=fake_client,
                output_dir=self.tmp_dir,
                modal_config={"system": "K"},
                node_id="1",
            )

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["improving_candidates"], [])
        self.assertEqual(structure[0]["wffstr"], "□A")
        self.assertGreaterEqual(len(records[0]["forced_choice_candidates"]), 1)
        self.assertEqual(records[0]["vlm_choice"], "□A")
        self.assertFalse(records[0]["accepted"])

    def test_recheck_suspicious_unary_formulas_accepts_vlm_selected_improving_candidate(self) -> None:
        crop_path = self.tmp_dir / "crop_1.jpg"
        cv2.imwrite(str(crop_path), np.full((40, 40, 3), 255, dtype=np.uint8))
        structure = [
            {"line_label": "4", "wffstr": "□A", "jstr": "DN 1", "_ocr_node_id": "1"},
        ]

        def fake_run_checker(answer_text, autograde_config):
            payload = json.loads(answer_text)
            sentence = payload["lines"][0]["wffstr"]
            score_map = {
                "□A": (0, 2),
                "¬A": (0, 3),
                "¬¬A": (0, 1),
                "¬¬¬A": (0, 2),
                "□¬A": (0, 3),
                "¬□A": (0, 3),
                "¬□¬A": (0, 2),
            }
            score = score_map.get(sentence, (0, 4))
            return type(
                "Result",
                (),
                {
                    "parse_errors": [object()] * score[0],
                    "check_errors": [object()] * score[1],
                },
            )()

        fake_client = FakeModalVisionClient(
            [json.dumps({"line_label": "4", "selected_formula": "¬¬A", "confidence": "medium"}, ensure_ascii=False)]
        )
        with patch.object(debug_modal_proof_ocr, "run_modal_proof_checker", side_effect=fake_run_checker):
            records = debug_modal_proof_ocr._recheck_suspicious_unary_formulas(
                proof_structure=structure,
                crop_path=crop_path,
                client=fake_client,
                output_dir=self.tmp_dir,
                modal_config={"system": "K"},
                node_id="1",
            )

        self.assertEqual(structure[0]["wffstr"], "¬¬A")
        self.assertEqual(records[0]["vlm_choice"], "¬¬A")
        self.assertTrue(records[0]["accepted"])

    def test_recheck_suspicious_unary_formulas_can_use_near_tie_candidate_on_visual_choice(self) -> None:
        crop_path = self.tmp_dir / "crop_1.jpg"
        cv2.imwrite(str(crop_path), np.full((120, 120, 3), 255, dtype=np.uint8))
        structure = [
            {"line_label": "", "wffstr": "¬□¬A", "jstr": "MC 3", "_ocr_node_id": "1", "_ocr_entry_index": 3},
        ]

        def fake_run_checker(answer_text, autograde_config):
            payload = json.loads(answer_text)
            sentence = payload["lines"][0]["wffstr"]
            score_map = {
                "¬□¬A": (0, 3),
                "¬□¬¬A": (0, 4),
                "¬□A": (0, 3),
                "¬A": (0, 5),
                "¬¬A": (0, 5),
                "¬¬¬A": (0, 5),
                "□A": (0, 5),
            }
            score = score_map.get(sentence, (0, 6))
            return type(
                "Result",
                (),
                {
                    "parse_errors": [object()] * score[0],
                    "check_errors": [object()] * score[1],
                },
            )()

        fake_client = FakeModalVisionClient(
            [json.dumps({"selected_formula": "¬□¬¬A", "confidence": "high"}, ensure_ascii=False)]
        )
        with patch.object(debug_modal_proof_ocr, "run_modal_proof_checker", side_effect=fake_run_checker):
            records = debug_modal_proof_ocr._recheck_suspicious_unary_formulas(
                proof_structure=structure,
                crop_path=crop_path,
                client=fake_client,
                output_dir=self.tmp_dir,
                modal_config={"system": "K"},
                node_id="1",
            )

        self.assertEqual(structure[0]["wffstr"], "¬□¬¬A")
        self.assertIn("¬□¬¬A", records[0]["forced_choice_candidates"])
        self.assertTrue(records[0]["accepted"])

    def test_recheck_suspicious_unary_formulas_keeps_original_when_vlm_is_uncertain(self) -> None:
        crop_path = self.tmp_dir / "crop_1.jpg"
        cv2.imwrite(str(crop_path), np.full((40, 40, 3), 255, dtype=np.uint8))
        structure = [
            {"line_label": "4", "wffstr": "¬□¬A", "jstr": "MC 3", "_ocr_node_id": "1"},
        ]

        def fake_run_checker(answer_text, autograde_config):
            payload = json.loads(answer_text)
            sentence = payload["lines"][0]["wffstr"]
            score_map = {
                "¬□¬A": (0, 4),
                "¬¬¬A": (0, 2),
                "□¬A": (0, 3),
                "¬□A": (0, 3),
                "¬□¬A": (0, 4),
            }
            score = score_map.get(sentence, (0, 5))
            return type(
                "Result",
                (),
                {
                    "parse_errors": [object()] * score[0],
                    "check_errors": [object()] * score[1],
                },
            )()

        fake_client = FakeModalVisionClient(
            [json.dumps({"line_label": "4", "selected_formula": "uncertain", "confidence": "uncertain"}, ensure_ascii=False)]
        )
        with patch.object(debug_modal_proof_ocr, "run_modal_proof_checker", side_effect=fake_run_checker):
            records = debug_modal_proof_ocr._recheck_suspicious_unary_formulas(
                proof_structure=structure,
                crop_path=crop_path,
                client=fake_client,
                output_dir=self.tmp_dir,
                modal_config={"system": "K"},
                node_id="1",
            )

        self.assertEqual(structure[0]["wffstr"], "¬□¬A")
        self.assertEqual(records[0]["vlm_choice"], "uncertain")
        self.assertFalse(records[0]["accepted"])
        self.assertTrue((self.tmp_dir / debug_modal_proof_ocr.UNARY_RECHECK_FILENAME).exists())

    def test_recursive_stitch_recovers_missing_child_token_before_box_intro(self) -> None:
        root = debug_modal_proof_ocr.SubproofNode("ROOT", (0, 0, 100, 100))
        parent = debug_modal_proof_ocr.SubproofNode("1", (10, 0, 80, 90))
        child = debug_modal_proof_ocr.SubproofNode("2", (20, 20, 60, 40))
        parent.parent = root
        child.parent = parent
        root.children.append(parent)
        parent.children.append(child)

        library = {
            "ROOT": [
                {"formula_text": "<SUBPROOF_ID_1>", "scope_role": "subproof_token"},
            ],
            "1": [
                {"formula_text": "□P", "scope_role": "premise", "line_label": "1"},
                {"formula_text": "□(P→Q)", "scope_role": "premise", "line_label": "2"},
                {"formula_text": "□Q", "justification_text": "□I 3-6", "scope_role": "derived_line", "line_label": "3"},
            ],
            "2": [
                {"formula_text": "□", "scope_role": "strict_open", "line_label": "1"},
                {"formula_text": "P", "justification_text": "□E 1", "scope_role": "derived_line", "line_label": "2"},
                {"formula_text": "P→Q", "justification_text": "□E 2", "scope_role": "derived_line", "line_label": "3"},
                {"formula_text": "Q", "justification_text": "→E 4 5", "scope_role": "derived_line", "line_label": "4"},
            ],
        }

        stitched = debug_modal_proof_ocr.recursive_stitch(library["ROOT"], library, root)
        while isinstance(stitched, list) and len(stitched) == 1 and isinstance(stitched[0], list):
            stitched = stitched[0]

        self.assertEqual(stitched[0]["wffstr"], "□P")
        self.assertEqual(stitched[1]["wffstr"], "□(P→Q)")
        self.assertIsInstance(stitched[2], list)
        self.assertEqual(stitched[2][0]["wffstr"], "□")
        self.assertEqual(stitched[3]["wffstr"], "□Q")

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

    def test_repair_modal_subproof_boundary_lines_fills_box_intro_and_dedupes(self) -> None:
        structure = [
            {"line_label": "1", "wffstr": "□(A → B)", "jstr": "PR", "line": 1},
            [
                {"line_label": "1", "wffstr": "□A", "jstr": "PR", "line": 2},
                [
                    {"line_label": "1", "wffstr": "□", "jstr": "PR", "line": 3},
                    {"line_label": "2", "wffstr": "A", "jstr": "□E 2", "line": 4},
                    {"line_label": "3", "wffstr": "A → B", "jstr": "□E 1", "line": 5},
                    {"line_label": "4", "wffstr": "B", "jstr": "→E 4 5", "line": 6},
                ],
                {"line_label": "7", "wffstr": "□B", "jstr": "derived_line", "line": 7},
                {"line_label": "8", "wffstr": "□A → □B", "jstr": "→I 2-7", "line": 8},
            ],
            {"line_label": "8", "wffstr": "□A → □B", "jstr": "→I 2-7", "line": 9},
        ]

        debug_modal_proof_ocr.repair_modal_subproof_boundary_lines(structure)

        self.assertEqual(structure[1][2]["jstr"], "□I 3-6")
        self.assertEqual(len(structure[1]), 3)
        self.assertEqual(structure[2]["wffstr"], "□A → □B")

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
                        {"formula_text": "¬□¬P", "justification_text": "MC 1", "scope_role": "derived_line", "line_label": "4"},
                        {"formula_text": "□□P", "justification_text": "□I 2-3", "scope_role": "derived_line", "line_label": "5"},
                    ],
                    ensure_ascii=False,
                ),
                json.dumps({"line_label": "4", "selected_formula": "¬□¬¬P", "confidence": "medium"}, ensure_ascii=False),
            ]
        )

        original_detect = debug_modal_proof_ocr.detect_vertical_bars
        original_run_checker = debug_modal_proof_ocr.run_modal_proof_checker
        try:
            debug_modal_proof_ocr.detect_vertical_bars = lambda img: [(0, 0, 2, 100)]
            def fake_run_checker(answer_text, autograde_config):
                payload = json.loads(answer_text)
                def iter_lines(items):
                    for item in items:
                        if isinstance(item, list):
                            yield from iter_lines(item)
                        elif isinstance(item, dict):
                            yield item

                target_sentence = next(
                    (line.get("wffstr", "") for line in iter_lines(payload["lines"]) if line.get("line_label") == "4"),
                    "",
                )
                score = (0, 1) if target_sentence == "¬□¬¬P" else (0, 3)
                return type(
                    "Result",
                    (),
                    {
                        "request_payload": {"system": "S4", "lines": []},
                        "response_payload": {"ok": True, "parse_errors": [], "check_errors": []},
                        "parse_errors": [object()] * score[0],
                        "check_errors": [object()] * score[1],
                    },
                )()

            debug_modal_proof_ocr.run_modal_proof_checker = fake_run_checker

            state = debug_modal_proof_ocr.transcribe_image_to_modal_debug_state(
                image_path=image_path,
                output_dir=output_dir,
                modal_config={"system": "S4", "conclusion": "□□P"},
                client=fake_client,
                vision_base_url="http://127.0.0.1:1234/v1",
                vision_model="google/gemma-3-12b-it",
            )
        finally:
            debug_modal_proof_ocr.detect_vertical_bars = original_detect
            debug_modal_proof_ocr.run_modal_proof_checker = original_run_checker

        self.assertEqual(state["system"], "S4")
        self.assertEqual(state["vision_model"], "google/gemma-3-12b-it")
        self.assertEqual(state["source_image_name"], "sample.png")
        self.assertIn("¬□¬¬P", state["proof_json_text"])
        self.assertIn("□□P", state["proof_json_text"])
        self.assertTrue((output_dir / "state.json").exists())
        self.assertTrue((output_dir / "final_structure.json").exists())
        self.assertTrue((output_dir / debug_modal_proof_ocr.UNARY_RECHECK_FILENAME).exists())

    def test_build_vlm_client_uses_explicit_overrides(self) -> None:
        client = debug_modal_proof_ocr.build_vlm_client(
            base_url="http://127.0.0.1:4321/v1",
            model="google/gemma-3-12b-it",
        )

        self.assertEqual(client.base_url, "http://127.0.0.1:4321/v1")
        self.assertEqual(client.model, "google/gemma-3-12b-it")


if __name__ == "__main__":
    unittest.main()
