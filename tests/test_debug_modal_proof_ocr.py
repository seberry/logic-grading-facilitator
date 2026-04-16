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
        self.assertIn("contradiction line may be written as ⊥, _|_, _ | _", prompt)
        self.assertIn("return formula_text as ⊥", prompt)

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

    def test_clear_output_dir_preserves_input_image_and_autograde_file(self) -> None:
        output_dir = self.tmp_dir / "example"
        output_dir.mkdir(parents=True, exist_ok=True)
        image_path = output_dir / "source_image.png"
        autograde_path = output_dir / "state.json"
        artifact_path = output_dir / "final_structure.json"
        nested_dir = output_dir / "stale_artifacts"

        image_path.write_bytes(b"png")
        autograde_path.write_text('{"system":"K"}', encoding="utf-8")
        artifact_path.write_text("old", encoding="utf-8")
        nested_dir.mkdir()
        (nested_dir / "extra.txt").write_text("stale", encoding="utf-8")

        debug_modal_proof_ocr._clear_output_dir(
            output_dir,
            preserve_paths=[image_path, autograde_path],
        )

        self.assertTrue(image_path.exists())
        self.assertTrue(autograde_path.exists())
        self.assertFalse(artifact_path.exists())
        self.assertFalse(nested_dir.exists())

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

    def test_normalize_formula_text_normalizes_bottom_variants(self) -> None:
        self.assertEqual(debug_modal_proof_ocr._normalize_formula_text("_|_", "derived_line"), "⊥")
        self.assertEqual(debug_modal_proof_ocr._normalize_formula_text("_ | _", "derived_line"), "⊥")
        self.assertEqual(debug_modal_proof_ocr._normalize_formula_text("bottom", "derived_line"), "⊥")

    def test_should_verify_unary_formula_targets_modal_prefix_lines(self) -> None:
        self.assertTrue(
            debug_modal_proof_ocr._should_verify_unary_formula(
                {
                    "formula_text": "¬□¬A",
                    "justification_text": "MC 3",
                    "scope_role": "derived_line",
                    "line_label": "4",
                }
            )
        )
        self.assertFalse(
            debug_modal_proof_ocr._should_verify_unary_formula(
                {
                    "formula_text": "A ∧ B",
                    "justification_text": "∧I 1 2",
                    "scope_role": "derived_line",
                    "line_label": "5",
                }
            )
        )

    def test_verifier_node_crop_masks_child_region_without_token_overlay(self) -> None:
        img = np.full((120, 160, 3), 255, dtype=np.uint8)
        img[30:90, 60:120] = 0
        root = debug_modal_proof_ocr.SubproofNode("ROOT", (0, 0, 160, 120))
        child = debug_modal_proof_ocr.SubproofNode("1", (60, 30, 60, 60))
        child.parent = root
        root.children.append(child)

        cropped = debug_modal_proof_ocr._verifier_node_crop_image(node=root, full_img=img)

        self.assertTrue(np.all(cropped[35:85, 65:115] == 255))

    def test_verifier_node_crop_masks_nested_child_to_right_edge(self) -> None:
        img = np.full((180, 220, 3), 255, dtype=np.uint8)
        img[65:120, 90:205] = 0
        parent = debug_modal_proof_ocr.SubproofNode("1", (60, 40, 2, 120))
        child = debug_modal_proof_ocr.SubproofNode("2", (80, 70, 2, 55))
        child.parent = parent
        parent.children.append(child)

        cropped = debug_modal_proof_ocr._verifier_node_crop_image(node=parent, full_img=img)

        self.assertTrue(np.all(cropped[40:80, 35:140] == 255))

    def test_verifier_node_crop_does_not_mutate_full_image_for_nested_nodes(self) -> None:
        img = np.full((180, 220, 3), 255, dtype=np.uint8)
        img[65:120, 90:205] = 0
        parent = debug_modal_proof_ocr.SubproofNode("1", (60, 40, 2, 120))
        child = debug_modal_proof_ocr.SubproofNode("2", (80, 70, 2, 55))
        child.parent = parent
        parent.children.append(child)
        original = img.copy()

        debug_modal_proof_ocr._verifier_node_crop_image(node=parent, full_img=img)

        self.assertTrue(np.array_equal(img, original))

    def test_extract_single_line_crop_avoids_adjacent_lines(self) -> None:
        crop_img = np.full((140, 120, 3), 255, dtype=np.uint8)
        cv2.putText(crop_img, "A", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(crop_img, "~A", (8, 73), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(crop_img, "[]A", (8, 118), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        line_crop = debug_modal_proof_ocr._extract_single_line_crop(node_crop_img=crop_img, line_order=1)

        self.assertLess(line_crop.shape[0], crop_img.shape[0])
        self.assertGreaterEqual(line_crop.shape[1], 80)

    def test_build_structured_unary_verifier_prompt_mentions_exact_counts(self) -> None:
        prompt = debug_modal_proof_ocr._build_structured_unary_verifier_prompt(
            {"formula_text": "¬□¬A", "justification_text": "MC 3", "scope_role": "derived_line"}
        )

        self.assertIn('main_modal_operator', prompt)
        self.assertIn("Count exactly how many negation marks", prompt)
        self.assertIn("Do not guess from proof validity", prompt)

    def test_build_forced_choice_unary_verifier_prompt_mentions_exact_counts(self) -> None:
        prompt = debug_modal_proof_ocr._build_forced_choice_unary_verifier_prompt(
            {"formula_text": "¬□¬A", "justification_text": "MC 3", "scope_role": "derived_line"},
            ["¬□¬A", "¬□¬¬A"],
        )

        self.assertIn("Count exactly how many negation marks", prompt)
        self.assertIn("Candidates: ¬□¬A ; ¬□¬¬A", prompt)
        self.assertIn('return "selected_formula": "uncertain"', prompt)

    def test_entry_to_line_preserves_scope_role_for_verifier_filters(self) -> None:
        line = debug_modal_proof_ocr._entry_to_line(
            {
                "formula_text": "□",
                "justification_text": "Assum",
                "scope_role": "strict_open",
                "line_label": "1",
            },
            node_id="2",
            entry_index=0,
            line_order=0,
        )

        self.assertIsNotNone(line)
        assert line is not None
        self.assertEqual(line["scope_role"], "strict_open")
        self.assertFalse(debug_modal_proof_ocr._should_verify_unary_formula(line))

    def test_best_verifier_attempt_prefers_expected_band_over_far_baseline_echo(self) -> None:
        best = debug_modal_proof_ocr._best_verifier_attempt(
            original_formula="□¬A",
            expected_line_order=4,
            attempts=[
                {
                    "band_index": 0,
                    "structured_formula": "□¬A",
                    "forced_choice": "□¬A",
                    "forced_confidence": 0.95,
                    "structured_counts": {"confidence": 0.95},
                    "line_crop_path": "far.jpg",
                },
                {
                    "band_index": 4,
                    "structured_formula": "□¬¬A",
                    "forced_choice": "□¬¬A",
                    "proposed_correction": "□¬¬A",
                    "forced_confidence": 0.98,
                    "structured_counts": {"confidence": 0.98},
                    "line_crop_path": "expected.jpg",
                },
            ],
        )

        self.assertEqual(best["line_crop_path"], "expected.jpg")

    def test_run_unary_verifier_experiment_records_proposed_correction(self) -> None:
        img = np.full((90, 120, 3), 255, dtype=np.uint8)
        cv2.putText(img, "~[]~~A", (4, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)
        root = debug_modal_proof_ocr.SubproofNode("ROOT", (0, 0, 120, 90))
        structure = [
            {
                "line_label": "4",
                "wffstr": "¬□¬A",
                "jstr": "MC 3",
                "scope_role": "derived_line",
                "_ocr_node_id": "ROOT",
                "_ocr_line_order": 0,
            }
        ]
        fake_client = FakeModalVisionClient(
            [
                json.dumps(
                    {
                        "main_modal_operator": "box",
                        "negations_before_modal": 1,
                        "negations_before_final_atom": 2,
                        "confidence": "high",
                    },
                    ensure_ascii=False,
                ),
                json.dumps({"selected_formula": "¬□¬¬A", "confidence": "high"}, ensure_ascii=False),
            ]
        )

        records = debug_modal_proof_ocr._run_unary_verifier_experiment(
            proof_structure=structure,
            tree_root=root,
            full_img=img,
            output_dir=self.tmp_dir,
            verifier_client=fake_client,
            checker_response={"parse_errors": [], "check_errors": [{"line": 4}]},
        )

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["structured_formula"], "¬□¬¬A")
        self.assertEqual(records[0]["forced_choice"], "¬□¬¬A")
        self.assertEqual(records[0]["proposed_correction"], "¬□¬¬A")
        self.assertTrue(records[0]["would_auto_apply"])
        self.assertTrue((self.tmp_dir / debug_modal_proof_ocr.UNARY_VERIFIER_FILENAME).exists())

    def test_run_unary_verifier_experiment_keeps_pure_negation_modal_free(self) -> None:
        img = np.full((90, 120, 3), 255, dtype=np.uint8)
        cv2.putText(img, "~~A", (6, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)
        root = debug_modal_proof_ocr.SubproofNode("ROOT", (0, 0, 120, 90))
        structure = [
            {
                "line_label": "3",
                "wffstr": "¬¬A",
                "jstr": "DN 1",
                "scope_role": "derived_line",
                "_ocr_node_id": "ROOT",
                "_ocr_line_order": 0,
            }
        ]
        fake_client = FakeModalVisionClient(
            [
                json.dumps(
                    {
                        "main_modal_operator": "none",
                        "negations_before_modal": 0,
                        "negations_before_final_atom": 2,
                        "confidence": "high",
                    },
                    ensure_ascii=False,
                ),
                json.dumps({"selected_formula": "¬¬A", "confidence": "high"}, ensure_ascii=False),
            ]
        )

        records = debug_modal_proof_ocr._run_unary_verifier_experiment(
            proof_structure=structure,
            tree_root=root,
            full_img=img,
            output_dir=self.tmp_dir,
            verifier_client=fake_client,
            checker_response={"parse_errors": [], "check_errors": [{"line": 3}]},
        )

        self.assertEqual(records[0]["structured_formula"], "¬¬A")
        self.assertEqual(records[0]["forced_choice"], "¬¬A")
        self.assertIsNone(records[0]["proposed_correction"])
        self.assertFalse(records[0]["would_auto_apply"])

    def test_apply_unary_verifier_corrections_updates_matching_lines(self) -> None:
        structure = [
            {"line": 4, "line_label": "4", "wffstr": "¬□¬A", "jstr": "MC 3"},
            {"line": 8, "line_label": "8", "wffstr": "□¬A", "jstr": "□I 5-7"},
        ]

        rewritten, rewrites = debug_modal_proof_ocr._apply_unary_verifier_corrections(
            structure,
            [
                {
                    "line_number": 4,
                    "node_id": "1",
                    "proposed_correction": "¬□¬¬A",
                    "would_auto_apply": True,
                },
                {
                    "line_number": 8,
                    "node_id": "1",
                    "proposed_correction": "□¬¬A",
                    "would_auto_apply": True,
                },
            ],
        )

        self.assertEqual(rewritten[0]["wffstr"], "¬□¬¬A")
        self.assertEqual(rewritten[1]["wffstr"], "□¬¬A")
        self.assertEqual([item["line_number"] for item in rewrites], [4, 8])
        self.assertEqual(rewrites[1]["original_formula"], "□¬A")
        self.assertEqual(rewrites[1]["rewritten_formula"], "□¬¬A")

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
        img = np.full((90, 140, 3), 255, dtype=np.uint8)
        cv2.putText(img, "~[]~~P", (4, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)
        cv2.imwrite(str(image_path), img)
        output_dir = self.tmp_dir / "out"

        fake_client = FakeModalVisionClient(
            responses=[
                json.dumps(
                    [
                        {"formula_text": "¬□¬P", "justification_text": "MC 1", "scope_role": "derived_line", "line_label": "4"},
                    ],
                    ensure_ascii=False,
                ),
            ]
        )
        fake_verifier_client = FakeModalVisionClient(
            responses=[
                json.dumps(
                    {
                        "line_centers_rel": [250],
                        "confidence": "low",
                    },
                    ensure_ascii=False,
                ),
                json.dumps(
                    {
                        "main_modal_operator": "box",
                        "negations_before_modal": 1,
                        "negations_before_final_atom": 2,
                        "confidence": "high",
                    },
                    ensure_ascii=False,
                ),
                json.dumps({"line_label": "4", "selected_formula": "¬□¬¬P", "confidence": "medium"}, ensure_ascii=False),
            ]
        )

        original_detect = debug_modal_proof_ocr.detect_vertical_bars
        original_run_checker = debug_modal_proof_ocr.run_modal_proof_checker
        try:
            debug_modal_proof_ocr.detect_vertical_bars = lambda img: []
            def fake_run_checker(answer_text, autograde_config):
                return type(
                    "Result",
                    (),
                    {
                        "request_payload": {"system": "S4", "lines": []},
                        "response_payload": {"ok": True, "parse_errors": [], "check_errors": [{"line": 4}]},
                        "parse_errors": [],
                        "check_errors": [object()],
                    },
                )()

            debug_modal_proof_ocr.run_modal_proof_checker = fake_run_checker

            state = debug_modal_proof_ocr.transcribe_image_to_modal_debug_state(
                image_path=image_path,
                output_dir=output_dir,
                modal_config={"system": "S4", "conclusion": "□□P"},
                client=fake_client,
                verifier_client=fake_verifier_client,
                vision_base_url="http://127.0.0.1:1234/v1",
                vision_model="google/gemma-3-12b-it",
                verifier_vision_model="qwen/qwen3-vl-8b",
            )
        finally:
            debug_modal_proof_ocr.detect_vertical_bars = original_detect
            debug_modal_proof_ocr.run_modal_proof_checker = original_run_checker

        self.assertEqual(state["system"], "S4")
        self.assertEqual(state["vision_model"], "google/gemma-3-12b-it")
        self.assertEqual(state["verifier_vision_model"], "qwen/qwen3-vl-8b")
        self.assertEqual(state["source_image_name"], "sample.png")
        self.assertIn("¬□¬¬P", state["proof_json_text"])
        self.assertIn("¬□¬P", state["original_proof_json_text"])
        self.assertEqual(state["verifier_rewrite_summary"][0]["rewritten_formula"], "¬□¬¬P")
        self.assertIn("verifier rewrote 1 line", state["run_summary"])
        self.assertTrue((output_dir / "state.json").exists())
        self.assertTrue((output_dir / debug_modal_proof_ocr.BASELINE_FINAL_STRUCTURE_FILENAME).exists())
        self.assertTrue((output_dir / "final_structure.json").exists())
        self.assertTrue((output_dir / debug_modal_proof_ocr.UNARY_VERIFIER_FILENAME).exists())
        self.assertTrue((output_dir / debug_modal_proof_ocr.VERIFIER_REWRITES_FILENAME).exists())

    def test_build_vlm_client_uses_explicit_overrides(self) -> None:
        client = debug_modal_proof_ocr.build_vlm_client(
            base_url="http://127.0.0.1:4321/v1",
            model="google/gemma-3-12b-it",
        )

        self.assertEqual(client.base_url, "http://127.0.0.1:4321/v1")
        self.assertEqual(client.model, "google/gemma-3-12b-it")


if __name__ == "__main__":
    unittest.main()
