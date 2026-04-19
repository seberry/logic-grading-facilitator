import unittest
from pathlib import Path
import shutil

import cv2
import numpy as np

from proofs_ocr.domains import modal
from proofs_ocr.domains import modal_line_zones
from modal_ocr.prompts import build_justification_ocr_prompt
from proofs_ocr.pipelines.modal_v2_bar_filters import filter_false_left_separator_bars
from proofs_ocr.pipelines.modal_v2_unary import run_structured_unary_probe
from proofs_ocr.pipelines import modal_v2
from proofs_ocr.pipelines import modal_v2_core
from proofs_ocr.pipelines.modal_v2 import run_modal_baseline_pipeline
from proofs_ocr.pipelines.modal_v2_core import ModalPipelineFeatureFlags
from proofs_ocr.pipelines.modal_v2_runtime import ModalBaselineArtifactLineReader
from proofs_ocr.pipelines.modal_v2_line_zones import extract_line_zone_crops
from proofs_ocr.common.types import SubproofNode


def _synthetic_recursive_modal_image() -> np.ndarray:
    image = np.full((220, 220, 3), 255, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.putText(image, "P", (12, 30), font, 0.8, (0, 0, 0), 2)
    cv2.line(image, (42, 54), (42, 150), (0, 0, 0), 2)
    cv2.putText(image, "A", (62, 78), font, 0.8, (0, 0, 0), 2)
    cv2.putText(image, "B", (62, 128), font, 0.8, (0, 0, 0), 2)
    cv2.putText(image, "Q", (12, 188), font, 0.8, (0, 0, 0), 2)

    return image


class ModalV2BaselinePipelineTests(unittest.TestCase):
    def test_node_crop_image_avoids_bottom_padding_for_child_nodes(self) -> None:
        image = np.zeros((200, 50, 3), dtype=np.uint8)
        child = SubproofNode("1", (10, 40, 2, 80))

        crop = modal_v2._node_crop_image(node=child, full_img=image)

        self.assertEqual(crop.shape[0], 90)

    def test_extract_ocr_line_crop_keeps_formula_side_visible(self) -> None:
        image = np.full((90, 900, 3), 255, dtype=np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, "1", (300, 42), font, 1.1, (0, 0, 0), 2)
        cv2.line(image, (345, 8), (345, 76), (0, 0, 0), 2)
        cv2.putText(image, "[]A", (395, 42), font, 1.1, (0, 0, 0), 2)

        crop = modal_v2._extract_ocr_line_crop(
            node_crop_img=image,
            row_record={"y1": 18, "y2": 46},
        )

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        formula_zone = gray[:, 380:520]
        self.assertGreater(int((formula_zone < 185).sum()), 25)

    def test_recover_line_entry_from_crop_recovers_tiny_strict_open(self) -> None:
        image = np.full((40, 220, 3), 255, dtype=np.uint8)
        image[:, 0:3] = 0
        cv2.rectangle(image, (34, 8), (56, 30), (0, 0, 0), 2)

        resolved, recovery = modal.recover_line_entry_from_crop(
            line_crop_img=image,
            parsed_entry={
                "line_label": "",
                "formula_text": "",
                "justification_text": "",
                "scope_role": "premise",
                "confidence": "1",
            },
        )

        self.assertEqual(recovery, "tiny_strict_open_crop")
        self.assertEqual(resolved["formula_text"], "□")
        self.assertEqual(resolved["scope_role"], "strict_open")

    def test_recover_line_entry_from_crop_does_not_fire_for_square_plus_formula(self) -> None:
        image = np.full((48, 260, 3), 255, dtype=np.uint8)
        image[:, 0:3] = 0
        cv2.rectangle(image, (34, 10), (56, 32), (0, 0, 0), 2)
        cv2.putText(image, "A", (78, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        resolved, recovery = modal.recover_line_entry_from_crop(
            line_crop_img=image,
            parsed_entry={
                "line_label": "",
                "formula_text": "",
                "justification_text": "",
                "scope_role": "premise",
                "confidence": "1",
            },
        )

        self.assertEqual(recovery, "")
        self.assertEqual(resolved["formula_text"], "")

    def test_recover_line_entry_from_crop_does_not_fire_when_line_label_is_present(self) -> None:
        image = np.full((40, 220, 3), 255, dtype=np.uint8)
        image[:, 0:3] = 0
        cv2.rectangle(image, (34, 8), (56, 30), (0, 0, 0), 2)

        resolved, recovery = modal.recover_line_entry_from_crop(
            line_crop_img=image,
            parsed_entry={
                "line_label": "1",
                "formula_text": "",
                "justification_text": "",
                "scope_role": "premise",
                "confidence": "1",
            },
        )

        self.assertEqual(recovery, "")
        self.assertEqual(resolved["formula_text"], "")

    def test_detect_label_only_crop_flags_wide_label_region_blob(self) -> None:
        image = np.full((52, 1433, 3), 255, dtype=np.uint8)
        cv2.putText(image, "5", (483, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)

        skip_reason = modal.detect_label_only_crop(image)

        self.assertEqual(skip_reason, "label_only_crop")

    def test_detect_label_only_crop_keeps_wide_formula_crop(self) -> None:
        image = np.full((52, 1433, 3), 255, dtype=np.uint8)
        cv2.putText(image, "3", (500, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
        cv2.putText(image, "~[]A", (610, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)

        skip_reason = modal.detect_label_only_crop(image)

        self.assertEqual(skip_reason, "")

    def test_modal_baseline_artifact_line_reader_skips_label_only_crop(self) -> None:
        class FailIfCalledClient:
            def chat_with_image_prompt(self, image_path, prompt, system_prompt=None):
                raise AssertionError("client should not be called for label-only crop")

        image = np.full((52, 1433, 3), 255, dtype=np.uint8)
        cv2.putText(image, "5", (483, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
        node = SubproofNode("ROOT", (0, 0, 0, 0))
        output_dir = Path("working/test_modal_v2_runtime_skip")
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            reader = ModalBaselineArtifactLineReader(
                output_dir=output_dir,
                client=FailIfCalledClient(),
                prompt="prompt",
            )

            resolved = reader(
                node=node,
                node_crop_img=image,
                line_crop_img=image,
                row_record={"row_order": 5, "band_index": 3, "y1": 10, "y2": 30},
                feature_flags=ModalPipelineFeatureFlags(),
            )

            self.assertEqual(resolved["formula_text"], "")
            payload_path = output_dir / "node_line_ROOT_5.json"
            self.assertTrue(payload_path.exists())
            payload = payload_path.read_text(encoding="utf-8")
            self.assertIn('"skip_reason": "label_only_crop"', payload)
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)

    def test_modal_baseline_artifact_line_reader_recovers_first_nonroot_strict_open_band(self) -> None:
        class FixedClient:
            def chat_with_image_prompt(self, image_path, prompt, system_prompt=None):
                del image_path, prompt, system_prompt
                return (
                    '{"line_label":"","formula_text":"Assum","justification_text":"",'
                    '"scope_role":"ordinary_assumption","confidence":"0.98"}'
                )

        image = np.full((44, 320, 3), 255, dtype=np.uint8)
        cv2.line(image, (6, 0), (6, 43), (0, 0, 0), 2)
        cv2.rectangle(image, (20, 4), (38, 22), (0, 0, 0), 2)
        cv2.line(image, (20, 27), (110, 27), (0, 0, 0), 2)
        cv2.putText(image, "Assum", (190, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        cv2.putText(image, "A", (24, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        node = SubproofNode("2", (0, 0, 0, 0))
        output_dir = Path("working/test_modal_v2_runtime_first_nonroot_strict_open")
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            reader = ModalBaselineArtifactLineReader(
                output_dir=output_dir,
                client=FixedClient(),
                prompt="prompt",
            )

            resolved = reader(
                node=node,
                node_crop_img=image,
                line_crop_img=image,
                row_record={"row_order": 1, "band_index": 0, "y1": 4, "y2": 26},
                feature_flags=ModalPipelineFeatureFlags(),
            )

            self.assertEqual(resolved["formula_text"], "□")
            self.assertEqual(resolved["scope_role"], "strict_open")
            self.assertEqual(resolved["justification_text"], "")
            payload = (output_dir / "node_line_2_1.json").read_text(encoding="utf-8")
            self.assertIn('"recovery_applied": "first_nonroot_strict_open_band"', payload)
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)

    def test_extract_line_zone_crops_splits_formula_and_justification_sides(self) -> None:
        image = np.full((52, 420, 3), 255, dtype=np.uint8)
        cv2.putText(image, "4", (12, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.putText(image, "[]A", (82, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
        cv2.putText(image, "[]E 1", (285, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        formula_crop, justification_crop, geometry = extract_line_zone_crops(image)

        self.assertGreater(geometry["split_x"], 120)
        self.assertLess(geometry["split_x"], 300)
        formula_gray = cv2.cvtColor(formula_crop, cv2.COLOR_BGR2GRAY)
        justification_gray = cv2.cvtColor(justification_crop, cv2.COLOR_BGR2GRAY)
        self.assertGreater(int((formula_gray[:, :180] < 185).sum()), 25)
        self.assertGreater(int((justification_gray[:, -140:] < 185).sum()), 25)

    def test_filter_false_left_separator_bars_removes_top_anchored_left_edge_bar(self) -> None:
        image = np.full((280, 260, 3), 255, dtype=np.uint8)

        filtered, payload = filter_false_left_separator_bars(
            image=image,
            bars=[(18, 0, 8, 262), (54, 157, 2, 128)],
        )

        self.assertEqual(filtered, [(54, 157, 2, 128)])
        self.assertTrue(payload["applied"])
        self.assertEqual(payload["reason"], "removed_false_left_separator_candidate")

    def test_filter_false_left_separator_bars_keeps_bar_when_not_top_anchored(self) -> None:
        image = np.full((280, 260, 3), 255, dtype=np.uint8)

        filtered, payload = filter_false_left_separator_bars(
            image=image,
            bars=[(18, 24, 8, 220), (54, 157, 2, 128)],
        )

        self.assertEqual(filtered, [(18, 24, 8, 220), (54, 157, 2, 128)])
        self.assertFalse(payload["applied"])
        self.assertEqual(payload["reason"], "leftmost_not_top_anchored")

    def test_run_modal_baseline_pipeline_can_filter_false_left_separator_before_tree_build(self) -> None:
        image = np.full((280, 260, 3), 255, dtype=np.uint8)
        rows_seen = []

        def read_line(*, node, node_crop_img, line_crop_img, row_record, feature_flags):
            del node_crop_img, line_crop_img, feature_flags
            rows_seen.append((node.id, int(row_record.get("row_order", 0))))
            return {"scope_role": "premise", "formula_text": "P"}

        original_detect_vertical_bars = modal_v2_core.tree.detect_vertical_bars
        try:
            modal_v2_core.tree.detect_vertical_bars = lambda _img: [(18, 0, 8, 262), (54, 157, 2, 128)]
            result = run_modal_baseline_pipeline(
                image=image,
                read_line=read_line,
                enable_false_left_separator_filter=True,
            )
        finally:
            modal_v2_core.tree.detect_vertical_bars = original_detect_vertical_bars

        self.assertEqual(result.raw_detected_bars, [(18, 0, 8, 262), (54, 157, 2, 128)])
        self.assertEqual(result.detected_bars, [(54, 157, 2, 128)])
        self.assertTrue(result.bar_filter_debug["applied"])

    def test_modal_line_zones_merge_recovers_justification_from_formula_label_spill(self) -> None:
        merged = modal_line_zones.merge_line_zone_results(
            formula_result={"line_label": "□E 2", "formula_text": "A", "confidence": "high"},
            justification_result={"justification_text": "", "confidence": "medium"},
            is_root_node=False,
            is_first_visible_row_of_nonroot=False,
        )

        self.assertEqual(merged["formula_text"], "A")
        self.assertEqual(merged["justification_text"], "□E 2")
        self.assertEqual(merged["scope_role"], "derived_line")

    def test_modal_line_zones_merge_marks_blank_root_line_as_premise(self) -> None:
        merged = modal_line_zones.merge_line_zone_results(
            formula_result={"line_label": "1", "formula_text": "A", "confidence": "high"},
            justification_result={"justification_text": "", "confidence": "medium"},
            is_root_node=True,
            is_first_visible_row_of_nonroot=False,
        )

        self.assertEqual(merged["scope_role"], "premise")
        self.assertEqual(merged["justification_text"], "")

    def test_modal_line_zones_merge_marks_first_nonroot_blank_line_as_assumption(self) -> None:
        merged = modal_line_zones.merge_line_zone_results(
            formula_result={"line_label": "", "formula_text": "A", "confidence": "high"},
            justification_result={"justification_text": "", "confidence": "medium"},
            is_root_node=False,
            is_first_visible_row_of_nonroot=True,
        )

        self.assertEqual(merged["scope_role"], "ordinary_assumption")
        self.assertEqual(merged["justification_text"], "")

    def test_modal_line_zones_merge_keeps_later_nonroot_blank_line_as_derived(self) -> None:
        merged = modal_line_zones.merge_line_zone_results(
            formula_result={"line_label": "", "formula_text": "B", "confidence": "high"},
            justification_result={"justification_text": "", "confidence": "medium"},
            is_root_node=False,
            is_first_visible_row_of_nonroot=False,
        )

        self.assertEqual(merged["scope_role"], "derived_line")
        self.assertEqual(merged["justification_text"], "")

    def test_resolve_modal_ocr_entry_normalizes_assumption_variants_to_pr(self) -> None:
        resolved = modal.resolve_modal_ocr_entry(
            {
                "scope_role": "ordinary_assumption",
                "formula_text": "A",
                "justification_text": "Assump",
                "line_label": "",
            },
            0,
        )

        self.assertIsNotNone(resolved)
        self.assertEqual(resolved["jstr"], "PR")

    def test_build_justification_ocr_prompt_mentions_assumption_and_intro_variants(self) -> None:
        prompt = build_justification_ocr_prompt({"system": "S4"})

        self.assertIn('"Assum"', prompt)
        self.assertIn('"Assump"', prompt)
        self.assertIn('"¬I 1-9"', prompt)
        self.assertIn('keep the printed tilde form', prompt)
        self.assertIn('do not copy numbers from the examples', prompt)
        self.assertIn('first line of a proof or subproof often has no printed justification', prompt)

    def test_modal_baseline_artifact_line_reader_can_separate_formula_and_justification_zones(self) -> None:
        class SequentialClient:
            def __init__(self) -> None:
                self.calls = []

            def chat_with_image_prompt(self, image_path, prompt, system_prompt=None):
                del system_prompt
                self.calls.append((image_path.name, prompt))
                if "formula side" in prompt:
                    return '{"line_label":"4","formula_text":"[]A","confidence":"high"}'
                return '{"justification_text":"[]E 1","confidence":"medium"}'

        image = np.full((52, 420, 3), 255, dtype=np.uint8)
        cv2.putText(image, "4", (12, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.putText(image, "[]A", (82, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
        cv2.putText(image, "[]E 1", (285, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        node = SubproofNode("ROOT", (0, 0, 0, 0))
        output_dir = Path("working/test_modal_v2_runtime_line_zones")
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            client = SequentialClient()
            reader = ModalBaselineArtifactLineReader(
                output_dir=output_dir,
                client=client,
                prompt="unused",
                modal_config={"system": "S4"},
                enable_formula_justification_separation=True,
            )

            resolved = reader(
                node=node,
                node_crop_img=image,
                line_crop_img=image,
                row_record={"row_order": 4, "band_index": 0, "y1": 8, "y2": 42},
                feature_flags=ModalPipelineFeatureFlags(),
            )

            self.assertEqual(resolved["line_label"], "4")
            self.assertEqual(resolved["formula_text"], "□A")
            self.assertEqual(resolved["justification_text"], "□E 1")
            self.assertEqual(resolved["scope_role"], "derived_line")
            self.assertEqual(len(client.calls), 2)
            self.assertTrue((output_dir / "node_line_formula_ROOT_4.jpg").exists())
            self.assertTrue((output_dir / "node_line_justification_ROOT_4.jpg").exists())
            payload = (output_dir / "node_line_ROOT_4.json").read_text(encoding="utf-8")
            self.assertIn('"mode": "formula_justification_separation"', payload)
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)

    def test_run_structured_unary_probe_can_use_formula_zone_crops_only(self) -> None:
        class FixedVerifierClient:
            def __init__(self) -> None:
                self.calls = []

            def chat_with_image_prompt(self, image_path, prompt, system_prompt=None):
                del prompt, system_prompt
                self.calls.append(image_path.name)
                return (
                    '{"main_modal_operator":"box","negations_before_modal":0,'
                    '"negations_before_final_atom":2,"confidence":1.0}'
                )

        output_dir = Path("working/test_modal_v2_unary_formula_probe")
        output_dir.mkdir(parents=True, exist_ok=True)
        formula_crop = np.full((40, 120, 3), 255, dtype=np.uint8)
        cv2.putText(formula_crop, "[]~A", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        whole_line_crop = np.full((40, 220, 3), 255, dtype=np.uint8)
        cv2.putText(whole_line_crop, "1 []~A []E 2", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        cv2.imwrite(str(output_dir / "node_line_formula_ROOT_1.jpg"), formula_crop)
        cv2.imwrite(str(output_dir / "node_line_ROOT_1.jpg"), whole_line_crop)

        try:
            client = FixedVerifierClient()
            records = run_structured_unary_probe(
                entries_by_node={
                    "ROOT": [
                        {
                            "formula_text": "□¬A",
                            "justification_text": "",
                            "scope_role": "derived_line",
                            "_ocr_row_order": 1,
                        }
                    ]
                },
                output_dir=output_dir,
                client=client,
                crop_mode="formula_crop_only",
            )

            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["probe_source_kind"], "formula_crop")
            self.assertEqual(records[0]["probe_source_filename"], "node_line_formula_ROOT_1.jpg")
            self.assertEqual(records[0]["structured_formula"], "□¬¬A")
            self.assertEqual(client.calls, ["unary_structured_formula_ROOT_1.jpg"])
            self.assertTrue((output_dir / "unary_structured_formula_probe.json").exists())
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)

    def test_run_modal_baseline_pipeline_stitches_child_scope(self) -> None:
        image = _synthetic_recursive_modal_image()
        line_calls: dict[str, list[dict[str, int | float | str]]] = {}

        def read_line(*, node, node_crop_img, line_crop_img, row_record, feature_flags):
            self.assertGreater(node_crop_img.size, 0)
            self.assertGreater(line_crop_img.size, 0)
            self.assertFalse(feature_flags.enable_boundary_repair)

            line_calls.setdefault(node.id, []).append(dict(row_record))
            line_index = len(line_calls[node.id])
            if node.id == "1":
                if line_index == 1:
                    return {"scope_role": "ordinary_assumption", "formula_text": "A"}
                return {"scope_role": "derived_line", "formula_text": "B", "justification_text": "R 1"}

            if line_index == 1:
                return {"scope_role": "premise", "formula_text": "P"}
            return {"scope_role": "derived_line", "formula_text": "Q", "justification_text": "R 2"}

        result = run_modal_baseline_pipeline(
            image=image,
            read_line=read_line,
        )

        self.assertEqual(len(result.detected_bars), 1)
        self.assertEqual([record["kind"] for record in result.row_records_by_node["1"]], ["line", "line"])
        self.assertEqual(
            [record["kind"] for record in result.row_records_by_node["ROOT"]],
            ["line", "subproof_token", "line"],
        )
        self.assertEqual(result.library["ROOT"][1]["formula_text"], "<SUBPROOF_ID_1>")
        self.assertEqual(
            result.final_structure,
            [
                {"line": 1, "wffstr": "P", "jstr": "PR"},
                [
                    {"line": 2, "wffstr": "A", "jstr": "PR"},
                    {"line": 3, "wffstr": "B", "jstr": "R 1"},
                ],
                {"line": 4, "wffstr": "Q", "jstr": "R 2"},
            ],
        )


if __name__ == "__main__":
    unittest.main()
