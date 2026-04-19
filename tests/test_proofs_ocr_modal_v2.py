import unittest
from pathlib import Path
import shutil

import cv2
import numpy as np

from proofs_ocr.domains import modal
from proofs_ocr.pipelines import modal_v2
from proofs_ocr.pipelines.modal_v2 import run_modal_baseline_pipeline
from proofs_ocr.pipelines.modal_v2_core import ModalPipelineFeatureFlags
from proofs_ocr.pipelines.modal_v2_runtime import ModalBaselineArtifactLineReader
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
