import unittest

import cv2
import numpy as np

from proofs_ocr.pipelines import modal_v2
from proofs_ocr.pipelines.modal_v2 import run_modal_baseline_pipeline
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
