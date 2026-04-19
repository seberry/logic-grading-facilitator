import json
import shutil
import unittest
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np

from proofs_ocr.common import artifacts
from proofs_ocr.pipelines.modal_v2_unary import run_blob_unary_probe, run_structured_unary_probe


class FakeUnaryVerifierClient:
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, str]] = []

    def chat_with_image_prompt(self, image_path: Path, prompt: str, system_prompt: str | None = None) -> str:
        del system_prompt
        self.calls.append((image_path.name, prompt))
        if not self.responses:
            raise AssertionError("No fake verifier responses left")
        return self.responses.pop(0)


class ModalV2UnaryProbeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = Path("working") / "test_tmp" / f"modal_v2_unary_{uuid4().hex}"
        self.tmp_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_run_structured_unary_probe_writes_records_for_unary_lines(self) -> None:
        crop_path = self.tmp_dir / artifacts.NODE_LINE_CROP_TEMPLATE.format(node_id="ROOT", row_order=1)
        image = np.full((40, 160, 3), 255, dtype=np.uint8)
        cv2.putText(image, "[]~A", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
        cv2.imwrite(str(crop_path), image)

        client = FakeUnaryVerifierClient(
            [
                json.dumps(
                    {
                        "main_modal_operator": "box",
                        "negations_before_modal": 0,
                        "negations_before_final_atom": 2,
                        "confidence": 1.0,
                    }
                )
            ]
        )

        records = run_structured_unary_probe(
            entries_by_node={
                "ROOT": [
                    {
                        "_ocr_row_order": 1,
                        "line_label": "1",
                        "formula_text": "□¬A",
                        "justification_text": "□I 5-7",
                        "scope_role": "derived_line",
                    },
                    {
                        "_ocr_row_order": 2,
                        "formula_text": "P",
                        "justification_text": "PR",
                        "scope_role": "premise",
                    },
                ]
            },
            output_dir=self.tmp_dir,
            client=client,
        )

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["baseline_formula"], "□¬A")
        self.assertEqual(records[0]["structured_formula"], "□¬¬A")
        self.assertTrue(records[0]["would_change_formula"])
        self.assertTrue((self.tmp_dir / artifacts.UNARY_STRUCTURED_PROBE_FILENAME).exists())
        self.assertTrue((self.tmp_dir / artifacts.UNARY_STRUCTURED_LINE_CROP_TEMPLATE.format(node_id="ROOT", row_order=1)).exists())
        self.assertEqual(len(client.calls), 1)

    def test_run_blob_unary_probe_writes_records_for_unary_lines(self) -> None:
        crop_path = self.tmp_dir / artifacts.NODE_LINE_CROP_TEMPLATE.format(node_id="ROOT", row_order=1)
        image = np.full((80, 320, 3), 255, dtype=np.uint8)
        cv2.putText(image, "1", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
        cv2.putText(image, "[]~~A", (45, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
        cv2.putText(image, "[]I 5-7", (220, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
        cv2.imwrite(str(crop_path), image)

        records = run_blob_unary_probe(
            entries_by_node={
                "ROOT": [
                    {
                        "_ocr_row_order": 1,
                        "line_label": "1",
                        "formula_text": "□¬A",
                        "justification_text": "□I 5-7",
                        "scope_role": "derived_line",
                    },
                    {
                        "_ocr_row_order": 2,
                        "formula_text": "P",
                        "justification_text": "PR",
                        "scope_role": "premise",
                    },
                ]
            },
            output_dir=self.tmp_dir,
        )

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["baseline_formula"], "□¬A")
        self.assertTrue((self.tmp_dir / artifacts.UNARY_BLOB_PROBE_FILENAME).exists())
        self.assertTrue((self.tmp_dir / artifacts.UNARY_BLOB_LINE_WINDOW_TEMPLATE.format(node_id="ROOT", row_order=1)).exists())
        self.assertGreaterEqual(records[0]["max_component_count"], records[0]["ocr_symbol_count"])


if __name__ == "__main__":
    unittest.main()
