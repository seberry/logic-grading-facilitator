import json
import shutil
import unittest
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np

from proofs_ocr.common import artifacts
from proofs_ocr.pipelines.modal_v2 import transcribe_image_to_modal_v2_debug_state


def _synthetic_recursive_modal_image() -> np.ndarray:
    image = np.full((220, 220, 3), 255, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.putText(image, "P", (12, 30), font, 0.8, (0, 0, 0), 2)
    cv2.line(image, (42, 54), (42, 150), (0, 0, 0), 2)
    cv2.putText(image, "A", (62, 78), font, 0.8, (0, 0, 0), 2)
    cv2.putText(image, "B", (62, 128), font, 0.8, (0, 0, 0), 2)
    cv2.putText(image, "Q", (12, 188), font, 0.8, (0, 0, 0), 2)

    return image


class FakeImagePromptClient:
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, str]] = []

    def chat_with_image_prompt(self, image_path: Path, prompt: str, system_prompt: str | None = None) -> str:
        del system_prompt
        self.calls.append((image_path.name, prompt))
        if not self.responses:
            raise AssertionError("No fake OCR responses left")
        return self.responses.pop(0)


class ModalV2DebugStateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = Path("working") / "test_tmp" / f"modal_v2_debug_{uuid4().hex}"
        self.tmp_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_transcribe_image_to_modal_v2_debug_state_writes_artifacts(self) -> None:
        image_path = self.tmp_dir / "synthetic.png"
        cv2.imwrite(str(image_path), _synthetic_recursive_modal_image())
        fake_client = FakeImagePromptClient(
            [
                json.dumps({"scope_role": "ordinary_assumption", "formula_text": "A", "justification_text": "", "line_label": "", "confidence": "high"}),
                json.dumps({"scope_role": "derived_line", "formula_text": "B", "justification_text": "R 1", "line_label": "", "confidence": "high"}),
                json.dumps({"scope_role": "premise", "formula_text": "P", "justification_text": "", "line_label": "", "confidence": "high"}),
                json.dumps({"scope_role": "derived_line", "formula_text": "Q", "justification_text": "R 2", "line_label": "", "confidence": "high"}),
            ]
        )

        state = transcribe_image_to_modal_v2_debug_state(
            image_path=image_path,
            output_dir=self.tmp_dir / "out",
            modal_config={"system": "S4", "conclusion": "Q"},
            client=fake_client,
            vision_base_url="http://127.0.0.1:1234/v1",
            vision_model="fake-model",
        )

        output_dir = self.tmp_dir / "out"
        self.assertEqual(len(fake_client.calls), 4)
        self.assertTrue((output_dir / artifacts.PROMPT_FILENAME).exists())
        self.assertTrue((output_dir / artifacts.DEBUG_BARS_FILENAME).exists())
        self.assertTrue((output_dir / artifacts.NODE_CROP_TEMPLATE.format(node_id="ROOT")).exists())
        self.assertTrue((output_dir / artifacts.NODE_CROP_TEMPLATE.format(node_id="1")).exists())
        self.assertTrue((output_dir / artifacts.NODE_ROWS_TEMPLATE.format(node_id="ROOT")).exists())
        self.assertTrue((output_dir / artifacts.NODE_ENTRIES_TEMPLATE.format(node_id="1")).exists())
        self.assertTrue((output_dir / artifacts.NODE_ROWS_OVERLAY_TEMPLATE.format(node_id="ROOT")).exists())
        self.assertTrue((output_dir / artifacts.NODE_LINE_CROP_TEMPLATE.format(node_id="ROOT", row_order=1)).exists())
        self.assertTrue((output_dir / artifacts.NODE_LINE_RESPONSE_TEMPLATE.format(node_id="1", row_order=1)).exists())
        self.assertTrue((output_dir / artifacts.BASELINE_FINAL_STRUCTURE_FILENAME).exists())
        self.assertTrue((output_dir / artifacts.FINAL_STRUCTURE_FILENAME).exists())
        self.assertTrue((output_dir / artifacts.STATE_FILENAME).exists())

        final_structure = json.loads((output_dir / artifacts.FINAL_STRUCTURE_FILENAME).read_text(encoding="utf-8"))
        self.assertEqual(
            final_structure,
            [
                {"line": 1, "wffstr": "P", "jstr": "PR"},
                [
                    {"line": 2, "wffstr": "A", "jstr": "PR"},
                    {"line": 3, "wffstr": "B", "jstr": "R 1"},
                ],
                {"line": 4, "wffstr": "Q", "jstr": "R 2"},
            ],
        )
        self.assertEqual(state["system"], "S4")
        self.assertEqual(state["vision_model"], "fake-model")
        self.assertIn("modal_v2 baseline OCR produced 4 visible lines", state["run_summary"])

        root_rows = json.loads((output_dir / artifacts.NODE_ROWS_TEMPLATE.format(node_id="ROOT")).read_text(encoding="utf-8"))
        self.assertEqual([record["kind"] for record in root_rows], ["line", "subproof_token", "line"])

    def test_transcribe_image_to_modal_v2_debug_state_can_write_passive_unary_probe_artifact(self) -> None:
        image_path = self.tmp_dir / "synthetic_unary.png"
        cv2.imwrite(str(image_path), _synthetic_recursive_modal_image())
        ocr_client = FakeImagePromptClient(
            [
                json.dumps({"scope_role": "ordinary_assumption", "formula_text": "A", "justification_text": "", "line_label": "", "confidence": "high"}),
                json.dumps({"scope_role": "derived_line", "formula_text": "B", "justification_text": "R 1", "line_label": "", "confidence": "high"}),
                json.dumps({"scope_role": "premise", "formula_text": "□¬A", "justification_text": "", "line_label": "", "confidence": "high"}),
                json.dumps({"scope_role": "derived_line", "formula_text": "Q", "justification_text": "R 2", "line_label": "", "confidence": "high"}),
            ]
        )
        verifier_client = FakeImagePromptClient(
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

        state = transcribe_image_to_modal_v2_debug_state(
            image_path=image_path,
            output_dir=self.tmp_dir / "out_unary",
            modal_config={"system": "S4", "conclusion": "Q"},
            client=ocr_client,
            verifier_client=verifier_client,
            vision_base_url="http://127.0.0.1:1234/v1",
            vision_model="fake-model",
            verifier_vision_model="fake-8b",
            enable_unary_structured_probe=True,
        )

        output_dir = self.tmp_dir / "out_unary"
        self.assertTrue((output_dir / artifacts.UNARY_STRUCTURED_PROBE_FILENAME).exists())
        probe_records = json.loads((output_dir / artifacts.UNARY_STRUCTURED_PROBE_FILENAME).read_text(encoding="utf-8"))
        self.assertEqual(len(probe_records), 1)
        self.assertEqual(probe_records[0]["baseline_formula"], "□¬A")
        self.assertEqual(probe_records[0]["structured_formula"], "□¬¬A")
        self.assertTrue(probe_records[0]["would_change_formula"])
        self.assertIn("Passive unary structured probe inspected 1 line", state["run_summary"])
        self.assertEqual(state["verifier_vision_model"], "fake-8b")


if __name__ == "__main__":
    unittest.main()
