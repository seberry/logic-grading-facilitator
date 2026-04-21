import json
import shutil
import unittest
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np

from proofs_ocr.common import artifacts
from proofs_ocr.pipelines.modal_v2 import transcribe_image_to_modal_v2_debug_state
from proofs_ocr.pipelines.modal_v2_debug import build_modal_v2_state


def _synthetic_recursive_modal_image() -> np.ndarray:
    image = np.full((220, 220, 3), 255, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.putText(image, "P", (12, 30), font, 0.8, (0, 0, 0), 2)
    cv2.line(image, (42, 54), (42, 150), (0, 0, 0), 2)
    cv2.putText(image, "A", (62, 78), font, 0.8, (0, 0, 0), 2)
    cv2.putText(image, "B", (62, 128), font, 0.8, (0, 0, 0), 2)
    cv2.putText(image, "Q", (12, 188), font, 0.8, (0, 0, 0), 2)

    return image


def _synthetic_false_separator_image() -> np.ndarray:
    image = np.full((280, 260, 3), 255, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.line(image, (18, 0), (18, 262), (0, 0, 0), 8)
    cv2.line(image, (54, 157), (54, 262), (0, 0, 0), 2)
    cv2.putText(image, "P", (72, 28), font, 0.8, (0, 0, 0), 2)
    cv2.putText(image, "Q", (72, 62), font, 0.8, (0, 0, 0), 2)
    cv2.putText(image, "A", (82, 188), font, 0.8, (0, 0, 0), 2)
    cv2.putText(image, "B", (82, 224), font, 0.8, (0, 0, 0), 2)
    cv2.putText(image, "R", (72, 254), font, 0.8, (0, 0, 0), 2)
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
        self.assertEqual(state["line_vision_model"], "fake-model")
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

    def test_transcribe_image_to_modal_v2_debug_state_can_write_formula_justification_zone_artifacts(self) -> None:
        image_path = self.tmp_dir / "synthetic_zones.png"
        cv2.imwrite(str(image_path), _synthetic_recursive_modal_image())
        ocr_client = FakeImagePromptClient(
            [
                json.dumps({"line_label": "", "formula_text": "A", "confidence": "high"}),
                json.dumps({"justification_text": "", "confidence": "high"}),
                json.dumps({"line_label": "", "formula_text": "B", "confidence": "high"}),
                json.dumps({"justification_text": "R 1", "confidence": "high"}),
                json.dumps({"line_label": "", "formula_text": "P", "confidence": "high"}),
                json.dumps({"justification_text": "", "confidence": "high"}),
                json.dumps({"line_label": "", "formula_text": "Q", "confidence": "high"}),
                json.dumps({"justification_text": "R 2", "confidence": "high"}),
            ]
        )

        state = transcribe_image_to_modal_v2_debug_state(
            image_path=image_path,
            output_dir=self.tmp_dir / "out_zones",
            modal_config={"system": "S4", "conclusion": "Q"},
            client=ocr_client,
            vision_base_url="http://127.0.0.1:1234/v1",
            vision_model="fake-model",
            enable_formula_justification_separation=True,
        )

        output_dir = self.tmp_dir / "out_zones"
        self.assertEqual(len(ocr_client.calls), 8)
        self.assertTrue((output_dir / artifacts.NODE_LINE_FORMULA_CROP_TEMPLATE.format(node_id="ROOT", row_order=1)).exists())
        self.assertTrue((output_dir / artifacts.NODE_LINE_JUSTIFICATION_CROP_TEMPLATE.format(node_id="ROOT", row_order=1)).exists())
        payload = json.loads((output_dir / artifacts.NODE_LINE_RESPONSE_TEMPLATE.format(node_id="ROOT", row_order=1)).read_text(encoding="utf-8"))
        self.assertEqual(payload["mode"], "formula_justification_separation")
        self.assertIn("Formula/justification separation was enabled.", state["run_summary"])

    def test_transcribe_image_to_modal_v2_debug_state_can_use_separate_line_vision_model(self) -> None:
        image_path = self.tmp_dir / "synthetic_line_model.png"
        cv2.imwrite(str(image_path), _synthetic_recursive_modal_image())

        built_clients: list[str | None] = []

        def fake_build_vlm_client(*, base_url=None, model=None, timeout=None):
            del base_url, timeout
            built_clients.append(model)
            return FakeImagePromptClient(
                [
                    json.dumps({"scope_role": "ordinary_assumption", "formula_text": "A", "justification_text": "", "line_label": "", "confidence": "high"}),
                    json.dumps({"scope_role": "derived_line", "formula_text": "B", "justification_text": "R 1", "line_label": "", "confidence": "high"}),
                    json.dumps({"scope_role": "premise", "formula_text": "P", "justification_text": "", "line_label": "", "confidence": "high"}),
                    json.dumps({"scope_role": "derived_line", "formula_text": "Q", "justification_text": "R 2", "line_label": "", "confidence": "high"}),
                ]
            )

        from unittest.mock import patch

        with patch("proofs_ocr.pipelines.modal_v2_debug.build_vlm_client", side_effect=fake_build_vlm_client):
            state = transcribe_image_to_modal_v2_debug_state(
                image_path=image_path,
                output_dir=self.tmp_dir / "out_line_model",
                modal_config={"system": "S4", "conclusion": "Q"},
                vision_base_url="http://127.0.0.1:1234/v1",
                vision_model="fake-structure-model",
                line_vision_model="fake-content-model",
            )

        self.assertEqual(built_clients, ["fake-content-model"])
        self.assertEqual(state["vision_model"], "fake-structure-model")
        self.assertEqual(state["line_vision_model"], "fake-content-model")
        self.assertIn("Final line OCR used fake-content-model while structure used fake-structure-model.", state["run_summary"])

    def test_build_modal_v2_state_persists_formula_consensus_flag(self) -> None:
        state = build_modal_v2_state(
            modal_config={"system": "S4", "conclusion": "Q"},
            final_structure=[{"line": 1, "wffstr": "Q", "jstr": "PR"}],
            run_summary="Formula-only consensus reread was enabled.",
            vision_base_url="http://127.0.0.1:1234/v1",
            vision_model="fake-model",
            verifier_vision_model="fake-8b",
            enable_formula_consensus_reread=True,
            source_image_name="source_image.png",
        )

        self.assertTrue(state["enable_formula_consensus_reread"])
        self.assertIn("Formula-only consensus reread was enabled.", state["run_summary"])

    def test_transcribe_image_to_modal_v2_debug_state_can_write_passive_unary_formula_crop_probe_artifact(self) -> None:
        image_path = self.tmp_dir / "synthetic_formula_probe.png"
        cv2.imwrite(str(image_path), _synthetic_recursive_modal_image())
        ocr_client = FakeImagePromptClient(
            [
                json.dumps({"line_label": "", "formula_text": "A", "confidence": "high"}),
                json.dumps({"justification_text": "", "confidence": "high"}),
                json.dumps({"line_label": "", "formula_text": "B", "confidence": "high"}),
                json.dumps({"justification_text": "R 1", "confidence": "high"}),
                json.dumps({"line_label": "", "formula_text": "□¬A", "confidence": "high"}),
                json.dumps({"justification_text": "", "confidence": "high"}),
                json.dumps({"line_label": "", "formula_text": "Q", "confidence": "high"}),
                json.dumps({"justification_text": "R 2", "confidence": "high"}),
            ]
        )
        verifier_client = FakeImagePromptClient(
            [
                json.dumps({"justification_text": "", "confidence": "low"}),
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
            output_dir=self.tmp_dir / "out_formula_probe",
            modal_config={"system": "S4", "conclusion": "Q"},
            client=ocr_client,
            verifier_client=verifier_client,
            vision_base_url="http://127.0.0.1:1234/v1",
            vision_model="fake-model",
            verifier_vision_model="fake-8b",
            enable_formula_justification_separation=True,
            enable_unary_structured_formula_crop_probe=True,
        )

        output_dir = self.tmp_dir / "out_formula_probe"
        self.assertTrue((output_dir / artifacts.UNARY_STRUCTURED_FORMULA_PROBE_FILENAME).exists())
        probe_records = json.loads(
            (output_dir / artifacts.UNARY_STRUCTURED_FORMULA_PROBE_FILENAME).read_text(encoding="utf-8")
        )
        self.assertEqual(len(probe_records), 1)
        self.assertEqual(probe_records[0]["baseline_formula"], "□¬A")
        self.assertEqual(probe_records[0]["structured_formula"], "□¬¬A")
        self.assertEqual(probe_records[0]["probe_source_kind"], "formula_crop")
        self.assertEqual(probe_records[0]["probe_source_filename"], "node_line_formula_ROOT_1.jpg")
        self.assertIn("Passive unary structured formula-crop probe inspected 1 line", state["run_summary"])

    def test_transcribe_image_to_modal_v2_debug_state_can_write_passive_unary_blob_artifact(self) -> None:
        image_path = self.tmp_dir / "synthetic_blob.png"
        cv2.imwrite(str(image_path), _synthetic_recursive_modal_image())
        ocr_client = FakeImagePromptClient(
            [
                json.dumps({"scope_role": "ordinary_assumption", "formula_text": "A", "justification_text": "", "line_label": "", "confidence": "high"}),
                json.dumps({"scope_role": "derived_line", "formula_text": "B", "justification_text": "R 1", "line_label": "", "confidence": "high"}),
                json.dumps({"scope_role": "premise", "formula_text": "□¬A", "justification_text": "", "line_label": "", "confidence": "high"}),
                json.dumps({"scope_role": "derived_line", "formula_text": "Q", "justification_text": "R 2", "line_label": "", "confidence": "high"}),
            ]
        )

        state = transcribe_image_to_modal_v2_debug_state(
            image_path=image_path,
            output_dir=self.tmp_dir / "out_blob",
            modal_config={"system": "S4", "conclusion": "Q"},
            client=ocr_client,
            vision_base_url="http://127.0.0.1:1234/v1",
            vision_model="fake-model",
            enable_unary_blob_probe=True,
        )

        output_dir = self.tmp_dir / "out_blob"
        self.assertTrue((output_dir / artifacts.UNARY_BLOB_PROBE_FILENAME).exists())
        probe_records = json.loads((output_dir / artifacts.UNARY_BLOB_PROBE_FILENAME).read_text(encoding="utf-8"))
        self.assertEqual(len(probe_records), 1)
        self.assertEqual(probe_records[0]["baseline_formula"], "□¬A")
        self.assertIn("Passive unary blob probe inspected 1 line", state["run_summary"])

    def test_transcribe_image_to_modal_v2_debug_state_writes_false_bar_verifier_artifacts(self) -> None:
        image_path = self.tmp_dir / "synthetic_false_bar.png"
        cv2.imwrite(str(image_path), _synthetic_false_separator_image())
        ocr_client = FakeImagePromptClient(
            [
                json.dumps({"scope_role": "ordinary_assumption", "formula_text": "A", "justification_text": "", "line_label": "", "confidence": "high"}),
                json.dumps({"scope_role": "derived_line", "formula_text": "B", "justification_text": "R 1", "line_label": "", "confidence": "high"}),
                json.dumps({"scope_role": "premise", "formula_text": "P", "justification_text": "", "line_label": "", "confidence": "high"}),
                json.dumps({"scope_role": "premise", "formula_text": "Q", "justification_text": "", "line_label": "", "confidence": "high"}),
                json.dumps({"scope_role": "derived_line", "formula_text": "R", "justification_text": "R 2", "line_label": "", "confidence": "high"}),
            ]
        )
        verifier_client = FakeImagePromptClient(
            [
                json.dumps(
                    {
                        "classification": "number_formula_separator",
                        "confidence": "high",
                        "reason": "margin separator beside line numbers",
                    }
                )
            ]
        )

        state = transcribe_image_to_modal_v2_debug_state(
            image_path=image_path,
            output_dir=self.tmp_dir / "out_false_bar",
            modal_config={"system": "S4", "conclusion": "R"},
            client=ocr_client,
            verifier_client=verifier_client,
            vision_base_url="http://127.0.0.1:1234/v1",
            vision_model="fake-model",
            verifier_vision_model="fake-8b",
            enable_false_left_separator_filter=True,
        )

        output_dir = self.tmp_dir / "out_false_bar"
        self.assertTrue((output_dir / artifacts.BAR_FILTER_DEBUG_FILENAME).exists())
        self.assertTrue((output_dir / artifacts.FALSE_LEFT_SEPARATOR_VERIFIER_CROP_FILENAME).exists())
        payload = json.loads((output_dir / artifacts.BAR_FILTER_DEBUG_FILENAME).read_text(encoding="utf-8"))
        self.assertEqual(payload["reason"], "verifier_rejected_false_left_separator")
        self.assertEqual(payload["verifier"]["classification"], "number_formula_separator")
        self.assertIn("False-left-separator filtering was enabled and removed one candidate bar.", state["run_summary"])


if __name__ == "__main__":
    unittest.main()
