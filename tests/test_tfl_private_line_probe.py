import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np


class TestPrivateTflLineProbe(unittest.TestCase):
    def test_estimate_row_split_x_uses_large_foreground_gap(self) -> None:
        from proofs_ocr.regression.private_tfl_line_probe import estimate_row_split_x

        img = np.ones((80, 500, 3), dtype=np.uint8) * 255
        for x in range(40, 170, 24):
            img[25:50, x : x + 10] = 0
        for x in range(310, 430, 24):
            img[25:50, x : x + 10] = 0

        split_x, source = estimate_row_split_x(img, fallback_ratio=0.34)

        self.assertGreater(split_x, 40)
        self.assertLess(split_x, 430)
        self.assertTrue(source.startswith("gap_"))

    def test_probe_summary_redacts_text_and_reports_rule_locations(self) -> None:
        from proofs_ocr.regression import private_tfl_line_probe

        class FakeClient:
            def __init__(self) -> None:
                self.calls = 0
                self.last_response_metadata = {}

            def build_messages_for_prompt(self, row_path, prompt):
                return [{"row_path": str(row_path), "prompt": prompt}]

            def chat(self, messages, **kwargs):
                self.calls += 1
                self.last_response_metadata = {
                    "thinking_chars": 0,
                    "content_chars": 20,
                    "finish_reason": "stop",
                }
                if self.calls == 1:
                    return json.dumps({"line": 1, "wffstr": "P", "jstr": ""})
                if self.calls == 2:
                    return json.dumps({"line": 2, "wffstr": "P ∧E 1", "jstr": "∧E 1"})
                return json.dumps([{"line": 3, "wffstr": "Q", "jstr": "→E 1,2"}])

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            crop_path = root / "crop.jpg"
            img = np.ones((300, 500, 3), dtype=np.uint8) * 255
            img[45:65, 40:420] = 0
            img[145:165, 40:420] = 0
            img[245:265, 40:420] = 0
            self.assertTrue(cv2.imwrite(str(crop_path), img))

            with patch.object(private_tfl_line_probe, "LMStudioQwenClient", FakeClient):
                private_tfl_line_probe.run_probe(
                    crop_path=crop_path,
                    output_dir=root / "probe",
                    expected_lines=3,
                    model=None,
                    timeout=10,
                    max_tokens=80,
                )

            summary = json.loads((root / "probe" / "shareable_summary.json").read_text())
            self.assertFalse(summary["privacy"]["contains_line_text"])
            self.assertEqual(
                [row["rule_head_location"] for row in summary["rows"]],
                ["neither", "both", "jstr"],
            )
            self.assertNotIn("P ∧E 1", json.dumps(summary))

    def test_split_probe_reports_side_rule_shape_without_text(self) -> None:
        from proofs_ocr.regression import private_tfl_line_probe

        class FakeClient:
            def __init__(self) -> None:
                self.calls = 0
                self.last_response_metadata = {}

            def build_messages_for_prompt(self, row_path, prompt):
                return [{"row_path": str(row_path), "prompt": prompt}]

            def chat(self, messages, **kwargs):
                self.calls += 1
                self.last_response_metadata = {
                    "thinking_chars": 0,
                    "content_chars": 20,
                    "finish_reason": "stop",
                }
                if self.calls % 2 == 1:
                    return json.dumps({"wffstr": "P", "jstr": ""})
                return json.dumps({"wffstr": "", "jstr": "∧E 1"})

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            crop_path = root / "crop.jpg"
            img = np.ones((200, 500, 3), dtype=np.uint8) * 255
            img[45:65, 40:420] = 0
            img[145:165, 40:420] = 0
            self.assertTrue(cv2.imwrite(str(crop_path), img))

            with patch.object(private_tfl_line_probe, "LMStudioQwenClient", FakeClient):
                private_tfl_line_probe.run_split_probe(
                    crop_path=crop_path,
                    output_dir=root / "probe",
                    expected_lines=2,
                    model=None,
                    timeout=10,
                    max_tokens=80,
                    split_strategy="auto-gap",
                )

            summary = json.loads((root / "probe" / "shareable_summary.json").read_text())
            self.assertEqual(summary["kind"], "private_tfl_split_line_probe_summary")
            self.assertEqual(summary["split_strategy"], "auto-gap")
            self.assertFalse(summary["privacy"]["contains_line_text"])
            self.assertIn("split_x", summary["rows"][0])
            self.assertIn("split_source", summary["rows"][0])
            self.assertTrue(all(row["right_rule_in_jstr"] for row in summary["rows"]))
            self.assertFalse(any(row["left_rule_leak"] for row in summary["rows"]))
            self.assertNotIn("∧E 1", json.dumps(summary))

    def test_justification_column_probe_reports_slot_shape_without_text(self) -> None:
        from proofs_ocr.regression import private_tfl_line_probe

        class FakeClient:
            last_response_metadata = {"thinking_chars": 0, "content_chars": 20, "finish_reason": "stop"}

            def build_messages_for_prompt(self, row_path, prompt):
                return [{"row_path": str(row_path), "prompt": prompt}]

            def chat(self, messages, **kwargs):
                return json.dumps({"justifications": ["∧E 1", "", "→E 1,2"]})

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            crop_path = root / "crop.jpg"
            img = np.ones((200, 500, 3), dtype=np.uint8) * 255
            img[45:65, 40:420] = 0
            img[145:165, 40:420] = 0
            self.assertTrue(cv2.imwrite(str(crop_path), img))

            with patch.object(private_tfl_line_probe, "LMStudioQwenClient", FakeClient):
                private_tfl_line_probe.run_justification_column_probe(
                    crop_path=crop_path,
                    output_dir=root / "probe",
                    expected_lines=4,
                    model=None,
                    timeout=10,
                    max_tokens=80,
                    split_ratio=0.34,
                    overlap_px=10,
                )

            summary = json.loads((root / "probe" / "shareable_summary.json").read_text())
            self.assertEqual(summary["kind"], "private_tfl_justification_column_probe_summary")
            self.assertFalse(summary["privacy"]["contains_line_text"])
            self.assertEqual([slot["has_rule_head"] for slot in summary["slots"]], [True, False, True])
            self.assertNotIn("→E 1,2", json.dumps(summary))


if __name__ == "__main__":
    unittest.main()
