import json
import shutil
import unittest
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np

from proofs_ocr.regression.modal_zone_review import (
    _parse_selected_row_ids,
    collect_modal_zone_review_records,
    filter_modal_zone_review_records,
    write_modal_zone_review_report,
)


class ModalZoneReviewTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = Path("working") / "test_tmp" / f"modal_zone_review_{uuid4().hex}"
        self.tmp_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_collect_modal_zone_review_records_flags_blank_justification_with_ink(self) -> None:
        image = np.full((40, 180, 3), 255, dtype=np.uint8)
        cv2.putText(image, "A", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        just = np.full((40, 120, 3), 255, dtype=np.uint8)
        cv2.putText(just, "Assum", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.imwrite(str(self.tmp_dir / "node_line_ROOT_1.jpg"), image)
        cv2.imwrite(str(self.tmp_dir / "node_line_formula_ROOT_1.jpg"), image)
        cv2.imwrite(str(self.tmp_dir / "node_line_justification_ROOT_1.jpg"), just)
        payload = {
            "node_id": "ROOT",
            "row_order": 1,
            "mode": "formula_justification_separation",
            "formula_crop_filename": "node_line_formula_ROOT_1.jpg",
            "justification_crop_filename": "node_line_justification_ROOT_1.jpg",
            "formula_resolved": {"formula_text": "A", "confidence": "high"},
            "justification_resolved": {"justification_text": "", "confidence": "low"},
            "resolved": {"formula_text": "A", "justification_text": "", "scope_role": "premise"},
            "formula_raw_response": '{"formula_text":"A","confidence":"high"}',
            "justification_raw_response": '{"justification_text":"","confidence":"low"}',
        }
        (self.tmp_dir / "node_line_ROOT_1.json").write_text(json.dumps(payload), encoding="utf-8")

        records = collect_modal_zone_review_records(output_dir=self.tmp_dir, case_id="demo_case")

        self.assertEqual(len(records), 1)
        self.assertIn("blank_justification_with_ink", records[0].suspicious_reasons)

    def test_write_modal_zone_review_report_includes_images_and_raw_responses(self) -> None:
        image = np.full((40, 180, 3), 255, dtype=np.uint8)
        cv2.putText(image, "B", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.imwrite(str(self.tmp_dir / "node_line_ROOT_2.jpg"), image)
        payload = {
            "node_id": "ROOT",
            "row_order": 2,
            "mode": "formula_justification_separation",
            "formula_crop_filename": "",
            "justification_crop_filename": "",
            "formula_resolved": {"formula_text": "B", "confidence": "medium"},
            "justification_resolved": {"justification_text": "PR", "confidence": "medium"},
            "resolved": {"formula_text": "B", "justification_text": "PR", "scope_role": "premise"},
            "formula_raw_response": '{"formula_text":"B"}',
            "justification_raw_response": '{"justification_text":"PR"}',
        }
        (self.tmp_dir / "node_line_ROOT_2.json").write_text(json.dumps(payload), encoding="utf-8")

        report_path = self.tmp_dir / "review.html"
        write_modal_zone_review_report(
            cases=[("demo_case", self.tmp_dir)],
            report_path=report_path,
            title="Demo Review",
        )

        report_text = report_path.read_text(encoding="utf-8")
        self.assertIn("Demo Review", report_text)
        self.assertIn("node_line_ROOT_2.jpg", report_text)
        self.assertIn("{&quot;formula_text&quot;:&quot;B&quot;}", report_text)
        self.assertIn("resolved justification", report_text)

    def test_filter_modal_zone_review_records_can_keep_only_suspicious_rows(self) -> None:
        blank_image = np.full((40, 180, 3), 255, dtype=np.uint8)
        just = np.full((40, 120, 3), 255, dtype=np.uint8)
        cv2.putText(just, "Assum", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.imwrite(str(self.tmp_dir / "node_line_ROOT_1.jpg"), blank_image)
        cv2.imwrite(str(self.tmp_dir / "node_line_formula_ROOT_1.jpg"), blank_image)
        cv2.imwrite(str(self.tmp_dir / "node_line_justification_ROOT_1.jpg"), just)
        payload_1 = {
            "node_id": "ROOT",
            "row_order": 1,
            "mode": "formula_justification_separation",
            "formula_crop_filename": "node_line_formula_ROOT_1.jpg",
            "justification_crop_filename": "node_line_justification_ROOT_1.jpg",
            "formula_resolved": {"formula_text": "A", "confidence": "high"},
            "justification_resolved": {"justification_text": "", "confidence": "low"},
            "resolved": {"formula_text": "A", "justification_text": "", "scope_role": "premise"},
        }
        (self.tmp_dir / "node_line_ROOT_1.json").write_text(json.dumps(payload_1), encoding="utf-8")

        ok_image = np.full((40, 180, 3), 255, dtype=np.uint8)
        cv2.putText(ok_image, "B", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.imwrite(str(self.tmp_dir / "node_line_ROOT_2.jpg"), ok_image)
        payload_2 = {
            "node_id": "ROOT",
            "row_order": 2,
            "mode": "formula_justification_separation",
            "formula_crop_filename": "",
            "justification_crop_filename": "",
            "formula_resolved": {"formula_text": "B", "confidence": "high"},
            "justification_resolved": {"justification_text": "PR", "confidence": "high"},
            "resolved": {"formula_text": "B", "justification_text": "PR", "scope_role": "premise"},
        }
        (self.tmp_dir / "node_line_ROOT_2.json").write_text(json.dumps(payload_2), encoding="utf-8")

        records = collect_modal_zone_review_records(output_dir=self.tmp_dir, case_id="demo_case")
        suspicious = filter_modal_zone_review_records(records, suspicious_only=True)

        self.assertEqual(len(records), 2)
        self.assertEqual(len(suspicious), 1)
        self.assertEqual(suspicious[0].row_order, 1)

    def test_filter_modal_zone_review_records_can_select_specific_rows(self) -> None:
        image = np.full((40, 180, 3), 255, dtype=np.uint8)
        cv2.imwrite(str(self.tmp_dir / "node_line_ROOT_1.jpg"), image)
        cv2.imwrite(str(self.tmp_dir / "node_line_ROOT_2.jpg"), image)
        for row_order in (1, 2):
            payload = {
                "node_id": "ROOT",
                "row_order": row_order,
                "mode": "formula_justification_separation",
                "formula_crop_filename": "",
                "justification_crop_filename": "",
                "formula_resolved": {"formula_text": f"L{row_order}", "confidence": "high"},
                "justification_resolved": {"justification_text": "", "confidence": "high"},
                "resolved": {"formula_text": f"L{row_order}", "justification_text": "", "scope_role": "premise"},
            }
            (self.tmp_dir / f"node_line_ROOT_{row_order}.json").write_text(json.dumps(payload), encoding="utf-8")

        records = collect_modal_zone_review_records(output_dir=self.tmp_dir, case_id="demo_case")
        selected = filter_modal_zone_review_records(
            records,
            selected_row_ids={("demo_case", "ROOT", 2)},
        )

        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0].row_order, 2)

    def test_parse_selected_row_ids(self) -> None:
        parsed = _parse_selected_row_ids(["case_a:ROOT:2", "case_b:1:4"])

        self.assertEqual(parsed, {("case_a", "ROOT", 2), ("case_b", "1", 4)})


if __name__ == "__main__":
    unittest.main()
