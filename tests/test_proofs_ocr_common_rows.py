import unittest

import cv2
import numpy as np

from proofs_ocr.common import rows
from proofs_ocr.common.types import SubproofNode


class ProofsOcrCommonRowsTests(unittest.TestCase):
    def test_merge_fragment_bands_absorbs_tiny_nearby_fragment(self) -> None:
        merged = rows.merge_fragment_bands(
            [(7, 29), (42, 67), (84, 94), (151, 167), (235, 261)]
        )

        self.assertEqual(merged, [(7, 29), (42, 94), (151, 167), (235, 261)])

    def test_segment_node_rows_builds_line_and_token_records(self) -> None:
        crop_img = np.full((220, 180, 3), 255, dtype=np.uint8)
        parent = SubproofNode("1", (0, 0, 180, 220))
        child = SubproofNode("2", (50, 85, 40, 60))
        child.parent = parent
        parent.children.append(child)

        writes = []
        band_writes = []

        row_records = rows.segment_node_rows(
            node=parent,
            node_crop_img=crop_img,
            suppress_left_scope_bar=lambda img: img,
            detect_text_row_bands_fn=lambda _img: [(20, 34), (108, 120)],
            band_has_textlike_ink_fn=lambda _img, _band: True,
            merge_fragment_bands_fn=lambda bands: bands,
            filter_bands_near_child_tokens_fn=lambda **_kwargs: [(20, 34)],
            token_rect_for_child_fn=lambda **_kwargs: (70, 146, 120, 171),
            write_row_artifacts=lambda records: writes.append(records),
            write_band_debug_artifacts=lambda payload: band_writes.append(payload),
        )

        self.assertEqual(len(writes), 1)
        self.assertEqual(len(band_writes), 1)
        self.assertEqual(
            band_writes[0]["raw_detected_bands"],
            [
                {
                    "y1": 20,
                    "y2": 34,
                    "height": 15,
                    "kept_after_textlike_filter": True,
                    "rejection_reason": "",
                },
                {
                    "y1": 108,
                    "y2": 120,
                    "height": 13,
                    "kept_after_textlike_filter": True,
                    "rejection_reason": "",
                },
            ],
        )
        self.assertEqual(
            band_writes[0]["rejected_after_merge"],
            [
                {
                    "y1": 108,
                    "y2": 120,
                    "height": 13,
                    "rejection_reason": "child_token_overlap",
                }
            ],
        )
        self.assertEqual(
            row_records,
            [
                {"kind": "line", "band_index": 0, "y1": 20, "y2": 34, "center_y": 27.0, "row_order": 1},
                {"kind": "subproof_token", "child_id": "2", "y1": 146, "y2": 171, "center_y": 158.5, "row_order": 2},
            ],
        )

    def test_detect_text_row_bands_recovers_compact_top_left_component(self) -> None:
        crop_img = np.full((140, 260, 3), 255, dtype=np.uint8)
        cv2.rectangle(crop_img, (36, 18), (56, 38), (0, 0, 0), 2)
        cv2.line(crop_img, (20, 68), (88, 68), (0, 0, 0), 2)
        cv2.putText(crop_img, "A", (30, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)

        bands = rows.detect_text_row_bands(crop_img)

        self.assertTrue(any(start <= 22 and end >= 34 for start, end in bands))


if __name__ == "__main__":
    unittest.main()
