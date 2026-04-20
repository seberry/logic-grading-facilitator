import unittest
from unittest.mock import patch

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

    def test_detect_text_row_bands_uses_justification_side_to_split_handwritten_child_rows(self) -> None:
        crop_img = np.full((138, 483, 3), 255, dtype=np.uint8)
        cv2.line(crop_img, (6, 0), (6, 137), (0, 0, 0), 2)
        cv2.rectangle(crop_img, (18, 10), (36, 28), (0, 0, 0), 2)
        cv2.line(crop_img, (18, 36), (120, 36), (0, 0, 0), 2)
        cv2.putText(crop_img, "A", (22, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
        cv2.putText(crop_img, "~~A", (12, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)

        cv2.putText(crop_img, "Assum", (232, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.putText(crop_img, "[]E 2", (228, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.putText(crop_img, "DN 6", (236, 106), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        bands = rows.detect_text_row_bands(crop_img)

        self.assertGreaterEqual(len(bands), 3)
        centers = [((start + end) / 2.0) for start, end in bands]
        self.assertTrue(any(center < 35 for center in centers))
        self.assertTrue(any(45 <= center <= 80 for center in centers))
        self.assertTrue(any(center > 85 for center in centers))

    def test_detect_text_row_bands_can_rescue_sparse_lower_half_compact_symbol_when_enabled(self) -> None:
        crop_img = np.full((180, 260, 3), 255, dtype=np.uint8)
        cv2.line(crop_img, (28, 126), (31, 126), (0, 0, 0), 1)
        cv2.line(crop_img, (35, 126), (38, 126), (0, 0, 0), 1)
        cv2.line(crop_img, (33, 121), (33, 126), (0, 0, 0), 1)

        default_bands = rows.detect_text_row_bands(crop_img)
        rescued_bands = rows.detect_text_row_bands(
            crop_img,
            enable_sparse_lower_compact_band_recovery=True,
        )

        self.assertEqual(default_bands, [])
        self.assertEqual(len(rescued_bands), 1)
        self.assertLessEqual(rescued_bands[0][0], 121)
        self.assertGreaterEqual(rescued_bands[0][1], 126)

    def test_recover_split_post_child_bands_from_right_window_refines_single_post_child_band(self) -> None:
        crop_img = np.full((340, 480, 3), 255, dtype=np.uint8)
        parent = SubproofNode("1", (0, 0, 480, 340))
        child = SubproofNode("2", (60, 205, 40, 30))
        child.parent = parent
        parent.children.append(child)

        with patch.object(rows, "_projection_bands_from_dark_mask", return_value=[(292, 312), (317, 338)]):
            refined = rows._recover_split_post_child_bands_from_right_window(
                bands=[(12, 29), (40, 51), (76, 103), (121, 136), (292, 314)],
                node=parent,
                node_crop_img=crop_img,
                token_rect_for_child_fn=lambda **_kwargs: (70, 204, 130, 234),
            )

        self.assertEqual(refined[-2:], [(292, 312), (317, 338)])

    def test_split_first_band_on_opener_underline_splits_merged_opener_band(self) -> None:
        gray = np.full((138, 483), 255, dtype=np.uint8)
        cv2.line(gray, (6, 0), (6, 137), 0, 2)
        cv2.rectangle(gray, (18, 10), (36, 28), 0, 2)
        cv2.line(gray, (18, 37), (120, 37), 0, 2)
        cv2.putText(gray, "A", (22, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 0, 2)
        cv2.putText(gray, "Assum", (232, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 0, 2)
        cv2.putText(gray, "[]E 2", (228, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 0, 2)

        split = rows._split_first_band_on_opener_underline(gray, [(22, 58), (77, 88)])

        self.assertEqual(split, [(22, 34), (43, 58), (77, 88)])


if __name__ == "__main__":
    unittest.main()
