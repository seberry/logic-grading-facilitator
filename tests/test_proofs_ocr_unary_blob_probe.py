import shutil
import unittest
from pathlib import Path
from uuid import uuid4

from PIL import Image, ImageDraw

from proofs_ocr.regression import unary_blob_probe, unary_probe_suite


def _render_line_crop(
    *,
    formula_text: str,
    output_path: Path,
    justification_text: str = "",
    line_label: str = "1",
) -> Path:
    font = unary_probe_suite.load_font()
    image = Image.new("RGB", (500, 80), "white")
    draw = ImageDraw.Draw(image)
    draw.text((8, 18), line_label, fill="black", font=font)
    draw.text((45, 18), formula_text, fill="black", font=font)
    if justification_text:
        draw.text((300, 18), justification_text, fill="black", font=font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return output_path


class UnaryBlobProbeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = Path("working") / "test_tmp" / f"unary_blob_probe_{uuid4().hex}"
        self.tmp_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_extract_formula_window_uses_fractional_middle_slice(self) -> None:
        image_path = _render_line_crop(
            formula_text="□¬¬A",
            output_path=self.tmp_dir / "line.png",
            justification_text="□I 5-7",
        )
        import cv2

        line_crop_img = cv2.imread(str(image_path))
        window, meta = unary_blob_probe.extract_formula_window(line_crop_img, left_fraction=0.09, right_fraction=0.36)

        self.assertEqual(meta["x1"], 45)
        self.assertEqual(meta["x2"], 180)
        self.assertEqual(window.shape[1], 135)

    def test_run_blob_probe_flags_missing_inner_box_negation(self) -> None:
        image_path = _render_line_crop(
            formula_text="□¬¬A",
            output_path=self.tmp_dir / "box_double.png",
            justification_text="□I 5-7",
        )

        payload = unary_blob_probe.run_blob_probe(
            image_path=image_path,
            current_ocr_formula="□¬A",
            output_dir=self.tmp_dir / "out_box",
        )

        self.assertTrue(payload["any_blob_flagged"])
        self.assertEqual(payload["ocr_symbol_count"], 3)
        self.assertGreaterEqual(payload["max_component_count"], 4)

    def test_run_blob_probe_does_not_flag_matching_single_negation_control(self) -> None:
        image_path = _render_line_crop(
            formula_text="□¬A",
            output_path=self.tmp_dir / "box_single.png",
            justification_text="□I 4-5",
        )

        payload = unary_blob_probe.run_blob_probe(
            image_path=image_path,
            current_ocr_formula="□¬A",
            output_dir=self.tmp_dir / "out_control",
        )

        self.assertFalse(payload["any_blob_flagged"])
        self.assertEqual(payload["ocr_symbol_count"], 3)


if __name__ == "__main__":
    unittest.main()
