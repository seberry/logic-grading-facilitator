import shutil
import unittest
from pathlib import Path
from uuid import uuid4

from PIL import Image, ImageDraw

from proofs_ocr.regression import line_char_count_probe, unary_probe_suite


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
    if line_label:
        draw.text((8, 18), line_label, fill="black", font=font)
    if formula_text:
        draw.text((45, 18), formula_text, fill="black", font=font)
    if justification_text:
        draw.text((300, 18), justification_text, fill="black", font=font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return output_path


class LineCharCountProbeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = Path("working") / "test_tmp" / f"line_char_count_probe_{uuid4().hex}"
        self.tmp_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_count_visible_char_slots_ignores_spaces(self) -> None:
        self.assertEqual(line_char_count_probe.count_visible_char_slots(""), 0)
        self.assertEqual(line_char_count_probe.count_visible_char_slots("□P"), 2)
        self.assertEqual(line_char_count_probe.count_visible_char_slots("□I 3-6"), 5)

    def test_probe_flags_formula_excess_when_current_ocr_is_missing_characters(self) -> None:
        image_path = _render_line_crop(
            formula_text="□P",
            output_path=self.tmp_dir / "line_formula_missing.png",
            justification_text="PR",
            line_label="",
        )

        payload = line_char_count_probe.run_line_char_count_probe(
            image_path=image_path,
            output_dir=self.tmp_dir / "out_formula_missing",
            line_label="",
            formula_text="□",
            justification_text="PR",
        )

        self.assertTrue(payload["zones"]["formula"]["any_excess_flagged"])
        self.assertGreaterEqual(payload["zones"]["formula"]["max_component_count"], 2)

    def test_probe_flags_justification_excess_when_current_ocr_is_missing_rule_head(self) -> None:
        image_path = _render_line_crop(
            formula_text="P→Q",
            output_path=self.tmp_dir / "line_just_missing.png",
            justification_text="□E 2",
            line_label="",
        )

        payload = line_char_count_probe.run_line_char_count_probe(
            image_path=image_path,
            output_dir=self.tmp_dir / "out_just_missing",
            line_label="",
            formula_text="P→Q",
            justification_text="2",
        )

        self.assertTrue(payload["zones"]["justification"]["any_large_excess_flagged"])


if __name__ == "__main__":
    unittest.main()
