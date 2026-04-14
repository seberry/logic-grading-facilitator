import shutil
import unittest
from pathlib import Path
from uuid import uuid4

from scripts import debug_fol_model_ocr


class FakeVisionClient:
    def chat_with_image_prompt(self, image_path: Path, prompt: str) -> str:
        return "```text\nDOMAIN: {1}\n\nCONSTANTS:\na: 1\n\nPREDICATES:\nP: {1}\n```"


class DebugFolModelOcrTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = Path("working") / "test_tmp" / f"fol_ocr_test_{uuid4().hex}"
        self.tmp_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_load_autograde_config_accepts_plain_yaml(self) -> None:
        path = self.tmp_dir / "autograde.yaml"
        path.write_text(
            "task_kind: truth_value\nprompt_data:\n  formulas:\n    - \"Pa\"\n",
            encoding="utf-8",
        )

        parsed = debug_fol_model_ocr._load_autograde_config(path)
        self.assertEqual(parsed["task_kind"], "truth_value")
        self.assertEqual(parsed["prompt_data"]["formulas"][0], "Pa")

    def test_load_autograde_config_accepts_state_json(self) -> None:
        path = self.tmp_dir / "state.json"
        path.write_text(
            '{\n'
            '  "problem_setup_text": "task_kind: non_equivalence\\nprompt_data:\\n  formulas:\\n    - \\"∀x Px\\"\\n    - \\"∃x Px\\"\\n"\n'
            '}',
            encoding="utf-8",
        )

        parsed = debug_fol_model_ocr._load_autograde_config(path)
        self.assertEqual(parsed["task_kind"], "non_equivalence")
        self.assertEqual(len(parsed["prompt_data"]["formulas"]), 2)

    def test_transcribe_image_to_canonical_text_writes_artifacts(self) -> None:
        image_path = self.tmp_dir / "sample.png"
        image_path.write_bytes(b"fake")
        output_dir = self.tmp_dir / "out"

        raw, cleaned = debug_fol_model_ocr.transcribe_image_to_canonical_text(
            image_path=image_path,
            output_dir=output_dir,
            autograde_config={"task_kind": "truth_value", "prompt_data": {"formulas": ["Pa"]}},
            client=FakeVisionClient(),
        )

        self.assertIn("DOMAIN: {1}", raw)
        self.assertEqual(cleaned, "DOMAIN: {1}\n\nCONSTANTS:\na: 1\n\nPREDICATES:\nP: {1}\n")
        self.assertTrue((output_dir / "ocr_prompt.txt").exists())
        self.assertTrue((output_dir / "ocr_raw_response.txt").exists())
        self.assertTrue((output_dir / "ocr_model_text.txt").exists())


if __name__ == "__main__":
    unittest.main()
