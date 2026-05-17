import json
import shutil
import unittest
from pathlib import Path
from uuid import uuid4

from scripts.generate_synthetic_logic_exam import generate
from scripts.run_synthetic_fol_ocr_eval import run_evaluation


class SyntheticFolOcrEvalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = Path("working") / "test_tmp" / f"synthetic_fol_ocr_eval_{uuid4().hex}"
        self.fixture_dir = self.tmp_dir / "fixture"
        self.output_dir = self.tmp_dir / "report"

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_run_evaluation_with_fake_transcriber_writes_report(self) -> None:
        generate(self.fixture_dir, style="printed", count=2, seed=22, force=True)

        def fake_transcribe(*, image_path: Path, output_dir: Path, autograde_config: dict):
            model_text = (output_dir / "gold_model_text.txt").read_text(encoding="utf-8")
            return model_text, model_text + "\n"

        results = run_evaluation(
            fixture_root=self.fixture_dir,
            output_dir=self.output_dir,
            limit=3,
            transcribe_fn=fake_transcribe,
        )

        self.assertEqual(len(results), 3)
        self.assertTrue(all(item.ocr_status == "ok" for item in results))
        self.assertTrue(all(item.checker_status == "scored" for item in results))
        self.assertTrue(all(item.verdict_match for item in results))
        self.assertTrue((self.output_dir / "summary.json").exists())
        self.assertTrue((self.output_dir / "summary.html").exists())

        summary = json.loads((self.output_dir / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(len(summary), 3)
        self.assertIn("ocr_model_text", summary[0])

    def test_run_evaluation_reports_verdict_mismatch(self) -> None:
        generate(self.fixture_dir, style="printed", count=1, seed=22, force=True)

        def fake_transcribe(*, image_path: Path, output_dir: Path, autograde_config: dict):
            model_text = "Domain: 1\nP: 1\nQ: 1\n"
            return model_text, model_text

        results = run_evaluation(
            fixture_root=self.fixture_dir,
            output_dir=self.output_dir,
            problem_ids=["Q1"],
            transcribe_fn=fake_transcribe,
        )

        self.assertEqual(len(results), 1)
        self.assertFalse(results[0].verdict_match)


if __name__ == "__main__":
    unittest.main()
