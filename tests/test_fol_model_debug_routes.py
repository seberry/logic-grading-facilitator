import json
import shutil
import unittest
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from fastapi.testclient import TestClient

from app import server
from app.routes import fol_model_debug as fol_model_debug_routes


class FolModelDebugRoutesTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(server.app)
        self.debug_dir = Path("working") / "test_tmp" / f"fol_model_{uuid4().hex}"
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        (self.debug_dir / "sample_scan").mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.debug_dir, ignore_errors=True)

    def test_fol_model_debug_page_renders(self) -> None:
        with patch.object(fol_model_debug_routes, "DEBUG_FOL_MODEL_DIR", self.debug_dir):
            response = self.client.get("/debug/fol-model")

        self.assertEqual(response.status_code, 200)
        self.assertIn("FOL Model Debugger", response.text)
        self.assertIn("DOMAIN: {1,2,3}", response.text)
        self.assertIn("Target sentences", response.text)
        self.assertIn("Run checker", response.text)
        self.assertIn("Save transcription", response.text)
        self.assertIn("Problem type", response.text)

    def test_save_fol_model_transcription_persists_state(self) -> None:
        with patch.object(fol_model_debug_routes, "DEBUG_FOL_MODEL_DIR", self.debug_dir):
            response = self.client.post(
                "/debug/fol-model/save",
                data={
                    "scan": "sample_scan",
                    "problem_type": "truth_value",
                    "problem_setup_text": "TYPE: truth_value",
                    "model_text": "DOMAIN: {1}\n\nCONSTANTS:\na: 1\n\nPREDICATES:\nP: {1}\n",
                    "sentences_text": "1. Pa",
                    "draft_comment": "saved comment",
                    "draft_grade": "2",
                },
                follow_redirects=False,
            )

            state_path = self.debug_dir / "sample_scan" / "state.json"
            saved = json.loads(state_path.read_text(encoding="utf-8"))

        self.assertEqual(response.status_code, 303)
        self.assertIn("/debug/fol-model?scan=sample_scan", response.headers["location"])
        self.assertEqual(saved["draft_comment"], "saved comment")
        self.assertEqual(saved["draft_grade"], "2")
        self.assertEqual(saved["problem_type"], "truth_value")

    def test_run_checker_renders_live_trace(self) -> None:
        with patch.object(fol_model_debug_routes, "DEBUG_FOL_MODEL_DIR", self.debug_dir):
            response = self.client.post(
                "/debug/fol-model/run_checker",
                data={
                    "scan": "sample_scan",
                    "problem_type": "truth_value",
                    "problem_setup_text": "TYPE: truth_value",
                    "model_text": "DOMAIN: {1,2}\n\nCONSTANTS:\na: 1\n\nPREDICATES:\nP: {1}\nQ: {2}\n",
                    "sentences_text": "1. Pa\n2. ∀x (Px → Qx)",
                    "draft_comment": "",
                    "draft_grade": "",
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertIn("Checker ran successfully.", response.text)
        self.assertIn("Sentence 2: ∀x (Px → Qx)", response.text)
        self.assertIn("counterexample", response.text)

    def test_non_equivalence_preview_renders_dummy_problem(self) -> None:
        with patch.object(fol_model_debug_routes, "DEBUG_FOL_MODEL_DIR", self.debug_dir):
            response = self.client.get("/debug/fol-model?problem_type=non_equivalence")

        self.assertEqual(response.status_code, 200)
        self.assertIn("TYPE: non_equivalence", response.text)
        self.assertIn("Show non-equivalence", response.text)
        self.assertIn("Dummy non-equivalence trace", response.text)


if __name__ == "__main__":
    unittest.main()
