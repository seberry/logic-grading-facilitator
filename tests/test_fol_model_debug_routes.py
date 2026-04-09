import json
import shutil
import unittest
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from fastapi.testclient import TestClient

from app import server
from app.models import GradingKey, ProblemSpec, RosterEntry, Session
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
                    "problem_setup_text": (
                        "task_kind: truth_value\n"
                        "prompt_data:\n"
                        "  formulas:\n"
                        "    - \"Pa\"\n"
                        "    - \"∀x (Px → Qx)\"\n"
                    ),
                    "model_text": "DOMAIN: {1,2}\n\nCONSTANTS:\na: 1\n\nPREDICATES:\nP: {1}\nQ: {2}\n",
                    "sentences_text": "1. Pa\n2. ∀x (Px → Qx)",
                    "draft_comment": "",
                    "draft_grade": "",
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertIn("Checker ran successfully through the shared autograde layer.", response.text)
        self.assertIn("Sentence 2: ∀x (Px → Qx)", response.text)
        self.assertIn("counterexample", response.text)

    def test_non_equivalence_preset_renders_runnable_problem(self) -> None:
        with patch.object(fol_model_debug_routes, "DEBUG_FOL_MODEL_DIR", self.debug_dir):
            response = self.client.get("/debug/fol-model?problem_type=non_equivalence")

        self.assertEqual(response.status_code, 200)
        self.assertIn("task_kind: non_equivalence", response.text)
        self.assertIn("Show non-equivalence", response.text)
        self.assertIn("∀x Px", response.text)

    def test_non_equivalence_runs_live_task_wrapper(self) -> None:
        with patch.object(fol_model_debug_routes, "DEBUG_FOL_MODEL_DIR", self.debug_dir):
            response = self.client.post(
                "/debug/fol-model/run_checker",
                data={
                    "scan": "sample_scan",
                    "problem_type": "non_equivalence",
                    "problem_setup_text": (
                        "task_kind: non_equivalence\n"
                        "prompt_data:\n"
                        "  formulas:\n"
                        "    - \"∀x Px\"\n"
                        "    - \"∃x Px\"\n"
                    ),
                    "model_text": "DOMAIN: {1,2}\n\nCONSTANTS:\n\nPREDICATES:\nP: {1}\n",
                    "sentences_text": "",
                    "draft_comment": "",
                    "draft_grade": "",
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertIn("Checker ran successfully through the shared autograde layer.", response.text)
        self.assertIn("Task summary: non-equivalence", response.text)
        self.assertIn("not logically equivalent", response.text)

    def test_rerun_ocr_loads_proposed_model_text(self) -> None:
        ocr_text_path = self.debug_dir / "sample_scan" / "ocr_model_text.txt"
        raw_response_path = self.debug_dir / "sample_scan" / "ocr_raw_response.txt"
        ocr_text_path.write_text("DOMAIN: {1}\n\nCONSTANTS:\na: 1\n\nPREDICATES:\nP: {1}\n", encoding="utf-8")
        raw_response_path.write_text("DOMAIN: {1}\n...", encoding="utf-8")

        async def fake_run(command):
            return 0, "mock ocr log"

        fake_image = self.debug_dir / "sample_scan" / "sample_scan.png"
        fake_image.write_bytes(b"fake-image")

        with (
            patch.object(fol_model_debug_routes, "DEBUG_FOL_MODEL_DIR", self.debug_dir),
            patch.object(fol_model_debug_routes, "RAW_SCANS_DIR", self.debug_dir / "raw_scans"),
            patch.object(fol_model_debug_routes, "_run_subprocess_with_logs", side_effect=fake_run),
        ):
            response = self.client.post(
                "/debug/fol-model/rerun",
                data={
                    "scan": "sample_scan",
                    "problem_type": "truth_value",
                    "trace_mode": "terse",
                    "problem_setup_text": (
                        "task_kind: truth_value\n"
                        "prompt_data:\n"
                        "  formulas:\n"
                        "    - \"Pa\"\n"
                    ),
                    "model_text": "",
                    "sentences_text": "1. Pa",
                    "draft_comment": "",
                    "draft_grade": "",
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertIn("OCR reran successfully.", response.text)
        self.assertIn("DOMAIN: {1}", response.text)
        self.assertIn("mock ocr log", response.text)

    def test_joint_consistency_runs_live_task_wrapper(self) -> None:
        with patch.object(fol_model_debug_routes, "DEBUG_FOL_MODEL_DIR", self.debug_dir):
            response = self.client.post(
                "/debug/fol-model/run_checker",
                data={
                    "scan": "sample_scan",
                    "problem_id": "",
                    "problem_type": "joint_consistency",
                    "problem_setup_text": (
                        "task_kind: joint_consistency\n"
                        "prompt_data:\n"
                        "  formulas:\n"
                        "    - \"∃x Px\"\n"
                        "    - \"∃x Qx\"\n"
                    ),
                    "model_text": "DOMAIN: {1,2}\n\nCONSTANTS:\n\nPREDICATES:\nP: {1}\nQ: {2}\n",
                    "sentences_text": "",
                    "draft_comment": "",
                    "draft_grade": "",
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertIn("Checker ran successfully through the shared autograde layer.", response.text)
        self.assertIn("Task summary: joint consistency", response.text)
        self.assertIn("jointly consistent", response.text)

    def test_debug_page_can_load_active_rubric_problem(self) -> None:
        session = Session(
            roster=[RosterEntry(student_id="s1", name="Berry, Sharon")],
            grading_key=GradingKey(
                quiz_id="quiz1",
                max_points=5,
                problems=[
                    ProblemSpec(
                        id="Q1",
                        label="Countermodel question",
                        points=5,
                        problem_type="fol_model",
                        autograde={
                            "task_kind": "non_validity",
                            "prompt_data": {
                                "premises": ["Pa"],
                                "conclusion": "Qa",
                            },
                        },
                    )
                ],
            ),
            work_units=[],
            unclaimed_pages=[],
        )

        with (
            patch.object(fol_model_debug_routes, "DEBUG_FOL_MODEL_DIR", self.debug_dir),
            patch.object(fol_model_debug_routes, "get_session", return_value=session),
        ):
            response = self.client.get("/debug/fol-model?problem_id=Q1")

        self.assertEqual(response.status_code, 200)
        self.assertIn("Loaded rubric config for Q1.", response.text)
        self.assertIn("task_kind: non_validity", response.text)
        self.assertIn("Countermodel question", response.text)


if __name__ == "__main__":
    unittest.main()
