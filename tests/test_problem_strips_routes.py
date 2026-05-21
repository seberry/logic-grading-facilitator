import unittest
import zipfile
from io import BytesIO
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4
import shutil
import json

from fastapi.testclient import TestClient
from PIL import Image

from app import server
from app import fol_model_review
from app.models import GradingKey, ProblemSpec, RosterEntry, Session, WorkUnit
from app.routes import modal_proof_debug as modal_proof_debug_routes
from app.routes import problem_strips as problem_strips_routes
from app.proof_ocr_review import fitch_ocr_state_path, public_fitch_ocr_status, _summarize_vlm_diagnostics
from app.storage import load_grading_key


def _build_session() -> Session:
    return Session(
        roster=[RosterEntry(student_id="s1", name="Berry, Sharon")],
        grading_key=GradingKey(
            quiz_id="quiz1",
            max_points=10,
            problems=[
                ProblemSpec(
                    id="Q1",
                    label="FOL model",
                    points=5,
                    section="Models",
                    problem_number="1",
                    prompt="Create a model making the target sentence false.",
                    segmentation_hint="Look in the Models section, problem 1.",
                    rubric_notes="Full credit for a model with the required evaluation.",
                ),
                ProblemSpec(id="Q2", label="Proof", points=5),
            ],
        ),
        work_units=[
            WorkUnit(
                unit_id=1,
                student_id="s1",
                page_filenames=["quiz1_berry_sharon_p01.jpg"],
                per_problem={"Q1": None, "Q2": None},
                comment=None,
                per_problem_comments={"Q1": None, "Q2": None},
            )
        ],
        unclaimed_pages=[],
    )


class ProblemStripsRoutesTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(server.app)
        self.tmp_dir = Path("working") / "test_tmp" / f"problem_strips_{uuid4().hex}"
        self.tmp_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_problem_strips_page_renders(self) -> None:
        session = _build_session()
        segments = [
            {
                "unit_id": 1,
                "page_filename": "quiz1_berry_sharon_p01.jpg",
                "page_index": 0,
                "bbox": [0.1, 0.2, 0.8, 0.9],
                "present": True,
                "comment": "found",
                "confidence": 0.9,
            }
        ]

        with (
            patch.object(problem_strips_routes, "get_session", return_value=session),
            patch.object(problem_strips_routes, "ensure_segments_cached", return_value=segments),
        ):
            response = self.client.get("/problem/Q1/strips")

        self.assertEqual(response.status_code, 200)
        self.assertIn("Problem Q1 Grading", response.text)
        self.assertIn("Berry, Sharon", response.text)
        self.assertIn("What the problem asked", response.text)
        self.assertIn("Create a model making the target sentence false.", response.text)
        self.assertNotIn("Autograde target", response.text)

    def test_problem_type_post_updates_session(self) -> None:
        session = _build_session()

        with (
            patch.object(problem_strips_routes, "get_session", return_value=session),
            patch.object(problem_strips_routes, "persist_session"),
        ):
            response = self.client.post(
                "/problem/Q1/type",
                data={"problem_type": "fol_model"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok", "problem_type": "fol_model"})
        self.assertEqual(session.problem_types["Q1"], "fol_model")

    def test_problem_strips_prefers_rubric_problem_type(self) -> None:
        session = _build_session()
        session.grading_key.problems[0].problem_type = "fol_model"
        session.grading_key.problems[0].autograde = {
            "task_kind": "non_equivalence",
            "prompt_data": {"formulas": ["∀x Px", "∃x Px"]},
        }
        session.problem_types["Q1"] = "fitch_proof"
        segments = [
            {
                "unit_id": 1,
                "page_filename": "quiz1_berry_sharon_p01.jpg",
                "page_index": 0,
                "bbox": [0.1, 0.2, 0.8, 0.9],
                "present": True,
                "comment": "found",
                "confidence": 0.9,
            }
        ]

        with (
            patch.object(problem_strips_routes, "get_session", return_value=session),
            patch.object(problem_strips_routes, "ensure_segments_cached", return_value=segments),
        ):
            response = self.client.get("/problem/Q1/strips")

        self.assertEqual(response.status_code, 200)
        self.assertIn("fol model", response.text.lower())
        self.assertIn('data-type="fol_model"', response.text)
        self.assertIn("FOL model checker", response.text)
        self.assertIn("Autograde target", response.text)
        self.assertGreaterEqual(response.text.count("Autograde target"), 2)
        self.assertIn("Need: the two target sentences have different truth values.", response.text)
        self.assertIn("Targets: ∀x Px; ∃x Px", response.text)

    def test_fol_model_check_endpoint_saves_and_runs_checker(self) -> None:
        session = _build_session()
        session.grading_key.problems[0].problem_type = "fol_model"
        session.grading_key.problems[0].autograde = {
            "task_kind": "non_equivalence",
            "default_score_correct": 3,
            "default_score_incorrect": 0,
            "prompt_data": {"formulas": ["∀x Px", "∃x Px"]},
        }
        grades_dir = self.tmp_dir / "grades"

        with (
            patch.object(problem_strips_routes, "get_session", return_value=session),
            patch.object(problem_strips_routes, "GRADES_DIR", grades_dir),
        ):
            response = self.client.post(
                "/problem/Q1/fol_model_ocr/1/check",
                data={"model_text": "Domain: 1,2 P: 1"},
            )

        self.assertEqual(response.status_code, 200)
        item = response.json()["item"]
        self.assertEqual(item["status"], "checker_passed")
        self.assertTrue(item["passed"])
        self.assertEqual(item["suggested_score"], 3.0)
        self.assertIn("not logically equivalent", item["summary"])
        saved = json.loads(fol_model_review.fol_model_state_path(grades_dir, "Q1").read_text(encoding="utf-8"))
        self.assertEqual(saved["items"]["1"]["model_text"], "Domain: 1,2 P: 1")

    def test_start_fol_model_ocr_endpoint_queues_cropped_jobs_without_blocking(self) -> None:
        session = _build_session()
        session.grading_key.problems[0].problem_type = "fol_model"
        session.grading_key.problems[0].autograde = {
            "task_kind": "non_equivalence",
            "prompt_data": {"formulas": ["∀x Px", "∃x Px"]},
        }
        raw_dir = self.tmp_dir / "raw_scans"
        debug_dir = self.tmp_dir / "debug_fol_model"
        grades_dir = self.tmp_dir / "grades"
        raw_dir.mkdir(parents=True, exist_ok=True)
        debug_dir.mkdir(parents=True, exist_ok=True)
        grades_dir.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (100, 200), color="white").save(raw_dir / "quiz1_berry_sharon_p01.jpg")
        segments = [
            {
                "unit_id": 1,
                "page_filename": "quiz1_berry_sharon_p01.jpg",
                "page_index": 0,
                "bbox": [0.1, 0.2, 0.8, 0.9],
                "present": True,
            }
        ]

        with (
            patch.object(problem_strips_routes, "get_session", return_value=session),
            patch.object(problem_strips_routes, "ensure_segments_cached", return_value=segments),
            patch.object(problem_strips_routes, "RAW_SCANS_DIR", raw_dir),
            patch.object(problem_strips_routes, "DEBUG_FOL_MODEL_DIR", debug_dir),
            patch.object(problem_strips_routes, "GRADES_DIR", grades_dir),
            patch("app.fol_model_review.submit_fol_model_ocr_job") as submit_job,
        ):
            response = self.client.post("/problem/Q1/fol_model_ocr/start", data={"force": "0"})

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["queued"], 1)
        submit_job.assert_called_once()

        slug_dir = debug_dir / "q1_1_quiz1_berry_sharon_p01"
        self.assertTrue(slug_dir.exists())
        self.assertTrue(any(path.suffix.lower() == ".jpg" for path in slug_dir.iterdir()))
        self.assertTrue((slug_dir / "state.json").exists())
        state = json.loads(fol_model_review.fol_model_state_path(grades_dir, "Q1").read_text(encoding="utf-8"))
        self.assertEqual(state["items"]["1"]["status"], "queued")

    def test_problem_strips_renders_fitch_action_when_problem_type_is_fitch(self) -> None:
        session = _build_session()
        session.problem_types["Q1"] = "fitch_proof"
        segments = [
            {
                "unit_id": 1,
                "page_filename": "quiz1_berry_sharon_p01.jpg",
                "page_index": 0,
                "bbox": [0.1, 0.2, 0.8, 0.9],
                "present": True,
                "comment": "found",
                "confidence": 0.9,
            }
        ]

        with (
            patch.object(problem_strips_routes, "get_session", return_value=session),
            patch.object(problem_strips_routes, "ensure_segments_cached", return_value=segments),
        ):
            response = self.client.get("/problem/Q1/strips")

        self.assertEqual(response.status_code, 200)
        self.assertIn("OCR Proof", response.text)
        self.assertIn("TFL/Fitch OCR review queue", response.text)
        self.assertIn('data-fitch-ocr-card', response.text)

    def test_fitch_ocr_status_endpoint_returns_existing_review_state(self) -> None:
        session = _build_session()
        session.problem_types["Q1"] = "fitch_proof"
        grades_dir = self.tmp_dir / "grades"
        state_path = fitch_ocr_state_path(grades_dir, "Q1")
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(
            json.dumps(
                {
                    "problem_id": "Q1",
                    "kind": "tfl_fitch",
                    "updated_at": "2026-05-14T12:00:00+00:00",
                    "items": {
                        "1": {
                            "unit_id": 1,
                            "page_filename": "quiz1_berry_sharon_p01.jpg",
                            "slug": "q1_1_quiz1_berry_sharon_p01",
                            "status": "needs_review",
                            "message": "OCR ready for review.",
                            "preview_lines": [
                                {"line": 1, "depth": 0, "wffstr": "A", "jstr": "Pr"}
                            ],
                        }
                    },
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        with (
            patch.object(problem_strips_routes, "get_session", return_value=session),
            patch.object(problem_strips_routes, "GRADES_DIR", grades_dir),
        ):
            response = self.client.get("/problem/Q1/fitch_ocr/status")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["counts"], {"needs_review": 1})
        self.assertEqual(payload["items"][0]["preview_lines"][0]["wffstr"], "A")

    def test_fitch_ocr_status_exposes_diagnostics_summary(self) -> None:
        grades_dir = self.tmp_dir / "grades"
        state_path = fitch_ocr_state_path(grades_dir, "Q1")
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(
            json.dumps(
                {
                    "items": {
                        "1": {
                            "unit_id": 1,
                            "status": "needs_review",
                            "ocr_diagnostics": {"note": "VLM calls 3; elapsed 42.0s"},
                        }
                    }
                }
            ),
            encoding="utf-8",
        )

        payload = public_fitch_ocr_status(grades_dir, "Q1")
        self.assertEqual(payload["items"][0]["ocr_diagnostics"]["note"], "VLM calls 3; elapsed 42.0s")

    def test_vlm_diagnostics_summary_reads_response_metadata(self) -> None:
        output_dir = self.tmp_dir / "debug_fitch" / "case"
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "response_metadata_ROOT.json").write_text(
            json.dumps({"elapsed_s": 1.5, "thinking_chars": 0, "content_chars": 50, "finish_reason": "stop"}),
            encoding="utf-8",
        )
        (output_dir / "response_metadata_2.json").write_text(
            json.dumps({"elapsed_s": 3.25, "thinking_chars": 40, "content_chars": 90, "finish_reason": "length"}),
            encoding="utf-8",
        )
        (output_dir / "thinking_2.txt").write_text("model thoughts", encoding="utf-8")

        summary = _summarize_vlm_diagnostics(output_dir)

        self.assertEqual(summary["call_count"], 2)
        self.assertEqual(summary["total_elapsed_s"], 4.75)
        self.assertEqual(summary["slowest_node"], "2")
        self.assertEqual(summary["thinking_chars"], 40)
        self.assertEqual(summary["thinking_file_count"], 1)
        self.assertEqual(summary["finish_reasons"], {"stop": 1, "length": 1})

    def test_start_fitch_ocr_endpoint_queues_cropped_jobs_without_blocking(self) -> None:
        session = _build_session()
        session.problem_types["Q1"] = "fitch_proof"
        raw_dir = self.tmp_dir / "raw_scans"
        debug_dir = self.tmp_dir / "debug_fitch"
        grades_dir = self.tmp_dir / "grades"
        raw_dir.mkdir(parents=True, exist_ok=True)
        debug_dir.mkdir(parents=True, exist_ok=True)
        grades_dir.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (100, 200), color="white").save(raw_dir / "quiz1_berry_sharon_p01.jpg")
        segments = [
            {
                "unit_id": 1,
                "page_filename": "quiz1_berry_sharon_p01.jpg",
                "page_index": 0,
                "bbox": [0.1, 0.2, 0.8, 0.9],
                "present": True,
            }
        ]

        with (
            patch.object(problem_strips_routes, "get_session", return_value=session),
            patch.object(problem_strips_routes, "ensure_segments_cached", return_value=segments),
            patch.object(problem_strips_routes, "RAW_SCANS_DIR", raw_dir),
            patch.object(problem_strips_routes, "DEBUG_FITCH_DIR", debug_dir),
            patch.object(problem_strips_routes, "GRADES_DIR", grades_dir),
            patch("app.proof_ocr_review.submit_fitch_ocr_job") as submit_job,
        ):
            response = self.client.post("/problem/Q1/fitch_ocr/start", data={"force": "0"})

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["queued"], 1)
        submit_job.assert_called_once()

        slug_dir = debug_dir / "q1_1_quiz1_berry_sharon_p01"
        self.assertTrue(slug_dir.exists())
        self.assertTrue(any(path.suffix.lower() == ".jpg" for path in slug_dir.iterdir()))
        state = json.loads(fitch_ocr_state_path(grades_dir, "Q1").read_text(encoding="utf-8"))
        self.assertEqual(state["items"]["1"]["status"], "queued")

    def test_accept_fitch_ocr_endpoint_marks_review_accepted(self) -> None:
        session = _build_session()
        session.problem_types["Q1"] = "fitch_proof"
        grades_dir = self.tmp_dir / "grades"
        state_path = fitch_ocr_state_path(grades_dir, "Q1")
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(
            json.dumps(
                {
                    "problem_id": "Q1",
                    "kind": "tfl_fitch",
                    "items": {
                        "1": {
                            "unit_id": 1,
                            "status": "needs_review",
                            "message": "OCR ready for review.",
                        }
                    },
                }
            ),
            encoding="utf-8",
        )

        with (
            patch.object(problem_strips_routes, "get_session", return_value=session),
            patch.object(problem_strips_routes, "GRADES_DIR", grades_dir),
        ):
            response = self.client.post("/problem/Q1/fitch_ocr/1/accept")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["item"]["status"], "accepted")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        self.assertEqual(state["items"]["1"]["status"], "accepted")

    def test_problem_strips_renders_modal_action_when_problem_type_is_modal(self) -> None:
        session = _build_session()
        session.problem_types["Q1"] = "modal_proof"
        segments = [
            {
                "unit_id": 1,
                "page_filename": "quiz1_berry_sharon_p01.jpg",
                "page_index": 0,
                "bbox": [0.1, 0.2, 0.8, 0.9],
                "present": True,
                "comment": "found",
                "confidence": 0.9,
            }
        ]

        with (
            patch.object(problem_strips_routes, "get_session", return_value=session),
            patch.object(problem_strips_routes, "ensure_segments_cached", return_value=segments),
        ):
            response = self.client.get("/problem/Q1/strips")

        self.assertEqual(response.status_code, 200)
        self.assertIn("Open Proof Editor", response.text)

    def test_problem_strips_renders_modal_ocr_preview_with_disagreement_and_checker_flags(self) -> None:
        session = _build_session()
        session.problem_types["Q1"] = "modal_proof"
        segments = [
            {
                "unit_id": 1,
                "page_filename": "quiz1_berry_sharon_p01.jpg",
                "page_index": 0,
                "bbox": [0.1, 0.2, 0.8, 0.9],
                "present": True,
                "comment": "found",
                "confidence": 0.9,
            }
        ]

        debug_dir = self.tmp_dir / "debug_modal_proof"
        example_dir = debug_dir / "q1_1_quiz1_berry_sharon_p01"
        example_dir.mkdir(parents=True, exist_ok=True)
        state = modal_proof_debug_routes._default_page_state()
        state["proof_json_text"] = json.dumps(
            {
                "lines": [
                    {"line": 1, "wffstr": "A", "jstr": "PR"},
                    {"line": 2, "wffstr": "¬□¬A", "jstr": "1"},
                ],
                "meta": {},
            },
            indent=2,
            ensure_ascii=False,
        )
        state["original_proof_json_text"] = state["proof_json_text"]
        state["checker_response"] = {
            "parse_errors": [],
            "check_errors": [
                {"line": 2, "code": "bad_rule", "message": "Derived line does not follow."}
            ],
            "normalized_lines": [],
        }
        (example_dir / "state.json").write_text(
            json.dumps(state, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (example_dir / "unary_verifier.json").write_text(
            json.dumps(
                [
                    {
                        "line_number": 2,
                        "baseline_formula": "¬□¬A",
                        "structured_formula": "¬□¬¬A",
                        "forced_choice": "¬□¬A",
                        "attempt_kind": "margin_anchor",
                    }
                ],
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        with (
            patch.object(problem_strips_routes, "get_session", return_value=session),
            patch.object(problem_strips_routes, "ensure_segments_cached", return_value=segments),
            patch.object(problem_strips_routes, "DEBUG_MODAL_PROOF_DIR", debug_dir),
            patch.object(modal_proof_debug_routes, "DEBUG_MODAL_PROOF_DIR", debug_dir),
        ):
            response = self.client.get("/problem/Q1/strips")

        self.assertEqual(response.status_code, 200)
        self.assertIn("Proposed OCR", response.text)
        self.assertIn("Open debug panel", response.text)
        self.assertIn("OCR disagree 1", response.text)
        self.assertIn("Checker 1", response.text)
        self.assertIn("ascii-proof-row is-both", response.text)
        self.assertIn("structured / forced disagree", response.text)
        self.assertIn("Derived line does not follow.", response.text)
        self.assertIn("checker: Derived line does not follow.", response.text)

    def test_fitch_strip_route_crops_scan_and_redirects_to_debug_page(self) -> None:
        session = _build_session()
        session.problem_types["Q2"] = "fitch_proof"

        raw_dir = self.tmp_dir / "raw_scans"
        debug_dir = self.tmp_dir / "debug_fitch"
        grades_dir = self.tmp_dir / "grades"
        raw_dir.mkdir(parents=True, exist_ok=True)
        debug_dir.mkdir(parents=True, exist_ok=True)
        grades_dir.mkdir(parents=True, exist_ok=True)

        image_path = raw_dir / "quiz1_berry_sharon_p01.jpg"
        Image.new("RGB", (100, 200), color="white").save(image_path)

        segments = [
            {
                "unit_id": 1,
                "page_filename": "quiz1_berry_sharon_p01.jpg",
                "page_index": 0,
                "bbox": [0.1, 0.2, 0.8, 0.9],
                "present": True,
            }
        ]
        (grades_dir / "segments_Q2.json").write_text(json.dumps(segments), encoding="utf-8")

        with (
            patch.object(problem_strips_routes, "get_session", return_value=session),
            patch.object(problem_strips_routes, "RAW_SCANS_DIR", raw_dir),
            patch.object(problem_strips_routes, "DEBUG_FITCH_DIR", debug_dir),
            patch.object(problem_strips_routes, "GRADES_DIR", grades_dir),
            patch.object(problem_strips_routes, "_run_subprocess_with_logs", return_value=(0, "ok")) as run_ocr,
        ):
            response = self.client.post(
                "/problem/Q2/fitch_debug",
                data={
                    "unit_id": "1",
                    "page_filename": "quiz1_berry_sharon_p01.jpg",
                    "bbox_json": "[0.1, 0.2, 0.8, 0.9]",
                },
                follow_redirects=False,
            )

        self.assertEqual(response.status_code, 303)
        self.assertIn("/debug/fitch?scan=", response.headers["location"])
        run_ocr.assert_called_once()

        debug_files = list((debug_dir / "q2_1_quiz1_berry_sharon_p01").glob("*"))
        self.assertTrue(any(path.suffix.lower() == ".jpg" for path in debug_files))

    def test_fitch_strip_route_uses_cached_debug_structure_without_rerunning_ocr(self) -> None:
        session = _build_session()
        session.problem_types["Q2"] = "fitch_proof"

        raw_dir = self.tmp_dir / "raw_scans"
        debug_dir = self.tmp_dir / "debug_fitch"
        grades_dir = self.tmp_dir / "grades"
        raw_dir.mkdir(parents=True, exist_ok=True)
        grades_dir.mkdir(parents=True, exist_ok=True)

        image_path = raw_dir / "quiz1_berry_sharon_p01.jpg"
        Image.new("RGB", (100, 200), color="white").save(image_path)

        cached_dir = debug_dir / "q2_1_quiz1_berry_sharon_p01"
        cached_dir.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (20, 20), color="white").save(cached_dir / "q2_1_quiz1_berry_sharon_p01.jpg")
        (cached_dir / "final_structure.json").write_text(
            '[{"line": 1, "wffstr": "P", "jstr": "Pr"}]',
            encoding="utf-8",
        )

        with (
            patch.object(problem_strips_routes, "get_session", return_value=session),
            patch.object(problem_strips_routes, "RAW_SCANS_DIR", raw_dir),
            patch.object(problem_strips_routes, "DEBUG_FITCH_DIR", debug_dir),
            patch.object(problem_strips_routes, "GRADES_DIR", grades_dir),
            patch.object(problem_strips_routes, "_run_subprocess_with_logs", return_value=(0, "ok")) as run_ocr,
        ):
            response = self.client.post(
                "/problem/Q2/fitch_debug",
                data={
                    "unit_id": "1",
                    "page_filename": "quiz1_berry_sharon_p01.jpg",
                    "bbox_json": "[0.1, 0.2, 0.8, 0.9]",
                },
                follow_redirects=False,
            )

        self.assertEqual(response.status_code, 303)
        self.assertIn("/debug/fitch?scan=q2_1_quiz1_berry_sharon_p01", response.headers["location"])
        run_ocr.assert_not_called()

    def test_fitch_strip_route_rejects_non_fitch_problem(self) -> None:
        session = _build_session()
        raw_dir = self.tmp_dir / "raw_scans"
        raw_dir.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (50, 50), color="white").save(raw_dir / "quiz1_berry_sharon_p01.jpg")

        with (
            patch.object(problem_strips_routes, "get_session", return_value=session),
            patch.object(problem_strips_routes, "RAW_SCANS_DIR", raw_dir),
        ):
            response = self.client.post(
                "/problem/Q2/fitch_debug",
                data={
                    "unit_id": "1",
                    "page_filename": "quiz1_berry_sharon_p01.jpg",
                    "bbox_json": "",
                },
            )

        self.assertEqual(response.status_code, 400)

    def test_modal_strip_route_crops_scan_and_redirects_to_modal_debug_page(self) -> None:
        session = _build_session()
        session.problem_types["Q2"] = "modal_proof"

        raw_dir = self.tmp_dir / "raw_scans"
        debug_dir = self.tmp_dir / "debug_modal_proof"
        grades_dir = self.tmp_dir / "grades"
        raw_dir.mkdir(parents=True, exist_ok=True)
        debug_dir.mkdir(parents=True, exist_ok=True)
        grades_dir.mkdir(parents=True, exist_ok=True)

        image_path = raw_dir / "quiz1_berry_sharon_p01.jpg"
        Image.new("RGB", (100, 200), color="white").save(image_path)

        segments = [
            {
                "unit_id": 1,
                "page_filename": "quiz1_berry_sharon_p01.jpg",
                "page_index": 0,
                "bbox": [0.1, 0.2, 0.8, 0.9],
                "present": True,
            }
        ]
        (grades_dir / "segments_Q2.json").write_text(json.dumps(segments), encoding="utf-8")

        with (
            patch.object(problem_strips_routes, "get_session", return_value=session),
            patch.object(problem_strips_routes, "RAW_SCANS_DIR", raw_dir),
            patch.object(problem_strips_routes, "DEBUG_MODAL_PROOF_DIR", debug_dir),
            patch.object(problem_strips_routes, "GRADES_DIR", grades_dir),
        ):
            response = self.client.post(
                "/problem/Q2/modal_proof_debug",
                data={
                    "unit_id": "1",
                    "page_filename": "quiz1_berry_sharon_p01.jpg",
                    "bbox_json": "[0.1, 0.2, 0.8, 0.9]",
                },
                follow_redirects=False,
            )

        self.assertEqual(response.status_code, 303)
        self.assertIn("/debug/modal-proof?example=", response.headers["location"])
        debug_files = list((debug_dir / "q2_1_quiz1_berry_sharon_p01").glob("*"))
        self.assertTrue(any(path.suffix.lower() == ".jpg" for path in debug_files))
        self.assertTrue(any(path.name == "state.json" for path in debug_files))


class RubricSchemaTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = Path("working") / "test_tmp" / f"rubric_schema_{uuid4().hex}"
        self.tmp_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_load_grading_key_preserves_problem_type_and_autograde(self) -> None:
        yaml_text = """
quiz_id: quiz1
max_points: 5
problems:
  - id: Q1
    label: FOL model
    points: 5
    problem_type: fol_model
    autograde:
      task_kind: non_validity
      prompt_data:
        premises:
          - "Pa"
        conclusion: "Qa"
"""
        rubric_path = self.tmp_dir / "quiz1.yaml"
        rubric_path.write_text(yaml_text, encoding="utf-8")
        grading_key = load_grading_key(rubric_path)

        self.assertEqual(grading_key.problems[0].problem_type, "fol_model")
        self.assertEqual(grading_key.problems[0].autograde["task_kind"], "non_validity")
        self.assertEqual(
            grading_key.problems[0].autograde["prompt_data"]["conclusion"],
            "Qa",
        )

    def test_load_grading_key_preserves_problem_context_fields(self) -> None:
        yaml_text = """
quiz_id: quiz1
max_points: 5
problems:
  - id: Q1
    label: Modal model
    points: 5
    section: Models
    problem_number: "3"
    prompt: Show that two sentences are not equivalent.
    segmentation_hint: Look under the Models heading for item 3.
    rubric_notes: Full credit needs a countermodel at w0.
"""
        rubric_path = self.tmp_dir / "quiz1_context.yaml"
        rubric_path.write_text(yaml_text, encoding="utf-8")
        grading_key = load_grading_key(rubric_path)

        problem = grading_key.problems[0]
        self.assertEqual(problem.section, "Models")
        self.assertEqual(problem.problem_number, "3")
        self.assertEqual(problem.prompt, "Show that two sentences are not equivalent.")
        self.assertEqual(problem.segmentation_hint, "Look under the Models heading for item 3.")
        self.assertEqual(problem.rubric_notes, "Full credit needs a countermodel at w0.")


class NameReviewSubmitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(server.app)
        self.tmp_dir = Path("working") / "test_tmp" / f"name_review_{uuid4().hex}"
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        (self.tmp_dir / "page1.jpg").write_bytes(b"fake-image-bytes")

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_name_review_submit_starts_grading_without_server_error(self) -> None:
        roster = [RosterEntry(student_id="s1", name="Berry, Sharon")]
        grading_key = GradingKey(
            quiz_id="quiz1",
            max_points=5,
            problems=[ProblemSpec(id="Q1", label="Problem 1", points=5)],
        )

        with (
            patch.object(server, "RAW_SCANS_DIR", self.tmp_dir),
            patch.object(server, "load_roster", return_value=roster),
            patch.object(server, "load_grading_key", return_value=grading_key),
            patch.object(server, "save_session"),
        ):
            response = self.client.post(
                "/name_review",
                data={
                    "quiz_id": "quiz1",
                    "raw_scans_subdir": "",
                    "page_count": "1",
                    "filename_0": "page1.jpg",
                    "student_0": "s1",
                    "next_action": "grade",
                },
                follow_redirects=False,
            )

        self.assertEqual(response.status_code, 303)
        self.assertEqual(response.headers["location"], "/problem/Q1/strips")


class CanvasZipImportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(server.app)
        self.tmp_dir = Path("working") / "test_tmp" / f"canvas_zip_{uuid4().hex}"
        self.raw_scans_dir = self.tmp_dir / "raw_scans"
        self.canvas_zip_dir = self.tmp_dir / "canvas_zips"
        self.manifest_dir = self.tmp_dir / "manifests"
        self.raw_scans_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _zip_bytes(self) -> bytes:
        buf = BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("studentone_submission_1.jpg", b"page-one")
            zf.writestr("studentone_submission_2.jpg", b"page-two")
            zf.writestr("studenttwo_submission_1.jpg", b"page-three")
        return buf.getvalue()

    def test_canvas_zip_import_without_roster_creates_anonymous_work_units(self) -> None:
        grading_key = GradingKey(
            quiz_id="quiz1",
            max_points=1,
            problems=[ProblemSpec(id="Q1", label="Problem 1", points=1, problem_type="fitch_proof")],
        )
        saved_sessions = []

        def capture_session(session, *args, **kwargs):
            saved_sessions.append(session)

        with (
            patch.object(server, "_has_existing_session_state", return_value=False),
            patch.object(server, "RAW_SCANS_DIR", self.raw_scans_dir),
            patch.object(server, "DATA_DIR", self.tmp_dir),
            patch.object(server, "CANVAS_REUPLOAD_ORIGINAL_ZIPS_DIR", self.canvas_zip_dir),
            patch.object(server, "load_roster", side_effect=FileNotFoundError("no roster")),
            patch.object(server, "load_grading_key", return_value=grading_key),
            patch.object(server, "save_session", side_effect=capture_session),
            patch("app.exam_shared.CANVAS_REUPLOAD_MANIFEST_DIR", self.manifest_dir),
        ):
            response = self.client.post(
                "/setup/import_canvas_zip",
                data={"quiz_id": "quiz1"},
                files={"zip_file": ("submissions.zip", self._zip_bytes(), "application/zip")},
                follow_redirects=False,
            )

        self.assertEqual(response.status_code, 303)
        self.assertEqual(response.headers["location"], "/problem/Q1/strips")
        self.assertEqual(len(saved_sessions), 1)

        session = saved_sessions[0]
        self.assertEqual([entry.student_id for entry in session.roster], ["anon_0001", "anon_0002"])
        self.assertEqual([entry.name for entry in session.roster], ["Anonymous 0001", "Anonymous 0002"])
        self.assertEqual(len(session.work_units), 2)
        self.assertEqual(session.work_units[0].student_id, "anon_0001")
        self.assertEqual(len(session.work_units[0].page_filenames), 2)
        self.assertEqual(session.work_units[1].student_id, "anon_0002")
        self.assertEqual(len(session.work_units[1].page_filenames), 1)
        self.assertEqual(len(list(self.raw_scans_dir.glob("*.jpg"))), 3)


if __name__ == "__main__":
    unittest.main()
