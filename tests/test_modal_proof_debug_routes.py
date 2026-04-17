import json
import os
import shutil
import unittest
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from fastapi.testclient import TestClient

os.environ.setdefault("DATA_DIR", str((Path("working") / "test_tmp" / "modal_proof_route_env").resolve()))
os.environ["ENV_MODE"] = ""

from app import server, storage
from app.autograde.base import AutoGradeResult
from app.routes import modal_proof_debug as modal_proof_debug_routes


class ModalProofDebugRoutesTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(server.app)
        self.debug_dir = Path("working") / "test_tmp" / f"modal_proof_{uuid4().hex}"
        self.debug_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.debug_dir, ignore_errors=True)

    def test_modal_proof_debug_page_renders_with_starter_example(self) -> None:
        with patch.object(modal_proof_debug_routes, "DEBUG_MODAL_PROOF_DIR", self.debug_dir):
            response = self.client.get("/debug/modal-proof")

        self.assertEqual(response.status_code, 200)
        self.assertIn("Modal Proof Debugger", response.text)
        self.assertIn("Proof image", response.text)
        self.assertIn("Proof editor", response.text)
        self.assertIn("Run OCR", response.text)
        self.assertIn("Proof JSON", response.text)
        self.assertIn("Run checker", response.text)
        self.assertIn("starter", response.text)
        self.assertIn("OCR review details", response.text)
        self.assertIn("Unary Verifier Experiment", response.text)

    def test_modal_proof_post_action_gets_redirect_back_to_debug_page(self) -> None:
        with patch.object(modal_proof_debug_routes, "DEBUG_MODAL_PROOF_DIR", self.debug_dir):
            response = self.client.get("/debug/modal-proof/run_ocr?example=starter", follow_redirects=False)

        self.assertEqual(response.status_code, 303)
        self.assertEqual(response.headers["location"], "/debug/modal-proof?example=starter")

    def test_save_modal_proof_state_persists_state_json(self) -> None:
        with patch.object(modal_proof_debug_routes, "DEBUG_MODAL_PROOF_DIR", self.debug_dir):
            response = self.client.post(
                "/debug/modal-proof/save",
                data={
                    "example": "starter",
                    "system": "S4",
                    "include_derived_rules": "true",
                    "conclusion": "A",
                    "vision_base_url": "http://127.0.0.1:1234/v1",
                    "vision_model": "qwen/qwen3-vl-4b",
                    "verifier_vision_model": "qwen/qwen3-vl-8b",
                    "proof_json_text": json.dumps([{"wffstr": "A", "jstr": "PR"}]),
                    "draft_comment": "saved comment",
                    "draft_grade": "2",
                },
                follow_redirects=False,
            )

            state_path = self.debug_dir / "starter" / "state.json"
            saved = json.loads(state_path.read_text(encoding="utf-8"))

        self.assertEqual(response.status_code, 303)
        self.assertIn("/debug/modal-proof?example=starter", response.headers["location"])
        self.assertEqual(saved["system"], "S4")
        self.assertTrue(saved["include_derived_rules"])
        self.assertEqual(saved["vision_base_url"], "http://127.0.0.1:1234/v1")
        self.assertEqual(saved["vision_model"], "qwen/qwen3-vl-4b")
        self.assertEqual(saved["verifier_vision_model"], "qwen/qwen3-vl-8b")
        self.assertEqual(saved["draft_comment"], "saved comment")
        self.assertEqual(saved["draft_grade"], "2")

    def test_save_modal_proof_state_persists_review_decisions(self) -> None:
        review_decisions = {
            "4": {
                "mode": "candidate",
                "formula": "¬□¬¬A",
                "source_key": "formula::¬□¬¬A",
            }
        }

        with patch.object(modal_proof_debug_routes, "DEBUG_MODAL_PROOF_DIR", self.debug_dir):
            self.client.post(
                "/debug/modal-proof/save",
                data={
                    "example": "starter",
                    "system": "K",
                    "conclusion": "",
                    "vision_base_url": "",
                    "vision_model": "qwen/qwen3-vl-4b",
                    "verifier_vision_model": "qwen/qwen3-vl-8b",
                    "proof_json_text": json.dumps({"lines": [{"line": 4, "wffstr": "¬□¬A", "jstr": "MC 3"}], "meta": {}}, ensure_ascii=False),
                    "review_decisions_json": json.dumps(review_decisions, ensure_ascii=False),
                    "draft_comment": "",
                    "draft_grade": "",
                },
                follow_redirects=False,
            )

        saved = json.loads((self.debug_dir / "starter" / "state.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["review_decisions"]["4"]["mode"], "candidate")
        self.assertEqual(saved["review_decisions"]["4"]["formula"], "¬□¬¬A")

    def test_run_ocr_updates_state_from_uploaded_image(self) -> None:
        ocr_state = {
            "system": "S5",
            "include_derived_rules": False,
            "conclusion": "□A",
            "vision_base_url": "http://127.0.0.1:1234/v1",
            "vision_model": "qwen/qwen3-vl-4b",
            "verifier_vision_model": "qwen/qwen3-vl-8b",
            "source_image_name": "source_image.png",
            "proof_json_text": json.dumps({"lines": [{"wffstr": "□A", "jstr": "PR"}], "meta": {}}),
            "original_proof_json_text": json.dumps({"lines": [{"wffstr": "A", "jstr": "PR"}], "meta": {}}),
            "verifier_rewrite_summary": [
                {"line_number": 1, "original_formula": "A", "rewritten_formula": "□A", "citation": "PR"}
            ],
            "draft_comment": "",
            "draft_grade": "",
            "checker_request": {"system": "S5", "lines": []},
            "checker_response": {"ok": True, "parse_errors": [], "check_errors": []},
            "run_summary": "OCR produced a modal proof structure and the checker ran.",
            "review_decisions": {"4": {"mode": "manual", "formula": "□A", "source_key": ""}},
        }

        with (
            patch.object(modal_proof_debug_routes, "DEBUG_MODAL_PROOF_DIR", self.debug_dir),
            patch.object(modal_proof_debug_routes, "build_vlm_client", return_value=object()),
            patch.object(modal_proof_debug_routes, "transcribe_image_to_modal_debug_state", return_value=ocr_state),
        ):
            response = self.client.post(
                "/debug/modal-proof/run_ocr",
                data={
                    "example": "starter",
                    "system": "S5",
                    "conclusion": "□A",
                    "vision_base_url": "http://127.0.0.1:1234/v1",
                    "vision_model": "qwen/qwen3-vl-4b",
                    "verifier_vision_model": "qwen/qwen3-vl-8b",
                    "proof_json_text": json.dumps([{"wffstr": "A", "jstr": "PR"}]),
                    "draft_comment": "keep me",
                    "draft_grade": "1",
                },
                files={"proof_image": ("proof.png", b"fakepng", "image/png")},
                follow_redirects=False,
            )
            page = self.client.get(response.headers["location"])

        saved = json.loads((self.debug_dir / "starter" / "state.json").read_text(encoding="utf-8"))

        self.assertEqual(response.status_code, 303)
        self.assertIn("/debug/modal-proof?example=starter", response.headers["location"])
        self.assertEqual(page.status_code, 200)
        self.assertIn("OCR ran and updated the modal proof state.", page.text)
        self.assertEqual(saved["vision_model"], "qwen/qwen3-vl-4b")
        self.assertEqual(saved["verifier_vision_model"], "qwen/qwen3-vl-8b")
        self.assertEqual(saved["draft_comment"], "keep me")
        self.assertEqual(saved["draft_grade"], "1")
        saved_lines = json.loads(saved["proof_json_text"])["lines"]
        self.assertEqual(saved_lines[0]["wffstr"], "□A")
        original_lines = json.loads(saved["original_proof_json_text"])["lines"]
        self.assertEqual(original_lines[0]["wffstr"], "A")
        self.assertEqual(saved["verifier_rewrite_summary"][0]["rewritten_formula"], "□A")
        self.assertEqual(saved["review_decisions"], {})

    def test_modal_proof_debug_page_shows_review_panel_and_unary_verifier_results(self) -> None:
        example_dir = self.debug_dir / "sample"
        example_dir.mkdir(parents=True, exist_ok=True)
        (example_dir / "state.json").write_text(
            json.dumps(
                {
                    "system": "K",
                    "include_derived_rules": False,
                    "conclusion": "",
                    "source_image_name": "sample_source.png",
                    "proof_json_text": json.dumps({"lines": [{"line": 4, "wffstr": "¬□¬A", "jstr": "MC 3"}], "meta": {}}, ensure_ascii=False),
                    "original_proof_json_text": json.dumps({"lines": [{"wffstr": "¬□¬A", "jstr": "MC 3"}], "meta": {}}, ensure_ascii=False),
                    "verifier_rewrite_summary": [
                        {"line_number": 4, "original_formula": "¬□¬A", "rewritten_formula": "¬□¬¬A", "citation": "MC 3"}
                    ],
                    "checker_response": {
                        "parse_errors": [],
                        "check_errors": [{"line": 4, "code": "BadUsage", "message": "cited a rule that was used incorrectly"}],
                    },
                    "run_summary": "ready",
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        (example_dir / "sample_source.png").write_bytes(b"fakepng")
        (example_dir / "unary_verifier.json").write_text(
            json.dumps(
                [
                    {
                        "node_id": "1",
                        "line_label": "4",
                        "node_line_order": 3,
                        "baseline_formula": "¬□¬A",
                        "structured_formula": "¬□¬¬A",
                        "forced_choice_candidates": ["¬□¬A", "¬□¬¬A"],
                        "forced_choice": "¬□¬¬A",
                        "proposed_correction": "¬□¬¬A",
                        "would_auto_apply": True,
                        "attempt_kind": "margin_anchor",
                        "strip_y1": 120,
                        "strip_y2": 148,
                        "line_crop_path": "working\\sample\\verifier_line_1_4.jpg",
                        "candidate_attempts": [
                            {
                                "attempt_kind": "margin_anchor",
                                "band_index": 3,
                                "band_distance": 0,
                                "strip_y1": 120,
                                "strip_y2": 148,
                                "structured_formula": "¬□¬¬A",
                                "forced_choice": "¬□¬¬A",
                                "line_crop_path": "working\\sample\\verifier_line_1_4.jpg",
                            },
                            {
                                "attempt_kind": "local_band",
                                "band_index": 2,
                                "band_distance": 1,
                                "structured_formula": "⋄¬A",
                                "forced_choice": "⋄¬A",
                                "line_crop_path": "working\\sample\\verifier_line_1_4_band3.jpg",
                            },
                        ],
                    }
                ],
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        (example_dir / "verifier_line_1_4.jpg").write_bytes(b"fakejpg")
        (example_dir / "verifier_line_1_4_band3.jpg").write_bytes(b"fakejpg")

        with patch.object(modal_proof_debug_routes, "DEBUG_MODAL_PROOF_DIR", self.debug_dir):
            response = self.client.get("/debug/modal-proof?example=sample")

        self.assertEqual(response.status_code, 200)
        self.assertIn("Proof image", response.text)
        self.assertIn("Proof editor", response.text)
        self.assertIn("Open image file", response.text)
        self.assertIn("ASCII Proof View", response.text)
        self.assertIn("OCR review details", response.text)
        self.assertIn("Flagged OCR lines: 1", response.text)
        self.assertIn("Flagged lines:", response.text)
        self.assertIn("Use ¬□¬¬A", response.text)
        self.assertIn("Use manual", response.text)
        self.assertIn("checker issue", response.text)
        self.assertIn("checker: cited a rule that was used incorrectly", response.text)
        self.assertIn("ocr: proposed correction, multiple candidate formulas", response.text)
        self.assertIn("Apply review choices to Proof JSON", response.text)
        self.assertIn("Unary Verifier Experiment", response.text)
        self.assertIn("¬□¬¬A", response.text)
        self.assertIn("verifier_line_1_4.jpg", response.text)
        self.assertIn('class="crop-preview"', response.text)
        self.assertIn("node slot 4", response.text)
        self.assertIn("Alternates", response.text)
        self.assertIn("verifier_line_1_4_band3.jpg", response.text)
        self.assertIn('class="crop-thumb"', response.text)
        self.assertIn("strategy: margin anchor", response.text)
        self.assertIn("y 120-148", response.text)
        self.assertIn("Verifier Rewrites", response.text)
        self.assertIn("Original OCR JSON", response.text)

    def test_build_review_panel_dedupes_candidates_and_flags_checker_only_lines(self) -> None:
        state = {
            "proof_json_text": json.dumps(
                {
                    "lines": [
                        {"line": 4, "wffstr": "¬□¬A", "jstr": "MC 3"},
                        {"line": 9, "wffstr": "⊥", "jstr": "¬E 4 8"},
                    ],
                    "meta": {},
                },
                ensure_ascii=False,
            ),
            "original_proof_json_text": json.dumps(
                {
                    "lines": [
                        {"line": 4, "wffstr": "¬□¬A", "jstr": "MC 3"},
                        {"line": 9, "wffstr": "⊥", "jstr": "¬E 4 8"},
                    ],
                    "meta": {},
                },
                ensure_ascii=False,
            ),
            "checker_response": {
                "parse_errors": [],
                "check_errors": [{"line": 9, "code": "BadUsage", "message": "cited a rule that was used incorrectly"}],
            },
            "review_decisions": {},
        }
        unary_results = [
            {
                "line_number": 4,
                "baseline_formula": "¬□¬A",
                "structured_formula": "¬□¬¬A",
                "forced_choice": "¬□¬¬A",
                "proposed_correction": "¬□¬¬A",
                "attempt_kind": "margin_anchor",
                "candidate_attempts": [
                    {"attempt_kind": "margin_anchor", "structured_formula": "¬□¬¬A", "forced_choice": "¬□¬¬A"},
                    {"attempt_kind": "local_band", "structured_formula": "¬□¬¬A", "forced_choice": "¬□¬A"},
                ],
            }
        ]

        panel = modal_proof_debug_routes._build_review_panel(
            state=state,
            unary_verifier_results=unary_results,
        )

        self.assertEqual(panel["flagged_count"], 2)
        line4 = next(item for item in panel["items"] if item["line_number"] == 4)
        formulas = [candidate["formula"] for candidate in line4["candidates"]]
        self.assertEqual(formulas.count("¬□¬¬A"), 1)
        self.assertIn("multiple candidate formulas", line4["flag_reasons"])

        line9 = next(item for item in panel["items"] if item["line_number"] == 9)
        self.assertEqual(line9["issues"][0]["kind"], "rule")
        self.assertEqual(line9["current_formula"], "⊥")

    def test_build_review_panel_ignores_stale_saved_decisions(self) -> None:
        state = {
            "proof_json_text": json.dumps(
                {"lines": [{"line": 4, "wffstr": "¬□¬A", "jstr": "MC 3"}], "meta": {}},
                ensure_ascii=False,
            ),
            "original_proof_json_text": json.dumps(
                {"lines": [{"line": 4, "wffstr": "¬□¬A", "jstr": "MC 3"}], "meta": {}},
                ensure_ascii=False,
            ),
            "checker_response": {"parse_errors": [], "check_errors": []},
            "review_decisions": {
                "4": {
                    "mode": "candidate",
                    "formula": "¬□¬¬A",
                    "source_key": "formula::stale",
                }
            },
        }
        unary_results = [
            {
                "line_number": 4,
                "baseline_formula": "¬□¬A",
                "structured_formula": "¬□¬¬A",
                "forced_choice": "uncertain",
                "proposed_correction": "",
                "attempt_kind": "local_band",
                "candidate_attempts": [],
            }
        ]

        panel = modal_proof_debug_routes._build_review_panel(
            state=state,
            unary_verifier_results=unary_results,
        )

        self.assertEqual(panel["flagged_count"], 1)
        self.assertIsNone(panel["items"][0]["selected_decision"])
        self.assertEqual(panel["unresolved_count"], 1)

    def test_run_checker_renders_summary_and_payloads(self) -> None:
        result = AutoGradeResult(
            problem_type="modal_proof",
            status="scored",
            passed=True,
            summary="Modal proof checked under system K; conclusion reached.",
            draft_comment="Modal proof checker system: K",
            score=2.0,
            metadata={
                "checker_request": {
                    "system": "K",
                    "lines": [{"depth": 0, "sentence": "A", "citation": "PR"}],
                },
                "checker_response": {
                    "ok": True,
                    "parse_errors": [],
                    "check_errors": [],
                    "conclusion_reached": True,
                    "normalized_lines": [{"depth": 0, "sentence": "A", "citation": "PR"}],
                    "metadata": {"system": "K"},
                },
            },
        )

        with (
            patch.object(modal_proof_debug_routes, "DEBUG_MODAL_PROOF_DIR", self.debug_dir),
            patch.object(modal_proof_debug_routes, "autograde_submission", return_value=result),
        ):
            response = self.client.post(
                "/debug/modal-proof/run_checker",
                data={
                    "example": "starter",
                    "system": "K",
                    "conclusion": "A",
                    "proof_json_text": json.dumps([{"wffstr": "A", "jstr": "PR"}]),
                    "draft_comment": "",
                    "draft_grade": "",
                },
                follow_redirects=False,
            )
            page = self.client.get(response.headers["location"])

        self.assertEqual(response.status_code, 303)
        self.assertIn("/debug/modal-proof?example=starter", response.headers["location"])
        self.assertEqual(page.status_code, 200)
        self.assertIn("Checker ran successfully through the shared autograde layer.", page.text)
        self.assertIn("Modal proof checked under system K; conclusion reached.", page.text)
        self.assertIn("Checker input", page.text)
        self.assertIn("ASCII Proof View", page.text)
        self.assertIn("1. A    PR", page.text)
        self.assertIn('"sentence": "A"', page.text)
        self.assertIn("No parse errors reported.", page.text)

    def test_malformed_json_stays_on_page_with_error_message(self) -> None:
        with patch.object(modal_proof_debug_routes, "DEBUG_MODAL_PROOF_DIR", self.debug_dir):
            response = self.client.post(
                "/debug/modal-proof/run_checker",
                data={
                    "example": "starter",
                    "system": "K",
                    "conclusion": "A",
                    "proof_json_text": "{bad json",
                    "draft_comment": "",
                    "draft_grade": "",
                },
                follow_redirects=False,
            )
            page = self.client.get(response.headers["location"])

        self.assertEqual(response.status_code, 303)
        self.assertEqual(page.status_code, 200)
        self.assertIn("Checker could not run.", page.text)
        self.assertIn("Modal proof answer_text is not valid JSON", page.text)

    def test_unavailable_checker_binary_stays_on_page_with_error_message(self) -> None:
        result = AutoGradeResult(
            problem_type="modal_proof",
            status="not_implemented",
            passed=None,
            summary="checker binary missing",
            draft_comment="Modal proof checker unavailable: checker binary missing",
            errors=["checker binary missing"],
            metadata={"checker_request": {"system": "K", "lines": []}},
        )

        with (
            patch.object(modal_proof_debug_routes, "DEBUG_MODAL_PROOF_DIR", self.debug_dir),
            patch.object(modal_proof_debug_routes, "autograde_submission", return_value=result),
        ):
            response = self.client.post(
                "/debug/modal-proof/run_checker",
                data={
                    "example": "starter",
                    "system": "K",
                    "conclusion": "A",
                    "proof_json_text": json.dumps([{"wffstr": "A", "jstr": "PR"}]),
                    "draft_comment": "",
                    "draft_grade": "",
                },
                follow_redirects=False,
            )
            page = self.client.get(response.headers["location"])

        self.assertEqual(response.status_code, 303)
        self.assertEqual(page.status_code, 200)
        self.assertIn("Checker could not run.", page.text)
        self.assertIn("checker binary missing", page.text)

    def test_archive_current_data_includes_debug_modal_proof(self) -> None:
        root = self.debug_dir / "data_root"
        data_dir = root / "data"
        archive_dir = root / "archive"
        raw_scans = data_dir / "raw_scans"
        grades = data_dir / "grades"
        rubric = data_dir / "rubric"
        name_review = data_dir / "name_review_cache"
        debug_fitch = data_dir / "debug_fitch"
        debug_fol_model = data_dir / "debug_fol_model"
        debug_modal_proof = data_dir / "debug_modal_proof"
        auto_grouped = data_dir / "auto_grouped"

        for path in (
            raw_scans,
            grades,
            rubric,
            name_review,
            debug_fitch,
            debug_fol_model,
            debug_modal_proof,
            auto_grouped,
        ):
            path.mkdir(parents=True, exist_ok=True)

        (data_dir / "roster.csv").write_text("student_id,name\ns1,Student One\n", encoding="utf-8")
        sample_state = debug_modal_proof / "sample"
        sample_state.mkdir(parents=True, exist_ok=True)
        (sample_state / "state.json").write_text('{"system": "K"}', encoding="utf-8")

        with (
            patch.object(storage, "DATA_DIR", data_dir),
            patch.object(storage, "DATA_ARCHIVE_DIR", archive_dir),
            patch.object(storage, "RAW_SCANS_DIR", raw_scans),
            patch.object(storage, "GRADES_DIR", grades),
            patch.object(storage, "RUBRIC_DIR", rubric),
            patch.object(storage, "NAME_REVIEW_CACHE_DIR", name_review),
            patch.object(storage, "DEBUG_FITCH_DIR", debug_fitch),
            patch.object(storage, "DEBUG_FOL_MODEL_DIR", debug_fol_model),
            patch.object(storage, "DEBUG_MODAL_PROOF_DIR", debug_modal_proof),
            patch.object(storage, "AUTO_GROUPED_DIR", auto_grouped),
            patch.object(storage, "DEFAULT_ROSTER_PATH", data_dir / "roster.csv"),
        ):
            snapshot_id = storage.archive_current_data("Modal Debug")

        archived_state = archive_dir / snapshot_id / "debug_modal_proof" / "sample" / "state.json"
        self.assertTrue(archived_state.exists())


if __name__ == "__main__":
    unittest.main()
