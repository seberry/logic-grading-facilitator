from __future__ import annotations

import io
import json
import shutil
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from fastapi.testclient import TestClient

from app import server
from app.models import GradingKey, ProblemSpec, RosterEntry, Session, WorkUnit
from app.quiz_variants import normalize_quiz_version, resolved_problem_context
from scripts.extract_quiz_reference_images import (
    P250_QUIZ1_REGIONS,
    P250_TRUTH_TABLE_QUIZ_REGIONS,
    extract_reference_images,
)
from app.routes import exam_exports as exam_exports_routes
from app.storage import load_grading_key
from app.vlm_classifier import PagePrediction, classify_page


def _variant_session() -> Session:
    return Session(
        roster=[RosterEntry(student_id="s1", name="Example, Ada")],
        grading_key=GradingKey(
            quiz_id="P250_F26_quiz1_ABC",
            problems=[
                ProblemSpec(
                    id="Q1",
                    label="Conclusion",
                    points=2,
                    variants={
                        "A": {
                            "prompt": "Version A prompt",
                            "answer_key": "Version A answer",
                            "rubric_notes": "TF chooses partial credit.",
                        },
                        "B": {
                            "prompt": "Version B prompt",
                            "answer_key": "Version B answer",
                        },
                    },
                )
            ],
        ),
        work_units=[
            WorkUnit(
                unit_id=1,
                page_filenames=["p250_quiz_1_ada_example_p01.jpg"],
                student_id="s1",
                quiz_version="A",
                per_problem={"Q1": None},
                per_problem_comments={"Q1": None},
                comment=None,
            )
        ],
        unclaimed_pages=[],
    )


class QuizVariantTests(unittest.TestCase):
    def test_normalizes_and_resolves_variant_context(self) -> None:
        session = _variant_session()
        context = resolved_problem_context(session.grading_key.problems[0], "Version a")
        self.assertEqual(normalize_quiz_version("v-b"), "B")
        self.assertEqual(context["quiz_version"], "A")
        self.assertEqual(context["answer_key"], "Version A answer")
        self.assertTrue(context["has_override"])

    def test_rubric_loader_preserves_variant_keys(self) -> None:
        root = Path("working") / "test_tmp" / f"variant_rubric_{uuid4().hex}"
        root.mkdir(parents=True, exist_ok=True)
        self.addCleanup(shutil.rmtree, root, True)
        rubric_path = root / "quiz.yaml"
        rubric_path.write_text(
            """quiz_id: quiz
max_points: 2
problems:
  - id: Q1
    label: Conclusion
    points: 2
    variants:
      A:
        answer_key: Expected A
""",
            encoding="utf-8",
        )
        key = load_grading_key(rubric_path)
        self.assertEqual(key.problems[0].variants["A"].answer_key, "Expected A")
        self.assertNotIn("max_points", key.model_dump())

    def test_setup_discards_model_generated_answer_keys(self) -> None:
        client = TestClient(server.app)
        root = Path("working") / "test_tmp" / f"answer_free_rubric_{uuid4().hex}"
        root.mkdir(parents=True, exist_ok=True)
        self.addCleanup(shutil.rmtree, root, True)
        yaml_text = """quiz_id: P250_F26_quiz1_ABC
max_points: 2
problems:
  - id: Q1
    label: Conclusion
    points: 2
    answer_key: Legacy base answer
    variants:
      A:
        prompt: Version A prompt
        answer_key: Model-generated answer
"""
        with patch.object(server, "RUBRIC_DIR", root):
            response = client.post(
                "/setup/save_rubric",
                data={"yaml_text": yaml_text},
                follow_redirects=False,
            )

        self.assertEqual(response.status_code, 303)
        saved = (root / "P250_F26_quiz1_ABC.yaml").read_text(encoding="utf-8")
        self.assertNotIn("answer_key", saved)
        self.assertNotIn("Model-generated answer", saved)
        self.assertNotIn("max_points", saved)

    def test_classifier_reads_printed_version(self) -> None:
        class FakeClient:
            def call_for_page(self, image_path, roster_text):
                return json.dumps(
                    {
                        "name_visible": True,
                        "raw_name": "Ada Example",
                        "student_id": "s1",
                        "quiz_version": "Version c",
                        "confidence": 0.98,
                        "reasoning": "Matched roster and printed title.",
                    }
                )

        prediction = classify_page(
            Path("phone_scan.jpg"),
            [RosterEntry(student_id="s1", name="Example, Ada")],
            FakeClient(),
        )
        self.assertEqual(prediction.quiz_version, "C")

    def test_classifier_recovers_fields_before_malformed_explanation(self) -> None:
        class FakeClient:
            def call_for_page(self, image_path, roster_text):
                return (
                    '{"name_visible": true, "raw_name": "Sample Student", '
                    '"student_id": "s1", "quiz_version": "A", '
                    '"confidence": 0.93, "reasoning": "Matched "Sample" and then rambled'
                )

        prediction = classify_page(
            Path("synthetic_scan.jpg"),
            [RosterEntry(student_id="s1", name="Student, Sample")],
            FakeClient(),
        )
        self.assertEqual(prediction.student_id, "s1")
        self.assertEqual(prediction.quiz_version, "A")
        self.assertAlmostEqual(prediction.confidence, 0.93)
        self.assertIn("verify in name review", prediction.reasoning)

    def test_classifier_rejects_student_id_not_in_roster(self) -> None:
        class FakeClient:
            def call_for_page(self, image_path, roster_text):
                return json.dumps(
                    {
                        "name_visible": True,
                        "raw_name": "Invented Person",
                        "student_id": "invented-id",
                        "quiz_version": "B",
                        "confidence": 0.99,
                    }
                )

        prediction = classify_page(
            Path("synthetic_scan.jpg"),
            [RosterEntry(student_id="s1", name="Student, Sample")],
            FakeClient(),
        )
        self.assertIsNone(prediction.student_id)
        self.assertEqual(prediction.confidence, 0.0)
        self.assertEqual(prediction.quiz_version, "B")

    def test_setup_auto_grouping_caches_printed_version(self) -> None:
        client = TestClient(server.app)
        root = Path("working") / "test_tmp" / f"version_cache_{uuid4().hex}"
        raw_dir = root / "raw_scans"
        raw_dir.mkdir(parents=True, exist_ok=True)
        self.addCleanup(shutil.rmtree, root, True)
        (raw_dir / "phone.jpg").write_bytes(b"fake-image")
        cache_path = root / "name_review.json"
        prediction = PagePrediction(
            filename="phone.jpg",
            student_id="s1",
            confidence=0.95,
            name_visible=True,
            raw_name="Ada Example",
            reasoning="Matched name and printed header.",
            quiz_version="B",
        )

        with (
            patch.object(server, "RAW_SCANS_DIR", raw_dir),
            patch.object(server, "load_roster", return_value=_variant_session().roster),
            patch.object(server, "run_name_recognition", return_value=[prediction]),
            patch.object(server, "name_review_cache_path", return_value=cache_path),
        ):
            response = client.post(
                "/setup/run_auto_grouping",
                data={"quiz_id": "P250_F26_quiz1_ABC"},
                follow_redirects=False,
            )

        self.assertEqual(response.status_code, 303)
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        self.assertEqual(cached[0]["quiz_version"], "B")

    def test_name_review_get_displays_cached_printed_version(self) -> None:
        client = TestClient(server.app)
        root = Path("working") / "test_tmp" / f"version_review_{uuid4().hex}"
        raw_dir = root / "raw_scans"
        raw_dir.mkdir(parents=True, exist_ok=True)
        self.addCleanup(shutil.rmtree, root, True)
        (raw_dir / "phone.jpg").write_bytes(b"fake-image")
        cache_path = root / "name_review.json"
        cache_path.write_text(
            json.dumps(
                [
                    {
                        "filename": "phone.jpg",
                        "student_id": "s1",
                        "raw_name": "Ada Example",
                        "confidence": 0.95,
                        "reasoning": "Printed Version C.",
                        "name_visible": True,
                        "quiz_version": "C",
                    }
                ]
            ),
            encoding="utf-8",
        )

        with (
            patch.object(server, "RAW_SCANS_DIR", raw_dir),
            patch.object(server, "load_roster", return_value=_variant_session().roster),
            patch.object(server, "name_review_cache_path", return_value=cache_path),
        ):
            response = client.get("/name_review?quiz=P250_F26_quiz1_ABC")

        self.assertEqual(response.status_code, 200)
        self.assertIn('option value="C" selected', response.text)
        self.assertIn("Printed version guess:", response.text)

    def test_name_review_collapses_handoff_copy_and_merges_name_and_version(self) -> None:
        client = TestClient(server.app)
        root = Path("working") / "test_tmp" / f"duplicate_review_{uuid4().hex}"
        raw_dir = root / "raw_scans"
        raw_dir.mkdir(parents=True, exist_ok=True)
        self.addCleanup(shutil.rmtree, root, True)
        scan_bytes = b"same-synthetic-scan"
        (raw_dir / "scan_001.jpg").write_bytes(scan_bytes)
        canonical = "p250_f26_quiz1_abc_example_ada_p01.jpg"
        (raw_dir / canonical).write_bytes(scan_bytes)
        cache_path = root / "name_review.json"
        cache_path.write_text(
            json.dumps(
                [
                    {
                        "filename": "scan_001.jpg",
                        "student_id": None,
                        "confidence": 0.8,
                        "quiz_version": "B",
                    },
                    {
                        "filename": canonical,
                        "student_id": "s1",
                        "raw_name": "Example, Ada",
                        "confidence": 1.0,
                        "quiz_version": None,
                    },
                ]
            ),
            encoding="utf-8",
        )

        with (
            patch.object(server, "RAW_SCANS_DIR", raw_dir),
            patch.object(server, "load_roster", return_value=_variant_session().roster),
            patch.object(server, "name_review_cache_path", return_value=cache_path),
        ):
            response = client.get("/name_review?quiz=P250_F26_quiz1_ABC")

        self.assertEqual(response.status_code, 200)
        self.assertIn('name="page_count" value="1"', response.text)
        self.assertIn('option value="s1"\n                selected', response.text)
        self.assertIn('option value="B" selected', response.text)

    def test_save_corrections_updates_cache_without_starting_grading(self) -> None:
        client = TestClient(server.app)
        root = Path("working") / "test_tmp" / f"review_save_{uuid4().hex}"
        root.mkdir(parents=True, exist_ok=True)
        self.addCleanup(shutil.rmtree, root, True)
        cache_path = root / "name_review.json"
        cache_path.write_text(
            json.dumps([{"filename": "phone.jpg", "student_id": None}]),
            encoding="utf-8",
        )

        with patch.object(server, "name_review_cache_path", return_value=cache_path):
            response = client.post(
                "/name_review",
                data={
                    "quiz_id": "P250_F26_quiz1_ABC",
                    "raw_scans_subdir": "",
                    "page_count": "1",
                    "filename_0": "phone.jpg",
                    "student_0": "s1",
                    "quiz_version_0": "A",
                    "next_action": "save",
                },
                follow_redirects=False,
            )

        self.assertEqual(response.status_code, 303)
        self.assertIn("saved=1", response.headers["location"])
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        self.assertEqual(cached[0]["student_id"], "s1")
        self.assertEqual(cached[0]["quiz_version"], "A")
        self.assertTrue(cached[0]["name_reviewed"])

    def test_refresh_preserves_reviewed_name_and_adds_fresh_version(self) -> None:
        client = TestClient(server.app)
        root = Path("working") / "test_tmp" / f"review_refresh_{uuid4().hex}"
        raw_dir = root / "raw_scans"
        raw_dir.mkdir(parents=True, exist_ok=True)
        self.addCleanup(shutil.rmtree, root, True)
        (raw_dir / "scan_001.jpg").write_bytes(b"fake-image")
        cache_path = root / "name_review.json"
        cache_path.write_text(
            json.dumps(
                [
                    {
                        "filename": "scan_001.jpg",
                        "student_id": "s1",
                        "name_reviewed": True,
                        "quiz_version": None,
                    }
                ]
            ),
            encoding="utf-8",
        )
        fresh = PagePrediction(
            filename="scan_001.jpg",
            student_id="wrong-model-name",
            confidence=0.91,
            name_visible=True,
            raw_name="Model OCR",
            reasoning="Printed Version B.",
            quiz_version="B",
        )

        with (
            patch.object(server, "RAW_SCANS_DIR", raw_dir),
            patch.object(server, "load_roster", return_value=_variant_session().roster),
            patch.object(server, "name_review_cache_path", return_value=cache_path),
            patch.object(server, "create_vlm_client", return_value=type("FakeClient", (), {"model": "test"})()),
            patch.object(server, "classify_page", return_value=fresh),
        ):
            response = client.get(
                "/name_review?quiz=P250_F26_quiz1_ABC&refresh=1"
            )

        self.assertEqual(response.status_code, 200)
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        self.assertEqual(cached[0]["student_id"], "s1")
        self.assertEqual(cached[0]["quiz_version"], "B")
        self.assertTrue(cached[0]["name_reviewed"])

    def test_strip_page_shows_prompt_but_not_legacy_answer_key(self) -> None:
        client = TestClient(server.app)
        session = _variant_session()
        segments = [
            {
                "unit_id": 1,
                "page_filename": "p250_quiz_1_ada_example_p01.jpg",
                "page_index": 0,
                "bbox": [0.1, 0.2, 0.9, 0.4],
                "present": True,
                "comment": None,
                "confidence": 0.9,
            }
        ]
        with (
            patch("app.routes.problem_strips.get_session", return_value=session),
            patch("app.routes.problem_strips.ensure_segments_cached", return_value=segments),
            patch("app.routes.problem_strips.question_image_url", return_value=None),
        ):
            response = client.get("/problem/Q1/strips")
        self.assertEqual(response.status_code, 200)
        self.assertIn("Version A", response.text)
        self.assertIn("Version A prompt", response.text)
        self.assertIn("No supplied answer key", response.text)
        self.assertIn("Rotate / view page", response.text)
        self.assertIn('id="page-viewer-canvas"', response.text)
        self.assertNotIn("Version A answer", response.text)
        self.assertNotIn("TF chooses partial credit", response.text)
        self.assertEqual(response.text.count("Version A prompt"), 1)

    def test_strip_page_filters_students_and_prompt_by_version(self) -> None:
        client = TestClient(server.app)
        session = _variant_session()
        session.roster.append(RosterEntry(student_id="s2", name="Example, Bea"))
        session.work_units.append(
            WorkUnit(
                unit_id=2,
                page_filenames=["bea.jpg"],
                student_id="s2",
                quiz_version="B",
                per_problem={"Q1": None},
                per_problem_comments={"Q1": None},
                comment=None,
            )
        )
        segments = [
            {
                "unit_id": wu.unit_id,
                "page_filename": wu.page_filenames[0],
                "page_index": 0,
                "bbox": [0.1, 0.2, 0.9, 0.4],
                "present": True,
                "comment": None,
                "confidence": 0.9,
            }
            for wu in session.work_units
        ]
        with (
            patch("app.routes.problem_strips.get_session", return_value=session),
            patch("app.routes.problem_strips.ensure_segments_cached", return_value=segments),
            patch(
                "app.routes.problem_strips.question_image_url",
                return_value="/rubric_assets/P250_F26_quiz1_ABC/Q1_B.png",
            ),
        ):
            response = client.get("/problem/Q1/strips?version=B")

        self.assertEqual(response.status_code, 200)
        self.assertIn("Example, Bea", response.text)
        self.assertNotIn("Example, Ada", response.text)
        self.assertNotIn("Version B prompt", response.text)
        self.assertNotIn("Version A prompt", response.text)
        self.assertEqual(
            response.text.count("/rubric_assets/P250_F26_quiz1_ABC/Q1_B.png"),
            2,
        )

    def test_save_next_advances_to_next_version(self) -> None:
        client = TestClient(server.app)
        session = _variant_session()
        with (
            patch("app.routes.problem_strips.get_session", return_value=session),
            patch("app.routes.problem_strips.persist_session"),
        ):
            response = client.post(
                "/problem/Q1/strips",
                data={"active_version": "A", "action": "save_next"},
                follow_redirects=False,
            )
        self.assertEqual(response.status_code, 303)
        self.assertEqual(response.headers["location"], "/problem/Q1/strips?version=B")

    def test_version_filtered_save_preserves_hidden_comments_and_zero_displays(self) -> None:
        client = TestClient(server.app)
        session = _variant_session()
        session.roster.append(RosterEntry(student_id="s2", name="Example, Bea"))
        session.work_units.append(
            WorkUnit(
                unit_id=2,
                page_filenames=["bea.jpg"],
                student_id="s2",
                quiz_version="B",
                per_problem={"Q1": 1.0},
                per_problem_comments={"Q1": "Keep this hidden-version comment."},
                comment=None,
            )
        )
        with (
            patch("app.routes.problem_strips.get_session", return_value=session),
            patch("app.routes.problem_strips.persist_session"),
        ):
            saved = client.post(
                "/problem/Q1/strips",
                data={
                    "active_version": "A",
                    "action": "save_back",
                    "score_1": "0",
                    "comment_1": "No credit.",
                },
                follow_redirects=False,
            )

        self.assertEqual(saved.status_code, 303)
        self.assertEqual(session.work_units[0].per_problem["Q1"], 0.0)
        self.assertEqual(session.work_units[0].per_problem_comments["Q1"], "No credit.")
        self.assertEqual(
            session.work_units[1].per_problem_comments["Q1"],
            "Keep this hidden-version comment.",
        )

        segments = [
            {
                "unit_id": 1,
                "page_filename": session.work_units[0].page_filenames[0],
                "page_index": 0,
                "bbox": [0.1, 0.2, 0.9, 0.4],
                "present": True,
                "comment": None,
                "confidence": 0.9,
            },
            {
                "unit_id": 2,
                "page_filename": "bea.jpg",
                "page_index": 0,
                "bbox": [0.1, 0.2, 0.9, 0.4],
                "present": True,
                "comment": None,
                "confidence": 0.9,
            },
        ]
        with (
            patch("app.routes.problem_strips.get_session", return_value=session),
            patch("app.routes.problem_strips.ensure_segments_cached", return_value=segments),
        ):
            rendered = client.get("/problem/Q1/strips?version=A")
        self.assertIn('name="score_1"', rendered.text)
        self.assertIn('value="0.0"', rendered.text)
        self.assertIn('min="0"', rendered.text)
        self.assertIn('max="2.0"', rendered.text)

    def test_strip_save_rejects_scores_outside_problem_bounds(self) -> None:
        client = TestClient(server.app)

        for raw_score in ("-0.01", "2.01", "nan", "inf"):
            with self.subTest(raw_score=raw_score):
                session = _variant_session()
                with (
                    patch("app.routes.problem_strips.get_session", return_value=session),
                    patch("app.routes.problem_strips.persist_session") as persist,
                ):
                    response = client.post(
                        "/problem/Q1/strips",
                        data={"active_version": "A", "score_1": raw_score},
                    )

                self.assertEqual(response.status_code, 400)
                self.assertIn("must be between 0 and 2", response.text)
                self.assertIsNone(session.work_units[0].per_problem["Q1"])
                persist.assert_not_called()

    def test_blank_pdf_reference_extractor_writes_all_version_problem_images(self) -> None:
        root = Path("working") / "test_tmp" / f"reference_extract_{uuid4().hex}"
        root.mkdir(parents=True, exist_ok=True)
        self.addCleanup(shutil.rmtree, root, True)
        pdf_path = root / "blank.pdf"
        import fitz

        document = fitz.open()
        for version in ("A", "B", "C"):
            page = document.new_page(width=612, height=792)
            page.insert_text((72, 72), f"Synthetic Version {version}")
        document.save(pdf_path)
        document.close()

        written = extract_reference_images(
            pdf_path,
            root / "assets",
            "synthetic_quiz",
            ["A", "B", "C"],
            P250_QUIZ1_REGIONS,
            scale=1,
        )
        self.assertEqual(len(written), 15)
        self.assertTrue((root / "assets" / "synthetic_quiz" / "Q3_C.png").is_file())

    def test_blank_pdf_reference_extractor_supports_two_pages_per_version(self) -> None:
        root = Path("working") / "test_tmp" / f"reference_extract_{uuid4().hex}"
        root.mkdir(parents=True, exist_ok=True)
        self.addCleanup(shutil.rmtree, root, True)
        pdf_path = root / "blank.pdf"
        import fitz

        document = fitz.open()
        for page_number in range(6):
            page = document.new_page(width=612, height=792)
            page.insert_text((72, 72), f"Synthetic page {page_number + 1}")
        document.save(pdf_path)
        document.close()

        written = extract_reference_images(
            pdf_path,
            root / "assets",
            "synthetic_quiz",
            ["A", "B", "C"],
            P250_TRUTH_TABLE_QUIZ_REGIONS,
            scale=1,
            pages_per_version=2,
            page_offsets={"Q1": 0, "Q2": 0, "Q3": 1, "EC1": 1},
        )
        self.assertEqual(len(written), 12)
        self.assertTrue((root / "assets" / "synthetic_quiz" / "Q3_C.png").is_file())

    def test_reference_pin_persists_per_problem_and_version(self) -> None:
        client = TestClient(server.app)
        session = _variant_session()
        with (
            patch("app.routes.problem_strips.get_session", return_value=session),
            patch("app.routes.problem_strips.persist_session") as persist,
        ):
            response = client.post(
                "/problem/Q1/reference-pin",
                data={"action": "pin", "quiz_version": "Version A", "unit_id": "1"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["reference_pins"], {"A": 1})
        self.assertEqual(session.reference_pins, {"Q1": {"A": 1}})
        persist.assert_called_once()

    def test_reference_pin_can_be_cleared_without_affecting_other_versions(self) -> None:
        client = TestClient(server.app)
        session = _variant_session()
        session.reference_pins = {"Q1": {"A": 1, "B": 99}}
        with (
            patch("app.routes.problem_strips.get_session", return_value=session),
            patch("app.routes.problem_strips.persist_session"),
        ):
            response = client.post(
                "/problem/Q1/reference-pin",
                data={"action": "clear", "quiz_version": "A"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["reference_pins"], {"B": 99})
        self.assertEqual(session.reference_pins, {"Q1": {"B": 99}})

    def test_name_review_persists_corrected_version(self) -> None:
        client = TestClient(server.app)
        root = Path("working") / "test_tmp" / f"variant_name_review_{uuid4().hex}"
        root.mkdir(parents=True, exist_ok=True)
        self.addCleanup(shutil.rmtree, root, True)
        (root / "phone.jpg").write_bytes(b"fake-image")
        session = _variant_session()
        saved = []

        with (
            patch.object(server, "RAW_SCANS_DIR", root),
            patch.object(server, "load_roster", return_value=session.roster),
            patch.object(server, "load_grading_key", return_value=session.grading_key),
            patch.object(server, "save_session", side_effect=lambda value, *args: saved.append(value)),
        ):
            response = client.post(
                "/name_review",
                data={
                    "quiz_id": "P250_F26_quiz1_ABC",
                    "raw_scans_subdir": "",
                    "page_count": "1",
                    "filename_0": "phone.jpg",
                    "student_0": "s1",
                    "quiz_version_0": "Version B",
                    "next_action": "grade",
                },
                follow_redirects=False,
            )

        self.assertEqual(response.status_code, 303)
        self.assertEqual(saved[0].work_units[0].quiz_version, "B")


class TfBundleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(server.app)
        self.root = Path("working") / "test_tmp" / f"tf_bundle_{uuid4().hex}"
        self.raw_dir = self.root / "raw_scans"
        self.grades_dir = self.root / "grades"
        self.assets_dir = self.root / "rubric_assets"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.grades_dir.mkdir(parents=True, exist_ok=True)
        self.assets_dir.mkdir(parents=True, exist_ok=True)
        self.session = _variant_session()
        self.scan_name = self.session.work_units[0].page_filenames[0]
        (self.raw_dir / self.scan_name).write_bytes(b"fake-jpeg")

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def _write_segments(self) -> None:
        (self.grades_dir / "segments_Q1.json").write_text(
            json.dumps(
                [
                    {
                        "unit_id": 1,
                        "page_filename": self.scan_name,
                        "page_index": 0,
                        "bbox": [0.1, 0.2, 0.9, 0.4],
                        "present": True,
                        "comment": None,
                        "confidence": 0.9,
                    }
                ]
            ),
            encoding="utf-8",
        )

    def test_bundle_contains_model_free_grading_state(self) -> None:
        self._write_segments()
        self.session.reference_pins = {"Q1": {"A": 1}}
        question_asset = self.assets_dir / "P250_F26_quiz1_ABC" / "Q1_A.png"
        question_asset.parent.mkdir(parents=True, exist_ok=True)
        question_asset.write_bytes(b"blank-question-image")
        with (
            patch.object(exam_exports_routes, "get_session", return_value=self.session),
            patch.object(exam_exports_routes, "RAW_SCANS_DIR", self.raw_dir),
            patch.object(exam_exports_routes, "GRADES_DIR", self.grades_dir),
            patch.object(exam_exports_routes, "RUBRIC_ASSETS_DIR", self.assets_dir),
        ):
            response = self.client.get("/export_tf_bundle")

        self.assertEqual(response.status_code, 200)
        with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
            names = set(zf.namelist())
            root = "P250_F26_quiz1_ABC_tf_bundle"
            self.assertIn(f"{root}/data/grades/session.json", names)
            self.assertIn(f"{root}/data/grades/segments_Q1.json", names)
            self.assertIn(f"{root}/data/raw_scans/{self.scan_name}", names)
            self.assertIn(f"{root}/START_TF_GRADER.ps1", names)
            launcher = zf.read(f"{root}/START_TF_GRADER.ps1").decode()
            self.assertIn('TF_GRADER_MODE = "1"', launcher)
            self.assertIn("[int]$Port = 5000", launcher)
            self.assertIn('.venv\\Scripts\\python.exe', launcher)
            self.assertIn("Test-Path -LiteralPath $venvPython", launcher)
            self.assertIn("Get-Command py", launcher)
            self.assertIn(f"{root}/START_TF_GRADER.command", names)
            mac_launcher_info = zf.getinfo(f"{root}/START_TF_GRADER.command")
            mac_launcher = zf.read(mac_launcher_info).decode()
            self.assertIn('export TF_GRADER_MODE="1"', mac_launcher)
            self.assertIn('PORT="${PORT:-5001}"', mac_launcher)
            self.assertIn('VENV_PYTHON="$CODE_DIR/.venv/bin/python"', mac_launcher)
            self.assertTrue((mac_launcher_info.external_attr >> 16) & 0o100)
            readme = zf.read(f"{root}/READ_ME_FIRST.txt").decode()
            self.assertIn("python -m venv .venv", readme)
            self.assertIn(r".\.venv\Scripts\python.exe -m pip install", readme)
            self.assertIn("./.venv/bin/python -m pip install", readme)
            self.assertIn(
                f"{root}/data/rubric_assets/P250_F26_quiz1_ABC/Q1_A.png",
                names,
            )
            manifest = json.loads(zf.read(f"{root}/bundle_manifest.json"))
            self.assertFalse(manifest["local_model_required"])
            self.assertFalse(manifest["answer_keys_included"])
            self.assertEqual(manifest["version_counts"], {"A": 1})
            self.assertEqual(manifest["blank_question_image_count"], 1)
            bundled_session = json.loads(zf.read(f"{root}/data/grades/session.json"))
            self.assertEqual(bundled_session["reference_pins"], {"Q1": {"A": 1}})
            problem = bundled_session["grading_key"]["problems"][0]
            self.assertNotIn("answer_key", problem)
            self.assertNotIn("answer_key", problem["variants"]["A"])
            self.assertNotIn("rubric_notes", problem)
            self.assertNotIn("rubric_notes", problem["variants"]["A"])
            rubric_text = zf.read(f"{root}/data/rubric/P250_F26_quiz1_ABC.yaml").decode()
            self.assertNotIn("answer_key", rubric_text)
            self.assertNotIn("rubric_notes", rubric_text)
            self.assertNotIn("Version A answer", rubric_text)

    def test_bundle_refuses_missing_question_locations(self) -> None:
        with (
            patch.object(exam_exports_routes, "get_session", return_value=self.session),
            patch.object(exam_exports_routes, "RAW_SCANS_DIR", self.raw_dir),
            patch.object(exam_exports_routes, "GRADES_DIR", self.grades_dir),
        ):
            response = self.client.get("/export_tf_bundle")
        self.assertEqual(response.status_code, 400)
        self.assertIn("no saved question-location guesses", response.json()["detail"])


if __name__ == "__main__":
    unittest.main()
