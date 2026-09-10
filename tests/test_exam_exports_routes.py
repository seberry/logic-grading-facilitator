import unittest
import csv
import io
import os
import tempfile
import zipfile
from pathlib import Path
from unittest.mock import patch

import fitz  # type: ignore
from fastapi.testclient import TestClient
from PIL import Image

from app import server
from app.models import GradingKey, ProblemSpec, RosterEntry, Session, WorkUnit
from app.routes import exam_exports as exam_exports_routes


def _build_session() -> Session:
    return Session(
        roster=[RosterEntry(student_id="s1", name="Berry, Sharon")],
        grading_key=GradingKey(
            quiz_id="quiz1",
            max_points=10,
            problems=[
                ProblemSpec(id="Q1", label="FOL model", points=4),
                ProblemSpec(id="Q2", label="Proof", points=6),
            ],
        ),
        work_units=[
            WorkUnit(
                unit_id=1,
                student_id="s1",
                page_filenames=["quiz1_berry_sharon_p01.jpg"],
                per_problem={"Q1": 4.0, "Q2": 5.0},
                comment="Good work",
                per_problem_comments={"Q1": "Correct", "Q2": "One minor issue"},
            )
        ],
        unclaimed_pages=[],
    )


class ExamExportsRoutesTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(server.app)

    def test_finish_page_renders_summary(self) -> None:
        session = _build_session()

        with patch.object(exam_exports_routes, "get_session", return_value=session):
            response = self.client.get("/finish")

        self.assertEqual(response.status_code, 200)
        self.assertIn("Berry, Sharon", response.text)
        self.assertIn("Q1", response.text)
        self.assertIn("Q2", response.text)
        self.assertIn("Prepare all question locations", response.text)
        self.assertIn('/problem/${encodeURIComponent(problemId)}/prefetch', response.text)

    def test_tf_finish_page_only_shows_tf_deliverables(self) -> None:
        session = _build_session()

        with (
            patch.object(exam_exports_routes, "get_session", return_value=session),
            patch.dict(os.environ, {"TF_GRADER_MODE": "1"}),
        ):
            response = self.client.get("/finish")

        self.assertEqual(response.status_code, 200)
        self.assertIn("What to return to the instructor", response.text)
        self.assertIn("Download grades CSV", response.text)
        self.assertIn("Download scans + feedback PDFs", response.text)
        self.assertNotIn("Gradebook merge tool", response.text)
        self.assertNotIn("Classwork viewer", response.text)
        self.assertNotIn("Canvas reupload", response.text)
        self.assertNotIn("Download renamed scans", response.text)
        self.assertNotIn("Prepare model-free TF bundle", response.text)
        self.assertNotIn("Prepare all question locations", response.text)
        self.assertNotIn("Back to setup", response.text)

    def test_export_scores_returns_csv(self) -> None:
        session = _build_session()

        with patch.object(exam_exports_routes, "get_session", return_value=session):
            response = self.client.get("/export")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "text/csv; charset=utf-8")
        self.assertIn('filename="quiz1_scores.csv"', response.headers["content-disposition"])
        self.assertIn("student_id,quiz1_score", response.text)
        self.assertIn("s1,9.0", response.text)

    def test_export_scores_can_rescale(self) -> None:
        session = _build_session()

        with patch.object(exam_exports_routes, "get_session", return_value=session):
            response = self.client.get("/export?scale_to=10")

        self.assertEqual(response.status_code, 200)
        self.assertIn('filename="quiz1_scores_out_of_10.csv"', response.headers["content-disposition"])
        self.assertIn("s1,9.0", response.text)

    def test_feedback_zip_contains_named_scan_and_feedback_pdf(self) -> None:
        session = _build_session()
        session.work_units[0].quiz_version = "A"
        session.work_units[0].page_filenames = ["page_1.jpg", "page_2.jpg"]

        with tempfile.TemporaryDirectory() as temp_dir:
            scans_dir = Path(temp_dir)
            Image.new("RGB", (120, 180), "white").save(scans_dir / "page_1.jpg")
            Image.new("RGB", (180, 120), "gray").save(scans_dir / "page_2.jpg")

            with (
                patch.object(exam_exports_routes, "get_session", return_value=session),
                patch.object(exam_exports_routes, "RAW_SCANS_DIR", scans_dir),
            ):
                response = self.client.get("/export_feedback_zip")

        self.assertEqual(response.status_code, 200)
        self.assertIn(
            'filename="quiz1_scans_with_feedback.zip"',
            response.headers["content-disposition"],
        )

        with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
            names = archive.namelist()
            pdf_names = [name for name in names if name.endswith(".pdf")]
            self.assertEqual(
                pdf_names,
                ["quiz1_berry_sharon_s1_vA_graded.pdf"],
            )
            self.assertIn("quiz1_feedback_manifest.csv", names)

            pdf_bytes = archive.read(pdf_names[0])
            with fitz.open(stream=pdf_bytes, filetype="pdf") as document:
                self.assertEqual(document.page_count, 3)
                feedback_text = document[-1].get_text()
                self.assertIn("Berry, Sharon (s1)", feedback_text)
                self.assertIn("Total: 9.0 / 10.0", feedback_text)
                self.assertIn("One minor issue", feedback_text)

            manifest_rows = list(
                csv.DictReader(
                    io.StringIO(
                        archive.read("quiz1_feedback_manifest.csv").decode("utf-8")
                    )
                )
            )
            self.assertEqual(len(manifest_rows), 1)
            self.assertEqual(manifest_rows[0]["scan_pages"], "2")
            self.assertEqual(manifest_rows[0]["feedback_pages"], "1")
            self.assertEqual(manifest_rows[0]["scan_warnings"], "")

    def test_finish_page_can_show_scaled_totals(self) -> None:
        session = _build_session()

        with patch.object(exam_exports_routes, "get_session", return_value=session):
            response = self.client.get("/finish?scale_to=20")

        self.assertEqual(response.status_code, 200)
        self.assertIn("Scaled Total (20.0)", response.text)
        self.assertIn("Showing raw totals out of 10.0 and scaled totals out of 20.0.", response.text)
        self.assertIn(">9.0</td>", response.text)
        self.assertIn(">18.0</td>", response.text)

    def test_finalize_assignment_archives_and_renders_confirmation(self) -> None:
        with patch.object(
            exam_exports_routes,
            "archive_current_data",
            return_value="20260402_exam1_archive",
        ):
            response = self.client.post("/finalize", data={"label": "exam1"})

        self.assertEqual(response.status_code, 200)
        self.assertIn("Assignment finalized", response.text)
        self.assertIn("20260402_exam1_archive", response.text)


if __name__ == "__main__":
    unittest.main()
