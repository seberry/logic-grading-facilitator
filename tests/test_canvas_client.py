import unittest
from pathlib import Path
import shutil
import uuid
from unittest.mock import Mock, patch

import httpx

from app.canvas_client import CanvasClient


TEST_TMP_ROOT = Path("working") / "test_canvas_client"


def _submission_payload() -> list[dict]:
    return [
        {
            "workflow_state": "submitted",
            "user": {
                "id": 101,
                "sis_user_id": "s1",
                "name": "Berry, Sharon",
            },
            "attachments": [
                {
                    "url": "https://files.example.test/submission.pdf",
                    "filename": "submission.pdf",
                }
            ],
        }
    ]


class CanvasClientDownloadTests(unittest.TestCase):
    def _make_temp_dir(self) -> Path:
        path = TEST_TMP_ROOT / uuid.uuid4().hex
        path.mkdir(parents=True, exist_ok=True)
        return path

    def test_download_assignment_submissions_retries_request_errors(self) -> None:
        client = CanvasClient(base_url="https://canvas.example.test", api_key="test-token")

        temp_dir = self._make_temp_dir()
        try:
            with patch.object(client, "_make_request", return_value=_submission_payload()):
                with patch.object(
                    client,
                    "_download_attachment_with_retry",
                    side_effect=[
                        httpx.ReadError("connection reset", request=httpx.Request("GET", "https://files.example.test/submission.pdf")),
                        b"pdf-bytes",
                    ],
                ) as download_mock:
                    with patch.object(client, "_reset_http_client") as reset_mock:
                        with patch("app.canvas_client.time.sleep"):
                            student_files, canvas_user_ids, download_warnings = client.download_assignment_submissions(
                                course_id=1,
                                assignment_id=2,
                                output_dir=temp_dir,
                            )
        finally:
            client.close()
            shutil.rmtree(temp_dir, ignore_errors=True)

        self.assertEqual(canvas_user_ids, {"s1": 101})
        self.assertEqual(download_warnings, [])
        self.assertIn("s1", student_files)
        self.assertEqual(len(student_files["s1"]), 1)
        self.assertEqual(student_files["s1"][0].name, "submission.pdf")
        self.assertEqual(download_mock.call_count, 2)
        reset_mock.assert_called_once()

    def test_download_assignment_submissions_records_warning_after_retries(self) -> None:
        client = CanvasClient(base_url="https://canvas.example.test", api_key="test-token")

        temp_dir = self._make_temp_dir()
        try:
            with patch.object(client, "_make_request", return_value=_submission_payload()):
                with patch.object(
                    client,
                    "_download_attachment_with_retry",
                    side_effect=[
                        httpx.ReadError("connection reset", request=httpx.Request("GET", "https://files.example.test/submission.pdf")),
                        httpx.ReadError("connection reset", request=httpx.Request("GET", "https://files.example.test/submission.pdf")),
                    ],
                ) as download_mock:
                    with patch.object(client, "_reset_http_client") as reset_mock:
                        with patch("app.canvas_client.time.sleep"):
                            student_files, canvas_user_ids, download_warnings = client.download_assignment_submissions(
                                course_id=1,
                                assignment_id=2,
                                output_dir=temp_dir,
                            )
        finally:
            client.close()
            shutil.rmtree(temp_dir, ignore_errors=True)

        self.assertEqual(student_files, {})
        self.assertEqual(canvas_user_ids, {"s1": 101})
        self.assertEqual(len(download_warnings), 1)
        self.assertEqual(download_warnings[0]["student_id"], "s1")
        self.assertEqual(download_warnings[0]["filename"], "submission.pdf")
        self.assertIn("connection reset", download_warnings[0]["error"])
        self.assertEqual(download_mock.call_count, 2)
        reset_mock.assert_called_once()


if __name__ == "__main__":
    unittest.main()
