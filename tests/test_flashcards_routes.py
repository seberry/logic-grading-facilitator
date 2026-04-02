import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from app import server
from app.routes import flashcards as flashcards_routes
import shutil
from uuid import uuid4


class FlashcardsRoutesTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(server.app)
        self.flashcards_dir = (
            Path("working") / "test_tmp" / f"flashcards_{uuid4().hex}"
        )
        self.flashcards_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.flashcards_dir, ignore_errors=True)

    def _write_state(self, session_id: str, state: dict) -> None:
        session_dir = self.flashcards_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        (session_dir / "state.json").write_text(
            flashcards_routes.json.dumps(state, indent=2),
            encoding="utf-8",
        )

    def test_flashcards_setup_page_renders(self) -> None:
        response = self.client.get("/flashcards")

        self.assertEqual(response.status_code, 200)
        self.assertIn("Flashcards Dataset Builder", response.text)

    def test_flashcards_review_renders_existing_session(self) -> None:
        session_id = "test_session"
        self._write_state(
            session_id,
            {
                "session_id": session_id,
                "created_at": "2026-04-02T11:30:00",
                "roster_path": None,
                "items": [
                    {
                        "id": 1,
                        "source_image": "sources/example.jpg",
                        "source_text": "",
                        "name": "Berry, Sharon",
                        "facts": "Teaches logic",
                        "bbox": [0.1, 0.2, 0.8, 0.9],
                        "skip": False,
                    }
                ],
            },
        )

        with patch.object(flashcards_routes, "FLASHCARDS_DIR", self.flashcards_dir):
            response = self.client.get(f"/flashcards/review?session={session_id}&index=0")

        self.assertEqual(response.status_code, 200)
        self.assertIn("Flashcards Review", response.text)
        self.assertIn("Berry, Sharon", response.text)

    def test_flashcards_review_submit_updates_state_and_redirects(self) -> None:
        session_id = "update_session"
        self._write_state(
            session_id,
            {
                "session_id": session_id,
                "created_at": "2026-04-02T11:30:00",
                "roster_path": None,
                "items": [
                    {
                        "id": 1,
                        "source_image": "sources/example.jpg",
                        "source_text": "",
                        "name": "",
                        "facts": "",
                        "bbox": None,
                        "skip": False,
                    }
                ],
            },
        )

        with patch.object(flashcards_routes, "FLASHCARDS_DIR", self.flashcards_dir):
            response = self.client.post(
                "/flashcards/review",
                data={
                    "session_id": session_id,
                    "index": 0,
                    "name": "Berry, Sharon",
                    "facts": "Logic professor",
                    "bbox": "0.1,0.2,0.8,0.9",
                    "action": "next",
                },
                follow_redirects=False,
            )

            saved = flashcards_routes._load_flashcards_state(session_id)

        self.assertEqual(response.status_code, 303)
        self.assertEqual(
            response.headers["location"],
            f"/flashcards/review?session={session_id}&index=1",
        )
        self.assertEqual(saved["items"][0]["name"], "Berry, Sharon")
        self.assertEqual(saved["items"][0]["facts"], "Logic professor")
        self.assertEqual(saved["items"][0]["bbox"], [0.1, 0.2, 0.8, 0.9])


if __name__ == "__main__":
    unittest.main()
