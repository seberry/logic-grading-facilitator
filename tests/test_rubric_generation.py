from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from app import server
from app.models import GradingKey, ProblemSpec
from app.rubric_generation import (
    RubricGenerationError,
    configured_rubric_providers,
    extract_assessment_text,
    generate_rubric_draft,
)


def _draft_json() -> str:
    return json.dumps(
        {
            "quiz_id": "quiz3",
            "problems": [
                {
                    "id": "Q1",
                    "label": "Truth table",
                    "points": 10,
                    "rubric_notes": "Give partial credit for a substantially correct table.",
                    "variants": {"A": {"prompt": "Version A prompt"}},
                },
                {
                    "id": "EC",
                    "label": "Extra credit",
                    "points": 3,
                    "extra_credit": True,
                },
            ],
        }
    )


class RubricGenerationCoreTests(unittest.IsolatedAsyncioTestCase):
    def test_reports_only_configured_provider_names(self) -> None:
        with patch.dict(
            os.environ,
            {"OPENAI_API_KEY": "secret", "DEEPSEEK_API_KEY": "secret"},
            clear=True,
        ):
            self.assertEqual(
                configured_rubric_providers(),
                [
                    {"id": "openai", "label": "OpenAI"},
                    {"id": "deepseek", "label": "DeepSeek"},
                ],
            )

    def test_extracts_tex_without_storing_it(self) -> None:
        text = extract_assessment_text("quiz.tex", b"\\question[3] Truth table")
        self.assertEqual(text, "\\question[3] Truth table")

    async def test_openai_uses_responses_schema_and_validates_result(self) -> None:
        response = {
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": _draft_json()}],
                }
            ]
        }
        post = AsyncMock(return_value=response)
        with (
            patch.dict(
                os.environ,
                {
                    "OPENAI_API_KEY": "secret",
                    "OPENAI_MODEL": "test-model",
                    "OPENAI_BASE_URL": "https://example.test/v1",
                },
                clear=True,
            ),
            patch("app.rubric_generation._post_json", post),
        ):
            rubric = await generate_rubric_draft("exam text", "ten points", "openai")

        self.assertEqual(rubric.quiz_id, "quiz3")
        self.assertTrue(rubric.problems[1].extra_credit)
        kwargs = post.await_args.kwargs
        self.assertEqual(post.await_args.args[0], "https://example.test/v1/responses")
        self.assertFalse(kwargs["payload"]["store"])
        self.assertEqual(
            kwargs["payload"]["text"]["format"]["type"],
            "json_schema",
        )

    async def test_deepseek_uses_json_mode_and_validates_result(self) -> None:
        post = AsyncMock(
            return_value={"choices": [{"message": {"content": _draft_json()}}]}
        )
        with (
            patch.dict(
                os.environ,
                {"DEEPSEEK_API_KEY": "secret", "DEEPSEEK_MODEL": "test-model"},
                clear=True,
            ),
            patch("app.rubric_generation._post_json", post),
        ):
            rubric = await generate_rubric_draft("exam text", "ten points", "deepseek")

        self.assertEqual(rubric.problems[0].points, 10)
        kwargs = post.await_args.kwargs
        self.assertEqual(
            kwargs["payload"]["response_format"],
            {"type": "json_object"},
        )
        self.assertIn("JSON", kwargs["payload"]["messages"][1]["content"])

    async def test_invalid_provider_output_is_a_safe_error(self) -> None:
        post = AsyncMock(return_value={"output_text": '{"quiz_id":"x","problems":[]}'})
        with (
            patch.dict(
                os.environ,
                {"OPENAI_API_KEY": "secret", "OPENAI_MODEL": "test-model"},
                clear=True,
            ),
            patch("app.rubric_generation._post_json", post),
        ):
            with self.assertRaisesRegex(RubricGenerationError, "not a valid rubric"):
                await generate_rubric_draft("exam text", "ten points", "openai")


class RubricGenerationRouteTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(server.app)

    def test_generated_draft_is_reviewed_before_save(self) -> None:
        generated = GradingKey(
            quiz_id="generated_quiz",
            problems=[
                ProblemSpec(id="Q1", label="Regular", points=10),
                ProblemSpec(id="EC", label="Extra", points=3, extra_credit=True),
            ],
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with (
                patch.object(server, "DATA_DIR", root),
                patch.object(server, "RUBRIC_DIR", root / "rubric"),
                patch.object(server, "DEFAULT_ROSTER_PATH", root / "roster.csv"),
                patch.object(server, "DEFAULT_GRADING_KEY_PATH", root / "rubric" / "quiz1.yaml"),
                patch.object(server, "_has_existing_session_state", return_value=False),
                patch.object(
                    server,
                    "generate_rubric_draft",
                    new=AsyncMock(return_value=generated),
                ),
            ):
                response = self.client.post(
                    "/setup/generate_rubric",
                    data={"provider": "openai", "instructor_requests": "Ten points."},
                    files={"assessment_file": ("quiz.tex", b"quiz source", "text/plain")},
                )

                self.assertEqual(response.status_code, 200)
                self.assertIn("Generated an editable draft", response.text)
                self.assertIn("quiz_id: generated_quiz", response.text)
                self.assertIn("10 regular points and 3 extra-credit points", response.text)
                self.assertIn(
                    '<details class="rubric-editor" id="rubric-editor" open>',
                    response.text,
                )
                self.assertFalse((root / "rubric" / "generated_quiz.yaml").exists())


if __name__ == "__main__":
    unittest.main()
