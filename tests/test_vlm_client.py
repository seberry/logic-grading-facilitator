import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from app.vlm_client import LMStudioQwenClient


class _FakeResponse:
    text = ""

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return {
            "choices": [
                {
                    "message": {
                        "content": "[{\"wffstr\":\"A\",\"jstr\":\"\"}]",
                        "reasoning_content": "brief thinking",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 3},
        }


class LMStudioQwenClientTests(unittest.TestCase):
    def test_chat_merges_extra_payload_and_records_metadata(self) -> None:
        captured = {}

        def fake_post(url, json, timeout):
            captured["url"] = url
            captured["json"] = json
            captured["timeout"] = timeout
            return _FakeResponse()

        client = LMStudioQwenClient(base_url="http://example.test/v1", model="base-model", timeout=5)
        with patch("app.vlm_client.requests.post", side_effect=fake_post):
            content = client.chat(
                [{"role": "user", "content": "hi"}],
                model="override-model",
                timeout=9,
                max_tokens=512,
                extra_payload={"chat_template_kwargs": {"enable_thinking": False}},
            )

        self.assertEqual(content, "[{\"wffstr\":\"A\",\"jstr\":\"\"}]")
        self.assertEqual(captured["url"], "http://example.test/v1/chat/completions")
        self.assertEqual(captured["timeout"], 9)
        self.assertEqual(captured["json"]["model"], "override-model")
        self.assertEqual(captured["json"]["max_tokens"], 512)
        self.assertEqual(captured["json"]["max_completion_tokens"], 512)
        self.assertEqual(captured["json"]["chat_template_kwargs"], {"enable_thinking": False})
        self.assertEqual(client.last_thinking, "brief thinking")
        self.assertEqual(client.last_response_metadata["thinking_chars"], len("brief thinking"))
        self.assertEqual(client.last_response_metadata["finish_reason"], "stop")
        self.assertEqual(client.last_response_metadata["sent_extra_payload_keys"], ["chat_template_kwargs"])

    def test_build_single_user_image_message_has_no_system_message(self) -> None:
        image_path = Path("working") / "test_vlm_client" / "tiny.jpg"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (8, 8), "white").save(image_path)

        client = LMStudioQwenClient()
        messages = client.build_single_user_image_message(image_path, "Read this.")

        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["role"], "user")
        content = messages[0]["content"]
        self.assertEqual(content[0], {"type": "text", "text": "Read this."})
        self.assertEqual(content[1]["type"], "image_url")
        self.assertTrue(content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,"))

    def test_name_classifier_requests_short_structured_json_without_thinking(self) -> None:
        image_path = Path("working") / "test_vlm_client" / "name_page.jpg"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (8, 8), "white").save(image_path)
        captured = {}

        def fake_post(url, json, timeout):
            captured["json"] = json
            return _FakeResponse()

        client = LMStudioQwenClient(base_url="http://example.test/v1")
        with patch("app.vlm_client.requests.post", side_effect=fake_post):
            client.call_for_page(image_path, "s1, Student, Sample")

        payload = captured["json"]
        self.assertEqual(payload["max_tokens"], 160)
        self.assertEqual(payload["chat_template_kwargs"], {"enable_thinking": False})
        self.assertEqual(payload["response_format"]["type"], "json_schema")
        schema = payload["response_format"]["json_schema"]["schema"]
        self.assertFalse(schema["additionalProperties"])
        self.assertNotIn("reasoning", schema["properties"])


if __name__ == "__main__":
    unittest.main()
