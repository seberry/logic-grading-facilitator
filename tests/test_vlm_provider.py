from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest
import requests

from app.vlm_provider import IUClient, IU_BASE_URL, IU_GEMMA, IU_QWEN, create_vlm_client
from app.segmentation_core import segment_page_for_problem


class Response:
    status_code = 200
    text = ""

    def json(self):
        return {"model": IU_GEMMA, "choices": [{"message": {"content": '{"ok":true}'},
                 "finish_reason": "stop"}], "usage": {"completion_tokens": 5}}


def test_default_is_gemma_and_qwen_is_explicit_without_reading_credentials():
    with patch.dict(os.environ, {}, clear=True), patch("app.vlm_provider.reallms_api_key") as key:
        assert create_vlm_client().model == IU_GEMMA
        assert create_vlm_client("qwen").model == IU_QWEN
        assert create_vlm_client("local").base_url == "http://127.0.0.1:1234/v1"
        key.assert_not_called()
    with pytest.raises(ValueError):
        create_vlm_client("https://untrusted.example")


def test_iu_transport_pins_endpoint_authenticates_and_disables_redirects():
    with patch("app.vlm_provider.requests.post", return_value=Response()) as post:
        client = IUClient(key="test-secret")
        assert client.chat([{"role": "user", "content": "hello"}]) == '{"ok":true}'
    assert post.call_args.args[0] == IU_BASE_URL + "/chat/completions"
    kwargs = post.call_args.kwargs
    assert kwargs["headers"]["Authorization"] == "Bearer test-secret"
    assert kwargs["allow_redirects"] is False
    assert kwargs["json"]["model"] == IU_GEMMA
    assert kwargs["json"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert "test-secret" not in str(client.last_response_metadata)


def test_http_errors_do_not_expose_provider_echoes():
    response = Response()
    response.status_code = 429
    response.text = "secret-name-and-roster"
    with patch("app.vlm_provider.requests.post", return_value=response):
        with pytest.raises(requests.HTTPError) as error:
            IUClient(key="test-secret").chat([])
    assert "429" in str(error.value)
    assert "secret-name" not in str(error.value)


def test_truncated_completion_is_rejected():
    response = Response()
    response.json = lambda: {"choices": [{"message": {"content": "partial"}, "finish_reason": "length"}]}
    with patch("app.vlm_provider.requests.post", return_value=response):
        with pytest.raises(RuntimeError, match="incomplete"):
            IUClient(key="test-secret").chat([])


def test_segmentation_errors_are_not_disguised_as_missing_questions():
    class FailedClient:
        def chat_with_image_prompt(self, **kwargs):
            raise RuntimeError("private roster echo")
    with pytest.raises(RuntimeError, match="preserved") as error:
        segment_page_for_problem(Path("fake.png"), "Q1", "fake question", FailedClient())
    assert "private roster" not in str(error.value)


def test_invalid_segmentation_bounds_are_rejected():
    class InvalidClient:
        def chat_with_image_prompt(self, **kwargs):
            return '{"present":true,"bbox":[0,0,1000,1000]}'
    with pytest.raises(RuntimeError, match="bounds"):
        segment_page_for_problem(Path("fake.png"), "Q1", "fake question", InvalidClient())


def test_failed_location_refresh_keeps_existing_cache(tmp_path):
    from types import SimpleNamespace
    from app.routes import problem_strips
    cache = tmp_path / "segments_Q1.json"
    original = '[{"unit_id":1,"present":true,"bbox":[0,0,1,1]}]'
    cache.write_text(original, encoding="utf-8")
    session = SimpleNamespace(work_units=[SimpleNamespace(unit_id=1, page_filenames=["fake.jpg"])])
    spec = SimpleNamespace(id="Q1", label="Fake question")
    with patch.object(problem_strips, "create_vlm_client", return_value=SimpleNamespace(model=IU_GEMMA)), patch.object(
        problem_strips, "segment_page_for_problem", side_effect=RuntimeError("provider unavailable")
    ):
        with pytest.raises(RuntimeError):
            problem_strips._compute_segments_data(session, spec, cache)
    assert cache.read_text(encoding="utf-8") == original
