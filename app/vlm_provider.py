"""Model selection for grading: IU Gemma by default, explicit Qwen/local choices.

The IU endpoint is pinned so credentials and protected scans cannot be redirected
by a form field. Constructing a client never requires a key; cached/model-free
grading remains usable without API access.
"""
from __future__ import annotations

import os
from pathlib import Path
import time

from dotenv import dotenv_values
import requests

from .vlm_client import LMStudioQwenClient

IU_BASE_URL = "https://reallms.rescloud.iu.edu/direct/v1"
IU_GEMMA = "gemma-4-31B-it"
IU_QWEN = "qwen3.8-27b"
IU_MODELS = (IU_GEMMA, IU_QWEN)


def reallms_api_key() -> str:
    key = os.getenv("REALLMS_API_KEY", "").strip()
    if not key:
        key = str(dotenv_values(Path(__file__).resolve().parents[1] / ".env").get("REALLMS_API_KEY") or "").strip()
    if key:
        return key
    if os.name == "nt":
        import winreg
        for hive, location in ((winreg.HKEY_CURRENT_USER, r"Environment"),
                (winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment")):
            try:
                with winreg.OpenKey(hive, location) as registry:
                    key = str(winreg.QueryValueEx(registry, "REALLMS_API_KEY")[0]).strip()
                if key:
                    return key
            except OSError:
                continue
    raise RuntimeError("Set REALLMS_API_KEY in your private environment or .env file to use IU models.")


class IUClient(LMStudioQwenClient):
    provider = "reallms"

    def __init__(self, model: str = IU_GEMMA, *, key: str | None = None, timeout: int = 180):
        if model not in IU_MODELS:
            raise ValueError("Choose IU Gemma or IU Qwen.")
        super().__init__(base_url=IU_BASE_URL, model=model, timeout=timeout)
        self._key = key
        self.responses: list[dict] = []

    def chat(self, messages, *, model=None, timeout=None, max_tokens=None, extra_payload=None):
        selected = model or self.model
        if selected not in IU_MODELS:
            raise ValueError("Choose IU Gemma or IU Qwen.")
        payload = {"model": selected, "messages": messages, "temperature": 0.0,
                   "max_tokens": max_tokens if max_tokens is not None else 8192,
                   "chat_template_kwargs": {"enable_thinking": False}}
        payload.update(extra_payload or {})
        # Never let generic extra parameters change the selected route or model.
        payload["model"] = selected
        started = time.monotonic()
        try:
            response = requests.post(IU_BASE_URL + "/chat/completions", json=payload,
                headers={"Authorization": "Bearer " + (self._key or reallms_api_key())},
                timeout=timeout or self.timeout, allow_redirects=False)
        except requests.RequestException:
            raise RuntimeError("IU REALLMS could not be reached. Your saved grading data is unchanged; retry later.") from None
        seconds = round(time.monotonic() - started, 3)
        if not 200 <= response.status_code < 300:
            self.responses.append({"status": response.status_code, "seconds": seconds})
            error = requests.HTTPError(f"IU REALLMS returned HTTP {response.status_code}; retry later or select Qwen.")
            error.response = response
            raise error
        try:
            data = response.json()
            choice = data["choices"][0]
            message = choice.get("message") or {}
            content = message.get("content")
        except (ValueError, KeyError, IndexError, TypeError):
            raise RuntimeError("IU REALLMS returned an unreadable completion.") from None
        self.last_thinking = str(message.get("reasoning_content") or "")
        self.last_response_metadata = {"provider": self.provider, "model": selected,
            "returned_model": data.get("model"), "seconds": seconds, "elapsed_s": seconds,
            "thinking_chars": len(self.last_thinking), "content_chars": len(content or ""),
            "finish_reason": choice.get("finish_reason"), "usage": data.get("usage") or {},
            "sent_max_tokens": payload["max_tokens"]}
        self.responses.append({**self.last_response_metadata, "content": content})
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("IU REALLMS returned an empty completion.")
        if choice.get("finish_reason") == "length":
            raise RuntimeError("IU REALLMS reached the response limit; do not use the incomplete result.")
        return content


def create_vlm_client(model: str | None = None) -> LMStudioQwenClient:
    aliases = {"gemma": IU_GEMMA, "qwen": IU_QWEN, **{m: m for m in IU_MODELS}}
    if model:
        if model in aliases:
            return IUClient(aliases[model])
        if model != "local":
            raise ValueError("Choose gemma, qwen, or local.")
    provider = "lmstudio" if model == "local" else os.getenv("GRADING_VLM_PROVIDER", "reallms").strip().lower()
    if provider == "reallms":
        return IUClient(os.getenv("REALLMS_MODEL", IU_GEMMA).strip() or IU_GEMMA)
    if provider == "lmstudio":
        return LMStudioQwenClient()
    raise ValueError("GRADING_VLM_PROVIDER must be reallms or lmstudio.")


def default_grading_base_url() -> str:
    return create_vlm_client().base_url


def default_grading_model() -> str:
    return create_vlm_client().model
