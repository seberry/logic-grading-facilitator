"""Prompt and model-call helpers for verifier-guided TFL row rereads."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping


def env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def env_int(name: str, default: int | None = None) -> int | None:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def get_27b_model(model: str | None = None) -> str | None:
    explicit = (model or "").strip()
    if explicit:
        return explicit
    env_model = os.getenv("TFL_FITCH_VERIFIER_REREAD_MODEL") or os.getenv("LMSTUDIO_MODEL_27B")
    if env_model is None:
        env_model = "qwen/qwen3.6-27b"
    env_model = env_model.strip()
    return env_model or None


def get_27b_extra_payload() -> dict[str, object]:
    extra_payload: dict[str, object] = {}
    if env_flag("TFL_FITCH_27B_ENABLE_THINKING_FALSE"):
        extra_payload["chat_template_kwargs"] = {"enable_thinking": False}
    return extra_payload


def build_single_line_reread_prompt(suspicion: Mapping[str, Any], *, target_kind: str = "rejected") -> str:
    issues = "; ".join(str(item) for item in suspicion.get("issues", []))
    target_label = "rejected verifier line" if target_kind == "rejected" else "line cited by the rejected line"
    return (
        "You are rereading one cropped row from a handwritten Fitch proof because "
        f"the first OCR pass produced a verifier warning. Target: {target_label}.\n"
        f"Final proof line: {suspicion.get('line')}.\n"
        f"First OCR formula: {suspicion.get('wffstr', '')}\n"
        f"First OCR justification: {suspicion.get('jstr', '')}\n"
        f"Verifier warning: {issues}\n\n"
        "Read only the visible text in this single row image. Return exactly one JSON object "
        "with keys \"line\", \"wffstr\", and \"jstr\". Put only the formula in "
        "\"wffstr\" and only the printed rule/citations in \"jstr\". Use the visible "
        "left-gutter line number if present. Preserve visible citation numbers exactly "
        "as written. Do not prove, repair, or validate the line semantically. Do not "
        "infer a valid proof from logic alone. Return JSON only."
    )


def json_from_text(text: str) -> Any:
    content = text.strip()
    if "```json" in content:
        content = content.split("```json", 1)[1].split("```", 1)[0]
    elif "```" in content:
        content = content.split("```", 1)[1].split("```", 1)[0]
    return json.loads(content)


def call_single_line_reread(
    *,
    client: Any,
    row_path: Path,
    prompt: str,
    model: str | None,
    timeout: int,
    max_tokens: int,
    extra_payload: Mapping[str, object] | None = None,
) -> tuple[str, Any]:
    if (
        model
        and env_flag("TFL_FITCH_27B_SINGLE_USER_MESSAGE")
        and hasattr(client, "build_single_user_image_message")
    ):
        messages = client.build_single_user_image_message(row_path, prompt)
    else:
        messages = client.build_messages_for_prompt(row_path, prompt)
    chat_kwargs: dict[str, Any] = {
        "model": model,
        "timeout": timeout,
        "max_tokens": max_tokens,
    }
    if extra_payload:
        chat_kwargs["extra_payload"] = extra_payload
    raw = client.chat(messages, **chat_kwargs)
    return raw, json_from_text(raw)
