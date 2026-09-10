# app/vlm_classifier.py
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from .models import RosterEntry
from .quiz_variants import normalize_quiz_version
from .vlm_client import LMStudioQwenClient


@dataclass
class PagePrediction:
    """
    Page-level name prediction returned by the VLM.

    This is the "duck type" that auto_group_core.group_pages and
    scripts/preview_page_predictions.py expect:

      - filename: str              original filename (basename)
      - student_id: Optional[str]  Canvas ID from roster, or None
      - confidence: float          model's confidence in this match (0–1)
      - name_visible: bool         whether any student name is visible
      - raw_name: Optional[str]    the model's transcription of the name
      - quiz_version: Optional[str] printed quiz/exam version, or None
      - reasoning: str             short explanation from the model
    """

    filename: str
    student_id: Optional[str]
    confidence: float
    name_visible: bool
    raw_name: Optional[str]
    reasoning: str
    quiz_version: Optional[str] = None


def _build_roster_text(roster: List[RosterEntry]) -> str:
    """
    Convert roster entries into the plain-text form expected by
    LMStudioQwenClient.build_messages / call_for_page.
    """
    lines = ["Roster (student_id, Canvas name):"]
    for r in roster:
        lines.append(f"{r.student_id}, {r.name}")
    return "\n".join(lines)


def _strip_fences(text: str) -> str:
    """
    Remove ```...``` markdown fences if the model wraps its JSON.

    Qwen sometimes returns:

        ```json
        { ... }
        ```

    We want just the inner JSON.
    """
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3:
            # drop first and last fence lines
            text = "\n".join(lines[1:-1]).strip()
    return text


def _decode_first_json_object(text: str) -> Optional[dict]:
    """Decode the first complete object, tolerating harmless trailing prose."""
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            value, _ = decoder.raw_decode(text[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def _extract_json_scalar(text: str, field: str, scalar_pattern: str):
    match = re.search(rf'"{re.escape(field)}"\s*:\s*({scalar_pattern})', text, flags=re.I)
    if not match:
        raise ValueError(f"Missing classifier field: {field}")
    return json.loads(match.group(1))


def _recover_classifier_fields(text: str) -> dict:
    """Recover only the five typed fields that precede a malformed explanation."""
    json_string_or_null = r'null|"(?:\\.|[^"\\])*"'
    return {
        "name_visible": _extract_json_scalar(text, "name_visible", r"true|false"),
        "raw_name": _extract_json_scalar(text, "raw_name", json_string_or_null),
        "student_id": _extract_json_scalar(text, "student_id", json_string_or_null),
        "quiz_version": _extract_json_scalar(text, "quiz_version", json_string_or_null),
        "confidence": _extract_json_scalar(
            text,
            "confidence",
            r"-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?",
        ),
        "reasoning": "Recovered classifier fields from malformed model output; verify in name review.",
    }


def _parse_classifier_response(raw: str) -> dict:
    raw_clean = _strip_fences(raw)
    try:
        data = json.loads(raw_clean)
    except json.JSONDecodeError:
        data = _decode_first_json_object(raw_clean)
        if data is None:
            try:
                data = _recover_classifier_fields(raw_clean)
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise RuntimeError(
                    "The local vision model did not return usable classifier JSON "
                    f"({len(raw)} characters). Restart or reload the model and retry this page."
                ) from exc
    if not isinstance(data, dict):
        raise RuntimeError("The local vision model returned JSON, but it was not an object.")
    return data


def classify_page(
    image_path: Path,
    roster: List[RosterEntry],
    client: LMStudioQwenClient,
) -> PagePrediction:
    """
    Call the local VLM via LMStudioQwenClient to classify a single page.

    The LMStudioQwenClient is responsible for:
      - Using NAME_SYSTEM_PROMPT as the system prompt.
      - Building the messages with roster text + image.
      - Hitting the LM Studio /chat/completions endpoint.
      - Returning the *raw* assistant message content as a string.

    Expected model JSON schema (NAME_SYSTEM_PROMPT should enforce this):

        {
          "name_visible": true or false,
          "raw_name": string or null,
          "student_id": string or null,
          "quiz_version": string or null,
          "confidence": float,
          "reasoning": string
        }

    Malformed free-form explanations are discarded. If the five classifier
    fields cannot be recovered, a short RuntimeError is raised without echoing
    student-related model output into the terminal.
    """
    roster_text = _build_roster_text(roster)

    # This uses LMStudioQwenClient.build_messages(...) + chat(...) internally.
    raw = client.call_for_page(image_path, roster_text)

    data = _parse_classifier_response(raw)

    # Extract fields with sensible defaults.
    name_visible = data.get("name_visible") is True
    raw_name = data.get("raw_name")
    student_id = data.get("student_id")
    try:
        confidence = max(0.0, min(1.0, float(data.get("confidence", 0.0))))
    except (TypeError, ValueError):
        confidence = 0.0
    reasoning = str(data.get("reasoning", "")).strip()[:200]
    quiz_version = normalize_quiz_version(data.get("quiz_version"))

    roster_ids = {entry.student_id for entry in roster}
    if not name_visible or student_id not in roster_ids:
        student_id = None
        confidence = 0.0

    return PagePrediction(
        filename=image_path.name,
        student_id=student_id,
        confidence=confidence,
        name_visible=name_visible,
        raw_name=raw_name,
        reasoning=reasoning,
        quiz_version=quiz_version,
    )
