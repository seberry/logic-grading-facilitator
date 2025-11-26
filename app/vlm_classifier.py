# app/vlm_classifier.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from .models import RosterEntry
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
      - reasoning: str             short explanation from the model
    """

    filename: str
    student_id: Optional[str]
    confidence: float
    name_visible: bool
    raw_name: Optional[str]
    reasoning: str


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
          "confidence": float,
          "reasoning": string
        }

    Any deviations will raise a RuntimeError with the raw text for debugging.
    """
    roster_text = _build_roster_text(roster)

    # This uses LMStudioQwenClient.build_messages(...) + chat(...) internally.
    raw = client.call_for_page(image_path, roster_text)

    # Be tolerant of ```json ... ``` fences.
    raw_clean = _strip_fences(raw)

    try:
        data = json.loads(raw_clean)
    except Exception as e:
        raise RuntimeError(f"Failed to parse JSON from VLM response: {raw}") from e

    # Extract fields with sensible defaults.
    name_visible = bool(data.get("name_visible", False))
    raw_name = data.get("raw_name")
    student_id = data.get("student_id")
    confidence = float(data.get("confidence", 0.0))
    reasoning = str(data.get("reasoning", ""))

    return PagePrediction(
        filename=image_path.name,
        student_id=student_id,
        confidence=confidence,
        name_visible=name_visible,
        raw_name=raw_name,
        reasoning=reasoning,
    )
