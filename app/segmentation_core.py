# app/segmentation_core.py
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import requests

from .vlm_client import LMStudioQwenClient
from .vlm_client import SINGLE_QUESTION_PROMPT  # or copy the prompt text here
from .vlm_provider import create_vlm_client


@dataclass
class QuestionSegment:
    problem_id: str
    page_filename: str
    page_index: int   # index into WorkUnit.page_filenames
    bbox: Optional[list[float]]  # [x_min, y_min, x_max, y_max] normalized
    present: bool
    comment: Optional[str]
    confidence: float


def segment_page_for_problem(
    image_path: Path,
    problem_id: str,
    problem_description: str,
    client: Optional[LMStudioQwenClient] = None,
    prompt_override: Optional[str] = None,
) -> QuestionSegment:
    if client is None:
        client = create_vlm_client()

    prompt = prompt_override or SINGLE_QUESTION_PROMPT.format(
        problem_id=problem_id,
        problem_description=problem_description,
    )

    try:
        raw = client.chat_with_image_prompt(image_path=image_path, prompt=prompt)
    except requests.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else "unknown"
        provider = "IU REALLMS" if getattr(client, "provider", "") == "reallms" else "vision model"
        raise RuntimeError(
            f"{provider} question-location request returned HTTP {status}. "
            "Saved locations were preserved. Retry with IU Qwen if Gemma continues to fail."
        ) from exc
    except RuntimeError as exc:
        message = str(exc)
        safe_prefixes = (
            "IU REALLMS could not be reached.",
            "IU REALLMS returned an unreadable completion.",
            "IU REALLMS returned an empty completion.",
            "IU REALLMS reached the response limit;",
            "Set REALLMS_API_KEY",
        )
        detail = message if message.startswith(safe_prefixes) and len(message) < 300 else "The vision provider failed."
        raise RuntimeError(
            f"Question-location request failed: {detail} Saved locations were preserved."
        ) from exc
    except Exception as exc:
        raise RuntimeError(
            f"Question-location request failed during image/model processing ({type(exc).__name__}). "
            "Saved locations were preserved."
        ) from exc

    # Allow ```json fences, same trick as in vlm_classifier
    if not isinstance(raw, str):
        raise RuntimeError("The model returned invalid question-location text. Saved locations were preserved.")
    raw_str = raw.strip()
    if raw_str.startswith("```"):
        lines = raw_str.splitlines()
        if len(lines) >= 3:
            raw_str = "\n".join(lines[1:-1]).strip()

    try:
        data = json.loads(raw_str)
    except Exception:
        raise RuntimeError("The model returned invalid question-location JSON. Saved locations were preserved.") from None

    if not isinstance(data, dict) or not isinstance(data.get("present"), bool):
        raise RuntimeError("The model returned invalid question-location fields.")
    confidence = data.get("confidence", 0.0)
    if (isinstance(confidence, bool) or not isinstance(confidence, (int, float))
            or not math.isfinite(confidence) or not 0 <= confidence <= 1):
        raise RuntimeError("The model returned invalid question-location confidence. Saved locations were preserved.")
    comment = data.get("comment")
    if comment is not None and not isinstance(comment, str):
        raise RuntimeError("The model returned invalid question-location comment. Saved locations were preserved.")
    if data["present"]:
        bbox = data.get("bbox")
        if (not isinstance(bbox, list) or len(bbox) != 4
                or any(isinstance(n, bool) or not isinstance(n, (int, float)) or not 0 <= n <= 1 for n in bbox)
                or not (bbox[0] < bbox[2] and bbox[1] < bbox[3])):
            raise RuntimeError("The model returned invalid question bounds. Saved locations were preserved.")

    return QuestionSegment(
        problem_id=data.get("target_problem_id", problem_id),
        page_filename=image_path.name,
        page_index=0,  # caller will set
        bbox=data.get("bbox"),
        present=bool(data.get("present", False)),
        comment=comment,
        confidence=float(confidence),
    )
