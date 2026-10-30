# app/segmentation_core.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

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
) -> QuestionSegment:
    if client is None:
        client = create_vlm_client()

    prompt = SINGLE_QUESTION_PROMPT.format(
        problem_id=problem_id,
        problem_description=problem_description,
    )

    try:
        raw = client.chat_with_image_prompt(image_path=image_path, prompt=prompt)
    except Exception as e:
        raise RuntimeError("Question-location request failed. Saved locations were preserved; retry later or with Qwen.") from None

    # Allow ```json fences, same trick as in vlm_classifier
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
        comment=data.get("comment"),
        confidence=float(data.get("confidence", 0.0)),
    )
