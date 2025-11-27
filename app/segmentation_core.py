# app/segmentation_core.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import json  # if not already present

from .storage import GRADES_DIR, RAW_SCANS_DIR


from .vlm_client import LMStudioQwenClient
from .vlm_client import SINGLE_QUESTION_PROMPT  # or copy the prompt text here


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
        client = LMStudioQwenClient(
            base_url="http://127.0.0.1:1234/v1",
            model="qwen/qwen3-vl-4b",
        )

    prompt = SINGLE_QUESTION_PROMPT.format(
        problem_id=problem_id,
        problem_description=problem_description,
    )

    raw = client.chat_with_image_prompt(image_path=image_path, prompt=prompt)

    # Allow ```json fences, same trick as in vlm_classifier
    raw_str = raw.strip()
    if raw_str.startswith("```"):
        lines = raw_str.splitlines()
        if len(lines) >= 3:
            raw_str = "\n".join(lines[1:-1]).strip()

    try:
        data = json.loads(raw_str)
    except Exception as e:
        return QuestionSegment(
            problem_id=problem_id,
            page_filename=image_path.name,
            page_index=0,  # we'll fill this later in the caller
            bbox=None,
            present=False,
            comment=f"JSON parse error: {e}",
            confidence=0.0,
        )

    return QuestionSegment(
        problem_id=data.get("target_problem_id", problem_id),
        page_filename=image_path.name,
        page_index=0,  # caller will set
        bbox=data.get("bbox"),
        present=bool(data.get("present", False)),
        comment=data.get("comment"),
        confidence=float(data.get("confidence", 0.0)),
    )
