"""Local-model verifier helpers for suspicious leftmost bar candidates."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from proofs_ocr.common import artifacts
from proofs_ocr.common.types import Box

_ALLOWED_CLASSIFICATIONS = {
    "scope_bar",
    "number_formula_separator",
    "margin_artifact_or_stamp",
    "uncertain",
}


def build_false_left_separator_verifier_prompt() -> str:
    return "\n".join(
        [
            "You are classifying the leftmost vertical mark in a handwritten Fitch-style proof image.",
            "Return only JSON. Do not include markdown fences or explanation.",
            (
                'Return one JSON object with exactly these keys: "classification", "confidence", and "reason".'
            ),
            'Allowed classification values: "scope_bar", "number_formula_separator", "margin_artifact_or_stamp", "uncertain".',
            "The candidate leftmost vertical mark is highlighted in red.",
            "Any next bar to its right is highlighted in blue.",
            (
                '"scope_bar" means the red mark is a real proof/subproof scope bar that organizes proof lines to its right.'
            ),
            (
                '"number_formula_separator" means the red mark is just a left separator beside line numbers or margin content, not a real scope bar.'
            ),
            '"margin_artifact_or_stamp" means the red mark is unrelated page junk, a stamp, or another artifact.',
            '"uncertain" means the crop is too ambiguous to trust.',
            "Do not transcribe the proof. Focus only on the role of the red vertical mark.",
            "Use confidence values high, medium, low, or uncertain.",
        ]
    )


def classify_false_left_separator_candidate(
    *,
    image: np.ndarray,
    candidate_bar: Box,
    neighboring_bars: Sequence[Box],
    verifier_client: Any | None,
    output_dir: Path | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "attempted": False,
        "candidate_crop_filename": "",
        "candidate_crop_path": "",
        "prompt": "",
        "raw_response": "",
        "classification": "",
        "confidence": "",
        "reason": "",
        "error": "",
    }
    if verifier_client is None:
        payload["error"] = "verifier_client_unavailable"
        return payload
    if output_dir is None:
        payload["error"] = "output_dir_unavailable"
        return payload

    prompt = build_false_left_separator_verifier_prompt()
    crop_path = _write_candidate_crop(
        image=image,
        candidate_bar=candidate_bar,
        neighboring_bars=neighboring_bars,
        output_dir=output_dir,
    )
    payload["attempted"] = True
    payload["candidate_crop_filename"] = crop_path.name
    payload["candidate_crop_path"] = str(crop_path)
    payload["prompt"] = prompt

    raw_response = verifier_client.chat_with_image_prompt(crop_path, prompt)
    payload["raw_response"] = raw_response
    parsed = _parse_verifier_response(raw_response)
    payload["classification"] = parsed.get("classification") or ""
    payload["confidence"] = parsed.get("confidence") or ""
    payload["reason"] = parsed.get("reason") or ""
    payload["error"] = parsed.get("error") or ""
    return payload


def _write_candidate_crop(
    *,
    image: np.ndarray,
    candidate_bar: Box,
    neighboring_bars: Sequence[Box],
    output_dir: Path,
) -> Path:
    img_h, img_w = image.shape[:2]
    x, y, w, h = candidate_bar
    next_bar = min(neighboring_bars, key=lambda box: (box[0], box[1])) if neighboring_bars else candidate_bar
    right_limit = min(
        img_w,
        max(
            x + w + 120,
            next_bar[0] + next_bar[2] + 140,
            int(img_w * 0.48),
        ),
    )
    bottom_limit = min(
        img_h,
        max(
            y + h + 12,
            next_bar[1] + next_bar[3] + 24,
            int(img_h * 0.70),
        ),
    )
    crop = image[0:bottom_limit, 0:right_limit].copy()
    rel_candidate = (x, y, w, h)
    cv2.rectangle(
        crop,
        (rel_candidate[0], rel_candidate[1]),
        (rel_candidate[0] + rel_candidate[2], rel_candidate[1] + rel_candidate[3]),
        (0, 0, 255),
        2,
    )
    cv2.putText(
        crop,
        "candidate",
        (max(4, rel_candidate[0]), max(18, rel_candidate[1] + 16)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 255),
        2,
    )
    for index, bar in enumerate(sorted(neighboring_bars, key=lambda item: (item[0], item[1])), start=1):
        bx, by, bw, bh = bar
        cv2.rectangle(crop, (bx, by), (bx + bw, by + bh), (255, 0, 0), 2)
        cv2.putText(
            crop,
            f"bar{index}",
            (bx, max(18, by + 16)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            2,
        )

    crop_path = output_dir / artifacts.FALSE_LEFT_SEPARATOR_VERIFIER_CROP_FILENAME
    cv2.imwrite(str(crop_path), crop)
    return crop_path


def _parse_verifier_response(raw_response: str) -> dict[str, str]:
    normalized = str(raw_response or "").strip()
    if not normalized:
        return {"classification": "", "confidence": "", "reason": "", "error": "empty_response"}

    try:
        parsed = json.loads(normalized)
    except json.JSONDecodeError:
        parsed = {}

    if isinstance(parsed, Mapping):
        classification = str(parsed.get("classification") or "").strip()
        if classification in _ALLOWED_CLASSIFICATIONS:
            return {
                "classification": classification,
                "confidence": str(parsed.get("confidence") or "").strip(),
                "reason": str(parsed.get("reason") or "").strip(),
                "error": "",
            }

    compact = normalized.strip().strip('"').strip().lower()
    for classification in _ALLOWED_CLASSIFICATIONS:
        if compact == classification:
            return {
                "classification": classification,
                "confidence": "",
                "reason": "",
                "error": "",
            }

    return {"classification": "", "confidence": "", "reason": "", "error": "unparseable_response"}
