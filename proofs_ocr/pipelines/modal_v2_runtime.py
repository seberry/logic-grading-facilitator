"""Runtime adapters for the modal_v2 baseline pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import cv2

from proofs_ocr.common import artifacts
from proofs_ocr.domains import modal
from proofs_ocr.pipelines.modal_v2_core import ImagePromptClient, ModalPipelineFeatureFlags

try:
    from app.vlm_client import LMStudioQwenClient, default_lmstudio_base_url, default_lmstudio_model
except ImportError:  # pragma: no cover - standalone fallback
    LMStudioQwenClient = None  # type: ignore[assignment]

    def default_lmstudio_base_url() -> str:
        return "http://127.0.0.1:1234/v1"

    def default_lmstudio_model() -> str:
        return "qwen/qwen3-vl-4b"


class ModalBaselineArtifactLineReader:
    """Real line reader that writes per-line baseline artifacts."""

    def __init__(
        self,
        *,
        output_dir: Path,
        client: ImagePromptClient,
        prompt: str,
    ) -> None:
        self.output_dir = output_dir
        self.client = client
        self.prompt = prompt

    def __call__(
        self,
        *,
        node,
        node_crop_img,
        line_crop_img,
        row_record,
        feature_flags: ModalPipelineFeatureFlags,
    ) -> Mapping[str, str]:
        del node_crop_img, feature_flags
        row_order = int(row_record.get("row_order", 0))
        crop_name = artifacts.NODE_LINE_CROP_TEMPLATE.format(node_id=node.id, row_order=row_order)
        crop_path = self.output_dir / crop_name
        cv2.imwrite(str(crop_path), line_crop_img)

        raw_response = self.client.chat_with_image_prompt(crop_path, self.prompt)
        parse_error = ""
        try:
            resolved = modal.parse_line_ocr_response(raw_response)
        except (ValueError, json.JSONDecodeError) as exc:
            parse_error = str(exc)
            resolved = {
                "line_label": "",
                "formula_text": "",
                "justification_text": "",
                "scope_role": "derived_line",
                "confidence": "",
            }

        payload = {
            "node_id": node.id,
            "row_order": row_order,
            "band_index": row_record.get("band_index"),
            "y1": row_record.get("y1"),
            "y2": row_record.get("y2"),
            "raw_response": raw_response,
            "parse_error": parse_error,
            "resolved": resolved,
        }
        (self.output_dir / artifacts.NODE_LINE_RESPONSE_TEMPLATE.format(node_id=node.id, row_order=row_order)).write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return resolved


def build_vlm_client(
    *,
    base_url: str | None = None,
    model: str | None = None,
    timeout: int = 60,
) -> LMStudioQwenClient:
    """Build the LM Studio client used by the baseline harness."""

    if LMStudioQwenClient is None:
        raise RuntimeError("LM Studio client is not available")
    return LMStudioQwenClient(
        base_url=(base_url or default_lmstudio_base_url()).strip(),
        model=(model or default_lmstudio_model()).strip(),
        timeout=timeout,
    )
