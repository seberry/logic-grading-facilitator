"""Runtime adapters for the modal_v2 baseline pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import cv2

from proofs_ocr.common import artifacts
from proofs_ocr.domains import modal
from proofs_ocr.pipelines.modal_v2_core import ImagePromptClient, ModalPipelineFeatureFlags
from proofs_ocr.pipelines.modal_v2_formula_rereads import maybe_reread_formula_text
from proofs_ocr.pipelines.modal_v2_line_rereads import maybe_reread_contradiction_line, maybe_reread_justification_head
from proofs_ocr.pipelines.modal_v2_line_zones import read_line_with_zone_separation

try:
    from app.vlm_client import LMStudioQwenClient, default_lmstudio_base_url, default_lmstudio_model, default_lmstudio_timeout
except ImportError:  # pragma: no cover - standalone fallback
    LMStudioQwenClient = None  # type: ignore[assignment]

    def default_lmstudio_base_url() -> str:
        return "http://127.0.0.1:1234/v1"

    def default_lmstudio_model() -> str:
        return "qwen/qwen3-vl-4b"

    def default_lmstudio_timeout() -> int:
        return 180


class ModalBaselineArtifactLineReader:
    """Real line reader that writes per-line baseline artifacts."""

    def __init__(
        self,
        *,
        output_dir: Path,
        client: ImagePromptClient,
        prompt: str,
        modal_config: Mapping[str, object] | None = None,
        enable_formula_justification_separation: bool = False,
        verifier_client: ImagePromptClient | None = None,
    ) -> None:
        self.output_dir = output_dir
        self.client = client
        self.prompt = prompt
        self.enable_formula_justification_separation = enable_formula_justification_separation
        self.verifier_client = verifier_client
        self.modal_config = modal_config
        self.formula_prompt = ""
        self.justification_prompt = ""
        if enable_formula_justification_separation:
            from proofs_ocr.domains import modal_line_zones

            self.formula_prompt = modal_line_zones.build_formula_zone_prompt(modal_config)
            self.justification_prompt = modal_line_zones.build_justification_zone_prompt(modal_config)

    def __call__(
        self,
        *,
        node,
        node_crop_img,
        line_crop_img,
        row_record,
        feature_flags: ModalPipelineFeatureFlags,
    ) -> Mapping[str, str]:
        if self.enable_formula_justification_separation:
            return read_line_with_zone_separation(
                output_dir=self.output_dir,
                client=self.client,
                verifier_client=self.verifier_client,
                formula_prompt=self.formula_prompt,
                justification_prompt=self.justification_prompt,
                node=node,
                node_crop_img=node_crop_img,
                line_crop_img=line_crop_img,
                row_record=row_record,
                feature_flags=feature_flags,
                modal_config=self.modal_config,
            )
        row_order = int(row_record.get("row_order", 0))
        crop_name = artifacts.NODE_LINE_CROP_TEMPLATE.format(node_id=node.id, row_order=row_order)
        crop_path = self.output_dir / crop_name
        cv2.imwrite(str(crop_path), line_crop_img)

        skip_reason = modal.detect_label_only_crop(line_crop_img)
        raw_response = ""
        parse_error = ""
        resolved = {
            "line_label": "",
            "formula_text": "",
            "justification_text": "",
            "scope_role": "derived_line",
            "confidence": "",
        }
        recovery_applied = ""
        contradiction_reread = None
        justification_head_reread = None
        formula_consensus_reread = None
        if not skip_reason:
            raw_response = self.client.chat_with_image_prompt(crop_path, self.prompt)
            try:
                resolved = modal.parse_line_ocr_response(raw_response)
            except (ValueError, json.JSONDecodeError) as exc:
                parse_error = str(exc)
            resolved, recovery_applied = modal.recover_line_entry_from_crop(
                line_crop_img=line_crop_img,
                parsed_entry=resolved,
            )
            if str(node.id) != "ROOT" and row_order == 1:
                strict_open_resolved, strict_open_recovery = modal.recover_first_nonroot_strict_open_from_crop(
                    line_crop_img=line_crop_img,
                    parsed_entry=resolved,
                )
                if strict_open_recovery:
                    resolved = strict_open_resolved
                    recovery_applied = strict_open_recovery
            if feature_flags.enable_contradiction_reread:
                contradiction_resolved, contradiction_reread, contradiction_recovery = maybe_reread_contradiction_line(
                    output_dir=self.output_dir,
                    client=self.verifier_client,
                    node_id=str(node.id),
                    row_order=row_order,
                    node_crop_img=node_crop_img,
                    row_record=row_record,
                    line_crop_img=line_crop_img,
                    parsed_entry=resolved,
                )
                if contradiction_recovery:
                    resolved = contradiction_resolved
                    recovery_applied = contradiction_recovery
            if feature_flags.enable_justification_head_reread:
                justification_resolved, justification_head_reread, justification_recovery = maybe_reread_justification_head(
                    output_dir=self.output_dir,
                    client=self.verifier_client,
                    node_id=str(node.id),
                    row_order=row_order,
                    line_crop_img=line_crop_img,
                    parsed_entry=resolved,
                )
                if justification_recovery:
                    resolved = justification_resolved
                    recovery_applied = justification_recovery
            contradiction_locked = bool(contradiction_reread and contradiction_reread.get("applied"))
            if feature_flags.enable_formula_consensus_reread and not contradiction_locked:
                formula_resolved, formula_consensus_reread, formula_recovery = maybe_reread_formula_text(
                    output_dir=self.output_dir,
                    client=self.verifier_client,
                    node_id=str(node.id),
                    row_order=row_order,
                    line_crop_img=line_crop_img,
                    parsed_entry=resolved,
                    modal_config=self.modal_config,
                )
                if formula_recovery:
                    resolved = formula_resolved
                    recovery_applied = formula_recovery

        payload = {
            "node_id": node.id,
            "row_order": row_order,
            "band_index": row_record.get("band_index"),
            "y1": row_record.get("y1"),
            "y2": row_record.get("y2"),
            "raw_response": raw_response,
            "parse_error": parse_error,
            "resolved": resolved,
            "recovery_applied": recovery_applied,
            "skip_reason": skip_reason,
        }
        if not skip_reason and contradiction_reread is not None:
            payload["contradiction_reread"] = contradiction_reread
        if not skip_reason and justification_head_reread is not None:
            payload["justification_head_reread"] = justification_head_reread
        if not skip_reason and formula_consensus_reread is not None:
            payload["formula_consensus_reread"] = formula_consensus_reread
        (self.output_dir / artifacts.NODE_LINE_RESPONSE_TEMPLATE.format(node_id=node.id, row_order=row_order)).write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return resolved


def build_vlm_client(
    *,
    base_url: str | None = None,
    model: str | None = None,
    timeout: int | None = None,
) -> LMStudioQwenClient:
    """Build the LM Studio client used by the baseline harness."""

    if LMStudioQwenClient is None:
        raise RuntimeError("LM Studio client is not available")
    return LMStudioQwenClient(
        base_url=(base_url or default_lmstudio_base_url()).strip(),
        model=(model or default_lmstudio_model()).strip(),
        timeout=default_lmstudio_timeout() if timeout is None else timeout,
    )
