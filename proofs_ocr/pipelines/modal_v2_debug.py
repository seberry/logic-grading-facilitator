"""Debug-state harness for running the modal_v2 baseline pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import cv2

from proofs_ocr.common import artifacts
from proofs_ocr.domains import modal
from proofs_ocr.pipelines.modal_v2_core import (
    ImagePromptClient,
    ModalPipelineFeatureFlags,
    _index_tree,
    _masked_node_crop_image,
    _node_crop_image,
    run_modal_baseline_pipeline,
)
from proofs_ocr.pipelines.modal_v2_runtime import (
    ModalBaselineArtifactLineReader,
    build_vlm_client,
    default_lmstudio_base_url,
    default_lmstudio_model,
)
from proofs_ocr.pipelines.modal_v2_unary import run_blob_unary_probe, run_structured_unary_probe


def build_modal_v2_state(
    *,
    modal_config: Mapping[str, Any] | None,
    final_structure: list[Any],
    run_summary: str,
    vision_base_url: str | None = None,
    vision_model: str | None = None,
    line_vision_model: str | None = None,
    verifier_vision_model: str | None = None,
    enable_formula_justification_separation: bool = False,
    enable_formula_consensus_reread: bool = False,
    source_image_name: str = "",
) -> dict[str, Any]:
    """Build the minimal state.json payload for baseline debug runs."""

    canonical = modal.canonical_modal_config(modal_config)
    return {
        "system": canonical["system"],
        "include_derived_rules": canonical["include_derived_rules"],
        "conclusion": canonical["conclusion"],
        "vision_base_url": (vision_base_url or default_lmstudio_base_url()).strip(),
        "vision_model": (vision_model or default_lmstudio_model()).strip(),
        "line_vision_model": (line_vision_model or vision_model or default_lmstudio_model()).strip(),
        "verifier_vision_model": (verifier_vision_model or "").strip(),
        "enable_formula_justification_separation": bool(enable_formula_justification_separation),
        "enable_formula_consensus_reread": bool(enable_formula_consensus_reread),
        "source_image_name": source_image_name,
        "proof_json_text": json.dumps({"lines": final_structure, "meta": {}}, indent=2, ensure_ascii=False),
        "original_proof_json_text": json.dumps({"lines": final_structure, "meta": {}}, indent=2, ensure_ascii=False),
        "verifier_rewrite_summary": [],
        "draft_comment": "",
        "draft_grade": "",
        "checker_request": None,
        "checker_response": None,
        "run_summary": run_summary,
    }


def transcribe_image_to_modal_v2_debug_state(
    *,
    image_path: Path,
    output_dir: Path,
    modal_config: Mapping[str, Any] | None = None,
    client: ImagePromptClient | None = None,
    vision_base_url: str | None = None,
    vision_model: str | None = None,
    line_vision_model: str | None = None,
    verifier_client: ImagePromptClient | None = None,
    verifier_vision_model: str | None = None,
    enable_formula_justification_separation: bool = False,
    enable_unary_structured_probe: bool = False,
    enable_unary_structured_formula_crop_probe: bool = False,
    enable_unary_blob_probe: bool = False,
    enable_false_left_separator_filter: bool = False,
    enable_sparse_lower_compact_band_recovery: bool = False,
    enable_contradiction_reread: bool = False,
    enable_justification_head_reread: bool = False,
    enable_formula_consensus_reread: bool = False,
) -> dict[str, Any]:
    """Run the modal_v2 baseline end to end and write debug artifacts."""

    image_path = image_path.resolve()
    output_dir = output_dir.resolve()
    preserved = [image_path] if image_path.parent == output_dir else []
    artifacts.clear_debug_dir(output_dir, preserve_paths=preserved)

    if client is None:
        client = build_vlm_client(base_url=vision_base_url, model=(line_vision_model or vision_model))
    if (
        (enable_formula_justification_separation and bool(verifier_vision_model))
        or enable_unary_structured_probe
        or enable_unary_structured_formula_crop_probe
        or enable_false_left_separator_filter
        or enable_contradiction_reread
        or enable_justification_head_reread
        or enable_formula_consensus_reread
    ) and verifier_client is None:
        verifier_client = build_vlm_client(
            base_url=vision_base_url,
            model=(verifier_vision_model or line_vision_model or vision_model),
        )

    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image at {image_path}")

    source_copy_name = f"{artifacts.SOURCE_IMAGE_STEM}{image_path.suffix.lower() or '.png'}"
    source_copy_path = output_dir / source_copy_name
    if source_copy_path != image_path:
        source_copy_path.write_bytes(image_path.read_bytes())

    prompt = modal.build_line_ocr_prompt(modal_config)
    (output_dir / artifacts.PROMPT_FILENAME).write_text(prompt, encoding="utf-8")

    result = run_modal_baseline_pipeline(
        image=image,
        read_line=ModalBaselineArtifactLineReader(
            output_dir=output_dir,
            client=client,
            prompt=prompt,
            modal_config=modal_config,
            enable_formula_justification_separation=enable_formula_justification_separation,
            verifier_client=verifier_client,
        ),
        feature_flags=ModalPipelineFeatureFlags(
            enable_sparse_lower_compact_band_recovery=enable_sparse_lower_compact_band_recovery,
            enable_contradiction_reread=enable_contradiction_reread,
            enable_justification_head_reread=enable_justification_head_reread,
            enable_formula_consensus_reread=enable_formula_consensus_reread,
        ),
        enable_false_left_separator_filter=enable_false_left_separator_filter,
        false_left_separator_verifier_client=verifier_client if enable_false_left_separator_filter else None,
        false_left_separator_output_dir=output_dir if enable_false_left_separator_filter else None,
    )

    if result.bar_filter_debug:
        (output_dir / artifacts.BAR_FILTER_DEBUG_FILENAME).write_text(
            json.dumps(result.bar_filter_debug, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    artifacts.write_bar_overlay(
        image=image,
        bars=result.detected_bars,
        output_path=output_dir / artifacts.DEBUG_BARS_FILENAME,
    )

    node_lookup = _index_tree(result.tree_root)
    for node_id, node in node_lookup.items():
        node_crop_img = _node_crop_image(node=node, full_img=image)
        masked_crop_img = _masked_node_crop_image(node=node, full_img=image)
        cv2.imwrite(str(output_dir / artifacts.NODE_CROP_TEMPLATE.format(node_id=node_id)), node_crop_img)
        (output_dir / artifacts.INTERMEDIATE_TEMPLATE.format(node_id=node_id)).write_text(
            json.dumps(
                {
                    "node_id": node_id,
                    "box": list(node.box),
                    "child_ids": [child.id for child in node.children],
                    "row_records": result.row_records_by_node.get(node_id, []),
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        (output_dir / artifacts.RAW_RESPONSE_TEMPLATE.format(node_id=node_id)).write_text(
            json.dumps(result.library.get(node_id, []), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (output_dir / artifacts.NODE_ROWS_TEMPLATE.format(node_id=node_id)).write_text(
            json.dumps(result.row_records_by_node.get(node_id, []), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (output_dir / artifacts.NODE_ROW_BANDS_TEMPLATE.format(node_id=node_id)).write_text(
            json.dumps(result.band_debug_by_node.get(node_id, {}), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (output_dir / artifacts.NODE_ENTRIES_TEMPLATE.format(node_id=node_id)).write_text(
            json.dumps(result.library.get(node_id, []), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        artifacts.write_row_overlay(
            node_crop_img=masked_crop_img,
            row_records=result.row_records_by_node.get(node_id, []),
            output_path=output_dir / artifacts.NODE_ROWS_OVERLAY_TEMPLATE.format(node_id=node_id),
        )

    final_structure_text = json.dumps(result.final_structure, indent=2, ensure_ascii=False)
    (output_dir / artifacts.BASELINE_FINAL_STRUCTURE_FILENAME).write_text(final_structure_text, encoding="utf-8")
    (output_dir / artifacts.FINAL_STRUCTURE_FILENAME).write_text(final_structure_text, encoding="utf-8")

    unary_structured_probe_records: list[dict[str, Any]] = []
    if enable_unary_structured_probe and verifier_client is not None:
        unary_structured_probe_records = run_structured_unary_probe(
            entries_by_node=result.library,
            output_dir=output_dir,
            client=verifier_client,
        )
    unary_structured_formula_probe_records: list[dict[str, Any]] = []
    if enable_unary_structured_formula_crop_probe and verifier_client is not None:
        unary_structured_formula_probe_records = run_structured_unary_probe(
            entries_by_node=result.library,
            output_dir=output_dir,
            client=verifier_client,
            crop_mode="formula_crop_only",
        )
    unary_blob_probe_records: list[dict[str, Any]] = []
    if enable_unary_blob_probe:
        unary_blob_probe_records = run_blob_unary_probe(
            entries_by_node=result.library,
            output_dir=output_dir,
        )

    visible_line_count = sum(
        1
        for entries in result.library.values()
        for entry in entries
        if modal.has_visible_formula(entry)
    )
    run_summary = (
        f"modal_v2 baseline OCR produced {visible_line_count} visible lines across "
        f"{len(node_lookup)} nodes and detected {len(result.detected_bars)} subproof bars."
    )
    if enable_unary_structured_probe:
        run_summary += (
            f" Passive unary structured probe inspected {len(unary_structured_probe_records)} line"
            f"{'' if len(unary_structured_probe_records) == 1 else 's'}."
        )
    if enable_unary_structured_formula_crop_probe:
        run_summary += (
            " Passive unary structured formula-crop probe inspected "
            f"{len(unary_structured_formula_probe_records)} line"
            f"{'' if len(unary_structured_formula_probe_records) == 1 else 's'}."
        )
    if enable_formula_justification_separation:
        run_summary += " Formula/justification separation was enabled."
    if enable_false_left_separator_filter:
        filter_applied = bool(result.bar_filter_debug.get("applied")) if result.bar_filter_debug else False
        run_summary += (
            " False-left-separator filtering was enabled"
            + (" and removed one candidate bar." if filter_applied else ".")
        )
    if enable_sparse_lower_compact_band_recovery:
        run_summary += " Sparse lower-half compact-band recovery was enabled."
    if enable_contradiction_reread:
        run_summary += " Contradiction-line reread was enabled."
    if enable_justification_head_reread:
        run_summary += " Boxed justification-head reread was enabled."
    if enable_formula_consensus_reread:
        run_summary += " Formula-only consensus reread was enabled."
    if str(line_vision_model or "").strip() and str(line_vision_model).strip() != str(vision_model or "").strip():
        run_summary += f" Final line OCR used {str(line_vision_model).strip()} while structure used {str(vision_model or default_lmstudio_model()).strip()}."
    if enable_unary_blob_probe:
        flagged_count = sum(1 for record in unary_blob_probe_records if bool(record.get("any_blob_flagged")))
        run_summary += (
            f" Passive unary blob probe inspected {len(unary_blob_probe_records)} line"
            f"{'' if len(unary_blob_probe_records) == 1 else 's'} and flagged {flagged_count}."
        )
    state = build_modal_v2_state(
        modal_config=modal_config,
        final_structure=result.final_structure,
        run_summary=run_summary,
        vision_base_url=vision_base_url,
        vision_model=vision_model,
        line_vision_model=line_vision_model or vision_model,
        verifier_vision_model=verifier_vision_model or line_vision_model or vision_model,
        enable_formula_justification_separation=enable_formula_justification_separation,
        enable_formula_consensus_reread=enable_formula_consensus_reread,
        source_image_name=source_copy_name,
    )
    (output_dir / artifacts.STATE_FILENAME).write_text(
        json.dumps(state, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return state
