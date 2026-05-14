"""Node runtime orchestration for handwritten TFL/Fitch OCR."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import cv2
import numpy as np

from proofs_ocr.pipelines.tfl_fitch_vlm import (
    build_27b_fullwidth_prompt,
    build_nested_leaf_separated_prompt,
    get_4b_fullwidth_max_lines,
    get_model_27b,
    model_27b_available,
    stitch_separated_nested_leaf_ocr,
    use_separated_nested_leaf_ocr,
)
from proofs_ocr.pipelines.tfl_line_reread import detect_ocr_row_bounds, tag_ocr_result_lines


def prepare_node_image(node, img: np.ndarray) -> np.ndarray:
    """Return the unmasked crop for a proof-tree node."""
    from proofs_ocr.pipelines import tfl_fitch as core

    if node.id == "ROOT":
        current_img = img.copy()
        if node.children:
            leftmost_bar_x = min(c.box[0] for c in node.children)
            if leftmost_bar_x > 0:
                current_img[:, :leftmost_bar_x] = 255
        return current_img

    if node.parent and node.parent.id == "ROOT":
        extended_h = img.shape[0] - node.box[1]
        return core.crop_region(img, (node.box[0], node.box[1], node.box[2], extended_h))
    return core.crop_region(img, node.box)


def process_and_mask_children(
    *,
    node,
    img: np.ndarray,
    current_img: np.ndarray,
    output_dir: Path,
    library: Dict,
    client,
    call_vlm_fn: Callable,
) -> Tuple[np.ndarray, List, List]:
    """Process child nodes recursively, then mask them in the parent crop."""
    from proofs_ocr.pipelines import tfl_fitch as core

    sorted_children = sorted(node.children, key=lambda c: c.box[1])
    masked_child_nodes: List = []

    for child in sorted_children:
        process_node_recursive(
            child,
            img,
            output_dir,
            library,
            client,
            call_vlm_fn=call_vlm_fn,
        )

        mask_box = core.adjusted_child_box_for_parent_mask(child, img)
        mask_child = core.SubproofNode(child.id, mask_box)
        if node.id == "ROOT":
            current_img = core.mask_subproof_in_parent(current_img, mask_child)
            masked_child_nodes.append(mask_child)
        else:
            px, py, _, _ = node.box
            offset_x = max(0, px - 10)
            offset_y = max(0, py - 10)
            rel_box = (mask_box[0] - offset_x, mask_box[1] - offset_y, mask_box[2], mask_box[3])
            rel_child = core.SubproofNode(child.id, rel_box)
            current_img = core.mask_subproof_in_parent(current_img, rel_child)
            masked_child_nodes.append(rel_child)

    return current_img, sorted_children, masked_child_nodes


def save_node_crop(
    *,
    node,
    current_img: np.ndarray,
    output_dir: Path,
    is_leaf: bool,
    original_crop_h: int,
) -> Path:
    """Apply leaf letterboxing when needed and write the node crop artifact."""
    from proofs_ocr.pipelines import tfl_fitch as core

    if is_leaf and original_crop_h < core.SMALL_CROP_THRESHOLD:
        current_img = core.letterbox_crop(current_img)

    save_path = output_dir / f"crop_{node.id}.jpg"
    cv2.imwrite(str(save_path), current_img)
    return save_path


def route_vlm_image_and_prompt(
    *,
    node,
    img: np.ndarray,
    output_dir: Path,
    is_leaf: bool,
    expected_lines: int | None,
) -> Tuple[str, str | None, str | None, bool]:
    """Choose the crop, model override, prompt override, and separated OCR mode."""
    from proofs_ocr.pipelines import tfl_fitch as core

    model_override = None
    prompt_override = None
    use_separated = False
    vlm_image_path = str(output_dir / f"crop_{node.id}.jpg")

    if (
        is_leaf
        and core.SMALL_LEAF_MODEL_ROUTING_THRESHOLD > 0
        and expected_lines is not None
        and expected_lines <= core.SMALL_LEAF_MODEL_ROUTING_THRESHOLD
        and model_27b_available()
    ):
        model_override = get_model_27b()
        print(
            f"   [ModelRoute] Node {node.id}: routing to {model_override} "
            f"(<= {core.SMALL_LEAF_MODEL_ROUTING_THRESHOLD} lines)"
        )
        fw_img = core.crop_full_width(img, node)
        fw_img = core.trim_lined_paper_top_carryover_for_ocr(
            fw_img,
            expected_lines=expected_lines,
        )
        fw_filename = f"crop_{node.id}_fullwidth.jpg"
        fw_save_path = output_dir / fw_filename
        cv2.imwrite(str(fw_save_path), fw_img)
        print(f"   [FullWidth] Node {node.id}: saved full-width crop to {fw_filename}")
        vlm_image_path = str(fw_save_path)
        prompt_override = build_27b_fullwidth_prompt(expected_lines=expected_lines)
        return vlm_image_path, model_override, prompt_override, use_separated

    four_b_fullwidth_max_lines = get_4b_fullwidth_max_lines()
    is_nested_leaf = is_leaf and node.parent is not None and node.parent.id != "ROOT"
    if (
        is_nested_leaf
        and four_b_fullwidth_max_lines > 0
        and expected_lines is not None
        and expected_lines <= four_b_fullwidth_max_lines
    ):
        print(
            f"   [FullWidth4B] Node {node.id}: using full-width crop "
            f"(<= {four_b_fullwidth_max_lines} lines)"
        )
        fw_img = core.crop_full_width(img, node)
        fw_img = core.trim_lined_paper_top_carryover_for_ocr(
            fw_img,
            expected_lines=expected_lines,
        )
        fw_filename = f"crop_{node.id}_4b_fullwidth.jpg"
        fw_save_path = output_dir / fw_filename
        cv2.imwrite(str(fw_save_path), fw_img)
        vlm_image_path = str(fw_save_path)
        if expected_lines is not None and expected_lines >= 2 and use_separated_nested_leaf_ocr():
            use_separated = True
            prompt_override = build_nested_leaf_separated_prompt(expected_lines=expected_lines)
            print(f"   [SeparatedLeaf] Node {node.id}: requesting formulas/justifications lists")

    return vlm_image_path, model_override, prompt_override, use_separated


def handle_separated_leaf_response(
    *,
    node,
    output_dir: Path,
    ocr_result,
    thinking_text: str,
    expected_lines: int | None,
    vlm_image_path: str,
    client,
    has_children: bool,
    model_override: str | None,
    call_vlm_fn: Callable,
):
    """Accept separated-list OCR when shape matches, otherwise fall back."""
    separated_payload = ocr_result
    separated_path = output_dir / f"response_{node.id}_separated_raw.json"
    with open(separated_path, "w", encoding="utf-8") as f:
        json.dump(separated_payload, f, indent=2, ensure_ascii=False)
    stitched_separated = stitch_separated_nested_leaf_ocr(
        separated_payload,
        expected_lines=expected_lines,
    )
    if stitched_separated is not None:
        print(
            f"   [SeparatedLeaf] Node {node.id}: accepted "
            f"{len(stitched_separated)} stitched line(s)"
        )
        return stitched_separated, thinking_text

    print(f"   [SeparatedLeaf] Node {node.id}: count/shape mismatch; falling back to row-object OCR")
    fallback_result, fallback_thinking = call_vlm_fn(
        vlm_image_path,
        client,
        has_children=has_children,
        expected_lines=expected_lines,
        model_override=model_override,
    )
    if fallback_thinking:
        thinking_text = (
            f"{thinking_text}\n\n--- fallback ---\n\n{fallback_thinking}"
            if thinking_text
            else fallback_thinking
        )
    return fallback_result, thinking_text


def write_vlm_diagnostics(*, node, output_dir: Path, client, thinking_text: str) -> None:
    """Persist thinking, metadata, and error artifacts for a node."""
    if thinking_text:
        thinking_path = output_dir / f"thinking_{node.id}.txt"
        thinking_path.write_text(thinking_text, encoding="utf-8")
        print(f"   [Thinking] Node {node.id}: saved {len(thinking_text)} chars to {thinking_path.name}")

    response_metadata = getattr(client, "last_response_metadata", {}) or {}
    if response_metadata:
        metadata_path = output_dir / f"response_metadata_{node.id}.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(response_metadata, f, indent=2, ensure_ascii=False)

    vlm_error = str(getattr(client, "last_vlm_error", "") or "").strip()
    if vlm_error:
        error_path = output_dir / f"error_{node.id}.txt"
        error_path.write_text(vlm_error, encoding="utf-8")


def truncate_to_expected_lines(
    *,
    node,
    ocr_result,
    is_leaf: bool,
    expected_lines: int | None,
) -> object:
    """Drop extra dict rows when line-count constraint is active."""
    from proofs_ocr.pipelines import tfl_fitch as core

    if not core.USE_LINE_COUNT_CONSTRAINT or not is_leaf or expected_lines is None:
        return ocr_result
    dict_items = [item for item in ocr_result if isinstance(item, dict)]
    if len(dict_items) <= expected_lines:
        return ocr_result

    print(
        f"   [Truncate] Node {node.id}: VLM returned "
        f"{len(dict_items)} lines, truncating to {expected_lines}"
    )
    kept = []
    for item in ocr_result:
        if isinstance(item, dict):
            if len(kept) < expected_lines:
                kept.append(item)
        else:
            kept.append(item)
    return kept


def repair_parent_ocr_result(
    *,
    node,
    ocr_result,
    masked_child_nodes: List,
    current_img: np.ndarray,
) -> object:
    """Apply parent-token ordering, missing-token, and bleed repairs."""
    from proofs_ocr.pipelines import tfl_fitch as core

    ordered = core.repair_trailing_child_token_order(ocr_result)
    if ordered is not ocr_result:
        print(f"   [TokenRepair] Node {node.id}: moved trailing child token before intro row")
        ocr_result = ordered

    repaired = core.repair_missing_child_tokens(
        ocr_result,
        masked_child_nodes,
        img_shape=current_img.shape,
    )
    if repaired is not ocr_result:
        print(
            f"   [TokenRepair] Node {node.id}: restored "
            f"{len(repaired) - len(ocr_result)} missing child token(s)"
        )
        ocr_result = repaired

    filtered = core.drop_trailing_parent_bleed_artifact(ocr_result)
    if filtered is not ocr_result:
        print(f"   [BleedFilter] Node {node.id}: dropped trailing parent-line artifact")
        ocr_result = filtered
    return ocr_result


def save_response_and_tag(
    *,
    node,
    output_dir: Path,
    ocr_result,
    vlm_image_path: str,
    expected_lines: int | None,
    is_leaf: bool,
) -> None:
    """Write raw response artifact and attach transient OCR metadata."""
    with open(output_dir / f"response_{node.id}.json", "w") as f:
        json.dump(ocr_result, f, indent=2)

    row_count_for_bounds = len([item for item in ocr_result if isinstance(item, dict)])
    row_bounds = detect_ocr_row_bounds(Path(vlm_image_path), row_count=row_count_for_bounds)

    tag_ocr_result_lines(
        ocr_result,
        node_id=node.id,
        crop_path=vlm_image_path,
        expected_lines=expected_lines,
        is_leaf=is_leaf,
        node_box=node.box,
        parent_id=node.parent.id if node.parent else None,
        row_bounds=row_bounds,
    )


def process_node_recursive(
    node,
    img: np.ndarray,
    output_dir: Path,
    library: Dict,
    client,
    *,
    call_vlm_fn: Callable,
) -> List[Dict]:
    """Process one proof-tree node and populate ``library`` with its OCR result."""
    from proofs_ocr.pipelines import tfl_fitch as core

    current_img = prepare_node_image(node, img)
    current_img, sorted_children, masked_child_nodes = process_and_mask_children(
        node=node,
        img=img,
        current_img=current_img,
        output_dir=output_dir,
        library=library,
        client=client,
        call_vlm_fn=call_vlm_fn,
    )

    original_crop_h = current_img.shape[0]
    line_count_img = current_img
    is_leaf = not sorted_children
    expected_lines = None

    save_path = save_node_crop(
        node=node,
        current_img=current_img,
        output_dir=output_dir,
        is_leaf=is_leaf,
        original_crop_h=original_crop_h,
    )
    print(f"Processing Node {node.id} (Depth {node.depth})... Saved to {save_path.name}")

    if core.USE_LINE_COUNT_CONSTRAINT and is_leaf:
        expected_lines = core.estimate_line_count(line_count_img)
        print(f"   [LineCount] Node {node.id}: detected {expected_lines} text line(s)")

    vlm_image_path, model_override, prompt_override, use_separated = route_vlm_image_and_prompt(
        node=node,
        img=img,
        output_dir=output_dir,
        is_leaf=is_leaf,
        expected_lines=expected_lines,
    )

    ocr_result, thinking_text = call_vlm_fn(
        vlm_image_path,
        client,
        has_children=bool(sorted_children),
        expected_lines=expected_lines,
        model_override=model_override,
        prompt_override=prompt_override,
    )

    if use_separated:
        ocr_result, thinking_text = handle_separated_leaf_response(
            node=node,
            output_dir=output_dir,
            ocr_result=ocr_result,
            thinking_text=thinking_text,
            expected_lines=expected_lines,
            vlm_image_path=vlm_image_path,
            client=client,
            has_children=bool(sorted_children),
            model_override=model_override,
            call_vlm_fn=call_vlm_fn,
        )

    write_vlm_diagnostics(
        node=node,
        output_dir=output_dir,
        client=client,
        thinking_text=thinking_text,
    )

    ocr_result = truncate_to_expected_lines(
        node=node,
        ocr_result=ocr_result,
        is_leaf=is_leaf,
        expected_lines=expected_lines,
    )

    if not is_leaf:
        ocr_result = repair_parent_ocr_result(
            node=node,
            ocr_result=ocr_result,
            masked_child_nodes=masked_child_nodes,
            current_img=current_img,
        )

    save_response_and_tag(
        node=node,
        output_dir=output_dir,
        ocr_result=ocr_result,
        vlm_image_path=vlm_image_path,
        expected_lines=expected_lines,
        is_leaf=is_leaf,
    )

    library[node.id] = ocr_result
    return ocr_result
