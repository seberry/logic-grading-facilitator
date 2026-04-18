"""Minimal recursive modal proof OCR pipeline.

This is the first real pipeline spine for the refactor workspace:

1. detect subproof bars
2. build a scope tree
3. recurse into child scopes first
4. mask children with placeholder tokens
5. segment visible rows for the current scope
6. let an injected reader OCR each visible line
7. stitch child placeholders back into the final nested structure

The default baseline stays old-script-free. Later legacy-inspired layers should
only arrive as explicit feature-flagged additions.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Protocol

import cv2
import numpy as np

from proofs_ocr.common import artifacts, masking, rows, stitch, tree
from proofs_ocr.common.types import Box, SubproofNode
from proofs_ocr.domains import modal

try:
    from app.vlm_client import LMStudioQwenClient, default_lmstudio_base_url, default_lmstudio_model
except ImportError:  # pragma: no cover - standalone fallback
    LMStudioQwenClient = None  # type: ignore[assignment]

    def default_lmstudio_base_url() -> str:
        return "http://127.0.0.1:1234/v1"

    def default_lmstudio_model() -> str:
        return "qwen/qwen3-vl-4b"


@dataclass(frozen=True)
class ModalPipelineFeatureFlags:
    """Optional layers reserved for later experiments."""

    enable_strict_opener_classifier: bool = False
    enable_short_formula_reread: bool = False
    enable_retry_on_suspicious_line: bool = False
    enable_unary_verifier: bool = False
    enable_boundary_repair: bool = False

    def enabled_layers(self) -> List[str]:
        enabled: List[str] = []
        for field_name, value in self.__dict__.items():
            if bool(value):
                enabled.append(field_name)
        return enabled


@dataclass
class ModalPipelineResult:
    """Artifacts returned by one baseline pipeline run."""

    detected_bars: List[Box]
    tree_root: SubproofNode
    library: Dict[str, List[Dict[str, Any]]]
    row_records_by_node: Dict[str, List[rows.RowRecord]]
    final_structure: List[Any]
    feature_flags: ModalPipelineFeatureFlags


class ModalLineReader(Protocol):
    """Injected per-line OCR adapter for the baseline pipeline."""

    def __call__(
        self,
        *,
        node: SubproofNode,
        node_crop_img: np.ndarray,
        line_crop_img: np.ndarray,
        row_record: rows.RowRecord,
        feature_flags: ModalPipelineFeatureFlags,
    ) -> Mapping[str, Any]:
        ...


class ImagePromptClient(Protocol):
    """Minimal image+prompt client interface for baseline OCR runs."""

    def chat_with_image_prompt(
        self,
        image_path: Path,
        prompt: str,
        system_prompt: str | None = None,
    ) -> str:
        ...


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
        node: SubproofNode,
        node_crop_img: np.ndarray,
        line_crop_img: np.ndarray,
        row_record: rows.RowRecord,
        feature_flags: ModalPipelineFeatureFlags,
    ) -> Mapping[str, Any]:
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


def run_modal_baseline_pipeline(
    *,
    image: np.ndarray,
    read_line: ModalLineReader,
    feature_flags: ModalPipelineFeatureFlags | None = None,
) -> ModalPipelineResult:
    """Run the minimal recursive modal baseline on one proof image."""

    flags = feature_flags or ModalPipelineFeatureFlags()
    _ensure_baseline_feature_support(flags)

    detected_bars = tree.detect_vertical_bars(image)
    tree_root = tree.build_proof_tree(detected_bars, image.shape[1], image.shape[0])
    library: Dict[str, List[Dict[str, Any]]] = {}
    row_records_by_node: Dict[str, List[rows.RowRecord]] = {}

    _process_node_recursive(
        node=tree_root,
        full_img=image,
        library=library,
        row_records_by_node=row_records_by_node,
        read_line=read_line,
        feature_flags=flags,
    )

    root_entries = library.get(tree_root.id, [])
    final_structure = stitch.recursive_stitch(
        root_entries,
        library,
        resolve_item=modal.resolve_modal_ocr_entry,
    )
    while isinstance(final_structure, list) and len(final_structure) == 1 and isinstance(final_structure[0], list):
        final_structure = final_structure[0]
    stitch.renumber_lines(final_structure)

    return ModalPipelineResult(
        detected_bars=detected_bars,
        tree_root=tree_root,
        library=library,
        row_records_by_node=row_records_by_node,
        final_structure=final_structure,
        feature_flags=flags,
    )


def _ensure_baseline_feature_support(feature_flags: ModalPipelineFeatureFlags) -> None:
    enabled_layers = feature_flags.enabled_layers()
    if enabled_layers:
        joined = ", ".join(enabled_layers)
        raise NotImplementedError(
            f"modal_v2 baseline does not implement optional layers yet: {joined}"
        )


def _process_node_recursive(
    *,
    node: SubproofNode,
    full_img: np.ndarray,
    library: Dict[str, List[Dict[str, Any]]],
    row_records_by_node: Dict[str, List[rows.RowRecord]],
    read_line: ModalLineReader,
    feature_flags: ModalPipelineFeatureFlags,
) -> None:
    for child in sorted(node.children, key=lambda item: (item.box[1], item.box[0])):
        _process_node_recursive(
            node=child,
            full_img=full_img,
            library=library,
            row_records_by_node=row_records_by_node,
            read_line=read_line,
            feature_flags=feature_flags,
        )

    node_crop_img = _node_crop_image(node=node, full_img=full_img)
    masked_img = node_crop_img.copy()
    for child in sorted(node.children, key=lambda item: (item.box[1], item.box[0])):
        masked_img = _mask_child_scope_in_node_crop(node=node, node_crop_img=masked_img, child=child)

    row_records = _segment_node_rows(node=node, node_crop_img=masked_img)
    row_records_by_node[node.id] = [dict(record) for record in row_records]

    entries: List[Dict[str, Any]] = []
    for row_record in row_records:
        if str(row_record.get("kind") or "") == "subproof_token":
            child_id = str(row_record.get("child_id") or "").strip()
            entries.append(modal.build_subproof_token_entry(child_id))
            continue

        line_crop_img = _extract_ocr_line_crop(
            node_crop_img=masked_img,
            row_record=row_record,
        )
        raw_entry = dict(
            read_line(
                node=node,
                node_crop_img=masked_img,
                line_crop_img=line_crop_img,
                row_record=row_record,
                feature_flags=feature_flags,
            )
        )
        raw_entry.setdefault("scope_role", "derived_line")
        raw_entry["_ocr_node_id"] = node.id
        raw_entry["_ocr_row_order"] = int(row_record.get("row_order", 0))
        entries.append(raw_entry)

    library[node.id] = entries


def _node_crop_image(*, node: SubproofNode, full_img: np.ndarray) -> np.ndarray:
    if node.id == "ROOT":
        return full_img.copy()
    return tree.crop_region(full_img, node.box)


def _mask_child_scope_in_node_crop(
    *,
    node: SubproofNode,
    node_crop_img: np.ndarray,
    child: SubproofNode,
) -> np.ndarray:
    masked = node_crop_img.copy()
    x, y, _, h = masking.relative_child_box(node, child)
    img_h, img_w = masked.shape[:2]
    erase_left = max(0, int(x))
    erase_top = max(0, int(y))
    erase_bottom = min(img_h, int(y + h))
    cv2.rectangle(masked, (erase_left, erase_top), (img_w, erase_bottom), (255, 255, 255), -1)

    token = f"<SUBPROOF_ID_{child.id}>"
    start_x = min(max(8, erase_left + 20), max(8, img_w - 40))
    center_y = masking.token_center_y_for_child(node, child, img_h)
    max_width = max(60, img_w - start_x - 8)
    scale, thickness, (left, top, right, bottom) = masking.fit_token_layout(
        token,
        start_x=start_x,
        center_y=center_y,
        max_width=max_width,
    )
    cv2.rectangle(masked, (left, top), (right, bottom), (0, 0, 255), 2)
    cv2.putText(
        masked,
        token,
        (left + 8, center_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 255),
        thickness,
    )
    return masked


def _suppress_left_scope_bar(node_crop_img: np.ndarray) -> np.ndarray:
    if node_crop_img.size == 0:
        return node_crop_img

    cleaned = node_crop_img.copy()
    detected_bars = tree.detect_vertical_bars(cleaned)
    if not detected_bars:
        return cleaned

    img_h, img_w = cleaned.shape[:2]
    leftmost = min(detected_bars, key=lambda box: box[0])
    x, _, w, h = leftmost
    if x <= max(24, int(img_w * 0.18)) and h >= max(20, int(img_h * 0.45)):
        erase_right = min(img_w, x + w + 6)
        cv2.rectangle(cleaned, (0, 0), (erase_right, img_h), (255, 255, 255), -1)
        return cleaned

    gray = cv2.cvtColor(cleaned, cv2.COLOR_BGR2GRAY)
    dark_mask = gray < 170
    left_limit = min(img_w, max(12, int(img_w * 0.18)))
    if left_limit <= 0:
        return cleaned

    col_counts = dark_mask[:, :left_limit].sum(axis=0)
    threshold = max(8, int(img_h * 0.55))
    active_cols = [idx for idx, count in enumerate(col_counts) if int(count) >= threshold]
    if not active_cols:
        return cleaned

    groups: List[tuple[int, int]] = []
    start = active_cols[0]
    prev = active_cols[0]
    for idx in active_cols[1:]:
        if idx == prev + 1:
            prev = idx
            continue
        groups.append((start, prev))
        start = idx
        prev = idx
    groups.append((start, prev))

    left, right = max(groups, key=lambda group: (group[1], group[1] - group[0]))
    erase_right = min(img_w, right + 5)
    cv2.rectangle(cleaned, (0, 0), (erase_right, img_h), (255, 255, 255), -1)
    return cleaned


def _extract_band_crop(*, node_crop_img: np.ndarray, bands: List[rows.Band], band_index: int) -> np.ndarray:
    if not bands or band_index < 0 or band_index >= len(bands):
        return node_crop_img
    band_top, band_bottom = bands[band_index]
    return rows.extract_line_row_crop(
        node_crop_img=node_crop_img,
        y1=band_top,
        y2=band_bottom,
        suppress_left_scope_bar=_suppress_left_scope_bar,
    )


def _extract_ocr_line_crop(
    *,
    node_crop_img: np.ndarray,
    row_record: rows.RowRecord,
) -> np.ndarray:
    """Extract the line image used for OCR without reapplying bar suppression.

    Row segmentation benefits from whitening the left proof bar. The OCR crop
    should keep the original visible formula area intact; otherwise wide root
    rows can lose everything except the line number.
    """

    return rows.extract_line_row_crop(
        node_crop_img=node_crop_img,
        y1=int(row_record.get("y1", 0)),
        y2=int(row_record.get("y2", 0)),
        suppress_left_scope_bar=lambda crop: crop,
    )


def _band_has_textlike_ink(node_crop_img: np.ndarray, band: rows.Band) -> bool:
    return rows.band_has_textlike_ink(
        node_crop_img,
        band,
        extract_band_crop_fn=_extract_band_crop,
    )


def _band_matches_child_token_artifact(
    *,
    band: rows.Band,
    node: SubproofNode,
    node_crop_img: np.ndarray,
) -> bool:
    return rows.band_matches_child_token_artifact(
        band=band,
        node=node,
        node_crop_img=node_crop_img,
        token_rect_for_child=masking.token_rect_for_child,
        band_has_text_outside_vertical_window_fn=lambda image, candidate_band, *, window_y1, window_y2: rows.band_has_text_outside_vertical_window(
            image,
            candidate_band,
            window_y1=window_y1,
            window_y2=window_y2,
            suppress_left_scope_bar=_suppress_left_scope_bar,
        ),
    )


def _filter_bands_near_child_tokens(
    *,
    bands: List[rows.Band],
    node: SubproofNode,
    node_crop_img: np.ndarray,
) -> List[rows.Band]:
    return rows.filter_bands_near_child_tokens(
        bands=bands,
        node=node,
        node_crop_img=node_crop_img,
        band_matches_child_token_artifact_fn=_band_matches_child_token_artifact,
    )


def _segment_node_rows(*, node: SubproofNode, node_crop_img: np.ndarray) -> List[rows.RowRecord]:
    return rows.segment_node_rows(
        node=node,
        node_crop_img=node_crop_img,
        suppress_left_scope_bar=_suppress_left_scope_bar,
        detect_text_row_bands_fn=rows.detect_text_row_bands,
        band_has_textlike_ink_fn=_band_has_textlike_ink,
        merge_fragment_bands_fn=rows.merge_fragment_bands,
        filter_bands_near_child_tokens_fn=_filter_bands_near_child_tokens,
        token_rect_for_child_fn=masking.token_rect_for_child,
    )


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


def build_modal_v2_state(
    *,
    modal_config: Mapping[str, Any] | None,
    final_structure: List[Any],
    run_summary: str,
    vision_base_url: str | None = None,
    vision_model: str | None = None,
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
        "verifier_vision_model": "",
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
) -> dict[str, Any]:
    """Run the modal_v2 baseline end to end and write debug artifacts."""

    image_path = image_path.resolve()
    output_dir = output_dir.resolve()
    preserved = [image_path] if image_path.parent == output_dir else []
    artifacts.clear_debug_dir(output_dir, preserve_paths=preserved)

    if client is None:
        client = build_vlm_client(base_url=vision_base_url, model=vision_model)

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
        ),
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
    state = build_modal_v2_state(
        modal_config=modal_config,
        final_structure=result.final_structure,
        run_summary=run_summary,
        vision_base_url=vision_base_url,
        vision_model=vision_model,
        source_image_name=source_copy_name,
    )
    (output_dir / artifacts.STATE_FILENAME).write_text(
        json.dumps(state, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return state


def _index_tree(root: SubproofNode) -> Dict[str, SubproofNode]:
    lookup = {root.id: root}
    for child in root.children:
        lookup.update(_index_tree(child))
    return lookup


def _masked_node_crop_image(*, node: SubproofNode, full_img: np.ndarray) -> np.ndarray:
    masked = _node_crop_image(node=node, full_img=full_img)
    for child in sorted(node.children, key=lambda item: (item.box[1], item.box[0])):
        masked = _mask_child_scope_in_node_crop(node=node, node_crop_img=masked, child=child)
    return masked


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the modal_v2 baseline OCR pipeline on one image.")
    parser.add_argument("--image", required=True, help="Path to the proof image.")
    parser.add_argument("--output-dir", required=True, help="Directory for baseline debug artifacts.")
    parser.add_argument("--system", default="K", help="Modal system label, e.g. K, T, S4, or S5.")
    parser.add_argument("--include-derived-rules", action="store_true", help="Tell the OCR prompt derived rules are in play.")
    parser.add_argument("--conclusion", default="", help="Optional target conclusion shown in the OCR prompt.")
    parser.add_argument("--vision-base-url", default="", help="LM Studio base URL.")
    parser.add_argument("--vision-model", default="", help="LM Studio model identifier.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    state = transcribe_image_to_modal_v2_debug_state(
        image_path=Path(args.image),
        output_dir=Path(args.output_dir),
        modal_config={
            "system": args.system,
            "include_derived_rules": bool(args.include_derived_rules),
            "conclusion": args.conclusion,
        },
        vision_base_url=args.vision_base_url or None,
        vision_model=args.vision_model or None,
    )
    print(state["run_summary"])
    print(f"Artifacts written to {Path(args.output_dir).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
