"""Core recursive mechanics for the modal_v2 baseline pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Protocol

import cv2
import numpy as np

from proofs_ocr.common import masking, rows, stitch, tree
from proofs_ocr.common.types import Box, SubproofNode
from proofs_ocr.domains import modal
from proofs_ocr.pipelines.modal_v2_bar_filters import filter_false_left_separator_bars


@dataclass(frozen=True)
class ModalPipelineFeatureFlags:
    """Optional layers reserved for later experiments."""

    enable_strict_opener_classifier: bool = False
    enable_short_formula_reread: bool = False
    enable_retry_on_suspicious_line: bool = False
    enable_unary_verifier: bool = False
    enable_boundary_repair: bool = False
    enable_sparse_lower_compact_band_recovery: bool = False
    enable_contradiction_reread: bool = False
    enable_justification_head_reread: bool = False
    enable_formula_consensus_reread: bool = False

    def enabled_layers(self) -> List[str]:
        enabled: List[str] = []
        for field_name, value in self.__dict__.items():
            if bool(value):
                enabled.append(field_name)
        return enabled


@dataclass
class ModalPipelineResult:
    """Artifacts returned by one baseline pipeline run."""

    raw_detected_bars: List[Box]
    detected_bars: List[Box]
    tree_root: SubproofNode
    library: Dict[str, List[Dict[str, Any]]]
    row_records_by_node: Dict[str, List[rows.RowRecord]]
    band_debug_by_node: Dict[str, Dict[str, Any]]
    final_structure: List[Any]
    feature_flags: ModalPipelineFeatureFlags
    bar_filter_debug: Dict[str, Any]


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


def run_modal_baseline_pipeline(
    *,
    image: np.ndarray,
    read_line: ModalLineReader,
    feature_flags: ModalPipelineFeatureFlags | None = None,
    enable_false_left_separator_filter: bool = False,
    false_left_separator_verifier_client: ImagePromptClient | None = None,
    false_left_separator_output_dir: Path | None = None,
) -> ModalPipelineResult:
    """Run the minimal recursive modal baseline on one proof image."""

    flags = feature_flags or ModalPipelineFeatureFlags()
    _ensure_baseline_feature_support(flags)

    raw_detected_bars = tree.detect_vertical_bars(image)
    detected_bars = list(raw_detected_bars)
    bar_filter_debug: Dict[str, Any] = {}
    if enable_false_left_separator_filter:
        detected_bars, bar_filter_debug = filter_false_left_separator_bars(
            image=image,
            bars=raw_detected_bars,
            verifier_client=false_left_separator_verifier_client,
            output_dir=false_left_separator_output_dir,
        )
    tree_root = tree.build_proof_tree(detected_bars, image.shape[1], image.shape[0])
    library: Dict[str, List[Dict[str, Any]]] = {}
    row_records_by_node: Dict[str, List[rows.RowRecord]] = {}
    band_debug_by_node: Dict[str, Dict[str, Any]] = {}

    _process_node_recursive(
        node=tree_root,
        full_img=image,
        library=library,
        row_records_by_node=row_records_by_node,
        band_debug_by_node=band_debug_by_node,
        read_line=read_line,
        feature_flags=flags,
    )

    root_entries = _select_top_level_entries(tree_root=tree_root, library=library)
    final_structure = stitch.recursive_stitch(
        root_entries,
        library,
        resolve_item=modal.resolve_modal_ocr_entry,
    )
    while isinstance(final_structure, list) and len(final_structure) == 1 and isinstance(final_structure[0], list):
        final_structure = final_structure[0]
    stitch.renumber_lines(final_structure)

    return ModalPipelineResult(
        raw_detected_bars=raw_detected_bars,
        detected_bars=detected_bars,
        tree_root=tree_root,
        library=library,
        row_records_by_node=row_records_by_node,
        band_debug_by_node=band_debug_by_node,
        final_structure=final_structure,
        feature_flags=flags,
        bar_filter_debug=bar_filter_debug,
    )


def _select_top_level_entries(
    *,
    tree_root: SubproofNode,
    library: Mapping[str, List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """Choose the stitched top level, allowing a visually empty ROOT container."""

    root_entries = list(library.get(tree_root.id, []))
    root_has_visible_lines = any(modal.has_visible_formula(entry) for entry in root_entries)
    if not root_has_visible_lines and "1" in library:
        return list(library["1"])
    return root_entries


def _ensure_baseline_feature_support(feature_flags: ModalPipelineFeatureFlags) -> None:
    enabled_layers = [
        layer
        for layer in feature_flags.enabled_layers()
        if layer
        not in {
            "enable_sparse_lower_compact_band_recovery",
            "enable_contradiction_reread",
            "enable_justification_head_reread",
            "enable_formula_consensus_reread",
        }
    ]
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
    band_debug_by_node: Dict[str, Dict[str, Any]],
    read_line: ModalLineReader,
    feature_flags: ModalPipelineFeatureFlags,
) -> None:
    for child in sorted(node.children, key=lambda item: (item.box[1], item.box[0])):
        _process_node_recursive(
            node=child,
            full_img=full_img,
            library=library,
            row_records_by_node=row_records_by_node,
            band_debug_by_node=band_debug_by_node,
            read_line=read_line,
            feature_flags=feature_flags,
        )

    node_crop_img = _node_crop_image(node=node, full_img=full_img)
    masked_img = node_crop_img.copy()
    for child in sorted(node.children, key=lambda item: (item.box[1], item.box[0])):
        masked_img = _mask_child_scope_in_node_crop(node=node, node_crop_img=masked_img, child=child)

    band_debug: Dict[str, Any] = {}
    row_records = _segment_node_rows(
        node=node,
        node_crop_img=masked_img,
        feature_flags=feature_flags,
        write_band_debug_artifacts=lambda payload: band_debug.update(payload),
    )
    row_records_by_node[node.id] = [dict(record) for record in row_records]
    band_debug_by_node[node.id] = dict(band_debug)

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
    return tree.crop_region(full_img, node.box, bottom_padding=0)


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
    """Extract the line image used for OCR without reapplying bar suppression."""

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


def _segment_node_rows(
    *,
    node: SubproofNode,
    node_crop_img: np.ndarray,
    feature_flags: ModalPipelineFeatureFlags,
    write_band_debug_artifacts: Callable[[Dict[str, Any]], None] | None = None,
) -> List[rows.RowRecord]:
    return rows.segment_node_rows(
        node=node,
        node_crop_img=node_crop_img,
        suppress_left_scope_bar=_suppress_left_scope_bar,
        detect_text_row_bands_fn=lambda crop_img: rows.detect_text_row_bands(
            crop_img,
            enable_sparse_lower_compact_band_recovery=feature_flags.enable_sparse_lower_compact_band_recovery,
        ),
        band_has_textlike_ink_fn=_band_has_textlike_ink,
        merge_fragment_bands_fn=rows.merge_fragment_bands,
        filter_bands_near_child_tokens_fn=_filter_bands_near_child_tokens,
        token_rect_for_child_fn=masking.token_rect_for_child,
        write_band_debug_artifacts=write_band_debug_artifacts,
    )


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
