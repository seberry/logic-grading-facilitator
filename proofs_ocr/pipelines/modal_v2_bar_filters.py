"""Early bar-filtering experiments for modal_v2 tree construction."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

from proofs_ocr.common import masking, rows, tree
from proofs_ocr.common.types import Box, SubproofNode
from proofs_ocr.pipelines.modal_v2_bar_verifier import classify_false_left_separator_candidate


def filter_false_left_separator_bars(
    *,
    image: np.ndarray,
    bars: Sequence[Box],
    verifier_client: Any | None = None,
    output_dir: Path | None = None,
) -> tuple[list[Box], dict[str, Any]]:
    """Optionally drop one suspicious full-height left separator before tree build."""

    img_h, img_w = image.shape[:2]
    raw_bars = [tuple(int(value) for value in box) for box in bars]
    payload: dict[str, Any] = {
        "raw_bars": [list(box) for box in raw_bars],
        "filtered_bars": [list(box) for box in raw_bars],
        "rejected_bars": [],
        "applied": False,
        "reason": "",
        "candidate_bar": None,
        "code_signals": {},
        "verifier": {},
    }
    if len(raw_bars) < 2:
        payload["reason"] = "too_few_bars"
        return list(raw_bars), payload

    leftmost = min(raw_bars, key=lambda box: (box[0], box[1], -box[3]))
    x, y, w, h = leftmost
    payload["candidate_bar"] = list(leftmost)

    right_bars = [box for box in raw_bars if box != leftmost and box[0] >= x + max(12, w + 4)]
    next_bar = min(right_bars, key=lambda box: (box[0], box[1])) if right_bars else None

    code_signals = {
        "leftmost_near_edge": bool(x <= max(24, int(img_w * 0.12))),
        "leftmost_top_anchored": bool(y <= max(8, int(img_h * 0.05))),
        "leftmost_tall_enough": bool(h >= max(80, int(img_h * 0.78))),
        "has_right_bar": bool(next_bar is not None),
        "right_bar_count": len(right_bars),
    }
    payload["code_signals"] = code_signals

    if not code_signals["leftmost_near_edge"]:
        payload["reason"] = "leftmost_not_near_edge"
        return list(raw_bars), payload
    if not code_signals["leftmost_top_anchored"]:
        payload["reason"] = "leftmost_not_top_anchored"
        return list(raw_bars), payload
    if not code_signals["leftmost_tall_enough"]:
        payload["reason"] = "leftmost_not_tall_enough"
        return list(raw_bars), payload
    if next_bar is None:
        payload["reason"] = "no_right_bar"
        return list(raw_bars), payload

    bars_without_candidate = [box for box in raw_bars if box != leftmost]
    kept_ownership = _evaluate_root_ownership(image=image, bars=raw_bars)
    removed_ownership = _evaluate_root_ownership(image=image, bars=bars_without_candidate)
    root_suspicious = bool(
        kept_ownership["root_line_count"] == 0
        or kept_ownership["first_root_row_kind"] == "subproof_token"
        or kept_ownership["root_rows_before_first_child_token"] == 0
    )
    restores_root_rows = bool(
        removed_ownership["root_rows_before_first_child_token"] > kept_ownership["root_rows_before_first_child_token"]
        and removed_ownership["root_rows_before_first_child_token"] >= 1
    )
    strong_code_only_drop = bool(
        root_suspicious
        and restores_root_rows
        and x <= max(20, int(img_w * 0.09))
        and w >= max(6, next_bar[2] + 2)
        and kept_ownership["root_line_count"] == 0
    )

    code_signals.update(
        {
            "kept_root_ownership": kept_ownership,
            "removed_root_ownership": removed_ownership,
            "suspicious_root_ownership": root_suspicious,
            "restores_root_rows_above_first_child": restores_root_rows,
            "strong_code_only_drop": strong_code_only_drop,
        }
    )
    if not root_suspicious or not restores_root_rows:
        payload["reason"] = "root_ownership_not_suspicious"
        return list(raw_bars), payload

    verifier_payload = classify_false_left_separator_candidate(
        image=image,
        candidate_bar=leftmost,
        neighboring_bars=right_bars,
        verifier_client=verifier_client,
        output_dir=output_dir,
    )
    payload["verifier"] = verifier_payload

    classification = str(verifier_payload.get("classification") or "").strip()
    if classification in {"number_formula_separator", "margin_artifact_or_stamp"}:
        payload["filtered_bars"] = [list(box) for box in bars_without_candidate]
        payload["rejected_bars"] = [list(leftmost)]
        payload["applied"] = True
        payload["reason"] = "verifier_rejected_false_left_separator"
        return bars_without_candidate, payload

    if strong_code_only_drop:
        payload["filtered_bars"] = [list(box) for box in bars_without_candidate]
        payload["rejected_bars"] = [list(leftmost)]
        payload["applied"] = True
        payload["reason"] = "code_only_rejected_false_left_separator"
        return bars_without_candidate, payload

    if classification == "scope_bar":
        payload["reason"] = "verifier_kept_scope_bar"
    elif classification == "uncertain":
        payload["reason"] = "verifier_uncertain_keep_conservative"
    elif verifier_payload.get("error"):
        payload["reason"] = "verifier_unavailable_keep_conservative"
    else:
        payload["reason"] = "candidate_kept"
    return list(raw_bars), payload


def _evaluate_root_ownership(
    *,
    image: np.ndarray,
    bars: Sequence[Box],
) -> dict[str, Any]:
    root = tree.build_proof_tree(list(bars), image.shape[1], image.shape[0])
    masked = _masked_node_crop_image(node=root, full_img=image)
    row_records = _segment_node_rows(node=root, node_crop_img=masked)
    line_rows = [dict(record) for record in row_records if str(record.get("kind") or "") == "line"]
    token_rows = [dict(record) for record in row_records if str(record.get("kind") or "") == "subproof_token"]
    first_token_center = min((float(record.get("center_y", 0.0)) for record in token_rows), default=None)
    if first_token_center is None:
        rows_before_first_token = len(line_rows)
    else:
        rows_before_first_token = sum(
            1 for record in line_rows if float(record.get("center_y", 0.0)) < first_token_center
        )
    return {
        "node_count": _count_tree_nodes(root),
        "root_child_count": len(root.children),
        "root_row_records": [dict(record) for record in row_records],
        "root_line_count": len(line_rows),
        "root_token_count": len(token_rows),
        "root_rows_before_first_child_token": int(rows_before_first_token),
        "first_root_row_kind": str(row_records[0].get("kind") or "") if row_records else "",
    }


def _count_tree_nodes(node: SubproofNode) -> int:
    return 1 + sum(_count_tree_nodes(child) for child in node.children)


def _node_crop_image(*, node: SubproofNode, full_img: np.ndarray) -> np.ndarray:
    if node.id == "ROOT":
        return full_img.copy()
    return tree.crop_region(full_img, node.box, bottom_padding=0)


def _masked_node_crop_image(*, node: SubproofNode, full_img: np.ndarray) -> np.ndarray:
    masked = _node_crop_image(node=node, full_img=full_img)
    for child in sorted(node.children, key=lambda item: (item.box[1], item.box[0])):
        masked = _mask_child_scope_in_node_crop(node=node, node_crop_img=masked, child=child)
    return masked


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

    groups: list[tuple[int, int]] = []
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


def _extract_band_crop(*, node_crop_img: np.ndarray, bands: list[rows.Band], band_index: int) -> np.ndarray:
    if not bands or band_index < 0 or band_index >= len(bands):
        return node_crop_img
    band_top, band_bottom = bands[band_index]
    return rows.extract_line_row_crop(
        node_crop_img=node_crop_img,
        y1=band_top,
        y2=band_bottom,
        suppress_left_scope_bar=_suppress_left_scope_bar,
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
    bands: list[rows.Band],
    node: SubproofNode,
    node_crop_img: np.ndarray,
) -> list[rows.Band]:
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
) -> list[rows.RowRecord]:
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
