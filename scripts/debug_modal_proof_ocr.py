from __future__ import annotations

import argparse
import base64
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from modal_ocr.artifacts import (
    BASELINE_FINAL_STRUCTURE_FILENAME,
    CHECKER_REQUEST_FILENAME,
    CHECKER_RESPONSE_FILENAME,
    FINAL_STRUCTURE_FILENAME,
    INTERMEDIATE_TEMPLATE,
    NODE_ENTRIES_TEMPLATE,
    NODE_LINE_CROP_TEMPLATE,
    NODE_LINE_FORMULA_CROP_TEMPLATE,
    NODE_LINE_FORMULA_REREAD_CROP_TEMPLATE,
    NODE_LINE_FORMULA_RETRY_CROP_TEMPLATE,
    NODE_LINE_JUSTIFICATION_CROP_TEMPLATE,
    NODE_LINE_JUSTIFICATION_RETRY_CROP_TEMPLATE,
    NODE_LINE_RESPONSE_TEMPLATE,
    NODE_LINE_RETRY_CROP_TEMPLATE,
    NODE_ROWS_OVERLAY_TEMPLATE,
    NODE_ROWS_TEMPLATE,
    PROMPT_FILENAME,
    RAW_RESPONSE_TEMPLATE,
    SOURCE_IMAGE_FILENAME,
    STATE_FILENAME,
    STRICT_OPENER_LINE_TEMPLATE,
    STRICT_OPENER_RESPONSE_TEMPLATE,
    UNARY_RECHECK_FILENAME,
    UNARY_VERIFIER_FILENAME,
    VERIFIER_REWRITES_FILENAME,
)
from modal_ocr.prompts import (
    build_formula_ocr_prompt,
    build_justification_ocr_prompt,
    build_line_ocr_prompt,
    build_prompt,
)
from modal_ocr.shared import (
    JUSTIFICATION_HEAD_OPTIONS,
    JUSTIFICATION_HEAD_OPTION_SET,
    LINE_SCOPE_ROLE_OPTIONS,
    LINE_SCOPE_ROLE_OPTION_SET,
    SUPPORTED_SYSTEMS,
    TOKEN_PATTERN,
    SubproofNode,
    canonical_modal_config as _canonical_modal_config,
    load_modal_config as _load_modal_config,
)
from modal_ocr.verifier import (
    SHORT_FORMULA_REREAD_RULE_HEADS,
    UNARY_PREFIX_CHARS,
    UNARY_SENSITIVE_RULE_HEADS,
    apply_unary_verifier_corrections as _apply_unary_verifier_corrections_impl,
    best_verifier_attempt as _best_verifier_attempt_impl,
    build_forced_choice_unary_verifier_prompt as _build_forced_choice_unary_verifier_prompt_impl,
    build_node_lookup as _build_node_lookup_impl,
    build_structured_unary_verifier_prompt as _build_structured_unary_verifier_prompt_impl,
    candidate_band_indices as _candidate_band_indices_impl,
    confidence_score as _confidence_score_impl,
    leading_unary_prefix as _leading_unary_prefix_impl,
    margin_anchor_strip_bounds as _margin_anchor_strip_bounds_impl,
    merge_margin_candidate_boxes as _merge_margin_candidate_boxes_impl,
    run_unary_verifier_experiment as _run_unary_verifier_experiment_impl,
    run_unary_postprocess,
    should_verify_unary_formula as _should_verify_unary_formula_impl,
    verifier_node_crop_image as _verifier_node_crop_image_impl,
)

try:
    from app.modal_proof_deduct import build_modal_proof_request, run_modal_proof_checker
    from app.storage import DEBUG_MODAL_PROOF_DIR, RAW_SCANS_DIR
    from app.vlm_client import LMStudioQwenClient, default_lmstudio_base_url, default_lmstudio_model

    vlm_client = LMStudioQwenClient()
except ImportError as exc:  # pragma: no cover - fallback for standalone use
    print(f"Warning: Could not import app modules ({exc}).")
    RAW_SCANS_DIR = Path(project_root) / "data" / "raw_scans"
    DEBUG_MODAL_PROOF_DIR = Path(project_root) / "data" / "debug_modal_proof"
    vlm_client = None

    def default_lmstudio_base_url() -> str:
        return os.getenv("LMSTUDIO_BASE_URL", "http://127.0.0.1:1234/v1").strip() or "http://127.0.0.1:1234/v1"

    def default_lmstudio_model() -> str:
        return os.getenv("LMSTUDIO_MODEL_ID", "qwen/qwen3-vl-4b").strip() or "qwen/qwen3-vl-4b"


def default_unary_verifier_model() -> str:
    return os.getenv("LMSTUDIO_VERIFIER_MODEL_ID", "qwen/qwen3-vl-8b").strip() or "qwen/qwen3-vl-8b"


def build_vlm_client(
    *,
    base_url: str | None = None,
    model: str | None = None,
    timeout: int = 60,
) -> LMStudioQwenClient:
    return LMStudioQwenClient(
        base_url=(base_url or default_lmstudio_base_url()).strip(),
        model=(model or default_lmstudio_model()).strip(),
        timeout=timeout,
    )


def detect_vertical_bars(img: np.ndarray) -> List[Tuple[int, int, int, int]]:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        91,
        15,
    )
    dilate_kernel = np.ones((1, 6), np.uint8)
    dilated = cv2.dilate(binary, dilate_kernel, iterations=1)
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 80))
    vertical_mask = cv2.morphologyEx(dilated, cv2.MORPH_OPEN, vertical_kernel)
    contours, _ = cv2.findContours(vertical_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [cv2.boundingRect(c) for c in contours]
    boxes.sort(key=lambda box: (box[0], box[1]))

    merged: List[Tuple[int, int, int, int]] = []
    if not boxes:
        return merged

    curr_x, curr_y, curr_w, curr_h = boxes[0]
    for nx, ny, nw, nh in boxes[1:]:
        if abs(curr_x - nx) < 20 and (ny - (curr_y + curr_h) < 100):
            new_x = min(curr_x, nx)
            new_y = min(curr_y, ny)
            new_w = max(curr_x + curr_w, nx + nw) - new_x
            new_h = max(curr_y + curr_h, ny + nh) - new_y
            curr_x, curr_y, curr_w, curr_h = new_x, new_y, new_w, new_h
        else:
            merged.append((curr_x, curr_y, curr_w, curr_h))
            curr_x, curr_y, curr_w, curr_h = nx, ny, nw, nh
    merged.append((curr_x, curr_y, curr_w, curr_h))
    refined = [_refine_vertical_bar_box(img, box) for box in merged]
    return [box for box in refined if box[3] > 50]


def _refine_vertical_bar_box(img: np.ndarray, box: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
    x, y, w, h = box
    img_h, img_w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(img_w, x + w)
    y2 = min(img_h, y + h)
    roi = gray[y1:y2, x1:x2]
    if roi.size == 0:
        return box

    dark_mask = roi < 140
    col_counts = dark_mask.sum(axis=0)
    min_count = max(20, int(0.2 * roi.shape[0]))
    active_cols = [idx for idx, count in enumerate(col_counts) if count >= min_count]
    if not active_cols:
        return box

    groups: List[Tuple[int, int]] = []
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

    # Prefer the rightmost substantial vertical stroke inside the merged contour.
    best_left, best_right = max(
        groups,
        key=lambda group: (group[1], int(col_counts[group[0] : group[1] + 1].sum())),
    )
    group_mask = dark_mask[:, best_left : best_right + 1]
    active_rows = np.where(group_mask.any(axis=1))[0]
    if len(active_rows) == 0:
        return box

    refined_x = x1 + best_left
    refined_y = y1 + int(active_rows[0])
    refined_w = best_right - best_left + 1
    refined_h = int(active_rows[-1] - active_rows[0] + 1)
    return (refined_x, refined_y, refined_w, refined_h)


def build_proof_tree(boxes: List[Tuple[int, int, int, int]], img_w: int, img_h: int) -> SubproofNode:
    root = SubproofNode("ROOT", (0, 0, img_w, img_h))
    nodes = [SubproofNode(str(i + 1), box) for i, box in enumerate(sorted(boxes, key=lambda b: b[0]))]
    for node in nodes:
        best_parent = root
        nx, ny, _, nh = node.box
        center_y = ny + nh / 2
        for candidate in reversed(nodes):
            if candidate is node:
                continue
            if candidate.box[0] >= nx:
                continue
            cx, cy, _, ch = candidate.box
            if cy <= center_y <= (cy + ch):
                best_parent = candidate
                break
        node.parent = best_parent
        best_parent.children.append(node)
    return root


def crop_region(
    img: np.ndarray,
    box: Tuple[int, int, int, int],
    *,
    x_padding: int = 10,
    top_padding: int = 10,
    bottom_padding: int = 80,
) -> np.ndarray:
    x, y, w, h = box
    img_h, img_w = img.shape[:2]
    x1 = max(0, x - x_padding)
    y1 = max(0, y - top_padding)
    x2 = img_w
    y2 = min(img_h, y + h + bottom_padding)
    return img[y1:y2, x1:x2]


def _fit_token_layout(
    token: str,
    *,
    start_x: int,
    center_y: int,
    max_width: int,
) -> Tuple[float, int, Tuple[int, int, int, int]]:
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2
    pad = 8
    scale = 1.0
    min_scale = 0.4

    while scale >= min_scale:
        (tw, th), baseline = cv2.getTextSize(token, font, scale, thickness)
        total_width = tw + 2 * pad
        if total_width <= max_width:
            left = start_x
            top = max(0, center_y - th - pad)
            right = left + total_width
            bottom = center_y + pad
            return scale, thickness, (left, top, right, bottom)
        scale -= 0.1

    scale = min_scale
    (tw, th), baseline = cv2.getTextSize(token, font, scale, thickness)
    total_width = min(max_width, tw + 2 * pad)
    left = start_x
    top = max(0, center_y - th - pad)
    right = left + total_width
    bottom = center_y + pad
    return scale, thickness, (left, top, right, bottom)


def mask_subproof_in_parent(parent_img: np.ndarray, child_node: SubproofNode) -> np.ndarray:
    masked = parent_img.copy()
    x, y, _, h = child_node.box
    img_h, img_w = masked.shape[:2]
    cv2.rectangle(masked, (x, y), (img_w, y + h), (255, 255, 255), -1)

    token = f"<SUBPROOF_ID_{child_node.id}>"
    font = cv2.FONT_HERSHEY_SIMPLEX
    start_x = min(max(8, x + 20), max(8, img_w - 40))
    center_y = min(max(24, y + max(18, h // 2)), img_h - 12)
    max_width = max(60, img_w - start_x - 8)
    scale, thickness, (left, top, right, bottom) = _fit_token_layout(
        token,
        start_x=start_x,
        center_y=center_y,
        max_width=max_width,
    )
    cv2.rectangle(masked, (left, top), (right, bottom), (0, 0, 255), 2)
    cv2.putText(masked, token, (left + 8, center_y), font, scale, (0, 0, 255), thickness)
    return masked


def _strip_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        parts = stripped.split("```")
        if len(parts) >= 3:
            middle = parts[1]
            if "\n" in middle:
                middle = middle.split("\n", 1)[1]
            return middle.strip()
    return stripped


def _parse_intermediate_response(raw: str) -> List[Dict[str, Any]]:
    cleaned = _strip_fences(raw)
    parsed = json.loads(cleaned)
    if isinstance(parsed, dict):
        entries = parsed.get("entries") or parsed.get("lines") or []
    elif isinstance(parsed, list):
        entries = parsed
    else:
        raise ValueError("OCR response must be a JSON list or object containing entries/lines")
    if not isinstance(entries, list):
        raise ValueError("OCR entries must be a list")
    return [item for item in entries if isinstance(item, dict)]


def call_vlm(image_path: Path, prompt: str, client: LMStudioQwenClient | None) -> Tuple[str, List[Dict[str, Any]]]:
    if client is None:
        raise RuntimeError("VLM client is not available")

    with image_path.open("rb") as handle:
        base64_image = base64.b64encode(handle.read()).decode("utf-8")

    messages = [
        {"role": "system", "content": prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Return the JSON transcription for this proof crop."},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}},
            ],
        },
    ]

    raw = client.chat(messages)
    return raw, _parse_intermediate_response(raw)


def _call_vlm_json_object(
    *,
    image_path: Path,
    prompt: str,
    client: LMStudioQwenClient | None,
    user_text: str,
) -> Tuple[str, Dict[str, Any]]:
    if client is None:
        raise RuntimeError("VLM client is not available")

    with image_path.open("rb") as handle:
        base64_image = base64.b64encode(handle.read()).decode("utf-8")

    raw = client.chat(
        [
            {"role": "system", "content": "You are a careful vision model. Follow the user's instructions exactly."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "text", "text": user_text},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}} ,
                ],
            },
        ]
    )
    return raw, _parse_unary_recheck_choice(raw)


def _build_subproof_opener_prompt() -> str:
    lines = [
        "You are classifying the first visible line of a handwritten Fitch subproof.",
        "Return only JSON. Do not include markdown fences or explanation.",
        'Return one JSON object with keys: kind, formula_text, confidence.',
        'Set kind to exactly one of: "strict_open", "ordinary_assumption", "uncertain".',
        'Use "strict_open" only when the line itself is just a lone modal box symbol (□ or handwritten []).',
        'Use "ordinary_assumption" when the line shows an actual formula such as P, ¬Q, or P→Q.',
        "Ignore the vertical scope bar and other box-border decoration; classify only the first proof line's content.",
        'If kind is "strict_open", set formula_text to "□".',
        'If kind is "ordinary_assumption", set formula_text to the visible formula.',
        'If you cannot tell, return {"kind":"uncertain","formula_text":"","confidence":"low"}.',
    ]
    return "\n".join(lines)


def _build_subproof_opener_formula_probe_prompt() -> str:
    lines = [
        "This image shows the first line of a handwritten modal Fitch subproof.",
        "Return only JSON with key formula_text.",
        "Report exactly what symbol or formula appears on the line.",
        'If the line is only a modal box symbol, you may return "□" or "[]".',
        "Do not include justification text or explanation.",
    ]
    return "\n".join(lines)


def _normalize_subproof_opener_formula_probe(text: Any) -> str:
    raw = str(text or "").strip()
    if raw in {"□", "[]", "D", "d", "O", "o", "0"}:
        return "□"
    normalized = _normalize_formula_text(raw, "derived_line")
    return "□" if normalized == "□" else normalized


def _classify_subproof_opener(
    *,
    node_crop_img: np.ndarray,
    output_dir: Path,
    node_id: str,
    client: LMStudioQwenClient | None,
) -> Dict[str, Any] | None:
    if client is None or node_crop_img.size == 0:
        return None

    processed_crop = _suppress_left_scope_bar(node_crop_img)
    bands = _detect_text_row_bands(processed_crop)
    if bands:
        line_crop = _extract_band_crop(node_crop_img=processed_crop, bands=bands, band_index=0)
    else:
        line_crop = _extract_single_line_crop(node_crop_img=processed_crop, line_order=0)
    if line_crop.size == 0:
        return None

    crop_path = output_dir / STRICT_OPENER_LINE_TEMPLATE.format(node_id=node_id)
    cv2.imwrite(str(crop_path), line_crop)

    with crop_path.open("rb") as handle:
        base64_image = base64.b64encode(handle.read()).decode("utf-8")

    raw = client.chat(
        [
            {"role": "system", "content": "You are a careful vision model. Follow the user's instructions exactly."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": _build_subproof_opener_prompt()},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}} ,
                ],
            },
        ]
    )
    parsed = _parse_unary_recheck_choice(raw)
    kind = str(parsed.get("kind") or "").strip().lower()
    if kind not in {"strict_open", "ordinary_assumption", "uncertain"}:
        kind = "uncertain"
    formula_text = str(parsed.get("formula_text") or "").strip()
    if kind == "strict_open":
        formula_text = "□"
    elif kind != "ordinary_assumption":
        formula_text = ""
    result = {
        "kind": kind,
        "formula_text": formula_text,
        "confidence": str(parsed.get("confidence") or "").strip(),
    }
    artifact_payload: Dict[str, Any] = {
        "classification_raw": raw,
        "classification": result,
    }

    if result["kind"] == "uncertain":
        probe_raw = client.chat(
            [
                {"role": "system", "content": "You are a careful vision model. Follow the user's instructions exactly."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": _build_subproof_opener_formula_probe_prompt()},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}} ,
                    ],
                },
            ]
        )
        probe_parsed = _parse_unary_recheck_choice(probe_raw)
        probe_formula = _normalize_subproof_opener_formula_probe(probe_parsed.get("formula_text") or "")
        artifact_payload["formula_probe_raw"] = probe_raw
        artifact_payload["formula_probe"] = {
            "formula_text": str(probe_parsed.get("formula_text") or "").strip(),
            "normalized_formula_text": probe_formula,
        }
        if probe_formula == "□":
            result = {
                "kind": "strict_open",
                "formula_text": "□",
                "confidence": result["confidence"] or "fallback",
                "source": "formula_probe",
            }
            artifact_payload["resolved_from_probe"] = True
        else:
            artifact_payload["resolved_from_probe"] = False

    artifact_payload["resolved"] = result
    (output_dir / STRICT_OPENER_RESPONSE_TEMPLATE.format(node_id=node_id)).write_text(
        json.dumps(artifact_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return result


def _normalize_line_scope_role(
    raw_scope_role: Any,
    *,
    is_root_node: bool,
    is_first_visible_row_of_nonroot: bool,
    justification_head: str,
) -> str:
    raw = str(raw_scope_role or "").strip().lower()
    mapping = {
        "premise": "premise",
        "pr": "premise",
        "ordinary_assumption": "ordinary_assumption",
        "assumption": "ordinary_assumption",
        "assum": "ordinary_assumption",
        "subproof": "ordinary_assumption",
        "derived_line": "derived_line",
        "derived": "derived_line",
        "line": "derived_line",
        "strict_open": "strict_open",
        "strict": "strict_open",
        "subproof_token": "subproof_token",
        "token": "subproof_token",
    }
    normalized = mapping.get(raw, "")

    if is_first_visible_row_of_nonroot and normalized not in {"strict_open", "subproof_token"}:
        return "ordinary_assumption"
    if normalized in LINE_SCOPE_ROLE_OPTION_SET:
        return normalized
    if is_root_node and justification_head in {"", "PR"}:
        return "premise"
    return "derived_line"


def _normalize_justification_head(raw_head: Any) -> str:
    raw = str(raw_head or "").strip()
    lowered = raw.lower()
    if lowered in {"", "blank", "<blank>", "none", "null"}:
        return ""
    shorthand = {
        "pr": "PR",
        "premise": "PR",
        "assumption": "PR",
        "assum": "PR",
        "r": "R",
        "andi": "∧I",
        "ande": "∧E",
        "ori": "∨I",
        "ore": "∨E",
        "impi": "→I",
        "impe": "→E",
        "bici": "↔I",
        "bice": "↔E",
        "noti": "¬I",
        "note": "¬E",
        "ip": "IP",
        "dn": "DN",
        "mc": "MC",
        "boxi": "□I",
        "boxe": "□E",
        "r4": "R4",
        "rt": "RT",
        "defdiamond": "Def⋄",
        "def⋄": "Def⋄",
        "other": "other",
    }
    if lowered in shorthand:
        return shorthand[lowered]

    normalized = _normalize_justification_text(raw)
    head = normalized.split(" ", 1)[0] if normalized else ""
    if head in JUSTIFICATION_HEAD_OPTION_SET:
        return head
    return "other"


def _normalize_justification_tail(raw_tail: Any) -> str:
    tail = str(raw_tail or "").strip()
    tail = tail.replace(",", " ")
    tail = tail.replace("–", "-")
    tail = tail.replace("—", "-")
    tail = re.sub(r"\s+", " ", tail).strip()
    return tail


def _compose_line_justification_text(*, head: str, tail: str, raw_head: Any = None) -> str:
    if head == "":
        return ""
    if head == "other":
        raw = str(raw_head or "").strip()
        if tail and raw:
            return f"{raw} {tail}".strip()
        return tail or raw or "other"
    return f"{head} {tail}".strip() if tail else head


def _line_result_is_suspicious(result: Dict[str, Any]) -> bool:
    confidence = _confidence_score(result.get("confidence"))
    if confidence < 0.65:
        return True
    if str(result.get("justification_head") or "") == "other":
        return True
    tail = str(result.get("justification_tail") or "").strip()
    if tail and re.search(r"[A-Za-z□⋄¬∧∨→↔]", tail):
        return True
    if not str(result.get("formula_text") or "").strip():
        return True
    return False


def _build_modal_line_prompt_for_row(
    base_prompt: str,
    *,
    is_root_node: bool,
    is_first_visible_row_of_nonroot: bool,
) -> str:
    lines = [base_prompt]
    if is_first_visible_row_of_nonroot:
        lines.extend(
            [
                "",
                "Context: this is the first visible line of a non-root subproof and it is not a strict opener.",
                'Prefer scope_role "ordinary_assumption" unless the crop clearly contradicts that.',
            ]
        )
    elif is_root_node:
        lines.extend(
            [
                "",
                "Context: this is a top-level line from the main proof.",
                'If it has no real justification beyond being given, prefer scope_role "premise".',
            ]
        )
    return "\n".join(lines)


def _save_node_row_overlay(
    *,
    node_crop_img: np.ndarray,
    output_dir: Path,
    node_id: str,
    row_records: List[Dict[str, Any]],
) -> None:
    overlay = node_crop_img.copy()
    img_h, img_w = overlay.shape[:2]
    for index, record in enumerate(row_records, start=1):
        y1 = int(record.get("y1", 0))
        y2 = int(record.get("y2", 0))
        kind = str(record.get("kind") or "line")
        color = (0, 165, 255) if kind == "subproof_token" else (0, 180, 0)
        cv2.rectangle(overlay, (0, max(0, y1)), (max(1, img_w - 1), max(y1 + 1, min(img_h - 1, y2))), color, 2)
        cv2.putText(
            overlay,
            f"{index}:{'tok' if kind == 'subproof_token' else 'row'}",
            (8, max(20, y1 + 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
    cv2.imwrite(str(output_dir / NODE_ROWS_OVERLAY_TEMPLATE.format(node_id=node_id)), overlay)


def _relative_child_box(node: SubproofNode, child: SubproofNode) -> Tuple[int, int, int, int]:
    if node.id == "ROOT":
        return child.box
    px, py, _, _ = node.box
    return (child.box[0] - px + 10, child.box[1] - py + 10, child.box[2], child.box[3])


def _token_center_y_for_child(node: SubproofNode, child: SubproofNode, img_h: int) -> int:
    _, y, _, h = _relative_child_box(node, child)
    return int(min(max(24, y + max(18, h // 2)), max(24, img_h - 12)))


def _token_rect_for_child(
    *,
    node: SubproofNode,
    child: SubproofNode,
    img_shape: Tuple[int, int, int] | Tuple[int, int],
) -> Tuple[int, int, int, int]:
    img_h, img_w = img_shape[:2]
    x, y, _, h = _relative_child_box(node, child)
    token = f"<SUBPROOF_ID_{child.id}>"
    start_x = min(max(8, x + 20), max(8, img_w - 40))
    center_y = int(min(max(24, y + max(18, h // 2)), img_h - 12))
    max_width = max(60, img_w - start_x - 8)
    _, _, rect = _fit_token_layout(
        token,
        start_x=start_x,
        center_y=center_y,
        max_width=max_width,
    )
    return rect


def _band_height(band: Tuple[int, int]) -> int:
    return max(0, int(band[1]) - int(band[0]) + 1)


def _adaptive_band_padding(*, band: Tuple[int, int], img_h: int) -> int:
    height = _band_height(band)
    if height <= 10:
        return min(24, max(16, img_h // 12))
    if height <= 14:
        return min(20, max(14, img_h // 14))
    if height <= 20:
        return min(16, max(12, img_h // 16))
    return 12


def _band_gap(upper: Tuple[int, int], lower: Tuple[int, int]) -> int:
    return int(lower[0]) - int(upper[1]) - 1


def _merge_fragment_bands(bands: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if not bands:
        return []

    merged = [tuple(map(int, band)) for band in sorted(bands)]
    changed = True
    while changed and len(merged) > 1:
        changed = False
        for idx, band in enumerate(list(merged)):
            height = _band_height(band)
            if height > 12:
                continue

            prev_gap = _band_gap(merged[idx - 1], band) if idx > 0 else None
            next_gap = _band_gap(band, merged[idx + 1]) if idx + 1 < len(merged) else None
            candidate_idx: int | None = None
            candidate_gap: int | None = None

            if prev_gap is not None and prev_gap <= 18:
                candidate_idx = idx - 1
                candidate_gap = prev_gap
            if next_gap is not None and next_gap <= 18 and (candidate_gap is None or next_gap < candidate_gap):
                candidate_idx = idx + 1
                candidate_gap = next_gap
            if candidate_idx is None:
                continue

            other = merged[candidate_idx]
            new_band = (min(band[0], other[0]), max(band[1], other[1]))
            low_idx = min(idx, candidate_idx)
            high_idx = max(idx, candidate_idx)
            merged[low_idx] = new_band
            del merged[high_idx]
            changed = True
            break
    return merged


def _band_has_text_outside_vertical_window(
    node_crop_img: np.ndarray,
    band: Tuple[int, int],
    *,
    window_y1: int,
    window_y2: int,
) -> bool:
    if node_crop_img.size == 0:
        return False

    img_h, img_w = node_crop_img.shape[:2]
    pad_y = _adaptive_band_padding(band=band, img_h=img_h)
    top = max(0, int(band[0]) - pad_y)
    bottom = min(img_h, int(band[1]) + pad_y)
    if bottom <= top:
        return False

    crop = _suppress_left_scope_bar(node_crop_img[top:bottom, :img_w])
    if crop.size == 0:
        return False

    local_window_y1 = max(0, min(crop.shape[0], int(window_y1) - top))
    local_window_y2 = max(0, min(crop.shape[0], int(window_y2) - top))
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    left_ignore = min(20, max(12, gray.shape[1] // 8))
    right_limit = min(gray.shape[1], max(80, int(gray.shape[1] * 0.68)))
    if right_limit <= left_ignore:
        return False

    focused = gray[:, left_ignore:right_limit]
    mask = np.ones(focused.shape[:2], dtype=np.uint8)
    if local_window_y2 > local_window_y1:
        mask[local_window_y1:local_window_y2, :] = 0

    dark = ((focused < 185) & (mask > 0)).astype(np.uint8)
    return int(dark.sum()) >= 14


def _band_matches_child_token_artifact(
    *,
    band: Tuple[int, int],
    node: SubproofNode,
    node_crop_img: np.ndarray,
    center_tolerance: int = 2,
    overlap_ratio_threshold: float = 0.45,
) -> bool:
    band_height = _band_height(band)
    if band_height <= 0 or not node.children:
        return False

    band_center = (int(band[0]) + int(band[1])) / 2.0
    for child in node.children:
        _, token_top, _, token_bottom = _token_rect_for_child(node=node, child=child, img_shape=node_crop_img.shape)
        overlap_top = max(int(band[0]), int(token_top))
        overlap_bottom = min(int(band[1]), int(token_bottom))
        overlap = overlap_bottom - overlap_top + 1
        overlap_ratio = (overlap / band_height) if overlap > 0 else 0.0
        center_inside = (int(token_top) - center_tolerance) <= band_center <= (int(token_bottom) + center_tolerance)
        if center_inside:
            return True
        if overlap_ratio < overlap_ratio_threshold:
            continue
        if not _band_has_text_outside_vertical_window(
            node_crop_img,
            band,
            window_y1=int(token_top),
            window_y2=int(token_bottom) + 1,
        ):
            return True
    return False


def _filter_bands_near_child_tokens(
    *,
    bands: List[Tuple[int, int]],
    node: SubproofNode,
    node_crop_img: np.ndarray,
) -> List[Tuple[int, int]]:
    if not bands or not node.children:
        return bands

    filtered: List[Tuple[int, int]] = []
    for band in bands:
        if not _band_matches_child_token_artifact(
            band=band,
            node=node,
            node_crop_img=node_crop_img,
        ):
            filtered.append(band)
    return filtered


def _row_overlaps_child_token(
    *,
    node: SubproofNode,
    row_record: Dict[str, Any],
    node_crop_img: np.ndarray,
) -> bool:
    band = (int(row_record.get("y1", 0)), int(row_record.get("y2", 0)))
    return _band_matches_child_token_artifact(
        band=band,
        node=node,
        node_crop_img=node_crop_img,
    )


def _band_has_textlike_ink(node_crop_img: np.ndarray, band: Tuple[int, int]) -> bool:
    band_crop = _extract_band_crop(node_crop_img=node_crop_img, bands=[band], band_index=0)
    if band_crop.size == 0:
        return False
    gray = cv2.cvtColor(band_crop, cv2.COLOR_BGR2GRAY)
    left_ignore = min(20, max(12, gray.shape[1] // 8))
    focused = gray[:, left_ignore : min(gray.shape[1], max(80, int(gray.shape[1] * 0.60)))]
    dark = (focused < 185).astype(np.uint8)
    if dark.size == 0:
        return False
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(18, int(focused.shape[1] * 0.2)), 1))
    horizontal_mask = cv2.morphologyEx(dark, cv2.MORPH_OPEN, horizontal_kernel)
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(16, int(focused.shape[0] * 0.75))))
    vertical_mask = cv2.morphologyEx(dark, cv2.MORPH_OPEN, vertical_kernel)
    text_mask = dark.copy()
    text_mask[horizontal_mask > 0] = 0
    text_mask[vertical_mask > 0] = 0
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(text_mask, 8)
    tall_components = 0
    total_area = 0
    for label_index in range(1, num_labels):
        area = int(stats[label_index, cv2.CC_STAT_AREA])
        width = int(stats[label_index, cv2.CC_STAT_WIDTH])
        height = int(stats[label_index, cv2.CC_STAT_HEIGHT])
        if area < 8:
            continue
        total_area += area
        if height >= 6 and width >= 2:
            tall_components += 1
    band_height = int(band[1] - band[0] + 1)
    if band_height < 12 and total_area < 36:
        return False
    return tall_components >= 1 and total_area >= 18


def segment_node_rows(
    *,
    node: SubproofNode,
    node_crop_img: np.ndarray,
    output_dir: Path,
) -> List[Dict[str, Any]]:
    cleaned = _suppress_left_scope_bar(node_crop_img)
    bands = [band for band in _detect_text_row_bands(cleaned) if _band_has_textlike_ink(cleaned, band)]
    bands = _merge_fragment_bands(bands)
    bands = _filter_bands_near_child_tokens(bands=bands, node=node, node_crop_img=cleaned)
    row_records: List[Dict[str, Any]] = []
    for band_index, (y1, y2) in enumerate(bands):
        row_records.append(
            {
                "kind": "line",
                "band_index": band_index,
                "y1": int(y1),
                "y2": int(y2),
                "center_y": float((y1 + y2) / 2.0),
            }
        )
    img_h = node_crop_img.shape[0]
    for child in sorted(node.children, key=lambda c: c.box[1]):
        _, token_top, _, token_bottom = _token_rect_for_child(node=node, child=child, img_shape=node_crop_img.shape)
        center_y = float((token_top + token_bottom) / 2.0)
        row_records.append(
            {
                "kind": "subproof_token",
                "child_id": child.id,
                "y1": int(max(0, token_top)),
                "y2": int(min(img_h - 1, token_bottom)),
                "center_y": center_y,
            }
        )

    row_records.sort(
        key=lambda record: (
            float(record.get("center_y", 0.0)),
            0 if str(record.get("kind") or "line") == "line" else 1,
            int(record.get("band_index", 0)),
        )
    )
    for row_order, record in enumerate(row_records, start=1):
        record["row_order"] = row_order

    (output_dir / NODE_ROWS_TEMPLATE.format(node_id=node.id)).write_text(
        json.dumps(row_records, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _save_node_row_overlay(node_crop_img=node_crop_img, output_dir=output_dir, node_id=node.id, row_records=row_records)
    return row_records


def _extract_line_row_crop(
    *,
    node_crop_img: np.ndarray,
    y1: int,
    y2: int,
) -> np.ndarray:
    if node_crop_img.size == 0:
        return node_crop_img
    img_h, img_w = node_crop_img.shape[:2]
    pad_y = _adaptive_band_padding(band=(int(y1), int(y2)), img_h=img_h)
    top = max(0, int(y1) - pad_y)
    bottom = min(img_h, int(y2) + pad_y)
    crop = node_crop_img[top:bottom, :img_w]
    if crop.size == 0:
        return node_crop_img
    return _suppress_left_scope_bar(crop)


def _score_to_confidence_label(score: float) -> str:
    if score >= 0.85:
        return "high"
    if score >= 0.55:
        return "medium"
    if score > 0.0:
        return "low"
    return "uncertain"


def _choose_line_zone_split_x(crop_img: np.ndarray) -> int:
    if crop_img.size == 0:
        return 0

    gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    dark = (gray < 185).astype(np.uint8)
    if dark.size == 0:
        return max(0, int(crop_img.shape[1] * 0.6))

    col_counts = dark.sum(axis=0).astype(np.float32)
    if col_counts.size == 0:
        return max(0, int(crop_img.shape[1] * 0.6))

    kernel_width = max(9, min(31, int(crop_img.shape[1] * 0.06)))
    if kernel_width % 2 == 0:
        kernel_width += 1
    kernel = np.ones(kernel_width, dtype=np.float32) / float(kernel_width)
    smoothed = np.convolve(col_counts, kernel, mode="same")

    total_ink = float(col_counts.sum())
    img_w = int(crop_img.shape[1])
    search_start = max(12, int(img_w * 0.18))
    search_end = min(img_w - 12, int(img_w * 0.72))
    preferred = float(img_w * 0.42)

    prefix = np.cumsum(col_counts)
    best_x = max(search_start, min(search_end, int(preferred)))
    best_score = float("inf")

    for split_x in range(search_start, max(search_start + 1, search_end)):
        left_ink = float(prefix[split_x - 1]) if split_x > 0 else 0.0
        right_ink = total_ink - left_ink
        if left_ink < 10.0 or right_ink < 10.0:
            continue
        score = float(smoothed[split_x]) + (abs(float(split_x) - preferred) * 0.03)
        if score < best_score:
            best_score = score
            best_x = split_x

    min_split = max(18, int(img_w * 0.22))
    max_split = max(min_split + 1, int(img_w * 0.62))
    return max(min_split, min(max_split, best_x))


def _extract_line_zone_crops(crop_img: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    if crop_img.size == 0:
        empty = crop_img.copy()
        return empty, empty, {"split_x": 0, "formula_x1": 0, "formula_x2": 0, "justification_x1": 0, "justification_x2": 0}

    img_h, img_w = crop_img.shape[:2]
    split_x = _choose_line_zone_split_x(crop_img)
    overlap = max(10, int(img_w * 0.02))

    formula_x1 = 0
    formula_x2 = max(1, min(img_w, split_x + overlap))
    justification_x1 = max(0, min(img_w - 1, split_x - overlap))
    justification_x2 = img_w

    formula_crop = crop_img[:, formula_x1:formula_x2]
    justification_crop = crop_img[:, justification_x1:justification_x2]
    if formula_crop.size == 0:
        formula_crop = crop_img.copy()
    if justification_crop.size == 0:
        justification_crop = crop_img.copy()

    return formula_crop, justification_crop, {
        "split_x": int(split_x),
        "formula_x1": int(formula_x1),
        "formula_x2": int(formula_x2),
        "justification_x1": int(justification_x1),
        "justification_x2": int(justification_x2),
        "width": int(img_w),
        "height": int(img_h),
    }


def _normalize_formula_zone_result(parsed: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "formula_text": str(parsed.get("formula_text") or "").strip(),
        "line_label": str(parsed.get("line_label") or "").strip(),
        "confidence": str(parsed.get("confidence") or "").strip(),
    }


def _normalize_justification_zone_result(parsed: Dict[str, Any]) -> Dict[str, Any]:
    raw_text = str(
        parsed.get("justification_text")
        or parsed.get("jstr")
        or _compose_line_justification_text(
            head=parsed.get("justification_head") or "",
            tail=parsed.get("justification_tail") or "",
            raw_head=parsed.get("justification_head"),
        )
        or ""
    ).strip()
    normalized_text = _normalize_justification_text(raw_text)
    parts = normalized_text.split(" ", 1) if normalized_text else []
    head = _normalize_justification_head(parts[0] if parts else "")
    tail = _normalize_justification_tail(parts[1] if len(parts) > 1 else "")
    if head == "":
        tail = ""
    return {
        "justification_head": head,
        "justification_tail": tail,
        "raw_justification_text": raw_text,
        "confidence": str(parsed.get("confidence") or "").strip(),
    }


def _normalize_visible_line_label(raw_value: Any) -> str:
    raw = str(raw_value or "").strip()
    if not raw:
        return ""
    compact = re.sub(r"\s+", "", raw)
    return raw if re.fullmatch(r"\d+[.)]?", compact) else ""


def _recover_justification_from_formula_label(raw_value: Any) -> Tuple[str, str]:
    raw = str(raw_value or "").strip()
    if not raw:
        return "", ""
    normalized = _normalize_justification_text(raw)
    if not normalized:
        return "", ""
    head = _normalize_justification_head(normalized)
    if head == "other":
        return "", ""
    parts = normalized.split(" ", 1)
    return head, _normalize_justification_tail(parts[1] if len(parts) > 1 else "")


def _merge_line_zone_results(
    *,
    formula_result: Dict[str, Any],
    justification_result: Dict[str, Any],
    is_root_node: bool,
    is_first_visible_row_of_nonroot: bool,
) -> Dict[str, Any]:
    justification_head = str(justification_result.get("justification_head") or "").strip()
    justification_tail = str(justification_result.get("justification_tail") or "").strip()
    raw_justification_text = str(justification_result.get("raw_justification_text") or "").strip()
    raw_line_label = str(formula_result.get("line_label") or "").strip()
    line_label = _normalize_visible_line_label(raw_line_label)
    fallback_head = ""
    fallback_tail = ""
    if raw_line_label and not line_label:
        fallback_head, fallback_tail = _recover_justification_from_formula_label(raw_line_label)
        if fallback_head and justification_head in {"", "other"}:
            justification_head = fallback_head
            combined_tail_parts = [part for part in (fallback_tail, justification_tail) if part]
            justification_tail = _normalize_justification_tail(" ".join(combined_tail_parts))
    formula_conf = _confidence_score(formula_result.get("confidence"))
    justification_conf = _confidence_score(justification_result.get("confidence"))
    overall_conf = formula_conf
    if justification_head or justification_tail:
        overall_conf = min(formula_conf, justification_conf) if formula_conf > 0.0 and justification_conf > 0.0 else max(formula_conf, justification_conf)
    return {
        "formula_text": str(formula_result.get("formula_text") or "").strip(),
        "justification_head": justification_head,
        "justification_tail": justification_tail,
        "justification_text": _compose_line_justification_text(
            head=justification_head,
            tail=justification_tail,
            raw_head=raw_justification_text or justification_head,
        ),
        "line_label": line_label,
        "scope_role": _normalize_line_scope_role(
            "",
            is_root_node=is_root_node,
            is_first_visible_row_of_nonroot=is_first_visible_row_of_nonroot,
            justification_head=justification_head,
        ),
        "confidence": _score_to_confidence_label(overall_conf),
    }


def _normalize_line_ocr_result(
    parsed: Dict[str, Any],
    *,
    is_root_node: bool,
    is_first_visible_row_of_nonroot: bool,
) -> Dict[str, Any]:
    justification_head = _normalize_justification_head(parsed.get("justification_head"))
    justification_tail = _normalize_justification_tail(parsed.get("justification_tail"))
    scope_role = _normalize_line_scope_role(
        parsed.get("scope_role"),
        is_root_node=is_root_node,
        is_first_visible_row_of_nonroot=is_first_visible_row_of_nonroot,
        justification_head=justification_head,
    )
    line_label = str(parsed.get("line_label") or "").strip()
    formula_text = str(parsed.get("formula_text") or "").strip()
    return {
        "formula_text": formula_text,
        "justification_head": justification_head,
        "justification_tail": justification_tail,
        "justification_text": _compose_line_justification_text(
            head=justification_head,
            tail=justification_tail,
            raw_head=parsed.get("justification_head"),
        ),
        "line_label": line_label,
        "scope_role": scope_role,
        "confidence": str(parsed.get("confidence") or "").strip(),
    }


def ocr_node_row(
    *,
    node: SubproofNode,
    node_crop_img: np.ndarray,
    row_record: Dict[str, Any],
    output_dir: Path,
    formula_prompt: str,
    justification_prompt: str,
    client: LMStudioQwenClient | None,
    verifier_client: LMStudioQwenClient | None = None,
    is_first_visible_row_of_nonroot: bool = False,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    row_order = int(row_record.get("row_order") or 0)
    crop_img = _extract_line_row_crop(
        node_crop_img=node_crop_img,
        y1=int(row_record.get("y1", 0)),
        y2=int(row_record.get("y2", 0)),
    )
    crop_path = output_dir / NODE_LINE_CROP_TEMPLATE.format(node_id=node.id, row_order=row_order)
    cv2.imwrite(str(crop_path), crop_img)
    formula_crop, justification_crop, zone_geometry = _extract_line_zone_crops(crop_img)
    formula_crop_path = output_dir / NODE_LINE_FORMULA_CROP_TEMPLATE.format(node_id=node.id, row_order=row_order)
    justification_crop_path = output_dir / NODE_LINE_JUSTIFICATION_CROP_TEMPLATE.format(node_id=node.id, row_order=row_order)
    cv2.imwrite(str(formula_crop_path), formula_crop)
    cv2.imwrite(str(justification_crop_path), justification_crop)

    formula_raw, formula_parsed = _call_vlm_json_object(
        image_path=formula_crop_path,
        prompt=formula_prompt,
        client=client,
        user_text="Return the JSON transcription for this formula zone.",
    )
    justification_raw, justification_parsed = _call_vlm_json_object(
        image_path=justification_crop_path,
        prompt=justification_prompt,
        client=client,
        user_text="Return the JSON transcription for this justification zone.",
    )
    formula_result = _normalize_formula_zone_result(formula_parsed)
    justification_result = _normalize_justification_zone_result(justification_parsed)
    initial_formula_result = dict(formula_result)
    initial_justification_result = dict(justification_result)
    justification_text = _compose_line_justification_text(
        head=str(justification_result.get("justification_head") or "").strip(),
        tail=str(justification_result.get("justification_tail") or "").strip(),
        raw_head=justification_result.get("raw_justification_text"),
    )
    formula_reread_artifact: Dict[str, Any] | None = None
    if verifier_client is not None and _should_reread_short_formula(
        formula_text=str(formula_result.get("formula_text") or ""),
        justification_text=justification_text,
    ):
        reread_img = _upscale_for_verifier(formula_crop)
        reread_crop_path = output_dir / NODE_LINE_FORMULA_REREAD_CROP_TEMPLATE.format(node_id=node.id, row_order=row_order)
        cv2.imwrite(str(reread_crop_path), reread_img)
        reread_prompt = _build_short_formula_reread_prompt(
            current_formula=str(formula_result.get("formula_text") or "").strip(),
            justification_text=justification_text,
        )
        reread_raw, reread_parsed = _call_vlm_json_object(
            image_path=reread_crop_path,
            prompt=reread_prompt,
            client=verifier_client,
            user_text="Return the JSON transcription for this single formula.",
        )
        reread_result = _normalize_formula_probe_result(reread_parsed)
        reread_applied = _should_accept_short_formula_reread(
            original_formula=str(formula_result.get("formula_text") or ""),
            candidate_formula=str(reread_result.get("formula_text") or ""),
            justification_text=justification_text,
            confidence=str(reread_result.get("confidence") or ""),
        )
        formula_reread_artifact = {
            "crop_filename": reread_crop_path.name,
            "prompt": reread_prompt,
            "raw": reread_raw,
            "parsed": reread_parsed,
            "resolved": reread_result,
            "applied": reread_applied,
        }
        if reread_applied:
            formula_result["formula_text"] = str(reread_result.get("formula_text") or "").strip()
            formula_result["confidence"] = str(reread_result.get("confidence") or formula_result.get("confidence") or "").strip()
    normalized = _merge_line_zone_results(
        formula_result=formula_result,
        justification_result=justification_result,
        is_root_node=node.id == "ROOT",
        is_first_visible_row_of_nonroot=is_first_visible_row_of_nonroot,
    )

    artifact_payload: Dict[str, Any] = {
        "node_id": node.id,
        "row_order": row_order,
        "band_index": row_record.get("band_index"),
        "y1": row_record.get("y1"),
        "y2": row_record.get("y2"),
        "zone_geometry": zone_geometry,
        "formula_crop_filename": formula_crop_path.name,
        "justification_crop_filename": justification_crop_path.name,
        "initial_raw": {
            "formula": formula_raw,
            "justification": justification_raw,
        },
        "initial_parsed": {
            "formula": formula_parsed,
            "justification": justification_parsed,
        },
        "initial_resolved_zones": {
            "formula": initial_formula_result,
            "justification": initial_justification_result,
        },
        "resolved": normalized,
        "retry_used": False,
    }
    if formula_reread_artifact is not None:
        artifact_payload["formula_reread"] = formula_reread_artifact

    if verifier_client is not None and _line_result_is_suspicious(normalized):
        retry_formula_img = _upscale_for_verifier(formula_crop)
        retry_justification_img = _upscale_for_verifier(justification_crop)
        retry_formula_crop_path = output_dir / NODE_LINE_FORMULA_RETRY_CROP_TEMPLATE.format(node_id=node.id, row_order=row_order)
        retry_justification_crop_path = output_dir / NODE_LINE_JUSTIFICATION_RETRY_CROP_TEMPLATE.format(node_id=node.id, row_order=row_order)
        cv2.imwrite(str(retry_formula_crop_path), retry_formula_img)
        cv2.imwrite(str(retry_justification_crop_path), retry_justification_img)
        retry_formula_raw, retry_formula_parsed = _call_vlm_json_object(
            image_path=retry_formula_crop_path,
            prompt=formula_prompt,
            client=verifier_client,
            user_text="Return the JSON transcription for this formula zone.",
        )
        retry_justification_raw, retry_justification_parsed = _call_vlm_json_object(
            image_path=retry_justification_crop_path,
            prompt=justification_prompt,
            client=verifier_client,
            user_text="Return the JSON transcription for this justification zone.",
        )
        retry_formula_result = _normalize_formula_zone_result(retry_formula_parsed)
        retry_justification_result = _normalize_justification_zone_result(retry_justification_parsed)
        retry_normalized = _merge_line_zone_results(
            formula_result=retry_formula_result,
            justification_result=retry_justification_result,
            is_root_node=node.id == "ROOT",
            is_first_visible_row_of_nonroot=is_first_visible_row_of_nonroot,
        )
        artifact_payload["retry_used"] = True
        artifact_payload["retry_crop_filename"] = NODE_LINE_RETRY_CROP_TEMPLATE.format(node_id=node.id, row_order=row_order)
        artifact_payload["retry_formula_crop_filename"] = retry_formula_crop_path.name
        artifact_payload["retry_justification_crop_filename"] = retry_justification_crop_path.name
        artifact_payload["retry_raw"] = {
            "formula": retry_formula_raw,
            "justification": retry_justification_raw,
        }
        artifact_payload["retry_parsed"] = {
            "formula": retry_formula_parsed,
            "justification": retry_justification_parsed,
        }
        artifact_payload["retry_resolved_zones"] = {
            "formula": retry_formula_result,
            "justification": retry_justification_result,
        }
        artifact_payload["retry_resolved"] = retry_normalized
        if not _line_result_is_suspicious(retry_normalized):
            normalized = retry_normalized
            artifact_payload["resolved"] = retry_normalized

    (output_dir / NODE_LINE_RESPONSE_TEMPLATE.format(node_id=node.id, row_order=row_order)).write_text(
        json.dumps(artifact_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    entry = {
        "formula_text": normalized["formula_text"],
        "justification_text": normalized["justification_text"],
        "scope_role": normalized["scope_role"],
        "line_label": normalized["line_label"],
    }
    return entry, artifact_payload


def assemble_node_entries_from_rows(
    *,
    node: SubproofNode,
    node_crop_img: np.ndarray,
    output_dir: Path,
    formula_prompt: str | None = None,
    justification_prompt: str | None = None,
    client: LMStudioQwenClient | None,
    opener_client: LMStudioQwenClient | None = None,
    verifier_client: LMStudioQwenClient | None = None,
    base_prompt: str | None = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if formula_prompt is None:
        formula_prompt = base_prompt or ""
    if justification_prompt is None:
        justification_prompt = base_prompt or ""
    row_records = segment_node_rows(node=node, node_crop_img=node_crop_img, output_dir=output_dir)
    opener_result = None
    consumed_first_line_row_order: int | None = None
    if node.id != "ROOT":
        opener_result = _classify_subproof_opener(
            node_crop_img=node_crop_img,
            output_dir=output_dir,
            node_id=node.id,
            client=opener_client or verifier_client or client,
        )
        if opener_result and opener_result.get("kind") == "strict_open":
            first_line = next((record for record in row_records if str(record.get("kind") or "line") == "line"), None)
            if first_line is not None:
                consumed_first_line_row_order = int(first_line.get("row_order") or 0)

    entries: List[Dict[str, Any]] = []
    line_artifacts: List[Dict[str, Any]] = []
    visible_line_index = 0
    for row_record in row_records:
        kind = str(row_record.get("kind") or "line")
        if kind == "subproof_token":
            child_id = str(row_record.get("child_id") or "").strip()
            if child_id:
                entries.append(
                    {
                        "formula_text": f"<SUBPROOF_ID_{child_id}>",
                        "scope_role": "subproof_token",
                        "line_label": "",
                    }
                )
            line_artifacts.append(
                {
                    "kind": "subproof_token",
                    "row_order": row_record.get("row_order"),
                    "child_id": child_id,
                    "center_y": row_record.get("center_y"),
                }
            )
            continue

        row_order = int(row_record.get("row_order") or 0)
        if consumed_first_line_row_order is not None and row_order == consumed_first_line_row_order:
            line_artifacts.append(
                {
                    "kind": "strict_open_consumed",
                    "row_order": row_order,
                    "resolved_from": opener_result,
                }
            )
            continue
        if _row_overlaps_child_token(node=node, row_record=row_record, node_crop_img=node_crop_img):
            line_artifacts.append(
                {
                    "kind": "token_overlap_skipped",
                    "row_order": row_order,
                    "band_index": row_record.get("band_index"),
                    "y1": row_record.get("y1"),
                    "y2": row_record.get("y2"),
                    "skipped_reason": "row_overlaps_child_token_pre_ocr",
                }
            )
            continue

        entry, artifact = ocr_node_row(
            node=node,
            node_crop_img=node_crop_img,
            row_record=row_record,
            output_dir=output_dir,
            formula_prompt=formula_prompt,
            justification_prompt=justification_prompt,
            client=client,
            verifier_client=verifier_client,
            is_first_visible_row_of_nonroot=(node.id != "ROOT" and visible_line_index == 0 and consumed_first_line_row_order is None),
        )
        if (
            str(entry.get("scope_role") or "").strip() == "subproof_token"
            and _row_overlaps_child_token(node=node, row_record=row_record, node_crop_img=node_crop_img)
        ):
            artifact["kind"] = "token_overlap_skipped"
            artifact["skipped_reason"] = "row_overlaps_child_token"
            line_artifacts.append(artifact)
            continue
        entries.append(entry)
        line_artifacts.append(artifact)
        visible_line_index += 1

    if opener_result and opener_result.get("kind") == "strict_open":
        entries = _prepend_synthetic_strict_open_if_needed(entries, opener_result)

    (output_dir / NODE_ENTRIES_TEMPLATE.format(node_id=node.id)).write_text(
        json.dumps(entries, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    node_response = {
        "node_id": node.id,
        "row_records": row_records,
        "line_artifacts": line_artifacts,
        "opener_result": opener_result,
        "entries": entries,
    }
    return entries, node_response

def _leading_unary_prefix(formula_text: str) -> str:
    return _leading_unary_prefix_impl(formula_text)


def _should_recheck_unary_formula(entry: Dict[str, Any]) -> bool:
    scope_role = str(entry.get("scope_role") or "derived_line").strip()
    if scope_role in {"subproof_token", "strict_open"}:
        return False

    formula_text = _normalize_formula_text(entry.get("formula_text") or entry.get("wffstr") or "", scope_role)
    if not formula_text:
        return False

    prefix = _leading_unary_prefix(formula_text)
    body = formula_text[len(prefix) :].strip()
    if body and _has_top_level_binary_connective(body):
        return False

    justification = _normalize_justification_text(entry.get("justification_text") or entry.get("jstr") or "")
    rule_head = justification.split(" ", 1)[0] if justification else ""

    if "¬□¬" in formula_text or "¬⋄¬" in formula_text:
        return True
    if len(prefix) >= 2 and any(marker in prefix for marker in ("¬□", "¬⋄", "□¬", "⋄¬")):
        return True
    if rule_head in UNARY_SENSITIVE_RULE_HEADS and any(char in formula_text for char in UNARY_PREFIX_CHARS):
        return True
    return False


def _unary_formula_skeleton(formula_text: str) -> str:
    return re.sub(r"[¬□⋄()\s]", "", str(formula_text or ""))


def _parse_unary_recheck_choice(raw: str) -> Dict[str, Any]:
    cleaned = _strip_fences(raw)
    parsed = json.loads(cleaned)
    if isinstance(parsed, list):
        parsed = parsed[0] if parsed else {}
    if not isinstance(parsed, dict):
        raise ValueError("Unary recheck response must be a JSON object")
    return parsed

def _checker_payload_has_issues(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    return bool(payload.get("parse_errors") or payload.get("check_errors"))


def _should_verify_unary_formula(entry: Dict[str, Any]) -> bool:
    return _should_verify_unary_formula_impl(
        entry,
        normalize_formula_text=_normalize_formula_text,
        normalize_justification_text=_normalize_justification_text,
    )


def _has_top_level_binary_connective(formula_text: str) -> bool:
    depth = 0
    for char in str(formula_text or ""):
        if char == "(":
            depth += 1
            continue
        if char == ")":
            depth = max(0, depth - 1)
            continue
        if depth == 0 and char in {"∧", "∨", "→", "↔"}:
            return True
    return False


def _build_node_lookup(node: SubproofNode) -> Dict[str, SubproofNode]:
    return _build_node_lookup_impl(node)


def _verifier_node_crop_image(*, node: SubproofNode, full_img: np.ndarray) -> np.ndarray:
    return _verifier_node_crop_image_impl(
        node=node,
        full_img=full_img,
        crop_region=crop_region,
    )


def _extract_single_line_crop(
    *,
    node_crop_img: np.ndarray,
    line_order: int | None,
) -> np.ndarray:
    if line_order is None or node_crop_img.size == 0:
        return node_crop_img

    bands = _detect_text_row_bands(node_crop_img)
    if not bands or line_order < 0 or line_order >= len(bands):
        return node_crop_img

    return _extract_band_crop(node_crop_img=node_crop_img, bands=bands, band_index=line_order)


def _extract_band_crop(
    *,
    node_crop_img: np.ndarray,
    bands: List[Tuple[int, int]],
    band_index: int,
) -> np.ndarray:
    if node_crop_img.size == 0 or not bands or band_index < 0 or band_index >= len(bands):
        return node_crop_img

    band_top, band_bottom = bands[band_index]
    img_h, img_w = node_crop_img.shape[:2]
    pad_y = _adaptive_band_padding(band=(band_top, band_bottom), img_h=img_h)
    y1 = max(0, band_top - pad_y)
    y2 = min(img_h, band_bottom + pad_y)
    x2 = min(img_w, max(80, int(img_w * 0.68)))
    band_crop = node_crop_img[y1:y2, 0:x2]
    return band_crop if band_crop.size else node_crop_img


def _suppress_left_scope_bar(node_crop_img: np.ndarray) -> np.ndarray:
    if node_crop_img.size == 0:
        return node_crop_img

    cleaned = node_crop_img.copy()
    bars = detect_vertical_bars(cleaned)
    if not bars:
        return cleaned

    img_h, img_w = cleaned.shape[:2]
    leftmost = min(bars, key=lambda box: box[0])
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
    groups: List[Tuple[int, int]] = []
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


def _candidate_band_indices(
    *,
    line_order: int | None,
    band_count: int,
    bands: List[Tuple[int, int]] | None = None,
    projected_y: float | None = None,
) -> List[int]:
    return _candidate_band_indices_impl(
        line_order=line_order,
        band_count=band_count,
        bands=bands,
        projected_y=projected_y,
    )


def _confidence_score(value: Any) -> float:
    return _confidence_score_impl(value)


def _best_verifier_attempt(
    *,
    original_formula: str,
    expected_line_order: int | None,
    attempts: List[Dict[str, Any]],
) -> Dict[str, Any]:
    return _best_verifier_attempt_impl(
        original_formula=original_formula,
        expected_line_order=expected_line_order,
        attempts=attempts,
        unary_formula_skeleton=_unary_formula_skeleton,
    )


def _merge_margin_candidate_boxes(boxes: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int]]:
    return _merge_margin_candidate_boxes_impl(boxes)


def _margin_anchor_strip_bounds(
    anchor_centers: List[float],
    image_height: int,
    anchor_heights: List[float] | None = None,
) -> List[Tuple[int, int]]:
    return _margin_anchor_strip_bounds_impl(anchor_centers, image_height, anchor_heights)


def _upscale_for_verifier(crop_img: np.ndarray, scale: int = 4) -> np.ndarray:
    if crop_img.size == 0:
        return crop_img
    return cv2.resize(crop_img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)


def _int_field(value: Any, default: int = 0, minimum: int = 0, maximum: int = 3) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, parsed))


def _formula_components(formula_text: str) -> Tuple[str, str]:
    prefix = _leading_unary_prefix(formula_text)
    body = formula_text[len(prefix) :].strip()
    return prefix, body


def _justification_rule_head(justification_text: Any) -> str:
    normalized = _normalize_justification_text(justification_text)
    return normalized.split(" ", 1)[0] if normalized else ""


def _compose_unary_formula(*, modal: str, before_modal: int, before_atom: int, body: str) -> str:
    prefix = ("¬" * max(0, before_modal)) + modal + ("¬" * max(0, before_atom))
    return f"{prefix}{body}".strip()


def _build_short_formula_reread_prompt(*, current_formula: str, justification_text: str) -> str:
    lines = [
        "This image shows one handwritten modal logic formula from a single proof line.",
        'Return only JSON with keys: formula_text and confidence.',
        "Copy exactly the formula or symbol you see.",
        "Preserve unary operators exactly: ¬, □, ⋄.",
        "Short formulas such as Q, □Q, ¬Q, ¬□¬A, or □ are common.",
        "Do not include justification text, line numbers, or explanation.",
        f"Current OCR formula: {current_formula}",
    ]
    if justification_text:
        lines.append(f"Current OCR justification: {justification_text}")
    lines.append('If you cannot tell, return {"formula_text":"","confidence":"uncertain"}.')
    return "\n".join(lines)


def _normalize_formula_probe_result(parsed: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "formula_text": _normalize_formula_text(parsed.get("formula_text") or "", "derived_line"),
        "confidence": str(parsed.get("confidence") or "").strip(),
    }


def _should_reread_short_formula(*, formula_text: str, justification_text: str) -> bool:
    normalized = _normalize_formula_text(formula_text, "derived_line")
    if not normalized or TOKEN_PATTERN.search(normalized):
        return False
    prefix, body = _formula_components(normalized)
    if body and _has_top_level_binary_connective(body):
        return False
    rule_head = _justification_rule_head(justification_text)
    if prefix:
        return True
    if normalized in {"□", "⋄", "¬", "¬¬"}:
        return True
    return rule_head in SHORT_FORMULA_REREAD_RULE_HEADS


def _should_accept_short_formula_reread(
    *,
    original_formula: str,
    candidate_formula: str,
    justification_text: str,
    confidence: str,
) -> bool:
    original = _normalize_formula_text(original_formula, "derived_line")
    candidate = _normalize_formula_text(candidate_formula, "derived_line")
    if not candidate or candidate == original or candidate == "uncertain":
        return False
    if _confidence_score(confidence) < 0.55:
        return False
    if TOKEN_PATTERN.search(candidate):
        return False

    _, candidate_body = _formula_components(candidate)
    if candidate_body and _has_top_level_binary_connective(candidate_body):
        return False

    original_skeleton = _unary_formula_skeleton(original)
    candidate_skeleton = _unary_formula_skeleton(candidate)
    rule_head = _justification_rule_head(justification_text)

    if original_skeleton and candidate_skeleton and original_skeleton == candidate_skeleton:
        return True
    if (
        not original_skeleton
        and candidate_skeleton
        and len(candidate_skeleton) <= 2
        and rule_head in SHORT_FORMULA_REREAD_RULE_HEADS
    ):
        return True
    return False


def _build_structured_unary_verifier_prompt(entry: Dict[str, Any]) -> str:
    return _build_structured_unary_verifier_prompt_impl(
        entry,
        normalize_formula_text=_normalize_formula_text,
        normalize_justification_text=_normalize_justification_text,
    )


def _build_forced_choice_unary_verifier_prompt(entry: Dict[str, Any], candidates: List[str]) -> str:
    return _build_forced_choice_unary_verifier_prompt_impl(
        entry,
        candidates,
        normalize_formula_text=_normalize_formula_text,
        normalize_justification_text=_normalize_justification_text,
    )


def _run_unary_verifier_experiment(
    *,
    proof_structure: List[Any],
    tree_root: SubproofNode,
    full_img: np.ndarray,
    output_dir: Path,
    verifier_client: LMStudioQwenClient | None,
    checker_response: Dict[str, Any] | None,
) -> List[Dict[str, Any]]:
    return _run_unary_verifier_experiment_impl(
        proof_structure=proof_structure,
        tree_root=tree_root,
        full_img=full_img,
        output_dir=output_dir,
        verifier_client=verifier_client,
        checker_response=checker_response,
        normalize_formula_text=_normalize_formula_text,
        normalize_justification_text=_normalize_justification_text,
        parse_unary_recheck_choice=_parse_unary_recheck_choice,
        unary_formula_skeleton=_unary_formula_skeleton,
        crop_region=crop_region,
        detect_vertical_bars=detect_vertical_bars,
        detect_text_row_bands=_detect_text_row_bands,
        extract_band_crop=_extract_band_crop,
        upscale_for_verifier=_upscale_for_verifier,
        int_field=_int_field,
        formula_components=_formula_components,
        iter_proof_line_dicts=_iter_proof_line_dicts,
    )


def _candidate_unary_prefixes(formula_text: str, justification: str) -> List[str]:
    prefix = _leading_unary_prefix(formula_text)
    rule_head = justification.split(" ", 1)[0] if justification else ""
    if not prefix:
        return []

    contains_box = "□" in prefix
    contains_diamond = "⋄" in prefix
    negation_count = prefix.count("¬")

    candidates: List[str] = [prefix]
    if contains_diamond:
        pool = ("⋄", "⋄¬", "⋄¬¬", "¬⋄", "¬⋄¬", "¬⋄¬¬")
    elif contains_box or negation_count >= 2 or rule_head in UNARY_SENSITIVE_RULE_HEADS:
        pool = ("□", "□¬", "□¬¬", "¬□", "¬□¬", "¬□¬¬")
    else:
        pool = ("¬", "¬¬", "¬¬¬")

    for candidate in pool:
        if candidate not in candidates:
            candidates.append(candidate)
    if not contains_box and not contains_diamond and prefix.endswith("¬") and prefix.count("¬") < 4:
        extended_prefix = f"{prefix}¬"
        if extended_prefix not in candidates:
            candidates.append(extended_prefix)
    return candidates


def _build_unary_formula_candidates(entry: Dict[str, Any]) -> List[str]:
    scope_role = str(entry.get("scope_role") or "derived_line").strip()
    formula_text = _normalize_formula_text(entry.get("formula_text") or entry.get("wffstr") or "", scope_role)
    justification = _normalize_justification_text(entry.get("justification_text") or entry.get("jstr") or "")
    prefix = _leading_unary_prefix(formula_text)
    if not prefix:
        return []

    body = formula_text[len(prefix) :].strip()
    if not body:
        return [formula_text]

    candidates: List[str] = []
    for candidate_prefix in _candidate_unary_prefixes(formula_text, justification):
        candidate = f"{candidate_prefix}{body}"
        if _unary_formula_skeleton(candidate) != _unary_formula_skeleton(formula_text):
            continue
        if candidate not in candidates:
            candidates.append(candidate)
    return candidates


def _checker_score(result: Any) -> Tuple[int, int]:
    return (len(getattr(result, "parse_errors", []) or []), len(getattr(result, "check_errors", []) or []))


def _clone_jsonish(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False))


def _iter_proof_line_dicts(structure: List[Any]) -> List[Dict[str, Any]]:
    lines: List[Dict[str, Any]] = []
    for item in structure:
        if isinstance(item, list):
            lines.extend(_iter_proof_line_dicts(item))
        elif isinstance(item, dict):
            lines.append(item)
    return lines


def _line_matches_source(
    line: Dict[str, Any],
    source_node_id: str,
    source_line_label: str,
    source_entry_index: int | None,
) -> bool:
    if str(line.get("_ocr_node_id") or "") != source_node_id:
        return False
    if source_line_label:
        return str(line.get("line_label") or "").strip() == source_line_label
    if source_entry_index is not None:
        try:
            return int(line.get("_ocr_entry_index")) == source_entry_index
        except (TypeError, ValueError):
            return False
    return False


def _substitute_formula_for_source_line(
    structure: List[Any],
    *,
    source_node_id: str,
    source_line_label: str,
    source_entry_index: int | None,
    formula_text: str,
) -> List[Any]:
    cloned = _clone_jsonish(structure)
    for line in _iter_proof_line_dicts(cloned):
        if _line_matches_source(line, source_node_id, source_line_label, source_entry_index):
            line["wffstr"] = formula_text
            break
    return cloned


def _score_unary_formula_candidates(
    *,
    proof_structure: List[Any],
    source_node_id: str,
    source_line_label: str,
    source_entry_index: int | None,
    candidates: List[str],
    modal_config: Dict[str, Any],
) -> Dict[str, Tuple[int, int]]:
    scores: Dict[str, Tuple[int, int]] = {}
    canonical_config = _canonical_modal_config(modal_config)
    for candidate in candidates:
        candidate_structure = _substitute_formula_for_source_line(
            proof_structure,
            source_node_id=source_node_id,
            source_line_label=source_line_label,
            source_entry_index=source_entry_index,
            formula_text=candidate,
        )
        answer_text = json.dumps({"lines": candidate_structure, "meta": {}}, ensure_ascii=False)
        result = run_modal_proof_checker(answer_text=answer_text, autograde_config=canonical_config)
        scores[candidate] = _checker_score(result)
    return scores


def _detect_text_row_bands(crop_img: np.ndarray) -> List[Tuple[int, int]]:
    gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    img_h, img_w = gray.shape[:2]
    formula_w = min(img_w, max(80, int(img_w * 0.50)))
    focused = gray[:, :formula_w]

    dark_mask = (focused < 185).astype(np.uint8)
    if dark_mask.size == 0:
        return []

    # Remove long proof bars and scope underlines so they do not merge adjacent rows.
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(20, int(formula_w * 0.18)), 1))
    horizontal_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, horizontal_kernel)
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(20, int(img_h * 0.35))))
    vertical_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, vertical_kernel)
    text_mask = dark_mask.copy()
    text_mask[horizontal_mask > 0] = 0
    text_mask[vertical_mask > 0] = 0

    row_counts = text_mask.sum(axis=1).astype(np.float32)
    if row_counts.size == 0:
        return []

    kernel = np.ones(9, dtype=np.float32) / 9.0
    smoothed = np.convolve(row_counts, kernel, mode="same")
    threshold = max(6.0, float(smoothed.max()) * 0.16)
    active_rows = np.where(smoothed >= threshold)[0]
    if active_rows.size == 0:
        return []

    bands: List[Tuple[int, int]] = []
    start = int(active_rows[0])
    prev = int(active_rows[0])
    for idx in active_rows[1:]:
        idx = int(idx)
        if idx <= prev + 4:
            prev = idx
            continue
        bands.append((start, prev))
        start = idx
        prev = idx
    bands.append((start, prev))

    merged: List[Tuple[int, int]] = []
    for band_start, band_end in bands:
        if merged and band_start <= merged[-1][1] + 10:
            merged[-1] = (merged[-1][0], band_end)
        else:
            merged.append((band_start, band_end))

    split_bands: List[Tuple[int, int]] = []
    valley_threshold = max(2.0, threshold * 0.45)
    min_band_height = 6
    min_gap_height = 4
    for band_start, band_end in merged:
        if band_end - band_start < 22:
            split_bands.append((band_start, band_end))
            continue

        current_start = band_start
        gap_start: int | None = None
        gap_end: int | None = None
        for row in range(band_start, band_end + 1):
            if smoothed[row] < valley_threshold:
                if gap_start is None:
                    gap_start = row
                gap_end = row
                continue

            if gap_start is not None and gap_end is not None:
                gap_height = gap_end - gap_start + 1
                left_height = gap_start - current_start
                right_height = band_end - gap_end
                if gap_height >= min_gap_height and left_height >= min_band_height and right_height >= min_band_height:
                    split_bands.append((current_start, gap_start - 1))
                    current_start = gap_end + 1
                gap_start = None
                gap_end = None

        if band_end - current_start + 1 >= min_band_height:
            split_bands.append((current_start, band_end))

    normalized = [(start, end) for start, end in split_bands if end - start >= min_band_height]
    if not normalized:
        normalized = [(start, end) for start, end in merged if end - start >= 6]

    component_bands: List[Tuple[int, int]] = []
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(text_mask, 8)
    raw_component_bands: List[Tuple[int, int]] = []
    for label_index in range(1, num_labels):
        area = int(stats[label_index, cv2.CC_STAT_AREA])
        if area < 10:
            continue
        top = int(stats[label_index, cv2.CC_STAT_TOP])
        height = int(stats[label_index, cv2.CC_STAT_HEIGHT])
        raw_component_bands.append((top, top + height - 1))

    for band_start, band_end in sorted(raw_component_bands):
        if component_bands and band_start <= component_bands[-1][1] + 8:
            component_bands[-1] = (component_bands[-1][0], max(component_bands[-1][1], band_end))
        else:
            component_bands.append((band_start, band_end))
    component_bands = [(start, end) for start, end in component_bands if end - start >= 4]

    if len(component_bands) > len(normalized):
        return component_bands
    return normalized


def _focused_unary_line_crop_path(
    *,
    crop_path: Path,
    line_order: int | None,
    output_dir: Path,
    node_id: str,
) -> Path:
    if line_order is None:
        return crop_path

    crop_img = cv2.imread(str(crop_path))
    if crop_img is None:
        return crop_path

    bands = _detect_text_row_bands(crop_img)
    if not bands or line_order < 0 or line_order >= len(bands):
        return crop_path

    band_top, band_bottom = bands[line_order]
    img_h, img_w = crop_img.shape[:2]
    pad_y = 12
    y1 = max(0, band_top - pad_y)
    y2 = min(img_h, band_bottom + pad_y)
    x2 = min(img_w, max(80, int(img_w * 0.68)))
    focused = crop_img[y1:y2, 0:x2]
    if focused.size == 0:
        return crop_path

    focused_path = output_dir / f"unary_line_{node_id}_{line_order + 1}.jpg"
    cv2.imwrite(str(focused_path), focused)
    return focused_path


def _build_unary_formula_recheck_prompt(entry: Dict[str, Any], candidates: List[str]) -> str:
    formula_text = _normalize_formula_text(
        entry.get("formula_text") or entry.get("wffstr") or "",
        str(entry.get("scope_role") or "derived_line").strip(),
    )
    justification = _normalize_justification_text(entry.get("justification_text") or entry.get("jstr") or "")
    candidate_text = " ; ".join(candidates)
    has_box = any("□" in candidate for candidate in candidates)
    has_diamond = any("⋄" in candidate for candidate in candidates)

    if has_box and not has_diamond:
        lines = [
            "This image shows one handwritten modal logic formula.",
            'Return only JSON with keys: box_present, left_of_box_negations, right_of_box_negations, selected_formula, confidence.',
            "Count exactly how many negation marks (¬) appear immediately to the LEFT of the box symbol (□).",
            "Count exactly how many negation marks (¬) appear immediately to the RIGHT of the box symbol before the final propositional letter.",
            "Then choose the exact candidate that best matches the handwriting.",
            "Do not invent a formula outside the candidate list.",
            f"Current OCR formula: {formula_text}",
        ]
    elif has_diamond and not has_box:
        lines = [
            "This image shows one handwritten modal logic formula.",
            'Return only JSON with keys: diamond_present, left_of_diamond_negations, right_of_diamond_negations, selected_formula, confidence.',
            "Count exactly how many negation marks (¬) appear immediately to the LEFT of the diamond symbol (⋄).",
            "Count exactly how many negation marks (¬) appear immediately to the RIGHT of the diamond symbol before the final propositional letter.",
            "Then choose the exact candidate that best matches the handwriting.",
            "Do not invent a formula outside the candidate list.",
            f"Current OCR formula: {formula_text}",
        ]
    else:
        lines = [
            "This image shows one handwritten modal logic formula.",
            'Return only JSON with keys: negations_before_final_atom, selected_formula, confidence.',
            "Count exactly how many negation marks (¬) appear immediately before the final propositional letter.",
            "Then choose the exact candidate that best matches the handwriting.",
            "Do not invent a formula outside the candidate list.",
            f"Current OCR formula: {formula_text}",
        ]

    if justification:
        lines.append(f"Current OCR justification: {justification}")
    lines.append(f"Candidates: {candidate_text}")
    lines.append('If you cannot tell, return "selected_formula": "uncertain".')
    return "\n".join(lines)


def _select_unary_forced_choice_candidates(
    *,
    original_formula: str,
    generated_candidates: List[str],
    candidate_scores: Dict[str, Tuple[int, int]],
) -> List[str]:
    original_score = candidate_scores.get(original_formula)
    if original_score is None:
        return [original_formula]

    forced_candidates = [original_formula]
    ranked_alternatives = sorted(
        (
            (candidate, score)
            for candidate, score in candidate_scores.items()
            if candidate != original_formula and _unary_formula_skeleton(candidate) == _unary_formula_skeleton(original_formula)
        ),
        key=lambda item: (item[1][0], item[1][1], generated_candidates.index(item[0])),
    )

    for candidate, score in ranked_alternatives:
        parse_delta = score[0] - original_score[0]
        check_delta = score[1] - original_score[1]
        if parse_delta > 0:
            continue
        if check_delta > 1:
            continue
        forced_candidates.append(candidate)
        if len(forced_candidates) >= 4:
            break

    return forced_candidates


def _recheck_suspicious_unary_formulas(
    *,
    proof_structure: List[Any],
    crop_path: Path,
    client: LMStudioQwenClient | None,
    output_dir: Path,
    modal_config: Dict[str, Any],
    node_id: str,
) -> List[Dict[str, Any]]:
    if client is None:
        return []

    recheck_records: List[Dict[str, Any]] = []
    for entry in _iter_proof_line_dicts(proof_structure):
        if str(entry.get("_ocr_node_id") or "") != node_id:
            continue
        if not _should_recheck_unary_formula(entry):
            continue

        scope_role = str(entry.get("scope_role") or "derived_line").strip()
        line_label = str(entry.get("line_label") or "").strip()
        source_node_id = str(entry.get("_ocr_node_id") or "").strip()
        source_entry_index = entry.get("_ocr_entry_index")
        source_line_order = entry.get("_ocr_line_order")
        if not source_node_id:
            continue
        if not line_label and not isinstance(source_entry_index, int):
            continue
        original_formula = _normalize_formula_text(entry.get("formula_text") or entry.get("wffstr") or "", scope_role)
        generated_candidates = _build_unary_formula_candidates(entry)
        if len(generated_candidates) <= 1:
            continue

        try:
            candidate_scores = _score_unary_formula_candidates(
                proof_structure=proof_structure,
                source_node_id=source_node_id,
                source_line_label=line_label,
                source_entry_index=source_entry_index,
                candidates=generated_candidates,
                modal_config=modal_config,
            )
        except Exception as exc:
            recheck_records.append(
                {
                    "node_id": source_node_id,
                    "line_label": line_label,
                    "before": original_formula,
                    "generated_candidates": generated_candidates,
                    "checker_error": str(exc),
                    "accepted": False,
                    "final_formula": original_formula,
                }
            )
            continue

        original_score = candidate_scores.get(original_formula)
        if original_score is None:
            continue
        improving_candidates = [
            candidate for candidate in generated_candidates if candidate_scores.get(candidate, original_score) < original_score
        ]
        forced_choice_candidates = _select_unary_forced_choice_candidates(
            original_formula=original_formula,
            generated_candidates=generated_candidates,
            candidate_scores=candidate_scores,
        )
        if len(forced_choice_candidates) <= 1:
            recheck_records.append(
                {
                    "node_id": source_node_id,
                    "line_label": line_label,
                    "before": original_formula,
                    "generated_candidates": generated_candidates,
                    "candidate_scores": {candidate: list(score) for candidate, score in candidate_scores.items()},
                    "improving_candidates": [],
                    "forced_choice_candidates": [original_formula],
                    "vlm_choice": None,
                    "accepted": False,
                    "final_formula": original_formula,
                }
            )
            continue

        line_crop_path = _focused_unary_line_crop_path(
            crop_path=crop_path,
            line_order=source_line_order if isinstance(source_line_order, int) else None,
            output_dir=output_dir,
            node_id=source_node_id,
        )
        with line_crop_path.open("rb") as handle:
            line_base64_image = base64.b64encode(handle.read()).decode("utf-8")

        raw_response = client.chat(
            [
                {"role": "system", "content": "You are a careful vision model. Follow the user's instructions exactly."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": _build_unary_formula_recheck_prompt(entry, forced_choice_candidates)},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{line_base64_image}"}},
                    ],
                },
            ]
        )
        parsed = _parse_unary_recheck_choice(raw_response)
        selected_formula = str(parsed.get("selected_formula") or "").strip()
        if selected_formula and selected_formula != "uncertain":
            selected_formula = _normalize_formula_text(selected_formula, scope_role)
        accepted = bool(selected_formula and selected_formula not in {"uncertain", original_formula} and selected_formula in forced_choice_candidates)
        if accepted:
            entry["wffstr"] = selected_formula

        recheck_records.append(
            {
                "node_id": source_node_id,
                "line_label": line_label,
                "before": original_formula,
                "generated_candidates": generated_candidates,
                "candidate_scores": {candidate: list(score) for candidate, score in candidate_scores.items()},
                "improving_candidates": improving_candidates,
                "forced_choice_candidates": forced_choice_candidates,
                "vlm_choice": selected_formula or "uncertain",
                "accepted": accepted,
                "final_formula": selected_formula if accepted else original_formula,
                "line_crop_path": str(line_crop_path),
                "raw_response": raw_response,
            }
        )

    if recheck_records:
        artifact_path = output_dir / UNARY_RECHECK_FILENAME
        existing_records: List[Dict[str, Any]] = []
        if artifact_path.exists():
            try:
                existing_records = json.loads(artifact_path.read_text(encoding="utf-8"))
            except Exception:
                existing_records = []
        artifact_path.write_text(
            json.dumps(existing_records + recheck_records, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    return recheck_records


def process_node_recursive(
    node: SubproofNode,
    img: np.ndarray,
    output_dir: Path,
    library: Dict[str, List[Dict[str, Any]]],
    *,
    formula_prompt: str | None = None,
    justification_prompt: str | None = None,
    client: LMStudioQwenClient | None,
    opener_client: LMStudioQwenClient | None = None,
    verifier_client: LMStudioQwenClient | None = None,
    base_prompt: str | None = None,
) -> None:
    if formula_prompt is None:
        formula_prompt = base_prompt or ""
    if justification_prompt is None:
        justification_prompt = base_prompt or ""
    if node.id == "ROOT":
        current_img = img.copy()
    else:
        current_img = crop_region(
            img,
            node.box,
            bottom_padding=0,
        )

    for child in sorted(node.children, key=lambda c: c.box[1]):
        process_node_recursive(
            child,
            img,
            output_dir,
            library,
            formula_prompt=formula_prompt,
            justification_prompt=justification_prompt,
            client=client,
            opener_client=opener_client,
            verifier_client=verifier_client,
            base_prompt=base_prompt,
        )
        if node.id == "ROOT":
            current_img = mask_subproof_in_parent(current_img, child)
        else:
            px, py, _, _ = node.box
            rel_box = (child.box[0] - px + 10, child.box[1] - py + 10, child.box[2], child.box[3])
            current_img = mask_subproof_in_parent(current_img, SubproofNode(child.id, rel_box))

    save_path = output_dir / f"crop_{node.id}.jpg"
    cv2.imwrite(str(save_path), current_img)
    print(f"Processing Node {node.id} (Depth {node.depth})... Saved to {save_path.name}")

    parsed_entries, node_response = assemble_node_entries_from_rows(
        node=node,
        node_crop_img=current_img,
        output_dir=output_dir,
        formula_prompt=formula_prompt,
        justification_prompt=justification_prompt,
        client=client,
        opener_client=opener_client,
        verifier_client=verifier_client,
        base_prompt=base_prompt,
    )
    (output_dir / RAW_RESPONSE_TEMPLATE.format(node_id=node.id)).write_text(
        json.dumps(node_response, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / INTERMEDIATE_TEMPLATE.format(node_id=node.id)).write_text(
        json.dumps(parsed_entries, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    library[node.id] = parsed_entries


def _normalize_formula_text(text: Any, scope_role: str) -> str:
    raw = str(text or "").strip()
    if scope_role == "strict_open":
        return "□"
    collapsed = re.sub(r"\s+", "", raw)
    if collapsed in {"⊥", "_|_", "bot", "bottom"}:
        return "⊥"

    replacements = [
        ("<->", "↔"),
        ("->", "→"),
        ("=>", "→"),
        ("[]", "□"),
        ("<>", "⋄"),
        ("◇", "⋄"),
        ("◊", "⋄"),
        ("~", "¬"),
        ("&", "∧"),
        ("^", "∧"),
        (" v ", " ∨ "),
    ]
    normalized = raw
    for src, dst in replacements:
        normalized = normalized.replace(src, dst)
    normalized = re.sub(r"(?<![A-Za-z0-9_])[DO0](?=\()", "□", normalized)
    normalized = re.sub(r"(?<![A-Za-z0-9_])[DO0](?=[A-Z¬□⋄◇])", "□", normalized)
    return normalized


def _normalize_justification_text(text: Any) -> str:
    raw = str(text or "").strip()
    raw = re.sub(r"^\s*incorrect\s+use\s+of\s+", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"^\s*invalid\s+use\s+of\s+", "", raw, flags=re.IGNORECASE)
    raw = raw.replace(",", " ")
    raw = raw.replace("–", "-")
    raw = raw.replace("—", "-")
    raw = re.sub(r"^(?:[DO0])\s*([EI])\s*(.+)$", lambda m: f"□{m.group(1)} {m.group(2).strip()}", raw)
    raw = re.sub(r"\s+", " ", raw).strip()
    mapping = {
        "boxe": "□E",
        "boxi": "□I",
        "rfour": "R4",
        "rfive": "R5",
        "rt": "RT",
        "mc": "MC",
        "diadf": "Def⋄",
        "ai": "∧I",
        "ae": "∧E",
        "ci": "→I",
        "ce": "→E",
        "ni": "¬I",
        "ne": "¬E",
        "ip": "IP",
    }
    parts = raw.split(" ", 1)
    head = parts[0]
    tail = parts[1] if len(parts) > 1 else ""
    key = head.lower()
    if key in mapping:
        head = mapping[key]
    elif head == "[]E":
        head = "□E"
    elif head == "[]I":
        head = "□I"
    elif head in {"Def◇", "Def◊", "Def⋄"}:
        head = "Def⋄"
    return f"{head} {tail}".strip()


def _entry_to_line(
    entry: Dict[str, Any],
    *,
    node_id: str | None = None,
    entry_index: int | None = None,
    line_order: int | None = None,
) -> Optional[Dict[str, Any]]:
    scope_role = str(entry.get("scope_role") or "derived_line").strip()
    formula_text = str(entry.get("formula_text") or entry.get("wffstr") or "").strip()
    token_match = TOKEN_PATTERN.search(formula_text)
    if scope_role == "subproof_token" or token_match:
        return None

    if scope_role in {"premise", "ordinary_assumption", "strict_open"}:
        jstr = "PR"
    else:
        jstr = _normalize_justification_text(entry.get("justification_text") or entry.get("jstr") or "")

    line = {
        "line_label": str(entry.get("line_label") or "").strip(),
        "wffstr": _normalize_formula_text(formula_text, scope_role),
        "jstr": jstr,
        "scope_role": scope_role,
    }
    if node_id:
        line["_ocr_node_id"] = str(node_id)
    if entry_index is not None:
        line["_ocr_entry_index"] = int(entry_index)
    if line_order is not None:
        line["_ocr_line_order"] = int(line_order)
    return line


def _infer_missing_prior_line_label(entries: List[Dict[str, Any]]) -> str:
    for entry in entries:
        formula_text = str(entry.get("formula_text") or entry.get("wffstr") or "").strip()
        if TOKEN_PATTERN.search(formula_text):
            continue
        line_label = str(entry.get("line_label") or "").strip()
        if re.fullmatch(r"\d+", line_label):
            prior = max(1, int(line_label) - 1)
            return str(prior)
        return ""
    return ""


def _prepend_synthetic_strict_open_if_needed(
    entries: List[Dict[str, Any]],
    opener_result: Dict[str, Any] | None,
) -> List[Dict[str, Any]]:
    if not entries or not opener_result or opener_result.get("kind") != "strict_open":
        return entries

    for entry in entries:
        scope_role = str(entry.get("scope_role") or "derived_line").strip()
        formula_text = str(entry.get("formula_text") or entry.get("wffstr") or "").strip()
        if TOKEN_PATTERN.search(formula_text):
            continue
        if scope_role == "strict_open" or _normalize_formula_text(formula_text, scope_role) == "□":
            return entries
        break

    synthetic = {
        "formula_text": "□",
        "justification_text": "",
        "scope_role": "strict_open",
        "line_label": _infer_missing_prior_line_label(entries),
    }
    return [synthetic, *entries]


def _is_strict_subproof_block(block: Any) -> bool:
    return bool(
        isinstance(block, list)
        and block
        and isinstance(block[0], dict)
        and str(block[0].get("wffstr", "")).strip() == "□"
        and str(block[0].get("jstr", "")).strip() == "PR"
    )


def _fallback_insert_index_for_child(resolved: List[Any], child_block: List[Any]) -> int:
    if _is_strict_subproof_block(child_block):
        for index, item in enumerate(resolved):
            if not isinstance(item, dict):
                continue
            citation = str(item.get("jstr") or "").strip()
            if citation.startswith("□I"):
                return index

    for index, item in enumerate(resolved):
        if not isinstance(item, dict):
            continue
        citation = str(item.get("jstr") or "").strip()
        if citation and citation != "PR":
            return index
    return len(resolved)


def recursive_stitch(
    entries: List[Dict[str, Any]],
    library: Dict[str, List[Dict[str, Any]]],
    node: Optional[SubproofNode] = None,
) -> List[Any]:
    resolved: List[Any] = []
    consumed_child_ids = set()
    visible_line_order = 0
    for entry_index, entry in enumerate(entries):
        formula_text = str(entry.get("formula_text") or entry.get("wffstr") or "").strip()
        token_match = TOKEN_PATTERN.search(formula_text)
        if token_match:
            child_id = token_match.group(1)
            child_entries = library.get(child_id, [])
            child_node = None
            if node is not None:
                child_node = next((child for child in node.children if child.id == child_id), None)
            resolved.append(recursive_stitch(child_entries, library, child_node))
            consumed_child_ids.add(child_id)
            continue

        line = _entry_to_line(
            entry,
            node_id=node.id if node is not None else None,
            entry_index=entry_index,
            line_order=visible_line_order,
        )
        if line is not None:
            resolved.append(line)
            visible_line_order += 1

    if node is not None:
        for child in sorted(node.children, key=lambda c: c.box[1]):
            if child.id in consumed_child_ids:
                continue
            child_entries = library.get(child.id, [])
            if not child_entries:
                continue
            child_block = recursive_stitch(child_entries, library, child)
            insert_at = _fallback_insert_index_for_child(resolved, child_block)
            resolved.insert(insert_at, child_block)
    return resolved


def renumber_lines(proof_structure: List[Any], start_counter: int = 1) -> int:
    current_counter = start_counter
    for item in proof_structure:
        if isinstance(item, list):
            current_counter = renumber_lines(item, current_counter)
        elif isinstance(item, dict):
            item["line"] = current_counter
            current_counter += 1
    return current_counter


STRICT_EXTERNAL_REFERENCE_RULES = {"□E", "R4", "R5"}


def _strict_reference_tokens(
    tail: str,
) -> List[str]:
    return re.findall(r"\b\d+(?:-\d+)?\b", tail)


def _strict_token_numbers(
    token: str,
) -> Optional[List[int]]:
    if "-" in token:
        left_raw, right_raw = token.split("-", 1)
        try:
            return [int(left_raw), int(right_raw)]
        except ValueError:
            return None
    try:
        return [int(token)]
    except ValueError:
        return None


def _strict_tail_fits_scope(
    tail: str,
    *,
    strict_start_line: int,
    current_line: int,
    offset: int = 0,
) -> bool:
    def valid_number(number: int) -> bool:
        return strict_start_line <= number < current_line

    tokens = _strict_reference_tokens(tail)
    if not tokens:
        return False
    for token in tokens:
        numbers = _strict_token_numbers(token)
        if not numbers:
            return False
        if not all(valid_number(number + offset) for number in numbers):
            return False
    return True


def _rewrite_strict_reference_token(
    token: str,
    *,
    offset: int,
) -> str:
    if "-" in token:
        left_raw, right_raw = token.split("-", 1)
        return f"{int(left_raw) + offset}-{int(right_raw) + offset}"
    return str(int(token) + offset)


def _rewrite_citation_tail_for_strict_scope(
    tail: str,
    *,
    strict_start_line: int,
    current_line: int,
    offset: Optional[int],
) -> str:
    if _strict_tail_fits_scope(tail, strict_start_line=strict_start_line, current_line=current_line, offset=0):
        return tail
    if offset is not None and _strict_tail_fits_scope(
        tail,
        strict_start_line=strict_start_line,
        current_line=current_line,
        offset=offset,
    ):
        return re.sub(
            r"\b\d+(?:-\d+)?\b",
            lambda match: _rewrite_strict_reference_token(match.group(0), offset=offset),
            tail,
        )
    return tail


def rewrite_strict_subproof_local_citations(proof_structure: List[Any]) -> None:
    for item in proof_structure:
        if isinstance(item, list):
            is_strict = bool(
                item
                and isinstance(item[0], dict)
                and str(item[0].get("wffstr", "")).strip() == "□"
                and str(item[0].get("jstr", "")).strip() == "PR"
            )

            if is_strict:
                strict_start_line = item[0].get("line") if isinstance(item[0], dict) else None
                first_label_text = str(item[0].get("line_label") or "").strip() if isinstance(item[0], dict) else ""
                try:
                    first_local_label = int(first_label_text) if first_label_text else None
                except ValueError:
                    first_local_label = None
                offset = None
                if isinstance(strict_start_line, int) and first_local_label is not None:
                    offset = strict_start_line - first_local_label

                for line in item:
                    if not isinstance(line, dict):
                        continue
                    citation = str(line.get("jstr", "")).strip()
                    if not citation or citation == "PR":
                        continue
                    parts = citation.split(" ", 1)
                    rule = parts[0]
                    tail = parts[1] if len(parts) > 1 else ""
                    current_line = line.get("line")
                    if (
                        not tail
                        or rule in STRICT_EXTERNAL_REFERENCE_RULES
                        or not isinstance(strict_start_line, int)
                        or not isinstance(current_line, int)
                    ):
                        continue
                    line["jstr"] = (
                        f"{rule} "
                        f"{_rewrite_citation_tail_for_strict_scope(tail, strict_start_line=strict_start_line, current_line=current_line, offset=offset)}"
                    ).strip()

            rewrite_strict_subproof_local_citations(item)


def _boxed_formula_matches_inner_line(boxed: str, inner: str) -> bool:
    boxed_text = str(boxed or "").strip()
    inner_text = str(inner or "").strip()
    if not boxed_text.startswith("□"):
        return False
    remainder = boxed_text[1:].strip()
    if remainder.startswith("(") and remainder.endswith(")"):
        remainder = remainder[1:-1].strip()
    return remainder == inner_text


def repair_modal_subproof_boundary_lines(proof_structure: List[Any]) -> None:
    for item in proof_structure:
        if isinstance(item, list):
            repair_modal_subproof_boundary_lines(item)

    index = 0
    while index < len(proof_structure):
        item = proof_structure[index]
        if not isinstance(item, list):
            index += 1
            continue

        if item and isinstance(item[-1], dict) and index + 1 < len(proof_structure):
            next_item = proof_structure[index + 1]
            if (
                isinstance(next_item, dict)
                and item[-1].get("wffstr") == next_item.get("wffstr")
                and item[-1].get("jstr") == next_item.get("jstr")
            ):
                item.pop()

        for child_index, child in enumerate(item[:-1]):
            if not isinstance(child, list) or not child:
                continue
            next_line = item[child_index + 1]
            if not isinstance(next_line, dict):
                continue
            if str(next_line.get("jstr") or "").strip() not in {"", "derived_line"}:
                continue
            if not (
                isinstance(child[0], dict)
                and str(child[0].get("wffstr") or "").strip() == "□"
                and str(child[0].get("jstr") or "").strip() == "PR"
            ):
                continue
            last_child_line = child[-1]
            if not isinstance(last_child_line, dict):
                continue
            if not _boxed_formula_matches_inner_line(next_line.get("wffstr"), last_child_line.get("wffstr")):
                continue
            start_line = child[0].get("line")
            end_line = last_child_line.get("line")
            if isinstance(start_line, int) and isinstance(end_line, int):
                next_line["jstr"] = f"□I {start_line}-{end_line}"

        index += 1


def _strip_internal_ocr_metadata(proof_structure: List[Any]) -> None:
    for item in proof_structure:
        if isinstance(item, list):
            _strip_internal_ocr_metadata(item)
            continue
        if not isinstance(item, dict):
            continue
        item.pop("_ocr_node_id", None)
        item.pop("_ocr_entry_index", None)
        item.pop("_ocr_line_order", None)
        item.pop("scope_role", None)


def _apply_unary_verifier_corrections(
    proof_structure: List[Any],
    verifier_records: List[Dict[str, Any]],
) -> tuple[List[Any], List[Dict[str, Any]]]:
    return _apply_unary_verifier_corrections_impl(
        proof_structure,
        verifier_records,
        clone_jsonish=_clone_jsonish,
        iter_proof_line_dicts=_iter_proof_line_dicts,
    )


def build_modal_state(
    *,
    config: Dict[str, Any],
    final_structure: List[Any],
    original_final_structure: Optional[List[Any]] = None,
    verifier_rewrite_summary: Optional[List[Dict[str, Any]]] = None,
    checker_request: Optional[Dict[str, Any]],
    checker_response: Optional[Dict[str, Any]],
    run_summary: str,
    vision_base_url: str | None = None,
    vision_model: str | None = None,
    verifier_vision_model: str | None = None,
    source_image_name: str = "",
) -> Dict[str, Any]:
    canonical = _canonical_modal_config(config)
    return {
        "system": canonical["system"],
        "include_derived_rules": canonical["include_derived_rules"],
        "conclusion": canonical.get("conclusion") or "",
        "vision_base_url": (vision_base_url or default_lmstudio_base_url()).strip(),
        "vision_model": (vision_model or default_lmstudio_model()).strip(),
        "verifier_vision_model": (verifier_vision_model or default_unary_verifier_model()).strip(),
        "source_image_name": source_image_name,
        "proof_json_text": json.dumps({"lines": final_structure, "meta": {}}, indent=2, ensure_ascii=False),
        "original_proof_json_text": json.dumps(
            {"lines": original_final_structure or final_structure, "meta": {}},
            indent=2,
            ensure_ascii=False,
        ),
        "verifier_rewrite_summary": verifier_rewrite_summary or [],
        "draft_comment": "",
        "draft_grade": "",
        "checker_request": checker_request,
        "checker_response": checker_response,
        "run_summary": run_summary,
    }


def _clear_output_dir(output_dir: Path, *, preserve_paths: List[Path] | None = None) -> None:
    preserve = {path.resolve() for path in (preserve_paths or [])}
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for child in output_dir.iterdir():
        child_resolved = child.resolve()
        if child_resolved in preserve:
            continue
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def transcribe_image_to_modal_debug_state(
    *,
    image_path: Path,
    output_dir: Path,
    modal_config: Dict[str, Any],
    client: LMStudioQwenClient | None = None,
    verifier_client: LMStudioQwenClient | None = None,
    enable_unary_verifier: bool = False,
    vision_base_url: str | None = None,
    vision_model: str | None = None,
    verifier_vision_model: str | None = None,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / UNARY_VERIFIER_FILENAME).unlink(missing_ok=True)
    for path in output_dir.glob("verifier_line_*.jpg"):
        path.unlink(missing_ok=True)
    for pattern in ("node_line_*.jpg", "node_line_*.json", "node_rows_*.json", "node_rows_overlay_*.jpg", "node_entries_*.json"):
        for path in output_dir.glob(pattern):
            path.unlink(missing_ok=True)
    formula_prompt = build_formula_ocr_prompt(modal_config)
    justification_prompt = build_justification_ocr_prompt(modal_config)
    prompt_artifact = "\n\n".join(
        [
            "=== FORMULA ZONE OCR PROMPT ===",
            formula_prompt,
            "=== JUSTIFICATION ZONE OCR PROMPT ===",
            justification_prompt,
            "=== STRICT OPENER CLASSIFIER PROMPT ===",
            _build_subproof_opener_prompt(),
        ]
    )
    (output_dir / PROMPT_FILENAME).write_text(prompt_artifact, encoding="utf-8")

    if client is None:
        client = vlm_client
    if client is None:
        raise RuntimeError("VLM client is not available")

    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Could not read image at {image_path}")

    bars = detect_vertical_bars(img)
    debug_img = img.copy()
    for idx, (x, y, w, h) in enumerate(bars, start=1):
        cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(debug_img, str(idx), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imwrite(str(output_dir / "debug_bars.jpg"), debug_img)

    tree_root = build_proof_tree(bars, img.shape[1], img.shape[0])
    library: Dict[str, List[Dict[str, Any]]] = {}
    process_node_recursive(
        tree_root,
        img,
        output_dir,
        library,
        formula_prompt=formula_prompt,
        justification_prompt=justification_prompt,
        client=client,
        opener_client=verifier_client or client,
        verifier_client=verifier_client,
    )

    root_entries = library.get("ROOT", [])
    root_has_content = any(str(item.get("formula_text") or item.get("wffstr") or "").strip() for item in root_entries)
    if not root_has_content and "1" in library:
        top_level_entries = library["1"]
    else:
        top_level_entries = root_entries

    final_structure = recursive_stitch(top_level_entries, library, tree_root)
    while isinstance(final_structure, list) and len(final_structure) == 1 and isinstance(final_structure[0], list):
        final_structure = final_structure[0]

    renumber_lines(final_structure)
    rewrite_strict_subproof_local_citations(final_structure)
    repair_modal_subproof_boundary_lines(final_structure)
    renumber_lines(final_structure)

    baseline_persisted_structure = _clone_jsonish(final_structure)
    _strip_internal_ocr_metadata(baseline_persisted_structure)
    (output_dir / BASELINE_FINAL_STRUCTURE_FILENAME).write_text(
        json.dumps(baseline_persisted_structure, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    baseline_checker_response = None
    try:
        baseline_proof_json_text = json.dumps({"lines": baseline_persisted_structure, "meta": {}}, ensure_ascii=False)
        baseline_checker_result = run_modal_proof_checker(
            answer_text=baseline_proof_json_text,
            autograde_config=_canonical_modal_config(modal_config),
        )
        baseline_checker_response = baseline_checker_result.response_payload
    except Exception:
        baseline_checker_response = None

    rewritten_structure, verifier_records, verifier_rewrites = run_unary_postprocess(
        enabled=enable_unary_verifier,
        proof_structure=final_structure,
        tree_root=tree_root,
        full_img=img,
        output_dir=output_dir,
        verifier_client=verifier_client,
        checker_response=baseline_checker_response,
        run_experiment=_run_unary_verifier_experiment,
        apply_corrections=_apply_unary_verifier_corrections,
    )

    persisted_structure = _clone_jsonish(rewritten_structure)
    _strip_internal_ocr_metadata(persisted_structure)
    (output_dir / FINAL_STRUCTURE_FILENAME).write_text(
        json.dumps(persisted_structure, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / VERIFIER_REWRITES_FILENAME).write_text(
        json.dumps(verifier_rewrites, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    checker_request = None
    checker_response = None
    rewrite_count = len(verifier_rewrites)
    if rewrite_count:
        run_summary = (
            f"OCR produced a modal proof structure, verifier rewrote {rewrite_count} line"
            f"{'' if rewrite_count == 1 else 's'}, and the checker ran."
        )
    else:
        run_summary = "OCR produced a modal proof structure and the checker ran."

    try:
        proof_json_text = json.dumps({"lines": persisted_structure, "meta": {}}, ensure_ascii=False)
        checker_result = run_modal_proof_checker(
            answer_text=proof_json_text,
            autograde_config=_canonical_modal_config(modal_config),
        )
        checker_request = checker_result.request_payload
        checker_response = checker_result.response_payload
        if rewrite_count:
            run_summary = (
                f"OCR produced a modal proof structure, verifier rewrote {rewrite_count} line"
                f"{'' if rewrite_count == 1 else 's'}, and the checker ran."
            )
        else:
            run_summary = "OCR produced a modal proof structure and the checker ran."
        (output_dir / CHECKER_REQUEST_FILENAME).write_text(
            json.dumps(checker_request, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (output_dir / CHECKER_RESPONSE_FILENAME).write_text(
            json.dumps(checker_response, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    except Exception as exc:
        if rewrite_count:
            run_summary = (
                f"OCR produced a modal proof structure and verifier rewrote {rewrite_count} line"
                f"{'' if rewrite_count == 1 else 's'}, but checker execution failed: {exc}"
            )
        else:
            run_summary = f"OCR produced a modal proof structure, but checker execution failed: {exc}"

    state = build_modal_state(
        config=modal_config,
        final_structure=persisted_structure,
        original_final_structure=baseline_persisted_structure,
        verifier_rewrite_summary=verifier_rewrites,
        checker_request=checker_request,
        checker_response=checker_response,
        run_summary=run_summary,
        vision_base_url=vision_base_url or getattr(client, "base_url", None),
        vision_model=vision_model or getattr(client, "model", None),
        verifier_vision_model=verifier_vision_model or getattr(verifier_client, "model", None),
        source_image_name=image_path.name,
    )
    (output_dir / STATE_FILENAME).write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
    return state


def main() -> int:
    parser = argparse.ArgumentParser(description="Run recursive OCR for a modal proof crop.")
    parser.add_argument("--image", required=True, help="Path to the image to transcribe.")
    parser.add_argument("--output-dir", required=True, help="Directory where OCR artifacts should be written.")
    parser.add_argument("--autograde-file", help="YAML/JSON file with modal checker config or saved state.json.")
    parser.add_argument("--lmstudio-base-url", default=default_lmstudio_base_url(), help="LM Studio OpenAI-compatible base URL.")
    parser.add_argument("--lmstudio-model", default=default_lmstudio_model(), help="LM Studio model identifier.")
    parser.add_argument(
        "--lmstudio-verifier-model",
        default=default_unary_verifier_model(),
        help="LM Studio model identifier for the unary verifier experiment.",
    )
    parser.add_argument(
        "--enable-unary-verifier",
        action="store_true",
        help="Enable the terminal unary verifier postprocess that may rewrite OCR formulas after the baseline checker run.",
    )
    args = parser.parse_args()

    image_path = Path(args.image)
    output_dir = Path(args.output_dir)
    autograde_file = Path(args.autograde_file) if args.autograde_file else None

    if not image_path.exists():
        print(f"Error: image not found: {image_path}")
        return 1

    modal_config = _canonical_modal_config(_load_modal_config(autograde_file))
    preserve_paths = [image_path]
    if autograde_file is not None:
        preserve_paths.append(autograde_file)
    _clear_output_dir(output_dir, preserve_paths=preserve_paths)
    print(f"[OCR] Image: {image_path}")
    print(f"[OCR] Output dir: {output_dir}")
    print(f"[OCR] Modal system: {modal_config['system']}")
    print(f"[OCR] VLM base URL: {args.lmstudio_base_url}")
    print(f"[OCR] VLM model: {args.lmstudio_model}")
    print(f"[OCR] Unary verifier model: {args.lmstudio_verifier_model}")
    print(f"[OCR] Unary verifier postprocess enabled: {'yes' if args.enable_unary_verifier else 'no'}")

    try:
        client = build_vlm_client(
            base_url=args.lmstudio_base_url,
            model=args.lmstudio_model,
        )
        verifier_client = build_vlm_client(
            base_url=args.lmstudio_base_url,
            model=args.lmstudio_verifier_model,
        )
        state = transcribe_image_to_modal_debug_state(
            image_path=image_path,
            output_dir=output_dir,
            modal_config=modal_config,
            client=client,
            verifier_client=verifier_client,
            enable_unary_verifier=args.enable_unary_verifier,
            vision_base_url=args.lmstudio_base_url,
            vision_model=args.lmstudio_model,
            verifier_vision_model=args.lmstudio_verifier_model,
        )
    except Exception as exc:
        (output_dir / "ocr_error.txt").write_text(str(exc), encoding="utf-8")
        print(f"[OCR] Modal OCR failed: {exc}")
        return 1

    print("[OCR] Wrote modal OCR artifacts.")
    print(f"[OCR] State saved to {output_dir / STATE_FILENAME}")
    if state.get("checker_response"):
        print("[OCR] Checker response saved.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
