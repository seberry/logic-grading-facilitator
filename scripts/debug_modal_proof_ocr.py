from __future__ import annotations

import argparse
import base64
import json
import os
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

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


STATE_FILENAME = "state.json"
PROMPT_FILENAME = "ocr_prompt.txt"
FINAL_STRUCTURE_FILENAME = "final_structure.json"
BASELINE_FINAL_STRUCTURE_FILENAME = "final_structure_baseline.json"
CHECKER_REQUEST_FILENAME = "checker_request.json"
CHECKER_RESPONSE_FILENAME = "checker_response.json"
RAW_RESPONSE_TEMPLATE = "response_{node_id}.json"
INTERMEDIATE_TEMPLATE = "intermediate_{node_id}.json"
STRICT_OPENER_LINE_TEMPLATE = "strict_opener_line_{node_id}.jpg"
STRICT_OPENER_RESPONSE_TEMPLATE = "strict_opener_{node_id}.json"
UNARY_RECHECK_FILENAME = "unary_rechecks.json"
UNARY_VERIFIER_FILENAME = "unary_verifier.json"
VERIFIER_REWRITES_FILENAME = "verifier_rewrites.json"
SOURCE_IMAGE_FILENAME = "source_image"

SUPPORTED_SYSTEMS = {"K", "T", "S4", "S5"}
TOKEN_PATTERN = re.compile(r"<SUBPROOF_ID_([A-Za-z0-9_]+)>")


@dataclass
class SubproofNode:
    id: str
    box: Tuple[int, int, int, int]
    children: List["SubproofNode"]
    parent: Optional["SubproofNode"] = None

    def __init__(self, box_id: str, box: Tuple[int, int, int, int]) -> None:
        self.id = box_id
        self.box = box
        self.children = []
        self.parent = None

    @property
    def depth(self) -> int:
        depth = 0
        current = self.parent
        while current:
            depth += 1
            current = current.parent
        return depth


def _load_modal_config(path: Path | None) -> Dict[str, Any]:
    if path is None or not path.exists():
        return {}

    raw_text = path.read_text(encoding="utf-8")
    parsed = yaml.safe_load(raw_text) or {}
    if not isinstance(parsed, dict):
        raise ValueError("modal autograde file must parse to an object/map")

    if "proof_json_text" in parsed or "system" in parsed:
        return {
            "system": str(parsed.get("system") or "").strip() or "K",
            "include_derived_rules": bool(parsed.get("include_derived_rules", False)),
            "conclusion": str(parsed.get("conclusion") or "").strip() or None,
        }

    nested_setup = parsed.get("problem_setup_text")
    if isinstance(nested_setup, str) and nested_setup.strip():
        nested = yaml.safe_load(nested_setup) or {}
        if not isinstance(nested, dict):
            raise ValueError("problem_setup_text must parse to an object/map")
        parsed = nested

    return {
        "system": str(parsed.get("system") or "").strip() or "K",
        "include_derived_rules": bool(parsed.get("include_derived_rules", False)),
        "conclusion": str(parsed.get("conclusion") or "").strip() or None,
    }


def _canonical_modal_config(config: Dict[str, Any]) -> Dict[str, Any]:
    system = str(config.get("system") or "K").strip() or "K"
    if system not in SUPPORTED_SYSTEMS:
        system = "K"
    conclusion = config.get("conclusion")
    conclusion_text = str(conclusion).strip() if conclusion is not None else ""
    return {
        "system": system,
        "include_derived_rules": bool(config.get("include_derived_rules", False)),
        "conclusion": conclusion_text or None,
    }


def build_prompt(config: Dict[str, Any]) -> str:
    clean = _canonical_modal_config(config)
    conclusion = clean.get("conclusion")
    lines = [
        "You are transcribing a modal Fitch-style proof from an image.",
        "Return only JSON. Do not include markdown fences or explanation.",
        "",
        "Return a JSON array. Each item must be one of:",
        '1. A proof line object: {"formula_text": "...", "justification_text": "...", "scope_role": "...", "line_label": "..."}',
        '2. A masked child token object: {"formula_text": "<SUBPROOF_ID_X>", "scope_role": "subproof_token"}',
        "",
        "Allowed scope_role values:",
        '- "premise" for top-level premises',
        '- "ordinary_assumption" for the first line of a normal subproof',
        '- "strict_open" for a strict modal subproof opener; the formula_text should be just □',
        '- "derived_line" for any ordinary derived line',
        '- "subproof_token" for a red masked token placeholder',
        "",
        "Transcription rules:",
        "- Keep formulas concise and symbolic.",
        "- Prefer Unicode operators: ¬, ∧, ∨, →, ↔, □, ⋄.",
        "- For justifications, prefer forms like R 1, ∧I 4 5, ∧E 1, →I 2-3, □E 1, □I 3-6, R4 1, RT 1, MC 1, Def⋄ 1.",
        "- Copy the printed line labels exactly as they appear in the image. Do not renumber lines within a crop.",
        "- If a line starts a strict modal subproof, use scope_role strict_open and formula_text □.",
        "- Preserve the exact number of negation marks. Do not collapse ¬¬A into ¬A.",
        "- In handwritten work, double negation may appear as ¬¬A, ~~A, or mixed marks; if you see two negation strokes, transcribe two negations.",
        "- If a line cites DN, be especially careful to preserve doubled negation in the formula.",
        "- A contradiction line may be written as ⊥, _|_, _ | _, or a short vertical stroke over a horizontal bar; transcribe any of these as ⊥.",
        "- If a line's entire formula is just contradiction/absurdity, return formula_text as ⊥.",
        "- If you see a red token like <SUBPROOF_ID_2>, return it exactly as formula_text with scope_role subproof_token.",
        "- After a red subproof token, keep listing any further visible lines in the current crop. Do not stop early just because a child subproof was masked.",
        "- If a current crop contains both a masked child token and a later closing-result line such as □B or A → B, include both in order.",
        "- Do not invent nested JSON structure yourself; just list the visible lines in order for the current crop.",
        "",
        f"Modal system: {clean['system']}",
        f"Include derived rules: {'yes' if clean['include_derived_rules'] else 'no'}",
    ]
    if conclusion:
        lines.append(f"Target conclusion: {conclusion}")
    return "\n".join(lines)


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


UNARY_PREFIX_CHARS = {"¬", "□", "⋄"}
UNARY_SENSITIVE_RULE_HEADS = {"MC", "DN", "Def⋄"}
UNARY_CANDIDATE_PREFIXES = ("¬", "¬¬", "¬¬¬", "□", "□¬", "¬□", "¬□¬", "⋄", "⋄¬", "¬⋄", "¬⋄¬")


def _leading_unary_prefix(formula_text: str) -> str:
    prefix_chars: List[str] = []
    for char in formula_text:
        if char not in UNARY_PREFIX_CHARS:
            break
        prefix_chars.append(char)
    return "".join(prefix_chars)


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


UNARY_VERIFIER_RULE_HEADS = {"MC", "DN", "Def⋄", "□I", "□E"}


def _checker_payload_has_issues(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    return bool(payload.get("parse_errors") or payload.get("check_errors"))


def _should_verify_unary_formula(entry: Dict[str, Any]) -> bool:
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

    if formula_text.startswith(tuple(UNARY_PREFIX_CHARS)):
        return True
    if "□" in formula_text or "⋄" in formula_text:
        return True
    if prefix.count("¬") >= 2:
        return True
    return rule_head in UNARY_VERIFIER_RULE_HEADS


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
    lookup = {node.id: node}
    for child in node.children:
        lookup.update(_build_node_lookup(child))
    return lookup


def _verifier_node_crop_image(*, node: SubproofNode, full_img: np.ndarray) -> np.ndarray:
    if node.id == "ROOT":
        crop_img = full_img.copy()
    else:
        crop_img = crop_region(full_img, node.box, bottom_padding=0).copy()

    if crop_img.size == 0:
        return crop_img

    for child in sorted(node.children, key=lambda c: c.box[1]):
        if node.id == "ROOT":
            child_x, child_y, _, child_h = child.box
        else:
            parent_x, parent_y, _, _ = node.box
            child_x = max(0, child.box[0] - parent_x + 10)
            child_y = max(0, child.box[1] - parent_y + 10)
            child_h = child.box[3]
        x1 = max(0, child_x)
        y1 = max(0, child_y)
        x2 = crop_img.shape[1]
        y2 = min(crop_img.shape[0], child_y + child_h)
        cv2.rectangle(crop_img, (x1, y1), (x2, y2), (255, 255, 255), -1)
    return crop_img


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
    pad_y = 12
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
    if x > max(24, int(img_w * 0.18)) or h < max(40, int(img_h * 0.45)):
        return cleaned

    erase_right = min(img_w, x + w + 6)
    cv2.rectangle(cleaned, (0, 0), (erase_right, img_h), (255, 255, 255), -1)
    return cleaned


def _candidate_band_indices(
    *,
    line_order: int | None,
    band_count: int,
    bands: List[Tuple[int, int]] | None = None,
    projected_y: float | None = None,
) -> List[int]:
    if band_count <= 0:
        return []

    if line_order is None and projected_y is not None and bands:
        ranked = sorted(
            range(band_count),
            key=lambda idx: (
                0 if bands[idx][0] <= projected_y <= bands[idx][1] else 1,
                abs(((bands[idx][0] + bands[idx][1]) / 2.0) - projected_y),
                abs(idx - line_order) if isinstance(line_order, int) else 0,
                idx,
            ),
        )
        return ranked[: min(5, band_count)]

    if line_order is None:
        return list(range(band_count))

    indices: List[int] = []
    for delta in range(0, min(3, band_count)):
        for candidate in (line_order - delta, line_order + delta):
            if candidate < 0 or candidate >= band_count or candidate in indices:
                continue
            indices.append(candidate)
    if not indices:
        fallback = min(max(0, line_order), band_count - 1)
        indices.append(fallback)
    return indices


def _confidence_score(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value or "").strip().lower()
    if text == "high":
        return 0.95
    if text == "medium":
        return 0.7
    if text == "low":
        return 0.35
    if text == "uncertain":
        return 0.0
    try:
        return float(text)
    except ValueError:
        return 0.0


def _unary_prefix_signature(formula_text: str) -> Tuple[str, int, int]:
    prefix = _leading_unary_prefix(formula_text)
    if "□" in prefix:
        modal_symbol = "□"
    elif "⋄" in prefix:
        modal_symbol = "⋄"
    else:
        modal_symbol = ""

    if modal_symbol:
        before_modal = prefix.split(modal_symbol, 1)[0].count("¬")
        before_atom = prefix.split(modal_symbol, 1)[1].count("¬")
    else:
        before_modal = 0
        before_atom = prefix.count("¬")
    return modal_symbol, before_modal, before_atom


def _unary_prefix_distance(original: str, candidate: str | None) -> int:
    if not candidate:
        return 99
    original_modal, original_before_modal, original_before_atom = _unary_prefix_signature(original)
    candidate_modal, candidate_before_modal, candidate_before_atom = _unary_prefix_signature(candidate)
    if original_modal != candidate_modal:
        return 10
    return abs(original_before_modal - candidate_before_modal) + abs(original_before_atom - candidate_before_atom)


def _best_verifier_attempt(
    *,
    original_formula: str,
    expected_line_order: int | None,
    attempts: List[Dict[str, Any]],
) -> Dict[str, Any]:
    if not attempts:
        return {}

    def score(attempt: Dict[str, Any]) -> Tuple[int, int, int, int, int, float]:
        forced_choice = str(attempt.get("forced_choice") or "").strip()
        structured_formula = str(attempt.get("structured_formula") or "").strip()
        candidate_formula = forced_choice if forced_choice and forced_choice != "uncertain" else structured_formula
        prefix_distance = _unary_prefix_distance(original_formula, candidate_formula)
        exact_original_agreement = forced_choice == original_formula and structured_formula == original_formula
        attempt_kind = str(attempt.get("attempt_kind") or "local_band").strip()
        margin_anchor_with_correction = attempt_kind == "margin_anchor" and bool(attempt.get("proposed_correction"))
        margin_anchor_tail_exact_match = (
            attempt_kind == "margin_anchor"
            and bool(attempt.get("tail_regularized"))
            and forced_choice == original_formula
            and prefix_distance <= 1
        )
        band_index = attempt.get("band_index")
        if isinstance(expected_line_order, int) and isinstance(band_index, int):
            band_distance = abs(band_index - expected_line_order)
            exact_band_match = band_distance == 0
            nearby_band_match = band_distance <= 1
            distance_score = -band_distance
        else:
            exact_band_match = False
            nearby_band_match = False
            distance_score = 0
        return (
            1 if margin_anchor_with_correction else 0,
            1 if margin_anchor_tail_exact_match else 0,
            1 if exact_band_match else 0,
            1 if nearby_band_match else 0,
            1 if exact_original_agreement else 0,
            1 if structured_formula == original_formula and prefix_distance == 0 else 0,
            1 if attempt.get("proposed_correction") and prefix_distance <= 1 else 0,
            1 if attempt.get("proposed_correction") else 0,
            1 if forced_choice == original_formula else 0,
            1 if structured_formula == original_formula else 0,
            1 if _is_structurally_close_unary_formula(original_formula, candidate_formula) else 0,
            -prefix_distance,
            distance_score,
            max(
                _confidence_score(attempt.get("structured_counts", {}).get("confidence")),
                _confidence_score(attempt.get("forced_confidence")),
            ),
        )

    return max(attempts, key=score)


def _build_full_image_line_anchor_prompt(total_lines: int) -> str:
    return "\n".join(
        [
            "This image shows a whole handwritten modal logic proof.",
            'Return only JSON with keys: line_centers_rel, confidence.',
            "line_centers_rel must be a list of approximate vertical centers for the proof lines in reading order from top to bottom.",
            "Use a 0-1000 scale, where 0 is the top of the image and 1000 is the bottom.",
            f"Return exactly {total_lines} numbers, one for each visible proof line in order.",
            "Do not return bounding boxes. Do not explain.",
        ]
    )


def _parse_full_image_line_anchors(raw: str, total_lines: int) -> Dict[int, float]:
    parsed = _parse_unary_recheck_choice(raw)
    values = parsed.get("line_centers_rel")
    if not isinstance(values, list):
        return {}

    anchors: Dict[int, float] = {}
    for index, value in enumerate(values[:total_lines], start=1):
        try:
            y_rel = float(value)
        except (TypeError, ValueError):
            continue
        anchors[index] = max(0.0, min(1000.0, y_rel))
    return anchors


def _request_full_image_line_anchors(
    *,
    full_img: np.ndarray,
    verifier_client: LMStudioQwenClient,
    total_lines: int,
) -> Dict[int, float]:
    if total_lines <= 0:
        return {}
    ok, encoded = cv2.imencode(".jpg", full_img)
    if not ok:
        return {}
    image_base64 = base64.b64encode(encoded.tobytes()).decode("utf-8")
    raw = verifier_client.chat(
        [
            {"role": "system", "content": "You are a careful vision model. Follow the user's instructions exactly."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": _build_full_image_line_anchor_prompt(total_lines)},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}} ,
                ],
            },
        ]
    )
    return _parse_full_image_line_anchors(raw, total_lines)


def _node_crop_origin(node: SubproofNode) -> Tuple[int, int]:
    if node.id == "ROOT":
        return (0, 0)
    x, y, _, _ = node.box
    return (max(0, x - 10), max(0, y - 10))


def _extract_left_margin_strip(full_img: np.ndarray, right_fraction: float = 1.0) -> Tuple[np.ndarray, int]:
    if full_img.size == 0:
        return full_img, 0
    bars = detect_vertical_bars(full_img)
    first_bar_x = bars[0][0] if bars else max(1, full_img.shape[1] // 8)
    scaled_right = int(first_bar_x * max(0.2, min(1.0, right_fraction)))
    if first_bar_x >= 30:
        scaled_right = max(24, scaled_right)
    margin_right = max(1, min(full_img.shape[1], scaled_right))
    return full_img[:, :margin_right].copy(), margin_right


def _candidate_box_iou(left: Tuple[int, int, int, int], right: Tuple[int, int, int, int]) -> float:
    lx, ly, lw, lh = left
    rx, ry, rw, rh = right
    inter_x1 = max(lx, rx)
    inter_y1 = max(ly, ry)
    inter_x2 = min(lx + lw, rx + rw)
    inter_y2 = min(ly + lh, ry + rh)
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0
    intersection = float((inter_x2 - inter_x1) * (inter_y2 - inter_y1))
    union = float(lw * lh + rw * rh) - intersection
    if union <= 0.0:
        return 0.0
    return intersection / union


def _merge_margin_candidate_boxes(boxes: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int]]:
    if not boxes:
        return []

    merged = sorted(boxes, key=lambda box: (box[1], box[0]))
    changed = True
    while changed:
        changed = False
        next_boxes: List[Tuple[int, int, int, int]] = []
        while merged:
            curr_x, curr_y, curr_w, curr_h = merged.pop(0)
            curr_x2 = curr_x + curr_w
            curr_y2 = curr_y + curr_h
            keep_searching = True
            while keep_searching:
                keep_searching = False
                for index, (nx, ny, nw, nh) in enumerate(merged):
                    nx2 = nx + nw
                    ny2 = ny + nh
                    vertical_overlap = max(0, min(curr_y2, ny2) - max(curr_y, ny))
                    min_height = max(1, min(curr_h, nh))
                    horizontal_overlap = max(0, min(curr_x2, nx2) - max(curr_x, nx))
                    min_width = max(1, min(curr_w, nw))
                    horizontal_gap = max(0, max(curr_x, nx) - min(curr_x2, nx2))
                    vertical_gap = max(0, max(curr_y, ny) - min(curr_y2, ny2))
                    same_row_merge = vertical_overlap >= int(0.5 * min_height) and horizontal_gap <= 24
                    stacked_fragment_merge = horizontal_overlap >= int(0.45 * min_width) and vertical_gap <= 18
                    if same_row_merge or stacked_fragment_merge:
                        curr_x = min(curr_x, nx)
                        curr_y = min(curr_y, ny)
                        curr_x2 = max(curr_x2, nx2)
                        curr_y2 = max(curr_y2, ny2)
                        curr_w = curr_x2 - curr_x
                        curr_h = curr_y2 - curr_y
                        merged.pop(index)
                        changed = True
                        keep_searching = True
                        break
            next_boxes.append((curr_x, curr_y, curr_w, curr_h))
        merged = sorted(next_boxes, key=lambda box: (box[1], box[0]))

    pruned: List[Tuple[int, int, int, int]] = []
    for box in sorted(merged, key=lambda item: item[2] * item[3], reverse=True):
        if any(_candidate_box_iou(box, existing) >= 0.6 for existing in pruned):
            continue
        pruned.append(box)
    return sorted(pruned, key=lambda box: (box[1], box[0]))


def _detect_margin_number_candidate_boxes(
    margin_upscaled_img: np.ndarray,
) -> Tuple[List[Tuple[int, int, int, int]], np.ndarray]:
    if margin_upscaled_img.size == 0:
        return [], np.zeros((0, 0), dtype=np.uint8)

    gray = cv2.cvtColor(margin_upscaled_img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 3)
    binary = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        11,
    )

    horizontal_len = max(12, gray.shape[1] // 2)
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_len, 1))
    horizontal_mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
    cleaned = cv2.subtract(binary, horizontal_mask)
    cleaned = cv2.morphologyEx(
        cleaned,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
    )

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
    boxes: List[Tuple[int, int, int, int]] = []
    max_h = gray.shape[0]
    max_w = gray.shape[1]
    for label in range(1, num_labels):
        x, y, w, h, area = stats[label]
        if area < 90:
            continue
        if h < 24 or h > int(max_h * 0.22):
            continue
        if w < 6 or w > max_w:
            continue
        if x > int(max_w * 0.85):
            continue
        boxes.append((int(x), int(y), int(w), int(h)))

    return _merge_margin_candidate_boxes(boxes), cleaned


def _score_margin_candidate_boxes(
    boxes: List[Tuple[int, int, int, int]],
    *,
    expected_count: int,
    strip_width: int,
) -> Tuple[int, int, int, int, int, int]:
    if not boxes:
        return (0, -expected_count, -9999, -9999, -9999, -9999)

    heights = [h for _, _, _, h in boxes]
    median_height = float(np.median(np.array(heights, dtype=np.float32))) if heights else 0.0
    tall_skinny_noise = sum(
        1
        for x, _, w, h in boxes
        if h > max(72.0, median_height * 2.4) and w < max(18.0, median_height * 0.55) and x > int(strip_width * 0.7)
    )
    far_right_noise = sum(
        1
        for x, _, w, _ in boxes
        if x > int(strip_width * 0.7) and w < int(strip_width * 0.28)
    )
    count_distance = abs(len(boxes) - expected_count)
    total_height_penalty = int(sum(max(0.0, h - max(72.0, median_height * 2.1)) for h in heights))
    return (
        1 if len(boxes) == expected_count else 0,
        -count_distance,
        -tall_skinny_noise,
        -far_right_noise,
        -total_height_penalty,
        -strip_width,
    )


def _margin_anchor_strip_bounds(
    anchor_centers: List[float],
    image_height: int,
    anchor_heights: List[float] | None = None,
) -> List[Tuple[int, int]]:
    if not anchor_centers or image_height <= 0:
        return []
    height_values = anchor_heights if anchor_heights and len(anchor_heights) == len(anchor_centers) else None
    if len(anchor_centers) == 1:
        center = anchor_centers[0]
        half_span = 24.0
        if height_values:
            half_span = max(half_span, float(height_values[0]) * 1.8)
        return [(max(0, int(round(center - half_span))), min(image_height, int(round(center + half_span))))]

    boundaries: List[float] = []
    first_gap = max(24.0, anchor_centers[1] - anchor_centers[0])
    boundaries.append(max(0.0, anchor_centers[0] - (first_gap / 2.0)))
    for left, right in zip(anchor_centers, anchor_centers[1:]):
        boundaries.append((left + right) / 2.0)
    last_gap = max(24.0, anchor_centers[-1] - anchor_centers[-2])
    boundaries.append(min(float(image_height), anchor_centers[-1] + (last_gap / 2.0)))

    bounds: List[Tuple[int, int]] = []
    for index in range(len(anchor_centers)):
        y1 = max(0, min(image_height - 1, int(round(boundaries[index]))))
        y2 = max(y1 + 1, min(image_height, int(round(boundaries[index + 1]))))
        bounds.append((y1, y2))

    gap_values = [right - left for left, right in zip(anchor_centers, anchor_centers[1:])]
    typical_gap = float(np.median(np.array(gap_values, dtype=np.float32))) if gap_values else 40.0
    typical_half_span = max(18.0, typical_gap * 0.55)
    if height_values:
        median_height = float(np.median(np.array(height_values, dtype=np.float32)))
        typical_half_span = max(typical_half_span, median_height * 1.5)
        tail_half_span = max(typical_half_span, float(height_values[-1]) * 1.8)
    else:
        tail_half_span = typical_half_span

    last_gap = max(24.0, anchor_centers[-1] - anchor_centers[-2])
    if last_gap > max(32.0, typical_gap * 1.25):
        tail_half_span = min(last_gap / 2.0, tail_half_span)
        tail_center = anchor_centers[-1]
        tail_y1 = max(0, min(image_height - 1, int(round(tail_center - tail_half_span))))
        tail_y2 = max(tail_y1 + 1, min(image_height, int(round(tail_center + tail_half_span))))
        bounds[-1] = (tail_y1, tail_y2)
    return bounds


def _build_margin_anchor_geometry(
    *,
    full_img: np.ndarray,
    total_lines: int,
    output_dir: Path,
) -> Dict[str, Any]:
    geometry: Dict[str, Any] = {
        "usable": False,
        "expected_lines": total_lines,
        "candidate_count": 0,
        "anchors": [],
        "strips": [],
    }
    if full_img.size == 0 or total_lines <= 0:
        return geometry

    margin_base_path = output_dir / "margin_anchor_margin_base.jpg"
    margin_upscaled_path = output_dir / "margin_anchor_margin_upscaled_4x.jpg"
    margin_binary_path = output_dir / "margin_anchor_binary.png"
    candidate_boxes_path = output_dir / "margin_anchor_candidate_boxes.jpg"
    strip_overlay_path = output_dir / "margin_anchor_strip_overlay.jpg"
    geometry_path = output_dir / "margin_anchor_geometry.json"

    candidate_variants: List[Tuple[Tuple[int, int, int, int, int, int], float, np.ndarray, int, np.ndarray, List[Tuple[int, int, int, int]], np.ndarray]] = []
    for right_fraction in (0.6, 0.65, 0.75, 0.85, 1.0):
        margin_img, margin_right = _extract_left_margin_strip(full_img, right_fraction=right_fraction)
        if margin_img.size == 0:
            continue
        margin_upscaled = _upscale_for_verifier(margin_img)
        candidate_boxes_upscaled, binary = _detect_margin_number_candidate_boxes(margin_upscaled)
        score = _score_margin_candidate_boxes(
            candidate_boxes_upscaled,
            expected_count=total_lines,
            strip_width=margin_upscaled.shape[1],
        )
        candidate_variants.append((score, right_fraction, margin_img, margin_right, margin_upscaled, candidate_boxes_upscaled, binary))

    if not candidate_variants:
        return geometry

    _, selected_fraction, margin_img, margin_right, margin_upscaled, candidate_boxes_upscaled, binary = max(
        candidate_variants,
        key=lambda item: item[0],
    )
    geometry["margin_right"] = margin_right
    geometry["selected_right_fraction"] = selected_fraction
    cv2.imwrite(str(margin_base_path), margin_img)
    cv2.imwrite(str(margin_upscaled_path), margin_upscaled)
    cv2.imwrite(str(margin_binary_path), binary)

    candidate_overlay = margin_upscaled.copy()
    scale = 4.0
    anchors: List[Dict[str, Any]] = []
    for index, (x, y, w, h) in enumerate(sorted(candidate_boxes_upscaled, key=lambda box: (box[1], box[0])), start=1):
        cv2.rectangle(candidate_overlay, (x, y), (x + w, y + h), (40, 160, 60), 2)
        cv2.putText(
            candidate_overlay,
            str(index),
            (x, max(18, y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 220),
            2,
            cv2.LINE_AA,
        )
        anchors.append(
            {
                "assigned_line_number": index,
                "bbox_upscaled": [int(x), int(y), int(w), int(h)],
                "bbox_original": [round(x / scale, 1), round(y / scale, 1), round(w / scale, 1), round(h / scale, 1)],
                "center_y_original": round((y + (h / 2.0)) / scale, 1),
                "height_original": round(h / scale, 1),
            }
        )
    cv2.imwrite(str(candidate_boxes_path), candidate_overlay)

    geometry["candidate_count"] = len(anchors)
    geometry["anchors"] = anchors

    if len(anchors) == total_lines:
        anchor_centers = [float(anchor["center_y_original"]) for anchor in anchors]
        anchor_heights = [float(anchor["height_original"]) for anchor in anchors]
        strip_bounds = _margin_anchor_strip_bounds(anchor_centers, full_img.shape[0], anchor_heights=anchor_heights)
        gap_values = [right - left for left, right in zip(anchor_centers, anchor_centers[1:])]
        typical_gap = float(np.median(np.array(gap_values, dtype=np.float32))) if gap_values else 0.0
        tail_regularized = bool(
            gap_values
            and typical_gap > 0.0
            and gap_values[-1] > max(32.0, typical_gap * 1.25)
        )
        strips: List[Dict[str, Any]] = []
        overlay = full_img.copy()
        for index, (anchor, (y1, y2)) in enumerate(zip(anchors, strip_bounds)):
            record = {
                "line_number": anchor["assigned_line_number"],
                "center_y_original": anchor["center_y_original"],
                "y1": int(y1),
                "y2": int(y2),
                "bbox_original": anchor["bbox_original"],
                "tail_regularized": bool(index == len(anchors) - 1 and tail_regularized),
            }
            strips.append(record)
            cv2.rectangle(overlay, (0, y1), (max(1, full_img.shape[1] - 1), max(y1 + 1, y2 - 1)), (0, 165, 255), 2)
            cv2.putText(
                overlay,
                f"L{anchor['assigned_line_number']}",
                (10, max(20, y1 + 18)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (220, 40, 40),
                2,
                cv2.LINE_AA,
            )
        cv2.imwrite(str(strip_overlay_path), overlay)
        geometry["usable"] = True
        geometry["strips"] = strips

    geometry["artifacts"] = {
        "margin_base": str(margin_base_path),
        "margin_upscaled": str(margin_upscaled_path),
        "binary": str(margin_binary_path),
        "candidate_boxes": str(candidate_boxes_path),
        "strip_overlay": str(strip_overlay_path),
    }
    geometry_path.write_text(json.dumps(geometry, indent=2, ensure_ascii=False), encoding="utf-8")
    return geometry


def _extract_margin_anchor_strip(full_img: np.ndarray, anchor_record: Dict[str, Any]) -> np.ndarray:
    if full_img.size == 0:
        return full_img
    try:
        y1 = int(anchor_record.get("y1"))
        y2 = int(anchor_record.get("y2"))
    except (TypeError, ValueError):
        return full_img
    y1 = max(0, min(full_img.shape[0] - 1, y1))
    y2 = max(y1 + 1, min(full_img.shape[0], y2))
    strip = full_img[y1:y2, :]
    return strip.copy() if strip.size else full_img


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


def _compose_unary_formula(*, modal: str, before_modal: int, before_atom: int, body: str) -> str:
    prefix = ("¬" * max(0, before_modal)) + modal + ("¬" * max(0, before_atom))
    return f"{prefix}{body}".strip()


def _build_structured_unary_verifier_prompt(entry: Dict[str, Any]) -> str:
    scope_role = str(entry.get("scope_role") or "derived_line").strip()
    formula_text = _normalize_formula_text(entry.get("formula_text") or entry.get("wffstr") or "", scope_role)
    justification = _normalize_justification_text(entry.get("justification_text") or entry.get("jstr") or "")
    lines = [
        "This image shows one handwritten modal logic proof formula on a single line.",
        'Return only JSON with keys: main_modal_operator, negations_before_modal, negations_before_final_atom, confidence.',
        'Set main_modal_operator to "box", "diamond", or "none".',
        "Count exactly how many negation marks (¬) appear immediately before the main modal operator, if any.",
        "Count exactly how many negation marks (¬) appear immediately before the final propositional letter.",
        "Do not guess from proof validity. Report only what the handwriting shows.",
        f"Current OCR formula: {formula_text}",
    ]
    if justification:
        lines.append(f"Current OCR justification: {justification}")
    return "\n".join(lines)


def _derive_formula_from_structured_counts(entry: Dict[str, Any], parsed: Dict[str, Any]) -> str | None:
    scope_role = str(entry.get("scope_role") or "derived_line").strip()
    original_formula = _normalize_formula_text(entry.get("formula_text") or entry.get("wffstr") or "", scope_role)
    prefix, body = _formula_components(original_formula)
    if not body:
        return None

    modal_value = str(parsed.get("main_modal_operator") or "").strip().lower()
    if modal_value == "box":
        modal_symbol = "□"
    elif modal_value == "diamond":
        modal_symbol = "⋄"
    else:
        modal_symbol = ""

    before_modal = _int_field(parsed.get("negations_before_modal"), default=0)
    before_atom = _int_field(parsed.get("negations_before_final_atom"), default=0)

    if not modal_symbol:
        return f"{'¬' * before_atom}{body}"
    return _compose_unary_formula(
        modal=modal_symbol,
        before_modal=before_modal,
        before_atom=before_atom,
        body=body,
    )


def _nearby_verifier_candidates(original_formula: str, proposed_formula: str | None) -> List[str]:
    prefix, body = _formula_components(original_formula)
    if not prefix or not body:
        return [original_formula]

    original_modal = "□" if "□" in prefix else "⋄" if "⋄" in prefix else ""
    original_before_modal = prefix.split(original_modal, 1)[0].count("¬") if original_modal else 0
    original_before_atom = prefix.split(original_modal, 1)[1].count("¬") if original_modal else prefix.count("¬")

    candidates: List[str] = [original_formula]
    if proposed_formula:
        candidates.append(proposed_formula)

    count_options = {
        max(0, original_before_atom - 1),
        original_before_atom,
        min(3, original_before_atom + 1),
    }
    modal_count_options = {
        max(0, original_before_modal - 1),
        original_before_modal,
        min(3, original_before_modal + 1),
    }
    for before_modal in sorted(modal_count_options):
        for before_atom in sorted(count_options):
            if original_modal:
                candidate = _compose_unary_formula(
                    modal=original_modal,
                    before_modal=before_modal,
                    before_atom=before_atom,
                    body=body,
                )
            else:
                candidate = f"{'¬' * before_atom}{body}"
            if candidate not in candidates:
                candidates.append(candidate)
            if len(candidates) >= 4:
                return candidates
    return candidates[:4]


def _build_forced_choice_unary_verifier_prompt(entry: Dict[str, Any], candidates: List[str]) -> str:
    scope_role = str(entry.get("scope_role") or "derived_line").strip()
    formula_text = _normalize_formula_text(entry.get("formula_text") or entry.get("wffstr") or "", scope_role)
    justification = _normalize_justification_text(entry.get("justification_text") or entry.get("jstr") or "")
    candidate_text = " ; ".join(candidates)
    has_box = any("□" in candidate for candidate in candidates)
    has_diamond = any("⋄" in candidate for candidate in candidates)
    if has_box and not has_diamond:
        lines = [
            "This image shows one handwritten modal logic formula on a single line.",
            'Return only JSON with keys: selected_formula, confidence, negations_before_modal, negations_before_final_atom.',
            "Count exactly how many negation marks (¬) appear immediately to the LEFT of the box symbol (□).",
            "Count exactly how many negation marks (¬) appear immediately to the RIGHT of the box symbol before the final propositional letter.",
            "Then choose the exact candidate that best matches the handwriting.",
            "Do not invent a formula outside the candidate list.",
            f"Current OCR formula: {formula_text}",
        ]
    elif has_diamond and not has_box:
        lines = [
            "This image shows one handwritten modal logic formula on a single line.",
            'Return only JSON with keys: selected_formula, confidence, negations_before_modal, negations_before_final_atom.',
            "Count exactly how many negation marks (¬) appear immediately to the LEFT of the diamond symbol (⋄).",
            "Count exactly how many negation marks (¬) appear immediately to the RIGHT of the diamond symbol before the final propositional letter.",
            "Then choose the exact candidate that best matches the handwriting.",
            "Do not invent a formula outside the candidate list.",
            f"Current OCR formula: {formula_text}",
        ]
    else:
        lines = [
            "This image shows one handwritten modal logic formula on a single line.",
            'Return only JSON with keys: selected_formula, confidence, negations_before_final_atom.',
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


def _normalize_selected_formula(value: Any, scope_role: str) -> str:
    selected = str(value or "").strip()
    if not selected or selected == "uncertain":
        return "uncertain"
    return _normalize_formula_text(selected, scope_role)


def _same_modal_class(left: str, right: str) -> bool:
    left_prefix = _leading_unary_prefix(left)
    right_prefix = _leading_unary_prefix(right)
    if "□" in left_prefix:
        return "□" in right_prefix
    if "⋄" in left_prefix:
        return "⋄" in right_prefix
    return "□" not in right_prefix and "⋄" not in right_prefix


def _is_structurally_close_unary_formula(original: str, candidate: str | None) -> bool:
    if not candidate:
        return False
    return _unary_formula_skeleton(original) == _unary_formula_skeleton(candidate) and _same_modal_class(original, candidate)


def _run_unary_verifier_attempt(
    *,
    entry: Dict[str, Any],
    original_formula: str,
    verifier_client: LMStudioQwenClient,
    crop_img: np.ndarray,
    verifier_path: Path,
    attempt_metadata: Dict[str, Any],
) -> Dict[str, Any] | None:
    if crop_img.size == 0:
        return None

    line_crop = _upscale_for_verifier(crop_img)
    cv2.imwrite(str(verifier_path), line_crop)

    with verifier_path.open("rb") as handle:
        line_base64_image = base64.b64encode(handle.read()).decode("utf-8")

    structured_raw = verifier_client.chat(
        [
            {"role": "system", "content": "You are a careful vision model. Follow the user's instructions exactly."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": _build_structured_unary_verifier_prompt(entry)},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{line_base64_image}"}} ,
                ],
            },
        ]
    )
    structured_parsed = _parse_unary_recheck_choice(structured_raw)
    structured_formula = _derive_formula_from_structured_counts(entry, structured_parsed)

    scope_role = str(entry.get("scope_role") or "derived_line").strip()
    forced_candidates = _nearby_verifier_candidates(original_formula, structured_formula)
    forced_raw = verifier_client.chat(
        [
            {"role": "system", "content": "You are a careful vision model. Follow the user's instructions exactly."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": _build_forced_choice_unary_verifier_prompt(entry, forced_candidates)},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{line_base64_image}"}} ,
                ],
            },
        ]
    )
    forced_parsed = _parse_unary_recheck_choice(forced_raw)
    forced_choice = _normalize_selected_formula(forced_parsed.get("selected_formula"), scope_role)

    proposed_correction = None
    if (
        structured_formula
        and forced_choice == structured_formula
        and structured_formula != original_formula
        and _is_structurally_close_unary_formula(original_formula, structured_formula)
    ):
        proposed_correction = structured_formula

    return {
        **attempt_metadata,
        "line_crop_path": str(verifier_path),
        "structured_counts": structured_parsed,
        "structured_formula": structured_formula,
        "forced_choice_candidates": forced_candidates,
        "forced_choice": forced_choice,
        "forced_confidence": forced_parsed.get("confidence"),
        "proposed_correction": proposed_correction,
        "structured_raw_response": structured_raw,
        "forced_raw_response": forced_raw,
    }


def _run_unary_verifier_experiment(
    *,
    proof_structure: List[Any],
    tree_root: SubproofNode,
    full_img: np.ndarray,
    output_dir: Path,
    verifier_client: LMStudioQwenClient | None,
    checker_response: Dict[str, Any] | None,
) -> List[Dict[str, Any]]:
    for path in output_dir.glob("verifier_line_*.jpg"):
        path.unlink(missing_ok=True)
    for path in output_dir.glob("margin_anchor_*"):
        path.unlink(missing_ok=True)
    (output_dir / UNARY_VERIFIER_FILENAME).unlink(missing_ok=True)

    if verifier_client is None or not _checker_payload_has_issues(checker_response):
        return []

    node_lookup = _build_node_lookup(tree_root)
    all_lines = _iter_proof_line_dicts(proof_structure)
    max_line_number = max(
        (int(line.get("line")) for line in all_lines if isinstance(line.get("line"), int)),
        default=0,
    )
    margin_anchor_geometry = _build_margin_anchor_geometry(
        full_img=full_img,
        total_lines=max_line_number,
        output_dir=output_dir,
    )
    margin_anchor_lookup = {
        int(record["line_number"]): record
        for record in margin_anchor_geometry.get("strips", [])
        if isinstance(record, dict) and isinstance(record.get("line_number"), int)
    }
    full_image_line_anchors = (
        {}
        if margin_anchor_geometry.get("usable")
        else _request_full_image_line_anchors(
            full_img=full_img,
            verifier_client=verifier_client,
            total_lines=max_line_number,
        )
    )
    records: List[Dict[str, Any]] = []
    for entry in all_lines:
        if not _should_verify_unary_formula(entry):
            continue

        node_id = str(entry.get("_ocr_node_id") or "").strip()
        scope_role = str(entry.get("scope_role") or "derived_line").strip()
        original_formula = _normalize_formula_text(entry.get("formula_text") or entry.get("wffstr") or "", scope_role)
        node = node_lookup.get(node_id)
        line_order = entry.get("_ocr_line_order")
        line_number = entry.get("line")
        attempt_records: List[Dict[str, Any]] = []
        if isinstance(line_number, int) and margin_anchor_geometry.get("usable") and line_number in margin_anchor_lookup:
            anchor_record = margin_anchor_lookup[line_number]
            anchor_crop = _extract_margin_anchor_strip(full_img, anchor_record)
            anchor_attempt = _run_unary_verifier_attempt(
                entry=entry,
                original_formula=original_formula,
                verifier_client=verifier_client,
                crop_img=anchor_crop,
                verifier_path=output_dir / f"verifier_line_anchor_{line_number}.jpg",
                attempt_metadata={
                    "attempt_kind": "margin_anchor",
                    "band_index": None,
                    "band_distance": None,
                    "anchor_rel": None,
                    "projected_crop_y": None,
                    "band_mid_y": float(anchor_record.get("center_y_original") or 0.0),
                    "anchor_center_y": anchor_record.get("center_y_original"),
                    "strip_y1": anchor_record.get("y1"),
                    "strip_y2": anchor_record.get("y2"),
                    "tail_regularized": bool(anchor_record.get("tail_regularized")),
                },
            )
            if anchor_attempt is not None:
                attempt_records.append(anchor_attempt)

        if node is not None and isinstance(line_order, int):
            node_crop = _verifier_node_crop_image(node=node, full_img=full_img)
            bands = _detect_text_row_bands(node_crop)
            _, node_origin_y = _node_crop_origin(node)
            anchor_rel = full_image_line_anchors.get(int(line_number)) if isinstance(line_number, int) else None
            projected_crop_y = None
            if anchor_rel is not None and full_img.shape[0] > 0:
                global_y = (anchor_rel / 1000.0) * float(full_img.shape[0])
                projected_crop_y = global_y - float(node_origin_y)
            candidate_indices = _candidate_band_indices(
                line_order=line_order,
                band_count=len(bands),
                bands=bands,
                projected_y=projected_crop_y,
            )
            for candidate_index in candidate_indices:
                line_crop = _extract_band_crop(
                    node_crop_img=node_crop,
                    bands=bands,
                    band_index=candidate_index,
                )
                suffix = f"{line_order + 1}"
                if candidate_index != line_order:
                    suffix = f"{suffix}_band{candidate_index + 1}"
                verifier_path = output_dir / f"verifier_line_{node_id}_{suffix}.jpg"
                attempt = _run_unary_verifier_attempt(
                    entry=entry,
                    original_formula=original_formula,
                    verifier_client=verifier_client,
                    crop_img=line_crop,
                    verifier_path=verifier_path,
                    attempt_metadata={
                        "attempt_kind": "local_band",
                        "band_index": candidate_index,
                        "band_distance": abs(candidate_index - line_order),
                        "anchor_rel": anchor_rel,
                        "projected_crop_y": projected_crop_y,
                        "band_mid_y": (bands[candidate_index][0] + bands[candidate_index][1]) / 2.0,
                        "anchor_center_y": None,
                        "strip_y1": None,
                        "strip_y2": None,
                    },
                )
                if attempt is not None:
                    attempt_records.append(attempt)

        if not attempt_records:
            records.append(
                {
                    "node_id": node_id,
                    "line_label": str(entry.get("line_label") or "").strip(),
                    "line_number": line_number,
                    "node_line_order": line_order if isinstance(line_order, int) else None,
                    "baseline_formula": original_formula,
                    "structured_error": "Could not isolate a single-line verifier crop.",
                    "would_auto_apply": False,
                }
            )
            continue

        best_attempt = _best_verifier_attempt(
            original_formula=original_formula,
            expected_line_order=line_order if isinstance(line_order, int) else None,
            attempts=attempt_records,
        )

        records.append(
            {
                "node_id": node_id,
                "line_label": str(entry.get("line_label") or "").strip(),
                "line_number": line_number,
                "node_line_order": line_order if isinstance(line_order, int) else None,
                "baseline_formula": original_formula,
                "band_count": len(_detect_text_row_bands(_verifier_node_crop_image(node=node, full_img=full_img))) if node is not None else 0,
                "anchor_rel": best_attempt.get("anchor_rel"),
                "projected_crop_y": best_attempt.get("projected_crop_y"),
                "structured_counts": best_attempt.get("structured_counts"),
                "structured_formula": best_attempt.get("structured_formula"),
                "forced_choice_candidates": best_attempt.get("forced_choice_candidates"),
                "forced_choice": best_attempt.get("forced_choice"),
                "proposed_correction": best_attempt.get("proposed_correction"),
                "would_auto_apply": bool(best_attempt.get("proposed_correction")),
                "attempt_kind": best_attempt.get("attempt_kind"),
                "anchor_center_y": best_attempt.get("anchor_center_y"),
                "strip_y1": best_attempt.get("strip_y1"),
                "strip_y2": best_attempt.get("strip_y2"),
                "line_crop_path": best_attempt.get("line_crop_path"),
                "structured_raw_response": best_attempt.get("structured_raw_response"),
                "forced_raw_response": best_attempt.get("forced_raw_response"),
                "candidate_attempts": attempt_records,
            }
        )

    if records:
        (output_dir / UNARY_VERIFIER_FILENAME).write_text(
            json.dumps(records, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    return records


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
    prompt: str,
    client: LMStudioQwenClient | None,
    opener_client: LMStudioQwenClient | None = None,
) -> None:
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
            prompt=prompt,
            client=client,
            opener_client=opener_client,
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

    opener_result = None
    if node.id != "ROOT":
        opener_result = _classify_subproof_opener(
            node_crop_img=current_img,
            output_dir=output_dir,
            node_id=node.id,
            client=opener_client or client,
        )

    raw_response, parsed_entries = call_vlm(save_path, prompt, client)
    parsed_entries = _prepend_synthetic_strict_open_if_needed(parsed_entries, opener_result)
    (output_dir / RAW_RESPONSE_TEMPLATE.format(node_id=node.id)).write_text(raw_response, encoding="utf-8")
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
    updated = _clone_jsonish(proof_structure)
    lines_by_number = {
        int(line.get("line")): line
        for line in _iter_proof_line_dicts(updated)
        if isinstance(line, dict) and isinstance(line.get("line"), int)
    }

    rewrites: List[Dict[str, Any]] = []
    for record in verifier_records:
        if not isinstance(record, dict):
            continue
        if not record.get("would_auto_apply"):
            continue
        line_number = record.get("line_number")
        proposed = str(record.get("proposed_correction") or "").strip()
        if not isinstance(line_number, int) or not proposed:
            continue
        target_line = lines_by_number.get(line_number)
        if not isinstance(target_line, dict):
            continue
        original_formula = str(target_line.get("wffstr") or "").strip()
        if not original_formula or original_formula == proposed:
            continue
        target_line["wffstr"] = proposed
        rewrites.append(
            {
                "line_number": line_number,
                "line_label": str(target_line.get("line_label") or "").strip(),
                "node_id": str(record.get("node_id") or "").strip(),
                "citation": str(target_line.get("jstr") or "").strip(),
                "original_formula": original_formula,
                "rewritten_formula": proposed,
            }
        )

    rewrites.sort(key=lambda item: (int(item.get("line_number") or 0), str(item.get("line_label") or "")))
    return updated, rewrites


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
    vision_base_url: str | None = None,
    vision_model: str | None = None,
    verifier_vision_model: str | None = None,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / UNARY_VERIFIER_FILENAME).unlink(missing_ok=True)
    for path in output_dir.glob("verifier_line_*.jpg"):
        path.unlink(missing_ok=True)
    prompt = build_prompt(modal_config)
    (output_dir / PROMPT_FILENAME).write_text(prompt, encoding="utf-8")

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
        prompt=prompt,
        client=client,
        opener_client=verifier_client or client,
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

    verifier_records = _run_unary_verifier_experiment(
        proof_structure=final_structure,
        tree_root=tree_root,
        full_img=img,
        output_dir=output_dir,
        verifier_client=verifier_client,
        checker_response=baseline_checker_response,
    )
    rewritten_structure, verifier_rewrites = _apply_unary_verifier_corrections(final_structure, verifier_records)

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
