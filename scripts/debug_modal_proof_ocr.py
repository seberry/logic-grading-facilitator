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


STATE_FILENAME = "state.json"
PROMPT_FILENAME = "ocr_prompt.txt"
FINAL_STRUCTURE_FILENAME = "final_structure.json"
CHECKER_REQUEST_FILENAME = "checker_request.json"
CHECKER_RESPONSE_FILENAME = "checker_response.json"
RAW_RESPONSE_TEMPLATE = "response_{node_id}.json"
INTERMEDIATE_TEMPLATE = "intermediate_{node_id}.json"
UNARY_RECHECK_FILENAME = "unary_rechecks.json"
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
        pool = ("¬", "¬¬", "¬¬¬", "⋄", "⋄¬", "¬⋄", "¬⋄¬")
    elif contains_box or negation_count >= 2 or rule_head in UNARY_SENSITIVE_RULE_HEADS:
        pool = ("¬", "¬¬", "¬¬¬", "□", "□¬", "¬□", "¬□¬")
    else:
        pool = ("¬", "¬¬", "¬¬¬")

    for candidate in pool:
        if candidate not in candidates:
            candidates.append(candidate)
    if prefix.endswith("¬") and prefix.count("¬") < 4:
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
    dark_mask = gray < 185
    row_counts = dark_mask.sum(axis=1).astype(np.float32)
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
    return [(start, end) for start, end in merged if end - start >= 6]


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
    lines = [
        "This image shows one handwritten modal logic proof formula.",
        'Return only a JSON object with keys: selected_formula, confidence, box_present, diamond_present, left_of_main_modal_negations, negations_before_final_atom.',
        "Count the visible unary marks carefully before choosing.",
        "A box is □, a diamond is ⋄, and a negation is ¬.",
        "Choose the exact candidate that best matches the handwriting.",
        "Do not invent a formula outside the candidate list.",
        f"Current OCR formula: {formula_text}",
    ]
    if justification:
        lines.append(f"Current OCR justification: {justification}")
    lines.append("Candidates:")
    for candidate in candidates:
        lines.append(f"- {candidate}")
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
        process_node_recursive(child, img, output_dir, library, prompt=prompt, client=client)
        if node.id == "ROOT":
            current_img = mask_subproof_in_parent(current_img, child)
        else:
            px, py, _, _ = node.box
            rel_box = (child.box[0] - px + 10, child.box[1] - py + 10, child.box[2], child.box[3])
            current_img = mask_subproof_in_parent(current_img, SubproofNode(child.id, rel_box))

    save_path = output_dir / f"crop_{node.id}.jpg"
    cv2.imwrite(str(save_path), current_img)
    print(f"Processing Node {node.id} (Depth {node.depth})... Saved to {save_path.name}")

    raw_response, parsed_entries = call_vlm(save_path, prompt, client)
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
    return normalized


def _normalize_justification_text(text: Any) -> str:
    raw = str(text or "").strip()
    raw = re.sub(r"^\s*incorrect\s+use\s+of\s+", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"^\s*invalid\s+use\s+of\s+", "", raw, flags=re.IGNORECASE)
    raw = raw.replace(",", " ")
    raw = raw.replace("–", "-")
    raw = raw.replace("—", "-")
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
    }
    if node_id:
        line["_ocr_node_id"] = str(node_id)
    if entry_index is not None:
        line["_ocr_entry_index"] = int(entry_index)
    if line_order is not None:
        line["_ocr_line_order"] = int(line_order)
    return line


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


def build_modal_state(
    *,
    config: Dict[str, Any],
    final_structure: List[Any],
    checker_request: Optional[Dict[str, Any]],
    checker_response: Optional[Dict[str, Any]],
    run_summary: str,
    vision_base_url: str | None = None,
    vision_model: str | None = None,
    source_image_name: str = "",
) -> Dict[str, Any]:
    canonical = _canonical_modal_config(config)
    return {
        "system": canonical["system"],
        "include_derived_rules": canonical["include_derived_rules"],
        "conclusion": canonical.get("conclusion") or "",
        "vision_base_url": (vision_base_url or default_lmstudio_base_url()).strip(),
        "vision_model": (vision_model or default_lmstudio_model()).strip(),
        "source_image_name": source_image_name,
        "proof_json_text": json.dumps({"lines": final_structure, "meta": {}}, indent=2, ensure_ascii=False),
        "draft_comment": "",
        "draft_grade": "",
        "checker_request": checker_request,
        "checker_response": checker_response,
        "run_summary": run_summary,
    }


def transcribe_image_to_modal_debug_state(
    *,
    image_path: Path,
    output_dir: Path,
    modal_config: Dict[str, Any],
    client: LMStudioQwenClient | None = None,
    vision_base_url: str | None = None,
    vision_model: str | None = None,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
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
    process_node_recursive(tree_root, img, output_dir, library, prompt=prompt, client=client)

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
    for crop_path in sorted(output_dir.glob("crop_*.jpg")):
        crop_node_id = crop_path.stem.removeprefix("crop_")
        _recheck_suspicious_unary_formulas(
            proof_structure=final_structure,
            crop_path=crop_path,
            client=client,
            output_dir=output_dir,
            modal_config=modal_config,
            node_id=crop_node_id,
        )

    persisted_structure = _clone_jsonish(final_structure)
    _strip_internal_ocr_metadata(persisted_structure)
    (output_dir / FINAL_STRUCTURE_FILENAME).write_text(
        json.dumps(persisted_structure, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    checker_request = None
    checker_response = None
    run_summary = "OCR produced a modal proof structure. Review it in /debug/modal-proof."

    try:
        proof_json_text = json.dumps({"lines": persisted_structure, "meta": {}}, ensure_ascii=False)
        checker_result = run_modal_proof_checker(
            answer_text=proof_json_text,
            autograde_config=_canonical_modal_config(modal_config),
        )
        checker_request = checker_result.request_payload
        checker_response = checker_result.response_payload
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
        run_summary = f"OCR produced a modal proof structure, but checker execution failed: {exc}"

    state = build_modal_state(
        config=modal_config,
        final_structure=persisted_structure,
        checker_request=checker_request,
        checker_response=checker_response,
        run_summary=run_summary,
        vision_base_url=vision_base_url or getattr(client, "base_url", None),
        vision_model=vision_model or getattr(client, "model", None),
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
    args = parser.parse_args()

    image_path = Path(args.image)
    output_dir = Path(args.output_dir)
    autograde_file = Path(args.autograde_file) if args.autograde_file else None

    if not image_path.exists():
        print(f"Error: image not found: {image_path}")
        return 1

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    modal_config = _canonical_modal_config(_load_modal_config(autograde_file))
    print(f"[OCR] Image: {image_path}")
    print(f"[OCR] Output dir: {output_dir}")
    print(f"[OCR] Modal system: {modal_config['system']}")
    print(f"[OCR] VLM base URL: {args.lmstudio_base_url}")
    print(f"[OCR] VLM model: {args.lmstudio_model}")

    try:
        client = build_vlm_client(
            base_url=args.lmstudio_base_url,
            model=args.lmstudio_model,
        )
        state = transcribe_image_to_modal_debug_state(
            image_path=image_path,
            output_dir=output_dir,
            modal_config=modal_config,
            client=client,
            vision_base_url=args.lmstudio_base_url,
            vision_model=args.lmstudio_model,
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
