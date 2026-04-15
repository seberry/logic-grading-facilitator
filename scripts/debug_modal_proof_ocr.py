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
    from app.vlm_client import LMStudioQwenClient

    vlm_client = LMStudioQwenClient()
except ImportError as exc:  # pragma: no cover - fallback for standalone use
    print(f"Warning: Could not import app modules ({exc}).")
    RAW_SCANS_DIR = Path(project_root) / "data" / "raw_scans"
    DEBUG_MODAL_PROOF_DIR = Path(project_root) / "data" / "debug_modal_proof"
    vlm_client = None


STATE_FILENAME = "state.json"
PROMPT_FILENAME = "ocr_prompt.txt"
FINAL_STRUCTURE_FILENAME = "final_structure.json"
CHECKER_REQUEST_FILENAME = "checker_request.json"
CHECKER_RESPONSE_FILENAME = "checker_response.json"
RAW_RESPONSE_TEMPLATE = "response_{node_id}.json"
INTERMEDIATE_TEMPLATE = "intermediate_{node_id}.json"

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
    return [box for box in merged if box[3] > 50]


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


def mask_subproof_in_parent(parent_img: np.ndarray, child_node: SubproofNode) -> np.ndarray:
    masked = parent_img.copy()
    x, y, _, h = child_node.box
    img_h, img_w = masked.shape[:2]
    cv2.rectangle(masked, (x, y), (img_w, y + h), (255, 255, 255), -1)

    token = f"<SUBPROOF_ID_{child_node.id}>"
    cx, cy = x + 150, y + h // 2
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.0
    thickness = 2
    (tw, th), _ = cv2.getTextSize(token, font, scale, thickness)
    pad = 10
    cv2.rectangle(masked, (cx - pad, cy - th - pad), (cx + tw + pad, cy + pad), (0, 0, 255), 2)
    cv2.putText(masked, token, (cx, cy), font, scale, (0, 0, 255), thickness)
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
            bottom_padding=55 if node.children else 20,
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
    return f"{head} {tail}".strip()


def _entry_to_line(entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    scope_role = str(entry.get("scope_role") or "derived_line").strip()
    formula_text = str(entry.get("formula_text") or entry.get("wffstr") or "").strip()
    token_match = TOKEN_PATTERN.search(formula_text)
    if scope_role == "subproof_token" or token_match:
        return None

    if scope_role in {"premise", "ordinary_assumption", "strict_open"}:
        jstr = "PR"
    else:
        jstr = _normalize_justification_text(entry.get("justification_text") or entry.get("jstr") or "")

    return {
        "line_label": str(entry.get("line_label") or "").strip(),
        "wffstr": _normalize_formula_text(formula_text, scope_role),
        "jstr": jstr,
    }


def recursive_stitch(entries: List[Dict[str, Any]], library: Dict[str, List[Dict[str, Any]]]) -> List[Any]:
    resolved: List[Any] = []
    for entry in entries:
        formula_text = str(entry.get("formula_text") or entry.get("wffstr") or "").strip()
        token_match = TOKEN_PATTERN.search(formula_text)
        if token_match:
            child_id = token_match.group(1)
            child_entries = library.get(child_id, [])
            resolved.append(recursive_stitch(child_entries, library))
            continue

        line = _entry_to_line(entry)
        if line is not None:
            resolved.append(line)
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


def build_modal_state(
    *,
    config: Dict[str, Any],
    final_structure: List[Any],
    checker_request: Optional[Dict[str, Any]],
    checker_response: Optional[Dict[str, Any]],
    run_summary: str,
) -> Dict[str, Any]:
    canonical = _canonical_modal_config(config)
    return {
        "system": canonical["system"],
        "include_derived_rules": canonical["include_derived_rules"],
        "conclusion": canonical.get("conclusion") or "",
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

    final_structure = recursive_stitch(top_level_entries, library)
    while isinstance(final_structure, list) and len(final_structure) == 1 and isinstance(final_structure[0], list):
        final_structure = final_structure[0]

    renumber_lines(final_structure)
    rewrite_strict_subproof_local_citations(final_structure)
    repair_modal_subproof_boundary_lines(final_structure)
    renumber_lines(final_structure)
    (output_dir / FINAL_STRUCTURE_FILENAME).write_text(
        json.dumps(final_structure, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    checker_request = None
    checker_response = None
    run_summary = "OCR produced a modal proof structure. Review it in /debug/modal-proof."

    try:
        proof_json_text = json.dumps({"lines": final_structure, "meta": {}}, ensure_ascii=False)
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
        final_structure=final_structure,
        checker_request=checker_request,
        checker_response=checker_response,
        run_summary=run_summary,
    )
    (output_dir / STATE_FILENAME).write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
    return state


def main() -> int:
    parser = argparse.ArgumentParser(description="Run recursive OCR for a modal proof crop.")
    parser.add_argument("--image", required=True, help="Path to the image to transcribe.")
    parser.add_argument("--output-dir", required=True, help="Directory where OCR artifacts should be written.")
    parser.add_argument("--autograde-file", help="YAML/JSON file with modal checker config or saved state.json.")
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

    try:
        state = transcribe_image_to_modal_debug_state(
            image_path=image_path,
            output_dir=output_dir,
            modal_config=modal_config,
            client=LMStudioQwenClient(),
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
