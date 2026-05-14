"""Importable Fitch/TFL OCR pipeline.

This module owns the old recursive chop-and-mask handwritten Fitch OCR flow so
tests, regression helpers, and CLI wrappers can call the same implementation
without shelling out through ``scripts/debug_fitch_ocr.py``.
"""

from __future__ import annotations

import os
import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from proofs_ocr.common.rows import band_height, detect_text_row_bands
from proofs_ocr.common.stitch import recursive_stitch, renumber_lines
from proofs_ocr.pipelines.tfl_line_reread import (
    collect_verifier_reread_candidates,
    strip_ocr_metadata,
)
from proofs_ocr.pipelines.tfl_fitch_vlm import (
    USE_SIMPLE_PROMPT,
    build_27b_fullwidth_prompt,
    build_nested_leaf_separated_prompt,
    build_ocr_prompt,
    call_vlm,
    env_flag as _env_flag,
    env_int as _env_int,
    get_27b_extra_payload as _get_27b_extra_payload,
    get_27b_max_tokens as _get_27b_max_tokens,
    get_27b_timeout_seconds as _get_27b_timeout_seconds,
    get_4b_fullwidth_max_lines as _get_4b_fullwidth_max_lines,
    get_model_27b as _get_model_27b,
    model_27b_available as _model_27b_available,
    stitch_separated_nested_leaf_ocr,
    use_separated_nested_leaf_ocr as _use_separated_nested_leaf_ocr,
)
from proofs_ocr.tricks.justification_snaps import (
    TFL_KNOWN_RULE_HEADS,
    TFL_VALID_RULE_HEADS,
    clean_justification_for_tfl,
    fuzzy_snap_justification,
    snap_justification_rule_head,
)

# Attempt to import project modules
try:
    from app.storage import DEBUG_FITCH_DIR, RAW_SCANS_DIR
    from app.vlm_client import LMStudioQwenClient
    from app.verifier import ProofLine, check_proof, flatten_proof

    vlm_client = LMStudioQwenClient()
    proof_checker_available = True
except ImportError as e:
    project_root = Path(__file__).resolve().parents[2]
    print(f"Warning: Could not import app modules ({e}). Using mocks for VLM/Verifier.")
    vlm_client = None
    proof_checker_available = False
    RAW_SCANS_DIR = project_root / "data" / "raw_scans"
    DEBUG_FITCH_DIR = project_root / "data" / "debug_fitch"

# renumber_lines and recursive_stitch are imported from proofs_ocr.common.stitch


def apply_default_justifications(proof_structure: list, numprems: int) -> None:
    """
    Fill in any remaining blank justifications after renumbering.

    - Lines whose numbers fall within the inferred premise block (<= numprems)
      are labeled as Premises (Pr).
    - All other blank justifications are treated as assumptions (Hyp) so the
      verifier can parse them instead of emitting "Justification left blank."
    """

    for item in proof_structure:
        if isinstance(item, list):
            apply_default_justifications(item, numprems)
        elif isinstance(item, dict):
            raw_j = str(item.get("jstr", "")).strip()
            if raw_j:
                continue

            line_no = item.get("line")
            if line_no is not None and line_no <= numprems:
                item["jstr"] = "Pr"
            else:
                item["jstr"] = "Hyp"


def remove_empty_lines(proof_structure: list) -> list:
    cleaned = []
    for item in proof_structure:
        if isinstance(item, list):
            nested = remove_empty_lines(item)
            if nested:
                cleaned.append(nested)
            continue
        if isinstance(item, dict):
            wff = str(item.get("wffstr", "")).strip()
            jstr = str(item.get("jstr", "")).strip()
            if not wff and not jstr:
                continue
            cleaned.append(item)
    proof_structure[:] = cleaned
    return proof_structure


def infer_missing_rule_heads(proof_structure: list) -> None:
    for item in proof_structure:
        if isinstance(item, list):
            infer_missing_rule_heads(item)
            continue
        if not isinstance(item, dict):
            continue
        wff = str(item.get("wffstr", "")).strip()
        jstr = str(item.get("jstr", "")).strip()
        if wff.startswith("¬") and re.fullmatch(r"I\s+\d+(?:-\d+)?", jstr):
            item["jstr"] = f"¬{jstr}"


def structure_to_proof_lines(structure: List[Union[Dict, List]]) -> List[Union[ProofLine, List]]:
    """
    Convert a nested list/dict structure into ProofLine objects that the verifier can consume.
    """
    if not proof_checker_available:
        return []

    proof_lines: List[Union[ProofLine, List]] = []

    for item in structure:
        if isinstance(item, list):
            proof_lines.append(structure_to_proof_lines(item))
        elif isinstance(item, dict):
            wffstr = str(item.get("wffstr", "")).strip()
            jstr = str(item.get("jstr", "")).strip()
            proof_lines.append(ProofLine(wffstr=wffstr, jstr=jstr, location=[]))

    return proof_lines


def infer_premises(flattened: List[ProofLine]) -> int:
    """
    Heuristically infer premises based on proof depth.

    - If the final line is at depth 0 (top level), assume no premises.
    - If the final line is at depth 1, treat the leading lines of that subproof
      with empty justifications as premises.
    """
    if not flattened:
        return 0

    conclusion = flattened[-1]
    conclusion_depth = len(conclusion.location) - 1

    if conclusion_depth == 1:
        return 0

    if conclusion_depth == 2:
        target_block = conclusion.location[0]
        subproof_lines = [
            line for line in flattened if len(line.location) == 2 and line.location[0] == target_block
        ]
        subproof_lines.sort(key=lambda l: l.location[1])

        numprems = 0
        for line in subproof_lines:
            if line.jstr.strip() == "":
                numprems += 1
            else:
                break
        return numprems

    # For deeper nesting, fall back to zero until a better heuristic is defined.
    return 0

def normalize_proof_structure(structure):
    """
    Walks the proof structure to:
    1. Convert ASCII logic symbols to Unicode (e.g., '~' -> '¬').
    2. Uppercase atomic formulas (TFL requires 'P', not 'p').
    3. Map loose justification names to verifier codes (e.g., 'Premise' -> 'Pr').
    4. Auto-label the first line of any subproof as 'Hyp' if blank.
    """
    
    # Map ASCII/Loose symbols to Verifier Unicode
    # Note: We do this replacement BEFORE uppercasing to protect 'v' (or) from becoming 'V'.
    symbol_map = {
        "~": "¬",
        "-": "¬",      # Sometimes OCR sees dash as negation
        "&": "∧",
        ".": "∧",      # VLM often uses dot for AND
        "v": "∨",      # Lowercase v is OR
        "V": "∨",
        "->": "→",
        "=>": "→",
        "<->": "↔",
        "=": "↔",      # Common OCR error for biconditional
        "_|_": "⊥",
        "F": "⊥",      # Falsum often written as F
    }

    contradiction_rule_heads = {"¬E", "⊥I"}

    def clean_wff(wff):
        if not wff: return ""
        if "<SUBPROOF_ID_" in str(wff):
            return str(wff)

        # Phase 0: Blanket OCR-artifact repair for TFL formulas.
        # TFL formulas never contain digits, epsilons, or element-of symbols.
        # Any occurrence is a VLM misread of a handwritten symbol.
        wff = wff.replace("∈", "E").replace("ε", "E")   # epsilon → E
        wff = wff.replace("8", "∧")                       # handwritten ∧ misread as 8
        wff = wff.replace("5", "S")                        # handwritten S misread as 5

        # 1. Replace multi-char operators first to avoid breaking them
        wff = wff.replace("->", "→").replace("<->", "↔").replace("<→", "↔")
        
        # 2. Character-by-character replacement for single symbols
        clean_chars = []
        for char in wff:
            if char in symbol_map:
                clean_chars.append(symbol_map[char])
            elif char.isalpha():
                # TFL atoms must be Uppercase. 
                # (Skip 'v' if it was mapped to OR, but here 'v' is already handled by symbol_map)
                clean_chars.append(char.upper())
            else:
                clean_chars.append(char)
        
        return "".join(clean_chars)

    def normalize_rule_candidate(token):
        compact = str(token or "").strip()
        compact = compact.replace("->", "→").replace("<->", "↔").replace("<→", "↔")
        compact = compact.replace("&", "∧").replace(".", "∧")
        compact = compact.replace("v", "∨").replace("V", "∨")
        compact = compact.replace("~", "¬").replace("-", "¬")
        return compact

    def looks_like_rule_token(token):
        if not token:
            return False
        compact = normalize_rule_candidate(token).replace(" ", "")
        return compact in TFL_KNOWN_RULE_HEADS

    def likely_rule_with_citations(token, next_token):
        if not looks_like_rule_token(token):
            return False
        next_text = str(next_token or "").strip()
        return bool(re.match(r"^[\da-zA-Z][\d,\-.]*$", next_text))

    def rule_token_at(tokens, index):
        token = tokens[index]
        if index + 1 < len(tokens):
            combined = f"{token}{tokens[index + 1]}"
            if likely_rule_with_citations(combined, tokens[index + 2] if index + 2 < len(tokens) else ""):
                return combined, 2
        if likely_rule_with_citations(token, tokens[index + 1] if index + 1 < len(tokens) else ""):
            return token, 1
        return None, 0

    def extract_embedded_justification(raw_wff, raw_j):
        wff_text = str(raw_wff or "").strip()
        j_text = str(raw_j or "").strip()

        # A bare line number or similar OCR noise in the justification column
        # should not block premise/assumption inference.
        if re.fullmatch(r"[\d,.\-]+", j_text):
            j_text = ""

        if j_text:
            return wff_text, j_text

        if not wff_text:
            return wff_text, j_text

        tokens = wff_text.split()
        if not tokens:
            return wff_text, j_text

        # Sometimes the model puts the entire justification in the formula field.
        head, consumed = rule_token_at(tokens, 0)
        if head is not None:
            justification_text = " ".join([head, *tokens[consumed:]]).strip()
            normalized_head = clean_justification_for_tfl(head)
            normalized_full = clean_justification_for_tfl(justification_text)
            if normalized_head in contradiction_rule_heads:
                return "⊥", normalized_full
            return "", normalized_full

        # More commonly the model appends the justification onto the formula.
        for i in range(1, len(tokens)):
            candidate_head, consumed = rule_token_at(tokens, i)
            if candidate_head is None:
                continue
            formula_part = " ".join(tokens[:i]).strip()
            justification_part = " ".join([candidate_head, *tokens[i + consumed:]]).strip()
            if not formula_part or not justification_part:
                continue
            return formula_part, justification_part

        return wff_text, j_text

    def recursive_normalize(sub_structure, is_subproof_start=False):
        for index, item in enumerate(sub_structure):
            
            # CASE A: It's a Subproof (List)
            if isinstance(item, list):
                # Recurse. The first item in this new list IS a subproof start.
                recursive_normalize(item, is_subproof_start=True)
                
            # CASE B: It's a Line (Dict)
            elif isinstance(item, dict):
                raw_wff = item.get("wffstr", "")
                raw_j = str(item.get("jstr", "")).strip()
                extracted_wff, extracted_j = extract_embedded_justification(raw_wff, raw_j)

                # 1. Clean Formulas
                item["wffstr"] = clean_wff(extracted_wff)
                
                # 3. Handle Implicit Assumptions
                # If this is the very first line of a subproof (List), it MUST be an assumption.
                if is_subproof_start and index == 0 and not extracted_j:
                    item["jstr"] = "Hyp"
                else:
                    normalized_j = snap_justification_rule_head(
                        clean_justification_for_tfl(extracted_j),
                        valid_rules=TFL_VALID_RULE_HEADS,
                    )
                    # Second pass: fuzzy combinatorial repair for stacked
                    # OCR confusions (e.g. "∧E S" → "∧E 5").
                    normalized_j = fuzzy_snap_justification(
                        normalized_j,
                        valid_rules=TFL_VALID_RULE_HEADS,
                    )
                    item["jstr"] = normalized_j

    # Start the recursion
    recursive_normalize(structure)
    return structure
    
# ==============================================================================
# 1. Computer Vision Core (Adapted from bold_fitch_lines.py)
# ==============================================================================

def _split_bars_on_gaps(
    img: np.ndarray, boxes: List[Tuple[int, int, int, int]]
) -> List[Tuple[int, int, int, int]]:
    """Post-process merged bars: split any bar with a significant vertical gap
    in its actual pixel data, indicating two separate subproofs that were
    erroneously merged because they sit at the same X position.

    Uses the same adaptive-threshold + morphological pipeline as bar detection
    so that the gap analysis is done on the same thick, smeared representation
    that caused the merge in the first place.
    """

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_h, img_w = gray.shape[:2]

    # Reproduce the same binary pipeline used by detect_vertical_bars.
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 91, 15,
    )
    dilate_width = 6
    if _env_flag("TFL_FITCH_SLANTED_BAR_REPAIR"):
        dilate_width = max(6, _env_int("TFL_FITCH_SLANTED_BAR_DILATE_WIDTH", 22) or 22)
    dilate_kernel = np.ones((1, dilate_width), np.uint8)
    dilated = cv2.dilate(binary, dilate_kernel, iterations=1)
    vert_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 80))
    vert_mask = cv2.morphologyEx(dilated, cv2.MORPH_OPEN, vert_kern)

    result: List[Tuple[int, int, int, int]] = []
    for x, y, w, h in boxes:
        # A bar must be tall enough to even consider splitting.
        if h < 60:
            result.append((x, y, w, h))
            continue

        # Extract the morphological mask within the bar's bounding box,
        # with a small horizontal margin.
        x1 = max(0, x - 4)
        x2 = min(img_w, x + w + 4)
        y1 = max(0, y)
        y2 = min(img_h, y + h)
        roi_mask = vert_mask[y1:y2, x1:x2]
        if roi_mask.size == 0:
            result.append((x, y, w, h))
            continue

        row_counts = (roi_mask > 0).sum(axis=1)
        min_row_count = max(1, int(0.15 * (x2 - x1)))

        active_rows = [idx for idx, count in enumerate(row_counts) if count >= min_row_count]
        if len(active_rows) < 2:
            result.append((x, y, w, h))
            continue

        # Find contiguous segments of active rows.
        segments: List[Tuple[int, int]] = []
        seg_start = active_rows[0]
        prev = active_rows[0]
        for idx in active_rows[1:]:
            if idx == prev + 1:
                prev = idx
                continue
            segments.append((seg_start, prev))
            seg_start = idx
            prev = idx
        segments.append((seg_start, prev))

        if len(segments) <= 1:
            result.append((x, y, w, h))
            continue

        # Only split on gaps that are proportionally significant.
        gap_threshold = max(15, int(h * 0.06))

        split_boxes: List[Tuple[int, int, int, int]] = []
        for i, (seg_s, seg_e) in enumerate(segments):
            seg_h = seg_e - seg_s + 1
            if seg_h < 15:
                continue
            if i == 0:
                split_boxes.append((x, y1 + seg_s, w, seg_h))
            else:
                prev_seg_end = segments[i - 1][1]
                gap = seg_s - prev_seg_end
                if gap >= gap_threshold:
                    split_boxes.append((x, y1 + seg_s, w, seg_h))
                else:
                    # Gap too small — merge back with the previous split box.
                    if split_boxes:
                        last = split_boxes[-1]
                        new_y = last[1]
                        new_h = (y1 + seg_e + 1) - new_y
                        split_boxes[-1] = (last[0], new_y, last[2], new_h)
                    else:
                        split_boxes.append((x, y1 + seg_s, w, seg_h))

        if len(split_boxes) <= 1:
            result.append((x, y, w, h))
        else:
            result.extend(split_boxes)

    return result


def _merge_vertical_bar_boxes(
    boxes: List[Tuple[int, int, int, int]],
    *,
    x_threshold: int = 20,
    y_gap_threshold: int = 100,
) -> List[Tuple[int, int, int, int]]:
    """Merge contour boxes that look like pieces of the same Fitch bar."""
    boxes = sorted(boxes, key=lambda b: (b[0], b[1]))
    if not boxes:
        return []

    merged: List[Tuple[int, int, int, int]] = []
    curr_x, curr_y, curr_w, curr_h = boxes[0]
    for nx, ny, nw, nh in boxes[1:]:
        if abs(curr_x - nx) < x_threshold and (ny - (curr_y + curr_h) < y_gap_threshold):
            new_x = min(curr_x, nx)
            new_y = min(curr_y, ny)
            new_w = max(curr_x + curr_w, nx + nw) - new_x
            new_h = max(curr_y + curr_h, ny + nh) - new_y
            curr_x, curr_y, curr_w, curr_h = new_x, new_y, new_w, new_h
        else:
            merged.append((curr_x, curr_y, curr_w, curr_h))
            curr_x, curr_y, curr_w, curr_h = nx, ny, nw, nh
    merged.append((curr_x, curr_y, curr_w, curr_h))
    return merged


def _dark_pixel_mask(img: np.ndarray) -> np.ndarray:
    """Return an inverted adaptive-threshold foreground mask."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 91, 15,
    )


def _extend_bars_to_nearby_openers(
    img: np.ndarray,
    boxes: List[Tuple[int, int, int, int]],
) -> List[Tuple[int, int, int, int]]:
    """Extend slanted bars upward to include nearby assumption/opening rows.

    Phone scans on lined paper can produce slanted proof bars whose detected
    vertical mask starts just below the assumption line.  This repair looks for
    a compact text row immediately above and to the right of each bar, then
    extends the bar upward to include it.
    """
    if not boxes:
        return boxes

    binary = _dark_pixel_mask(img)
    img_h, img_w = binary.shape[:2]
    repaired: List[Tuple[int, int, int, int]] = []

    for x, y, w, h in boxes:
        search_y1 = max(0, y - 90)
        search_y2 = y
        search_x1 = min(img_w - 1, x + w + 2)
        search_x2 = min(img_w, x + w + 220)
        new_y = y

        if search_y2 > search_y1 and search_x2 > search_x1:
            roi = binary[search_y1:search_y2, search_x1:search_x2]
            row_counts = (roi > 0).sum(axis=1)
            min_row_count = max(6, int(0.03 * (search_x2 - search_x1)))
            active_rows = [idx for idx, count in enumerate(row_counts) if count >= min_row_count]

            segments: List[Tuple[int, int]] = []
            if active_rows:
                seg_start = active_rows[0]
                prev = active_rows[0]
                for idx in active_rows[1:]:
                    if idx <= prev + 2:
                        prev = idx
                    else:
                        segments.append((seg_start, prev))
                        seg_start = idx
                        prev = idx
                segments.append((seg_start, prev))

            candidates = [
                (search_y1 + start, search_y1 + end)
                for start, end in segments
                if 3 <= (end - start + 1) <= 45
            ]
            if candidates:
                opener_y1, opener_y2 = max(candidates, key=lambda item: item[1])
                if y - opener_y2 <= 45:
                    new_y = opener_y1

        repaired.append((x, new_y, w, h + (y - new_y)))

    return repaired


def _trim_nested_bars_at_text_gap(
    img: np.ndarray,
    boxes: List[Tuple[int, int, int, int]],
) -> List[Tuple[int, int, int, int]]:
    """Trim nested bars before a large text-row gap to avoid swallowing conclusions.

    The slanted-bar repair can correctly extend a bar upward, but on lined
    paper a child bar may also be vertically smeared down toward the
    post-subproof conclusion.  For non-leftmost bars, look at text rows to the
    right of the bar and trim before a large late gap.
    """
    if not boxes:
        return boxes

    binary = _dark_pixel_mask(img)
    _img_h, img_w = binary.shape[:2]
    min_x = min(box[0] for box in boxes)
    repaired: List[Tuple[int, int, int, int]] = []

    for x, y, w, h in boxes:
        new_h = h
        x1 = min(img_w - 1, x + w + 5)
        x2 = min(img_w, x + w + 420)

        if x2 > x1 and h > 100 and x > min_x + 10:
            roi = binary[y:y + h, x1:x2]
            hline_mask = cv2.morphologyEx(
                roi,
                cv2.MORPH_OPEN,
                cv2.getStructuringElement(cv2.MORPH_RECT, (55, 1)),
            )
            text_mask = cv2.subtract(roi, hline_mask)
            row_counts = (text_mask > 0).sum(axis=1)
            min_row_count = max(4, int(0.02 * (x2 - x1)))
            active_rows = [idx for idx, count in enumerate(row_counts) if count >= min_row_count]

            segments: List[Tuple[int, int]] = []
            if active_rows:
                seg_start = active_rows[0]
                prev = active_rows[0]
                for idx in active_rows[1:]:
                    if idx <= prev + 3:
                        prev = idx
                    else:
                        segments.append((seg_start, prev))
                        seg_start = idx
                        prev = idx
                segments.append((seg_start, prev))

            segments = [
                (start, end)
                for start, end in segments
                if (end - start + 1) >= 10
            ]
            gaps = [
                (segments[idx - 1][1], segments[idx][0], segments[idx][0] - segments[idx - 1][1])
                for idx in range(1, len(segments))
            ]
            if gaps:
                prev_end, next_start, gap = max(gaps, key=lambda item: item[2])
                if gap >= 28 and next_start >= h * 0.55:
                    new_h = min(h, prev_end + 10)

        repaired.append((x, y, w, new_h))

    return repaired


def _detect_vertical_bars_standard(img: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """Standard vertical-bar detector for clean-ish handwritten fixtures."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Adaptive Threshold (same params as bold_fitch_lines.py)
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 91, 15
    )

    # 1. Horizontal smear to catch wobbles
    dilate_kernel = np.ones((1, 6), np.uint8)
    dilated = cv2.dilate(binary, dilate_kernel, iterations=1)

    # 2. Vertical filter to isolate long lines
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 80))
    vertical_mask = cv2.morphologyEx(dilated, cv2.MORPH_OPEN, vertical_kernel)

    # 3. Contours
    contours, _ = cv2.findContours(vertical_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 4. Merge Logic (Simplified version of bold_fitch_lines.merge_vertical_segments)
    boxes = [cv2.boundingRect(c) for c in contours]
    merged = _merge_vertical_bar_boxes(boxes)

    # Post-merge: split bars that have significant vertical gaps (e.g. two
    # adjacent subproofs at the same X position that were erroneously merged).
    split = _split_bars_on_gaps(img, merged)

    # Filter out noise (very short lines)
    final_boxes = [b for b in split if b[3] > 50]
    return final_boxes


def _detect_vertical_bars_slanted_repair(img: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """Slant-tolerant detector for normalized phone crops on lined paper.

    This is deliberately opt-in. It widens the horizontal smear before vertical
    opening so slightly slanted Fitch bars survive, extends bars upward to
    nearby opener rows, and trims nested bars before large row gaps so they do
    not swallow post-subproof conclusion lines.
    """
    binary = _dark_pixel_mask(img)

    dilate_width = max(6, _env_int("TFL_FITCH_SLANTED_BAR_DILATE_WIDTH", 22) or 22)

    def candidates(vertical_kernel_height: int, min_height: int) -> List[Tuple[int, int, int, int]]:
        dilated = cv2.dilate(binary, np.ones((1, dilate_width), np.uint8), iterations=1)
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_kernel_height))
        vertical_mask = cv2.morphologyEx(dilated, cv2.MORPH_OPEN, vertical_kernel)

        contours, _ = cv2.findContours(vertical_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = [cv2.boundingRect(c) for c in contours]
        merged = _merge_vertical_bar_boxes(boxes)
        split = [box for box in _split_bars_on_gaps(img, merged) if box[3] > min_height]
        extended = _extend_bars_to_nearby_openers(img, split)
        trimmed = _trim_nested_bars_at_text_gap(img, extended)
        return _filter_slanted_repair_bar_candidates(trimmed, img_shape=img.shape)

    repaired = candidates(vertical_kernel_height=80, min_height=50)
    if len(repaired) >= 2:
        return repaired

    # Some phone crops show only the top of the leftmost scope bar after ruling
    # cleanup, while the nested bar is broken into line-height fragments.  The
    # normal 80px opening correctly rejects handwriting in most cases, but it
    # also drops those fragmented bars.  Use the relaxed pass only as a fallback
    # when the strict pass cannot form a plausible nested structure.
    relaxed_kernel = max(20, _env_int("TFL_FITCH_SLANTED_RELAXED_VERTICAL_KERNEL", 40) or 40)
    relaxed = candidates(vertical_kernel_height=relaxed_kernel, min_height=35)
    if len(relaxed) > len(repaired):
        return relaxed
    return repaired


def _filter_slanted_repair_bar_candidates(
    boxes: List[Tuple[int, int, int, int]],
    *,
    img_shape: Tuple[int, ...],
) -> List[Tuple[int, int, int, int]]:
    """Drop obvious right-side handwriting strokes from phone-crop repair mode."""
    if len(boxes) <= 2:
        return boxes

    img_h, img_w = img_shape[:2]
    min_x = min(box[0] for box in boxes)
    max_scope_x = min_x + max(120, int(img_w * 0.18))
    min_scope_height = max(50, int(img_h * 0.12))
    filtered = [
        box
        for box in boxes
        if box[0] <= max_scope_x and box[3] >= min_scope_height
    ]
    return filtered or boxes


def detect_vertical_bars(img: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """
    Returns a list of (x, y, w, h) tuples for vertical proof bars.
    """
    if _env_flag("TFL_FITCH_SLANTED_BAR_REPAIR"):
        return _detect_vertical_bars_slanted_repair(img)
    return _detect_vertical_bars_standard(img)


def preprocess_slanted_phone_crop(img: np.ndarray) -> np.ndarray:
    """Normalize high-resolution lined-paper phone crops before geometry.

    The fake phone-crop probes that produced good geometry did not feed the raw
    camera crop directly to bar detection.  They first downscaled the image,
    thresholded it, and removed long horizontal ruling.  Keep this behavior
    behind the slanted repair flag so clean handwritten fixtures still use the
    historical image path.
    """
    if img.size == 0:
        return img

    if not _env_flag("TFL_FITCH_SLANTED_PHONE_PREPROCESS", True):
        return img

    img_h, img_w = img.shape[:2]
    target_max_dim = max(100, _env_int("TFL_FITCH_SLANTED_PREPROCESS_MAX_DIM", 714) or 714)
    scale = min(1.0, target_max_dim / max(img_h, img_w))
    if scale < 1.0:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

    def remove_horizontal_ruling(bw: np.ndarray) -> np.ndarray:
        cleaned = bw.copy()
        inv = 255 - cleaned
        h, w = cleaned.shape[:2]
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(30, int(w * 0.20)), 1))
        horizontal_mask = cv2.morphologyEx(inv, cv2.MORPH_OPEN, horizontal_kernel)
        cleaned[horizontal_mask > 0] = 255
        return cleaned

    threshold = max(0, min(255, _env_int("TFL_FITCH_SLANTED_PREPROCESS_THRESHOLD", 110) or 110))
    _, global_bw = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    global_bw = remove_horizontal_ruling(global_bw)

    mode = (os.getenv("TFL_FITCH_SLANTED_PHONE_PREPROCESS_MODE") or "auto").strip().lower()
    if mode in {"global", "threshold", "legacy"}:
        return cv2.cvtColor(global_bw, cv2.COLOR_GRAY2BGR)

    shadow_sigma = max(3, _env_int("TFL_FITCH_SHADOW_PREPROCESS_SIGMA", 25) or 25)
    shadow_threshold = max(1, _env_int("TFL_FITCH_SHADOW_PREPROCESS_THRESHOLD", 25) or 25)
    background = cv2.GaussianBlur(gray, (0, 0), sigmaX=shadow_sigma, sigmaY=shadow_sigma)
    foreground_strength = cv2.subtract(background, gray)
    shadow_bw = np.where(foreground_strength >= shadow_threshold, 0, 255).astype(np.uint8)
    shadow_bw = remove_horizontal_ruling(shadow_bw)

    if mode == "shadow":
        return cv2.cvtColor(shadow_bw, cv2.COLOR_GRAY2BGR)

    if mode != "auto":
        return cv2.cvtColor(global_bw, cv2.COLOR_GRAY2BGR)

    h, w = global_bw.shape[:2]
    y1 = int(h * 0.45)
    y2 = int(h * 0.85)
    x2 = int(w * 0.45)
    if y2 <= y1 or x2 <= 0:
        return cv2.cvtColor(global_bw, cv2.COLOR_GRAY2BGR)

    global_density = float((global_bw[y1:y2, :x2] == 0).mean())
    shadow_density = float((shadow_bw[y1:y2, :x2] == 0).mean())
    density_floor = float(os.getenv("TFL_FITCH_SHADOW_AUTO_DENSITY", "0.14") or "0.14")
    improvement_floor = float(os.getenv("TFL_FITCH_SHADOW_AUTO_IMPROVEMENT", "0.04") or "0.04")
    if global_density >= density_floor and (global_density - shadow_density) >= improvement_floor:
        return cv2.cvtColor(shadow_bw, cv2.COLOR_GRAY2BGR)

    return cv2.cvtColor(global_bw, cv2.COLOR_GRAY2BGR)


def filter_scaffold_bars(
    boxes: List[Tuple[int, int, int, int]],
    *,
    img_w: int,
    img_h: int,
) -> List[Tuple[int, int, int, int]]:
    """
    Ignore the outer full-height Fitch spine when it behaves like page scaffolding
    rather than the opening bar of a real subproof.
    """
    if not boxes:
        return boxes

    sorted_boxes = sorted(boxes, key=lambda b: (b[0], b[1]))

    # Drop narrow border artifacts stuck to the extreme page edges before
    # reasoning about the real proof scaffold vs. subproof bars.
    edge_margin = max(16, int(img_w * 0.02))
    narrow_width = max(24, int(img_w * 0.03))
    sorted_boxes = [
        box
        for box in sorted_boxes
        if not (
            box[2] <= narrow_width
            and (box[0] <= edge_margin or (box[0] + box[2]) >= (img_w - edge_margin))
        )
    ]
    if not sorted_boxes:
        return []

    # NOTE: scaffold-removal logic deliberately disabled (2026-04-24).
    # Handwritten Fitch proofs have a real outer subproof bar that spans
    # most of the image height at the left edge — exactly the profile that
    # scaffold removal was dropping.  The edge-artifact cleanup above is
    # sufficient for real-world scans; a human-in-the-loop can crop any
    # genuine page-border artifacts during scan prep.
    return sorted_boxes

def count_text_lines_in_region(img: np.ndarray) -> int:
    """Estimate the number of handwritten text lines in a crop image.

    Uses a horizontal projection profile: sum dark pixels per row, then
    count contiguous clusters of "active" rows that exceed a density
    threshold.  This reuses the same adaptive-threshold pipeline as bar
    detection but analyses rows instead of columns.

    Parameters
    ----------
    img:
        Cropped proof region (BGR or grayscale).

    Returns
    -------
    int
        Estimated number of text lines (always >= 1).
    """
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    h, w = gray.shape[:2]
    if h < 10 or w < 10:
        return 1

    # Binary: dark text on white background
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 91, 15,
    )

    # Horizontal projection: count dark pixels per row
    row_counts = (binary > 0).sum(axis=1)

    # A row is "active" if it has at least 5% of its pixels dark.
    # This ignores nearly-blank rows between lines.
    min_row_density = max(3, int(0.05 * w))
    active = row_counts >= min_row_density

    # Count contiguous runs of active rows
    runs = 0
    in_run = False
    for val in active:
        if val and not in_run:
            runs += 1
            in_run = True
        elif not val:
            in_run = False

    return max(1, runs)


class SubproofNode:
    def __init__(self, box_id, box):
        self.id = box_id
        self.box = box  # (x, y, w, h)
        self.children = []
        self.parent = None
    
    @property
    def depth(self):
        d = 0
        p = self.parent
        while p:
            d += 1
            p = p.parent
        return d

def build_proof_tree(boxes: List[Tuple[int, int, int, int]], img_w: int, img_h: int) -> SubproofNode:
    """
    Organizes vertical bars into a hierarchy.
    Root is the entire image. Children are bars contained within parent's Y-range
    and strictly to the right in X.
    """
    # Root represents the main document
    root = SubproofNode("ROOT", (0, 0, img_w, img_h))
    
    # Sort boxes by X position (left to right) to handle nesting naturally
    # Fitch proofs grow rightward as depth increases.
    sorted_boxes = sorted(boxes, key=lambda b: b[0])
    
    nodes = [SubproofNode(str(i+1), b) for i, b in enumerate(sorted_boxes)]

    # Assign parents
    # A simplified heuristic for Fitch:
    # A bar is a child of the closest bar to its LEFT that vertically spans it.
    for node in nodes:
        best_parent = root
        nx, ny, nw, nh = node.box
        ny_center = ny + nh/2
        
        # Look at all potential parents (bars to the left)
        # We search backwards to find the *closest* one
        for candidate in reversed(nodes):
            if candidate == node: continue
            if candidate.box[0] >= nx: continue # Must be to the left
            
            cx, cy, cw, ch = candidate.box
            # Check vertical containment (loose)
            if cy <= ny_center <= (cy + ch):
                best_parent = candidate
                break
        
        node.parent = best_parent
        best_parent.children.append(node)

    return root


def _flatten_subproof_nodes(node: SubproofNode) -> List[SubproofNode]:
    nodes: List[SubproofNode] = []
    for child in node.children:
        nodes.append(child)
        nodes.extend(_flatten_subproof_nodes(child))
    return nodes


def _subproof_tree_shape(node: SubproofNode) -> str:
    if not node.children:
        return str(node.id)
    return f"{node.id}->[" + ",".join(_subproof_tree_shape(child) for child in node.children) + "]"


def _subproof_tree_max_depth(node: SubproofNode) -> int:
    if not node.children:
        return int(node.depth)
    return max(_subproof_tree_max_depth(child) for child in node.children)


def _with_env_flag(name: str, value: str | None, fn):
    old_value = os.environ.get(name)
    try:
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value
        return fn()
    finally:
        if old_value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = old_value


def _estimate_leaf_rows_for_geometry(img: np.ndarray, node: SubproofNode, *, slanted: bool) -> Dict[str, object]:
    def estimate() -> Dict[str, object]:
        tight = crop_region(img, node.box)
        full = crop_full_width(img, node)
        return {
            "id": str(node.id),
            "box": [int(value) for value in node.box],
            "tight_rows": int(estimate_line_count(tight)),
            "full_rows": int(estimate_line_count(full)),
        }

    return _with_env_flag("TFL_FITCH_SLANTED_BAR_REPAIR", "1" if slanted else None, estimate)


def _geometry_candidate_summary(
    *,
    mode: str,
    img: np.ndarray,
    bars: List[Tuple[int, int, int, int]],
    tree: SubproofNode,
) -> Dict[str, object]:
    nodes = _flatten_subproof_nodes(tree)
    leaves = [node for node in nodes if not node.children]
    slanted = mode == "slanted"
    leaf_rows = [
        _estimate_leaf_rows_for_geometry(img, node, slanted=slanted)
        for node in leaves
    ]

    flags: List[str] = []
    if not bars:
        flags.append("no_bars")
    if len(bars) == 1:
        flags.append("single_bar")
    if len(bars) > max(2, _env_int("TFL_FITCH_PREFLIGHT_MAX_BARS", 8) or 8):
        flags.append("too_many_bars")

    row_values: List[int] = []
    for row in leaf_rows:
        for key in ("tight_rows", "full_rows"):
            value = row.get(key)
            if isinstance(value, int):
                row_values.append(value)
    if any(value >= (_env_int("TFL_FITCH_PREFLIGHT_HUGE_ROW_COUNT", 20) or 20) for value in row_values):
        flags.append("huge_leaf_row_count")
    if leaves and any(value == 0 for value in row_values):
        flags.append("zero_row_leaf")

    return {
        "mode": mode,
        "image_size": [int(img.shape[1]), int(img.shape[0])],
        "bars": len(bars),
        "bar_boxes": [[int(value) for value in bar] for bar in bars],
        "node_count": len(nodes),
        "leaf_count": len(leaves),
        "max_depth": _subproof_tree_max_depth(tree),
        "shape": _subproof_tree_shape(tree),
        "leaf_rows": leaf_rows,
        "flags": flags,
    }


def _has_major_geometry_failure(summary: Dict[str, object]) -> bool:
    flags = set(summary.get("flags") or [])
    return bool(flags.intersection({"no_bars", "too_many_bars", "huge_leaf_row_count"}))


def _choose_preflight_geometry(
    standard: Dict[str, object],
    slanted: Dict[str, object],
) -> str:
    std_major = _has_major_geometry_failure(standard)
    slanted_major = _has_major_geometry_failure(slanted)

    if std_major and not slanted_major:
        return "slanted"
    if not std_major:
        return "standard"
    if slanted_major:
        return "manual"
    return "standard"


def _detect_geometry_candidate(raw_img: np.ndarray, *, mode: str) -> Dict[str, object]:
    if mode == "slanted":
        img = preprocess_slanted_phone_crop(raw_img)
        bars = _with_env_flag(
            "TFL_FITCH_SLANTED_BAR_REPAIR",
            "1",
            lambda: _detect_vertical_bars_slanted_repair(img),
        )
    elif mode == "standard":
        img = raw_img
        bars = _detect_vertical_bars_standard(img)
    else:
        raise ValueError(f"Unknown geometry mode: {mode}")

    bars = filter_scaffold_bars(bars, img_w=img.shape[1], img_h=img.shape[0])
    tree = build_proof_tree(bars, img.shape[1], img.shape[0])
    return {
        "mode": mode,
        "img": img,
        "bars": bars,
        "tree": tree,
        "summary": _geometry_candidate_summary(mode=mode, img=img, bars=bars, tree=tree),
    }


def select_fitch_geometry(raw_img: np.ndarray) -> Dict[str, object]:
    """Select standard or slanted geometry before making any VLM calls.

    ``TFL_FITCH_GEOMETRY_MODE`` may force ``standard``, ``slanted``, or ``auto``.
    For backward compatibility, ``TFL_FITCH_SLANTED_BAR_REPAIR=1`` still forces
    the slanted path unless ``TFL_FITCH_GEOMETRY_MODE=auto`` is explicitly set.
    """
    requested = (os.getenv("TFL_FITCH_GEOMETRY_MODE") or "").strip().lower()
    if requested not in {"", "auto", "standard", "slanted"}:
        requested = ""

    if requested == "standard":
        candidate = _detect_geometry_candidate(raw_img, mode="standard")
        candidate["selected_mode"] = "standard"
        candidate["preflight"] = {"selected_mode": "standard", "standard": candidate["summary"]}
        return candidate

    if requested == "slanted" or (not requested and _env_flag("TFL_FITCH_SLANTED_BAR_REPAIR")):
        candidate = _detect_geometry_candidate(raw_img, mode="slanted")
        candidate["selected_mode"] = "slanted"
        candidate["preflight"] = {"selected_mode": "slanted", "slanted": candidate["summary"]}
        return candidate

    standard = _detect_geometry_candidate(raw_img, mode="standard")
    slanted = _detect_geometry_candidate(raw_img, mode="slanted")
    selected_mode = _choose_preflight_geometry(standard["summary"], slanted["summary"])
    preflight = {
        "selected_mode": selected_mode,
        "standard": standard["summary"],
        "slanted": slanted["summary"],
    }
    if selected_mode == "manual":
        return {
            "selected_mode": "manual",
            "img": standard["img"],
            "bars": standard["bars"],
            "tree": standard["tree"],
            "summary": standard["summary"],
            "preflight": preflight,
        }
    candidate = slanted if selected_mode == "slanted" else standard
    candidate["selected_mode"] = selected_mode
    candidate["preflight"] = preflight
    return candidate


# ==============================================================================
# 2. Image Manipulation (Masking & Cropping)
# ==============================================================================

def crop_region(img: np.ndarray, box: Tuple[int, int, int, int], padding=10) -> np.ndarray:
    x, y, w, h = box
    h_img, w_img = img.shape[:2]
    
    # For a subproof, we want the text to the RIGHT of the bar.
    # We'll take from the bar's X up to the image width (or a reasonable limit).
    # We'll take Y from top of bar to bottom of bar.
    
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = w_img # Extend to right edge
    y2 = min(h_img, y + h + padding)
    
    return img[y1:y2, x1:x2]

def crop_full_width(img: np.ndarray, node: SubproofNode, padding: int = 10) -> np.ndarray:
    """Crop a full-width horizontal slice at the node's Y range.

    Unlike :func:`crop_region` which starts from the bar's X position, this
    preserves the full image width including the line-number gutter on the
    left.  The 27b model uses the gutter numbers as visual context.
    """
    _, y, _, h = node.box
    h_img, _w_img = img.shape[:2]

    if node.parent and node.parent.id == "ROOT":
        y2 = h_img
    else:
        y2 = min(h_img, y + h)

    y1 = max(0, y - padding)
    return img[y1:y2, :]


def _subproof_token_baseline_y(*, y1: int, h: int, img_h: int, text_height: int) -> int:
    """Place the red child token near the masked subproof opener row."""
    top_offset = min(max(text_height + 10, 26), max(text_height + 10, h - 6))
    return max(text_height + 8, min(y1 + top_offset, img_h - 8))


def mask_subproof_in_parent(parent_img: np.ndarray, child_node: SubproofNode) -> np.ndarray:
    """
    Draws a white box over the child's area and stamps the token.
    """
    masked = parent_img.copy()
    x, y, w, h = child_node.box
    h_img, w_img = masked.shape[:2]
    
    # Mask area: From bar X to right edge, matching bar height
    # This effectively "deletes" the subproof content
    x1 = x
    y1 = y
    x2 = w_img
    y2 = y + h
    
    # 1. Draw solid white rectangle (Erasure)
    cv2.rectangle(masked, (x1, y1), (x2, y2), (255, 255, 255), -1)
    
    # 2. Draw a compact red token where the erased subproof belongs.  Keeping
    # it close to the scope bar makes the placeholder read like a proof row
    # instead of a floating annotation in the middle of the whiteout.
    token = f"<SUBPROOF_ID_{child_node.id}>"
    token_x = min(w_img - 40, x1 + max(42, int(w_img * 0.08)))
    
    # Calculate text size for centering
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.65
    thickness = 2
    (tw, th), _ = cv2.getTextSize(token, font, scale, thickness)
    token_x = max(0, min(token_x, w_img - tw - 12))
    token_y = _subproof_token_baseline_y(y1=y1, h=h, img_h=h_img, text_height=th)
    
    # Draw red border around token
    pad = 6
    cv2.rectangle(
        masked,
        (token_x - pad, token_y - th - pad),
        (token_x + tw + pad, token_y + pad),
        (0, 0, 255),
        2,
    )
    
    # 3. Write Token
    cv2.putText(masked, token, (token_x, token_y), font, scale, (0, 0, 255), thickness)
    
    return masked


def _subproof_token_id(value: object) -> str | None:
    match = re.search(r"<SUBPROOF_ID_([^>]+)>", str(value or ""))
    return match.group(1) if match else None


def repair_missing_child_tokens(
    ocr_result: object,
    child_nodes: List[SubproofNode],
    *,
    img_shape: Tuple[int, int, int] | Tuple[int, int],
) -> object:
    """Restore child placeholders omitted by parent OCR.

    The red tokens are synthetic structural markers drawn by this pipeline.
    If the VLM skips one, the child OCR is already present in ``library`` but
    cannot be stitched.  This repair only inserts missing immediate child
    placeholders; it does not alter formulas or justifications.
    """
    if not isinstance(ocr_result, list) or not child_nodes:
        return ocr_result

    existing_ids = {
        token_id
        for item in ocr_result
        if isinstance(item, dict)
        for token_id in [_subproof_token_id(item.get("wffstr"))]
        if token_id is not None
    }
    missing_children = [child for child in sorted(child_nodes, key=lambda c: c.box[1]) if child.id not in existing_ids]
    if not missing_children:
        return ocr_result

    child_y_by_id = {child.id: int(child.box[1]) for child in child_nodes}
    repaired = list(ocr_result)
    img_h = int(img_shape[0])

    def fallback_index(child: SubproofNode) -> int:
        if not repaired:
            return 0
        last = repaired[-1]
        if isinstance(last, dict) and re.search(r"[IΙ]\s*\d+\s*-\s*\d+", str(last.get("jstr") or "")):
            return len(repaired) - 1
        child_bottom = int(child.box[1] + child.box[3])
        post_child_room = img_h - child_bottom
        if len(repaired) >= 2 and post_child_room > max(48, int(img_h * 0.12)):
            return len(repaired) - 1
        return len(repaired)

    for child in missing_children:
        insert_at: int | None = None
        child_y = int(child.box[1])
        for index, item in enumerate(repaired):
            if not isinstance(item, dict):
                continue
            token_id = _subproof_token_id(item.get("wffstr"))
            if token_id is not None and child_y_by_id.get(token_id, child_y) > child_y:
                insert_at = index
                break
        if insert_at is None:
            insert_at = fallback_index(child)
        repaired.insert(
            insert_at,
            {
                "line": insert_at + 1,
                "wffstr": f"<SUBPROOF_ID_{child.id}>",
                "jstr": "",
            },
        )
    return repaired


def drop_trailing_parent_bleed_artifact(ocr_result: object) -> object:
    """Drop a narrow OCR row caused by bleed from the next parent line."""
    if not isinstance(ocr_result, list) or len(ocr_result) < 2:
        return ocr_result
    last = ocr_result[-1]
    if not isinstance(last, dict):
        return ocr_result

    wff = str(last.get("wffstr") or "").strip()
    jstr = str(last.get("jstr") or "").strip()
    if wff != "E":
        return ocr_result
    if not re.fullmatch(r"[\d,\-\s]+", jstr):
        return ocr_result
    return ocr_result[:-1]


def repair_trailing_child_token_order(ocr_result: object) -> object:
    """Move a child token before a trailing intro-conclusion row when OCR flips them."""
    if not isinstance(ocr_result, list) or len(ocr_result) < 2:
        return ocr_result
    last = ocr_result[-1]
    prev = ocr_result[-2]
    if not isinstance(last, dict) or not isinstance(prev, dict):
        return ocr_result
    if _subproof_token_id(last.get("wffstr")) is None:
        return ocr_result
    prev_jstr = str(prev.get("jstr") or "")
    if not re.search(r"[IΙ]\s*\d+\s*-\s*\d+", prev_jstr):
        return ocr_result
    repaired = list(ocr_result[:-2])
    repaired.extend([last, prev])
    return repaired


# ==============================================================================
# 2b. Context-Aware Crops for Small Subproofs
# ==============================================================================

# Threshold: leaf crops shorter than this (pixels) get a parent-context crop
# with a highlighted box instead of a tight white-masked crop.
SMALL_CROP_THRESHOLD = 250

def estimate_line_count(img: np.ndarray) -> int:
    """Estimate proof lines in a leaf crop using morphological band detection.

    Wraps ``detect_text_row_bands`` with post-filtering:
    * removes thin artifact bands (underline / rule-line residuals < 18 px)
    * estimates merged lines inside unusually tall bands.

    Returns 0 if the image is too small or no bands are found.
    """
    if _env_flag("TFL_FITCH_SLANTED_BAR_REPAIR"):
        lined_count = _estimate_lined_paper_line_count(img)
        if lined_count > 0:
            return lined_count

    bands = detect_text_row_bands(img)
    if not bands:
        return 0

    MIN_BAND_HEIGHT = 18
    real_bands = [b for b in bands if band_height(b) >= MIN_BAND_HEIGHT]
    if not real_bands:
        return len(bands)

    heights = sorted(band_height(b) for b in real_bands)
    ref_height = heights[len(heights) // 2]  # median

    MERGE_FACTOR = 1.6
    count = 0
    for b in real_bands:
        h = band_height(b)
        if h > ref_height * MERGE_FACTOR and ref_height > 0:
            count += max(1, round(h / ref_height))
        else:
            count += 1
    return count


def _estimate_lined_paper_line_count(img: np.ndarray) -> int:
    """Estimate text rows for phone crops on ruled paper.

    The default row-band detector works well on clean crops, but lined-paper
    phone crops can leave ruling/shadow residuals that merge several proof rows
    into oversized bands.  This opt-in counter removes long horizontal ruling
    and vertical scope bars before grouping the remaining handwriting pixels.
    It also drops a clipped top-edge band, which occurs when a child subproof
    crop begins just below the previous parent row but padding still includes
    its descenders/underline.
    """
    if img.size == 0:
        return 0

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    img_h, img_w = gray.shape[:2]
    if img_h < 20 or img_w < 40:
        return 0

    dark = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        91,
        15,
    )

    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(80, int(img_w * 0.12)), 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(80, int(img_h * 0.22))))
    horizontal_mask = cv2.morphologyEx(dark, cv2.MORPH_OPEN, horizontal_kernel)
    vertical_mask = cv2.morphologyEx(dark, cv2.MORPH_OPEN, vertical_kernel)

    text_mask = dark.copy()
    text_mask[horizontal_mask > 0] = 0
    text_mask[vertical_mask > 0] = 0

    left = min(img_w - 1, max(24, int(img_w * 0.04)))
    right = max(left + 1, min(img_w, int(img_w * 0.94)))
    focused = text_mask[:, left:right]
    if focused.size == 0:
        return 0

    row_counts = focused.sum(axis=1).astype(np.float32) / 255.0
    if float(row_counts.max(initial=0.0)) <= 0:
        return 0

    smooth_window = max(9, min(25, img_h // 45 if img_h >= 405 else 9))
    if smooth_window % 2 == 0:
        smooth_window += 1
    smoothed = np.convolve(row_counts, np.ones(smooth_window, dtype=np.float32) / smooth_window, mode="same")
    threshold = max(8.0, float(smoothed.max()) * 0.18)
    active_rows = np.where(smoothed >= threshold)[0]
    if active_rows.size == 0:
        return 0

    max_gap = max(5, img_h // 120)
    bands: List[Tuple[int, int]] = []
    start = int(active_rows[0])
    prev = int(active_rows[0])
    for row in active_rows[1:]:
        row = int(row)
        if row <= prev + max_gap:
            prev = row
            continue
        bands.append((start, prev))
        start = row
        prev = row
    bands.append((start, prev))

    min_height = max(9, img_h // 140)
    bands = [(start, end) for start, end in bands if (end - start + 1) >= min_height]
    if not bands:
        return 0

    if len(bands) >= 2 and bands[0][0] <= max(8, img_h // 80):
        next_gap = bands[1][0] - bands[0][1] - 1
        if next_gap >= max(18, img_h // 45):
            bands = bands[1:]

    heights = sorted(end - start + 1 for start, end in bands)
    ref_height = heights[len(heights) // 2]
    if ref_height <= 0:
        return len(bands)

    count = 0
    for start, end in bands:
        height = end - start + 1
        if height > ref_height * 1.8:
            count += max(1, round(height / ref_height))
        else:
            count += 1

    if count == 2:
        # Short phone-crop subproofs can merge two adjacent handwritten rows
        # into one smoothed band when a long justification bridges the rows.
        # Use a stricter unsmoothed projection only when it cleanly separates
        # the crop into exactly three substantial text rows.
        high_threshold = max(20.0, float(row_counts.max()) * 0.20)
        high_active = np.where(row_counts >= high_threshold)[0]
        high_bands: List[Tuple[int, int]] = []
        if high_active.size:
            start = int(high_active[0])
            prev = int(high_active[0])
            for row in high_active[1:]:
                row = int(row)
                if row <= prev + 3:
                    prev = row
                    continue
                high_bands.append((start, prev))
                start = row
                prev = row
            high_bands.append((start, prev))
        high_bands = [
            (start, end)
            for start, end in high_bands
            if (end - start + 1) >= 3
        ]
        if len(high_bands) == 3:
            return 3

    return count


def _top_carryover_trim_y(
    *,
    img_h: int,
    bands: List[Tuple[int, int]],
    expected_lines: int | None,
) -> int:
    if expected_lines is None or expected_lines <= 0:
        return 0
    if len(bands) <= expected_lines:
        return 0
    if len(bands) < 2:
        return 0
    if bands[0][0] > max(8, img_h // 80):
        return 0
    next_gap = bands[1][0] - bands[0][1] - 1
    if next_gap < max(18, img_h // 45):
        return 0
    return max(0, bands[1][0] - max(10, img_h // 30))


def trim_lined_paper_top_carryover_for_ocr(
    img: np.ndarray,
    *,
    expected_lines: int | None,
) -> np.ndarray:
    """Remove clipped parent-row carryover before sending a leaf crop to OCR."""
    if not _env_flag("TFL_FITCH_SLANTED_BAR_REPAIR") or img.size == 0:
        return img

    bands = detect_text_row_bands(img)
    trim_y = _top_carryover_trim_y(img_h=img.shape[0], bands=bands, expected_lines=expected_lines)
    if trim_y <= 0 or trim_y >= img.shape[0] - 8:
        return img
    return img[trim_y:, :]


def adjusted_child_box_for_parent_mask(
    child_node: SubproofNode,
    source_img: np.ndarray,
) -> Tuple[int, int, int, int]:
    """Shift a leaf child's parent-mask top below clipped carryover rows."""
    x, y, w, h = child_node.box
    if not _env_flag("TFL_FITCH_SLANTED_BAR_REPAIR") or child_node.children:
        return child_node.box

    child_crop = crop_region(source_img, child_node.box)
    expected_lines = estimate_line_count(child_crop)
    if expected_lines <= 0:
        return child_node.box

    fw_img = crop_full_width(source_img, child_node)
    bands = detect_text_row_bands(fw_img)
    trim_y = _top_carryover_trim_y(
        img_h=fw_img.shape[0],
        bands=bands,
        expected_lines=expected_lines,
    )
    if trim_y <= 0:
        return child_node.box

    fullwidth_y1 = max(0, y - 10)
    adjusted_y = max(y, fullwidth_y1 + trim_y)
    delta = adjusted_y - y
    if delta <= 0 or delta >= h - 8:
        return child_node.box
    return (x, adjusted_y, w, h - delta)


#: Enable line-count estimation for leaf crops using the shared morphological
#: band detector from proofs_ocr/common/rows.py, with post-filtering to remove
#: thin underline artifacts and estimate merged tall bands.
#: When True, the VLM prompt is constrained to the detected line count and
#: excess VLM responses are truncated.
USE_LINE_COUNT_CONSTRAINT = True

# Visual settings for the highlight rectangle drawn on context crops.
HIGHLIGHT_COLOR = (0, 255, 0)   # bright green (BGR)
HIGHLIGHT_THICKNESS = 3


def letterbox_crop(
    img: np.ndarray,
    target_ratio: float = 4 / 3,
    min_height: int = 250,
) -> np.ndarray:
    """Pad a small crop with white space to reach a target aspect ratio.

    Many VLMs (including Qwen-VL) internally resize images to fixed patch
    sizes.  A very short, wide crop can get distorted or severely
    down-sampled, degrading handwriting recognition.  Letterboxing adds
    white padding so the image looks like a clean "page" with a few lines.

    Parameters
    ----------
    img:
        The cropped proof image (BGR).
    target_ratio:
        Desired width / height ratio (default 4:3).
    min_height:
        Minimum height in pixels; if the crop is shorter, vertical padding
        is added first to reach this height before aspect-ratio adjustment.
    """
    h, w = img.shape[:2]

    # Ensure minimum height first
    if h < min_height:
        pad_v = min_height - h
        pad_top = pad_v // 2
        pad_bot = pad_v - pad_top
        img = cv2.copyMakeBorder(
            img, pad_top, pad_bot, 0, 0,
            cv2.BORDER_CONSTANT, value=(255, 255, 255),
        )
        h = img.shape[0]

    # Now pad horizontally to reach target_ratio
    target_w = int(h * target_ratio)
    if w >= target_w:
        return img  # already wide enough

    pad_total = target_w - w
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left
    return cv2.copyMakeBorder(
        img, 0, 0, pad_left, pad_right,
        cv2.BORDER_CONSTANT, value=(255, 255, 255),
    )


def build_context_crop(
    parent_img: np.ndarray,
    child_box_global: Tuple[int, int, int, int],
    parent_crop_offset: Tuple[int, int],
) -> np.ndarray:
    """Draw a colored highlight rectangle around the child region on a parent image.

    The parent image should already be cropped to the parent's region (unmasked).
    The child coordinates are in the *global* image frame and are translated
    into the parent-crop frame using *parent_crop_offset*.

    Parameters
    ----------
    parent_img:
        The parent's cropped image (no masking applied).
    child_box_global:
        The child's box ``(x, y, w, h)`` in global image coordinates.
    parent_crop_offset:
        ``(offset_x, offset_y)`` — the top-left corner of the parent crop
        in global coordinates (i.e. what was subtracted by ``crop_region``).

    Returns
    -------
    np.ndarray
        Copy of *parent_img* with a colored rectangle around the child region.
    """
    annotated = parent_img.copy()
    offset_x, offset_y = parent_crop_offset

    cx, cy, _cw, ch = child_box_global
    h_img, w_img = annotated.shape[:2]

    # Translate to parent-crop frame.  The highlight covers the same region
    # that mask_subproof_in_parent would white-out: from the child bar X to
    # the right edge of the image, and from child bar Y to Y+H.
    x1 = max(0, cx - offset_x)
    y1 = max(0, cy - offset_y)
    x2 = w_img  # extend to right edge
    y2 = min(h_img, cy + ch - offset_y)

    cv2.rectangle(annotated, (x1, y1), (x2, y2), HIGHLIGHT_COLOR, HIGHLIGHT_THICKNESS)

    return annotated


# ==============================================================================
# 3. VLM Interaction (Mockable)
# ==============================================================================

#: Model routing for small leaf crops.  When a leaf crop has <= this many
#: expected lines, the pipeline routes it to a larger model (27b) for better
#: handwriting recognition.  Set to 0 to disable routing.
SMALL_LEAF_MODEL_ROUTING_THRESHOLD = 3

# ==============================================================================
# 4. Orchestration
# ==============================================================================

def process_node_recursive(node: SubproofNode, img: np.ndarray, output_dir: Path, library: Dict, client) -> List[Dict]:
    """Process a proof-tree node while preserving the historical public helper."""
    from proofs_ocr.pipelines.tfl_fitch_runtime import process_node_recursive as _runtime_process_node_recursive

    return _runtime_process_node_recursive(
        node,
        img,
        output_dir,
        library,
        client,
        call_vlm_fn=call_vlm,
    )

# ==============================================================================
# 5. Stitching — uses shared recursive_stitch from proofs_ocr.common.stitch
# ==============================================================================

def label_and_count_premises(proof_structure: List[Union[Dict, List]]) -> int:
    """
    Walks the top-level of the stitched proof structure.
    If a line has an empty justification ('jstr') and appears before any 
    subproofs or justified lines, it is labeled as 'Pr' (for premise).
    
    Returns: The number of premises identified.
    """
    premise_count = 0
    
    for item in proof_structure:
        # 1. If we hit a subproof (list), the premise block is definitely over.
        #    (Premises cannot appear after or inside a subproof at the top level)
        if isinstance(item, list):
            break
            
        # 2. If it's a line (dict)...
        if isinstance(item, dict):
            jstr = item.get("jstr", "").strip()
            
            # If justification is empty, treat it as a Premise
            if not jstr:
                item["jstr"] = "Pr" # Inject the label
                premise_count += 1
            
            # If it already has a justification (e.g., "Taut Con"), 
            # the premise block is over.
            elif jstr.lower() != "pr": 
                break
                
    return premise_count

# ==============================================================================
# Main
# ==============================================================================

def run_fitch_ocr_pipeline(
    *,
    image_path: Path,
    output_dir: Path,
    client=None,
) -> Dict:
    """Run the full Fitch OCR pipeline on a single image.

    This is the programmatic entry point extracted from ``main()`` so that
    regression tests can call the pipeline directly without going through
    the CLI.

    Parameters
    ----------
    image_path:
        Path to the proof image (PNG/JPG).
    output_dir:
        Directory where intermediate crops and the final structure are
        written.  Created if it does not exist.
    client:
        VLM client used for OCR.  Pass ``None`` to fall back to the
        module-level ``vlm_client`` (or a mock placeholder).

    Returns
    -------
    dict
        ``{"run_summary": str, "line_count": int}``
    """
    if client is None:
        client = vlm_client

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load image
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    # 2. Select geometry, detect bars, and build tree before any VLM calls.
    geometry = select_fitch_geometry(img)
    preflight = geometry.get("preflight") or {}
    (output_dir / "geometry_preflight.json").write_text(
        json.dumps(preflight, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    selected_mode = str(geometry.get("selected_mode") or "standard")
    if selected_mode == "manual":
        raise RuntimeError(
            "Fitch geometry preflight could not choose a reliable OCR mode. "
            f"See {output_dir / 'geometry_preflight.json'} for standard/slanted diagnostics."
        )

    img = geometry["img"]
    bars = geometry["bars"]
    tree_root = geometry["tree"]

    # Draw debug bars
    debug_img = img.copy()
    for i, (x, y, w, h) in enumerate(bars):
        cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(debug_img, str(i + 1), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imwrite(str(output_dir / "debug_bars.jpg"), debug_img)

    max_slanted_bars = _env_int("TFL_FITCH_SLANTED_MAX_BARS_BEFORE_OCR", 6) or 6
    if selected_mode == "slanted" and len(bars) > max_slanted_bars:
        raise RuntimeError(
            "Slanted phone-crop geometry is implausible before OCR: "
            f"detected {len(bars)} bar candidates after preprocessing "
            f"(max {max_slanted_bars})."
        )

    # 3. Recursive process
    library: Dict = {}
    _with_env_flag(
        "TFL_FITCH_SLANTED_BAR_REPAIR",
        "1" if selected_mode == "slanted" else None,
        lambda: process_node_recursive(tree_root, img, output_dir, library, client=client),
    )

    # 4. Stitch & Renumber
    root_has_content = any(line.get("wffstr", "").strip() for line in library.get("ROOT", []))

    if not root_has_content and "1" in library:
        top_level_structure = library["1"]
    else:
        top_level_structure = library.get("ROOT", [])

    stitched_structure = recursive_stitch(top_level_structure, library)

    # Unwrap unnecessary nesting
    while (isinstance(stitched_structure, list)
           and len(stitched_structure) == 1
           and isinstance(stitched_structure[0], list)):
        stitched_structure = stitched_structure[0]

    # Normalize
    normalize_proof_structure(stitched_structure)
    remove_empty_lines(stitched_structure)

    numprems = label_and_count_premises(stitched_structure)

    total_lines = renumber_lines(stitched_structure)
    infer_missing_rule_heads(stitched_structure)
    apply_default_justifications(stitched_structure, numprems)

    if _env_flag("TFL_FITCH_VERIFIER_REREAD"):
        reread_records = collect_verifier_reread_candidates(
            stitched_structure,
            output_dir=output_dir,
            client=client,
            numprems=numprems,
            model=os.getenv("TFL_FITCH_VERIFIER_REREAD_MODEL") or None,
            timeout=max(1, _env_int("TFL_FITCH_VERIFIER_REREAD_TIMEOUT_SECONDS", 120) or 120),
            source_image=img,
        )
        reread_path = output_dir / "verifier_line_reread_candidates.json"
        reread_path.write_text(json.dumps(reread_records, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"   [VerifierReread] Wrote {len(reread_records)} record(s) to {reread_path.name}")

    strip_ocr_metadata(stitched_structure)

    # Save final structure
    final_json_path = output_dir / "final_structure.json"
    with open(final_json_path, "w", encoding="utf-8") as f:
        json.dump(stitched_structure, f, indent=2, ensure_ascii=False)

    line_count = total_lines - 1
    run_summary = (
        f"Fitch OCR pipeline: {len(bars)} bars, {line_count} lines, "
        f"{numprems} premises, geometry={selected_mode}"
    )

    return {"run_summary": run_summary, "line_count": line_count}


def _print_proof_checker_report(stitched_structure: list, numprems: int) -> None:
    print("\n--- Proof Checker ---")
    if not proof_checker_available:
        print("   [Verifier] Proof checker unavailable (imports failed).")
        return

    proofline_structure = structure_to_proof_lines(stitched_structure)
    if not proofline_structure:
        print("   [Verifier] No proof lines to verify.")
        return

    flattened = flatten_proof(proofline_structure, [])
    conc_guess = flattened[-1].wffstr if flattened else ""

    print(f"   [Verifier] Inferred premises: {numprems}")
    print(f"   [Verifier] Target conclusion: {conc_guess or '(empty)'}")

    result = check_proof(proofline_structure, numprems=numprems, conc=conc_guess)
    print(f"   [Verifier] Conclusion reached? {'Yes' if result.conc_reached else 'No'}")

    if result.issues:
        print("   [Verifier] Issues:")
        for issue in result.issues:
            print(f"      - {issue}")
    else:
        print("   [Verifier] No issues detected.")


def main():
    parser = argparse.ArgumentParser(description="Debug the Recursive Fitch OCR Pipeline")
    parser.add_argument(
        "filename",
        help="Name of file in raw_scans (e.g., quiz1_studentA_p1.jpg)",
    )
    args = parser.parse_args()

    raw_scan_path = RAW_SCANS_DIR / args.filename
    if not raw_scan_path.exists():
        print(f"Error: File not found at {raw_scan_path}")
        return

    output_base = DEBUG_FITCH_DIR / raw_scan_path.stem
    if output_base.exists():
        import shutil

        shutil.rmtree(output_base)
    output_base.mkdir(parents=True, exist_ok=True)

    print("=== Starting Fitch OCR Debug ===")
    print(f"Input: {raw_scan_path}")
    print(f"Output: {output_base}")

    result = run_fitch_ocr_pipeline(
        image_path=raw_scan_path,
        output_dir=output_base,
        client=vlm_client,
    )
    print(result["run_summary"])

    final_json_path = output_base / "final_structure.json"
    stitched_structure = json.loads(final_json_path.read_text(encoding="utf-8"))

    print("\n--- Final JSON Structure ---")
    print(json.dumps(stitched_structure, indent=2, ensure_ascii=False))
    print(f"Final stitched structure saved to {final_json_path}")

    premise_match = re.search(r"(\d+) premises", str(result.get("run_summary") or ""))
    numprems = int(premise_match.group(1)) if premise_match else 0
    _print_proof_checker_report(stitched_structure, numprems)
    
if __name__ == "__main__":
    main()
