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
from proofs_ocr.domains.shared import parse_vlm_json_response
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
    dilate_kernel = np.ones((1, 6), np.uint8)
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


def detect_vertical_bars(img: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """
    Returns a list of (x, y, w, h) tuples for vertical proof bars.
    """
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
    boxes.sort(key=lambda b: (b[0], b[1])) # Sort by X then Y

    merged = []
    if not boxes:
        return []

    curr_x, curr_y, curr_w, curr_h = boxes[0]
    for nx, ny, nw, nh in boxes[1:]:
        # Merge if close in X (same column) and Y (close vertically)
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

    # Post-merge: split bars that have significant vertical gaps (e.g. two
    # adjacent subproofs at the same X position that were erroneously merged).
    split = _split_bars_on_gaps(img, merged)

    # Filter out noise (very short lines)
    final_boxes = [b for b in split if b[3] > 50]
    return final_boxes


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
    
    # 2. Draw Red Box for Token (Attention)
    token = f"<SUBPROOF_ID_{child_node.id}>"
    cx, cy = x1 + 150, y1 + h // 2 # Place token somewhat to the right of the bar
    
    # Calculate text size for centering
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.0
    thickness = 2
    (tw, th), _ = cv2.getTextSize(token, font, scale, thickness)
    
    # Draw red border around token
    pad = 10
    cv2.rectangle(masked, (cx - pad, cy - th - pad), (cx + tw + pad, cy + pad), (0, 0, 255), 2)
    
    # 3. Write Token
    cv2.putText(masked, token, (cx, cy), font, scale, (0, 0, 255), thickness)
    
    return masked


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

#: When True, use a stripped-down prompt for leaf crops.  The full prompt
#: stacks ~6 instruction blocks which may overwhelm a 4b model; this variant
#: keeps only the essentials.  Toggle to compare regression scores.
USE_SIMPLE_PROMPT = False

#: Model routing for small leaf crops.  When a leaf crop has ≤ this many
#: expected lines, the pipeline routes it to a larger model (27b) for better
#: handwriting recognition.  Set to 0 to disable routing.
SMALL_LEAF_MODEL_ROUTING_THRESHOLD = 3

#: LM Studio model identifier for the larger VLM used on small leaf crops.
#: This is Qwen 3.6 27B (qwen/qwen3.6-27b), a significantly more capable
#: model than the default 4b for handwriting recognition on small crops.
#: Loaded from LMSTUDIO_MODEL_27B env var.
def _get_model_27b() -> str:
    return os.getenv("LMSTUDIO_MODEL_27B", "qwen/qwen3.6-27b").strip()

def _model_27b_available() -> bool:
    """Check if a 27b model is specified and differs from the default 4b."""
    return bool(_get_model_27b().strip())


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int | None = None) -> int | None:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _get_27b_timeout_seconds() -> int:
    return max(1, _env_int("TFL_FITCH_27B_TIMEOUT_SECONDS", 600) or 600)


def _get_27b_max_tokens() -> int | None:
    return _env_int("TFL_FITCH_27B_MAX_TOKENS")


def _get_27b_extra_payload() -> Dict[str, object]:
    extra_payload: Dict[str, object] = {}
    if _env_flag("TFL_FITCH_27B_ENABLE_THINKING_FALSE"):
        extra_payload["chat_template_kwargs"] = {"enable_thinking": False}
    return extra_payload


def _get_4b_fullwidth_max_lines() -> int:
    """Return max nested leaf line count for default-model full-width crops."""
    return max(0, _env_int("TFL_FITCH_4B_FULLWIDTH_MAX_LINES", 99) or 0)


def build_ocr_prompt(
    *,
    has_children: bool = True,
    highlighted_box: bool = False,
    expected_lines: int | None = None,
) -> str:
    """Build the VLM OCR prompt for a Fitch proof crop.

    Parameters
    ----------
    has_children:
        When *True* (the default), the crop contains red-box tokens that
        reference masked child subproofs, so the prompt instructs the VLM to
        copy those tokens verbatim.  When *False* the crop is a leaf node with
        no red stamps, so token instructions are omitted entirely — this
        prevents the VLM from hallucinating placeholder tokens that don't
        exist in the image.
    highlighted_box:
        When *True*, the image is a parent-context crop with a bright green
        rectangle drawn around the target subproof region.  The prompt
        instructs the VLM to OCR only the lines inside that box.
    expected_lines:
        When provided (and *has_children* is False), injects an explicit line
        count into the prompt so the VLM knows exactly how many line objects
        to produce.  This is derived from horizontal projection profiling of
        the crop and helps prevent the VLM from hallucinating extra lines.
    """
    # ---- Simple prompt path ------------------------------------------------
    if USE_SIMPLE_PROMPT and not has_children:
        parts = [
            "Transcribe this Fitch proof image into a JSON array.",
            "Each element: {\"line\": <num>, \"wffstr\": \"<formula>\", \"jstr\": \"<justification>\"}.",
            "wffstr = formula on the left.  jstr = rule + line refs on the right.  Keep them separate.",
        ]
        if expected_lines is not None and expected_lines >= 1:
            parts.append(f"Output exactly {expected_lines} line(s).")
        parts.append("Return JSON only. No markdown.")
        return " ".join(parts)

    # ---- Full prompt path (non-leaf or simple disabled) --------------------
    base = (
        "Transcribe the Fitch-style proof in this image into a JSON list of objects. "
        "Each object must have exactly these keys: 'line', 'wffstr', and 'jstr'. "
        "Use 'wffstr' only for the formula on that line. Use 'jstr' only for the justification. "
        "Do not place justification text inside 'wffstr'. "
        "If a line is a premise or assumption and no justification is printed, leave 'jstr' as an empty string. "
    )
    # Handwriting Translation Guide — helps the VLM correctly interpret
    # common sloppy handwritten symbols that it would otherwise misread
    # as digits or punctuation.  Validated on Q4 crop_3 (K∧W subproof)
    # with the 27b model: baseline+guide passed 2/2 formulas, 2/2 justifications.
    base += (
        "Handwriting Translation Guide (CRITICAL): "
        "Students have sloppy handwriting. Use these known distortions to correct your visual reading: "
        "- A sloppy K frequently looks like '1 <' or a vertical line next to a '<'. "
        "- A sloppy ampersand & (conjunction) frequently looks like an '8'. "
        "- A sloppy S frequently looks like a '5'. "
    )
    if has_children:
        base += (
            "If you see a red box token like <SUBPROOF_ID_X>, copy it exactly as the entire 'wffstr' value for that line and leave 'jstr' empty. "
        )
    if highlighted_box:
        base += (
            "OCR only the proof lines inside the bright green highlighted box. "
            "There are typically 1-3 lines inside the box. "
            "Ignore all other text outside the green box. "
        )
    if not has_children:
        base += "Transcribe only the lines actually visible in this image. "
        if expected_lines is not None and expected_lines >= 1:
            base += f"This image contains exactly {expected_lines} proof line(s). Output exactly {expected_lines} line objects. "
            if expected_lines <= 4:
                base += (
                    f"Each of the {expected_lines} line(s) has a formula on the LEFT and a justification on the RIGHT. "
                    "Formula goes in 'wffstr', justification goes in 'jstr'. Keep them separate. "
                )
    base += "Return JSON only, with no markdown fences or explanation."
    return base


def build_27b_fullwidth_prompt(expected_lines: int | None = None) -> str:
    """Build the proven manual prompt for 27b full-width crop OCR.

    This prompt was validated on Q4 crop_3 (K∧W subproof) with the 27b model
    and produces correct results after normalization.  It assumes the crop
    includes the line-number gutter (full image width, restricted Y range).
    """
    prompt = (
        "Role: You are a logic quiz transcriber performing OCR on subproof chunks "
        "cropped out from the middle of handwritten Fitch-style natural deduction proofs.\n"
        "Context: You will be given cropped images of specific subproofs (beginning with "
        "an assumption with no justification above a horizontal line, and deriving various "
        "consequences from that line and earlier lines). So don't worry that those are "
        "missing, and remember that the top line in the crop you are seeing might not be "
        "line 1 in the proof.\n"
        "Task: Extract the logical formulas and their justifications.\n"
        "Output format: JSON array of objects with keys 'line', 'wffstr', 'jstr'.\n"
        "Handwriting Translation Guide (CRITICAL):\n"
        "Students have sloppy handwriting. Use the following known distortions to correct "
        "your visual reading:\n"
        "- A sloppy K frequently looks like '1 <' or a vertical line next to a '<'.\n"
        "- A sloppy ampersand & (conjunction) frequently looks like an '8'.\n"
        "- A sloppy S frequently looks like a '5'.\n"
    )
    if expected_lines is not None and expected_lines >= 1:
        prompt += (
            f"\nThis image contains exactly {expected_lines} proof line(s). "
            f"Output exactly {expected_lines} line objects.\n"
        )
    prompt += "Return JSON only, with no markdown fences or explanation."
    return prompt


def call_vlm(
    image_path: str,
    client,
    *,
    has_children: bool = True,
    highlighted_box: bool = False,
    expected_lines: int | None = None,
    model_override: str | None = None,
    prompt_override: str | None = None,
) -> Tuple[List[Dict], str]:
    """
    Real VLM call. expects client to be compatible with typical OpenAI/LocalAI SDKs.

    Returns
    -------
    Tuple[List[Dict], str]
        (parsed OCR result, model thinking/reasoning text).

    Parameters
    ----------
    model_override:
        When set, passed through to ``client.chat(model=...)`` to route the
        request to a different LM Studio model (e.g. the 27b variant for
        small leaf crops).
    prompt_override:
        When set, use this string as the prompt instead of building one via
        :func:`build_ocr_prompt`.  Used for the 27b full-width crop path.
    """
    if prompt_override:
        prompt = prompt_override
    else:
        prompt = build_ocr_prompt(
        has_children=has_children,
        highlighted_box=highlighted_box,
        expected_lines=expected_lines,
    )
    is_27b_route = bool(model_override)
    if is_27b_route and _env_flag("TFL_FITCH_27B_NO_THINK"):
        prompt = f"{prompt} /no_think"

    model_tag = f" (model={model_override})" if model_override else ""
    print(f"   [VLM] Sending {os.path.basename(image_path)}{model_tag}...")

    # Mock response if no client provided
    if client is None:
        return [{"line": "?", "wffstr": "VLM_NOT_CONFIGURED", "jstr": "Error"}], ""

    try:
        image_path_obj = Path(image_path)
        if is_27b_route and _env_flag("TFL_FITCH_27B_SINGLE_USER_MESSAGE"):
            messages = client.build_single_user_image_message(image_path_obj, prompt)
        elif hasattr(client, "build_messages_for_prompt"):
            messages = client.build_messages_for_prompt(image_path_obj, "Provide the transcription as instructed.", system_prompt=prompt)
        else:
            import base64

            with open(image_path, "rb") as f:
                base64_image = base64.b64encode(f.read()).decode("utf-8")

            messages = [
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Provide the transcription as instructed."},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                        },
                    ],
                },
            ]

        # Use extended timeout for larger model requests
        extra_kwargs = {}
        if is_27b_route:
            extra_kwargs["timeout"] = _get_27b_timeout_seconds()
            max_tokens = _get_27b_max_tokens()
            if max_tokens is not None:
                extra_kwargs["max_tokens"] = max_tokens
            extra_payload = _get_27b_extra_payload()
            if extra_payload:
                extra_kwargs["extra_payload"] = extra_payload
        content = client.chat(messages, model=model_override, **extra_kwargs)

        # Extract thinking/reasoning text from the VLM client (populated by
        # LMStudioQwenClient.chat() from the reasoning_content field).
        thinking = getattr(client, "last_thinking", "") or ""
        metadata = getattr(client, "last_response_metadata", {}) or {}
        if metadata:
            print(
                "   [VLM] Metadata: "
                f"elapsed={metadata.get('elapsed_s')}s "
                f"thinking={metadata.get('thinking_chars')} "
                f"content={metadata.get('content_chars')} "
                f"finish={metadata.get('finish_reason')}"
            )

        # Extract JSON from markdown blocks if present
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]

        return json.loads(content), thinking

    except Exception as e:
        print(f"   [VLM Error] {e}")
        return [], ""

# ==============================================================================
# 4. Orchestration
# ==============================================================================

def process_node_recursive(node: SubproofNode, img: np.ndarray, output_dir: Path, library: Dict, client) -> List[Dict]:
    """
    Bottom-up processing:
    1. Process children first.
    2. Mask children on current image.
    3. Save current image -> OCR -> Store result in library.
    """
    
    # 1. Prepare the image for THIS node
    # Start with the raw crop of this node's area
    if node.id == "ROOT":
        current_img = img.copy()

        # Blot out line numbers to the LEFT of the outermost bar so the VLM
        # doesn't read them as formula content.  When we have detected bars,
        # the leftmost bar's X position marks the boundary — everything left
        # of that is line numbers (or blank margin).
        if node.children:
            leftmost_bar_x = min(c.box[0] for c in node.children)
            if leftmost_bar_x > 0:
                current_img[:, :leftmost_bar_x] = 255  # fill with white
    else:
        if node.parent and node.parent.id == "ROOT":
            # Extend crop to image bottom to capture post-subproof conclusion
            # lines that sit below the outermost Fitch bar.  Only done for
            # direct children of ROOT because deeper nesting never has
            # post-subproof content outside the parent bar's Y span.
            extended_h = img.shape[0] - node.box[1]
            current_img = crop_region(img, (node.box[0], node.box[1], node.box[2], extended_h))
        else:
            current_img = crop_region(img, node.box)

    # 2. Process Children (Deepest first)
    # We sort children by vertical position just in case, though order doesn't strictly matter for masking
    sorted_children = sorted(node.children, key=lambda c: c.box[1])
    
    for child in sorted_children:
        # RECURSE: This will populate the library with the child's JSON
        process_node_recursive(child, img, output_dir, library, client)
        
        # MASK: Now we hide the child on our current image
        # Note: We must translate child coordinates to the cropped frame of the parent
        if node.id == "ROOT":
            # Child coordinates are already global
            current_img = mask_subproof_in_parent(current_img, child)
        else:
            # Adjust child global coords to be relative to parent crop
            px, py, _, _ = node.box
            # Create a temporary relative node for masking
            rel_box = (child.box[0] - px + 10, child.box[1] - py + 10, child.box[2], child.box[3])
            rel_child = SubproofNode(child.id, rel_box)
            current_img = mask_subproof_in_parent(current_img, rel_child)

    # Capture the original crop height before any letterboxing changes it,
    # since the context-crop decision should be based on the real content size.
    original_crop_h = current_img.shape[0]
    is_leaf = not sorted_children

    # 3. Save & OCR
    filename = f"crop_{node.id}.jpg"
    save_path = output_dir / filename

    # Apply letterboxing to small leaf crops so the VLM's internal resizing
    # doesn't distort the handwriting.  Context crops are already large
    # enough (they embed the parent image) so they don't need letterboxing.
    if is_leaf and original_crop_h < SMALL_CROP_THRESHOLD:
        current_img = letterbox_crop(current_img)

    cv2.imwrite(str(save_path), current_img)

    # --- Context-aware crop for small leaf subproofs ---
    # DISABLED (2026-04-25): The VLM (qwen3-vl-4b) ignores the green box
    # instruction and reads the entire parent image, producing 2–3× more
    # lines than expected.  Using letterboxed tight crops instead limits
    # what the model can physically see.
    use_context_crop = False

    if False and use_context_crop:
        parent_node = node.parent
        px, py, pw, ph = parent_node.box

        # Build the parent's unmasked crop the same way it would be built
        # in process_node_recursive for the parent.
        if parent_node.parent and parent_node.parent.id == "ROOT":
            p_extended_h = img.shape[0] - py
            parent_crop = crop_region(img, (px, py, pw, p_extended_h))
        else:
            parent_crop = crop_region(img, parent_node.box)

        # Compute the offset that crop_region applied (padding-aware).
        p_offset_x = max(0, px - 10)
        p_offset_y = max(0, py - 10)

        # Draw the highlight rectangle on the parent crop.
        context_img = build_context_crop(parent_crop, node.box, (p_offset_x, p_offset_y))

        # Save context crop for debugging comparison.
        context_filename = f"crop_{node.id}_context.jpg"
        context_save_path = output_dir / context_filename
        cv2.imwrite(str(context_save_path), context_img)

        print(
            f"Processing Node {node.id} (Depth {node.depth})... "
            f"Using context-aware crop ({original_crop_h}px < {SMALL_CROP_THRESHOLD}px)"
        )

        ocr_result, _thinking = call_vlm(
            str(context_save_path), client,
            has_children=False,
            highlighted_box=True,
        )
    else:
        print(f"Processing Node {node.id} (Depth {node.depth})... Saved to {save_path.name}")

        # For leaf nodes, optionally estimate expected line count from
        # horizontal projection profiling to constrain the VLM and truncate
        # hallucinations.  Controlled by USE_LINE_COUNT_CONSTRAINT flag.
        expected_lines = None
        if USE_LINE_COUNT_CONSTRAINT and is_leaf:
            expected_lines = estimate_line_count(current_img)
            print(f"   [LineCount] Node {node.id}: detected {expected_lines} text line(s)")

        # Model routing: use the larger 27b model for small leaf crops where
        # the 4b model struggles with handwriting recognition.
        model_override = None
        prompt_override = None
        vlm_image_path = str(save_path)  # default: tight (letterboxed) crop

        if (
            is_leaf
            and SMALL_LEAF_MODEL_ROUTING_THRESHOLD > 0
            and expected_lines is not None
            and expected_lines <= SMALL_LEAF_MODEL_ROUTING_THRESHOLD
            and _model_27b_available()
        ):
            model_override = _get_model_27b()
            print(f"   [ModelRoute] Node {node.id}: routing to {model_override} (<={SMALL_LEAF_MODEL_ROUTING_THRESHOLD} lines)")

            # Generate a full-width crop for 27b — preserves the line-number
            # gutter for citation context.  No letterboxing needed since the
            # aspect ratio is already reasonable at full image width.
            fw_img = crop_full_width(img, node)
            fw_filename = f"crop_{node.id}_fullwidth.jpg"
            fw_save_path = output_dir / fw_filename
            cv2.imwrite(str(fw_save_path), fw_img)
            print(f"   [FullWidth] Node {node.id}: saved full-width crop to {fw_filename}")

            vlm_image_path = str(fw_save_path)
            prompt_override = build_27b_fullwidth_prompt(expected_lines=expected_lines)
        else:
            four_b_fullwidth_max_lines = _get_4b_fullwidth_max_lines()
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
                fw_img = crop_full_width(img, node)
                fw_filename = f"crop_{node.id}_4b_fullwidth.jpg"
                fw_save_path = output_dir / fw_filename
                cv2.imwrite(str(fw_save_path), fw_img)
                vlm_image_path = str(fw_save_path)

        # Call VLM — leaf nodes (no masked children) get a simpler prompt that
        # omits the <SUBPROOF_ID_X> instructions, preventing hallucinated tokens.
        ocr_result, thinking_text = call_vlm(
            vlm_image_path, client,
            has_children=bool(sorted_children),
            expected_lines=expected_lines,
            model_override=model_override,
            prompt_override=prompt_override,
        )

        # Save thinking/reasoning text for debugging (especially useful for
        # diagnosing when the 27b model gets stuck in infinite thinking loops).
        if thinking_text:
            thinking_path = output_dir / f"thinking_{node.id}.txt"
            thinking_path.write_text(thinking_text, encoding="utf-8")
            print(f"   [Thinking] Node {node.id}: saved {len(thinking_text)} chars to {thinking_path.name}")

        # Truncate if VLM returned more dict items than expected.
        # Only applies when USE_LINE_COUNT_CONSTRAINT is enabled and we
        # have a line estimate for this leaf node.
        if USE_LINE_COUNT_CONSTRAINT and is_leaf and expected_lines is not None:
            dict_items = [item for item in ocr_result if isinstance(item, dict)]
            if len(dict_items) > expected_lines:
                print(
                    f"   [Truncate] Node {node.id}: VLM returned "
                    f"{len(dict_items)} lines, truncating to {expected_lines}"
                )
                # Keep only the first expected_lines dict items (preserve
                # any non-dict entries like sub-lists, though leaf nodes
                # shouldn't have them).
                kept = []
                for item in ocr_result:
                    if isinstance(item, dict):
                        if len(kept) < expected_lines:
                            kept.append(item)
                    else:
                        kept.append(item)
                ocr_result = kept

    # Save raw VLM response for debug
    with open(output_dir / f"response_{node.id}.json", "w") as f:
        json.dump(ocr_result, f, indent=2)

    # Add to library
    library[node.id] = ocr_result

    return ocr_result

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

    # 2. Detect bars & build tree
    bars = detect_vertical_bars(img)
    bars = filter_scaffold_bars(bars, img_w=img.shape[1], img_h=img.shape[0])

    # Draw debug bars
    debug_img = img.copy()
    for i, (x, y, w, h) in enumerate(bars):
        cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(debug_img, str(i + 1), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imwrite(str(output_dir / "debug_bars.jpg"), debug_img)

    tree_root = build_proof_tree(bars, img.shape[1], img.shape[0])

    # 3. Recursive process
    library: Dict = {}
    process_node_recursive(tree_root, img, output_dir, library, client=client)

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

    # Save final structure
    final_json_path = output_dir / "final_structure.json"
    with open(final_json_path, "w", encoding="utf-8") as f:
        json.dump(stitched_structure, f, indent=2, ensure_ascii=False)

    line_count = total_lines - 1
    run_summary = f"Fitch OCR pipeline: {len(bars)} bars, {line_count} lines, {numprems} premises"

    return {"run_summary": run_summary, "line_count": line_count}


def main():
    parser = argparse.ArgumentParser(description="Debug the Recursive Fitch OCR Pipeline")
    parser.add_argument(
        "filename",
        help="Name of file in raw_scans (e.g., quiz1_studentA_p1.jpg)",
    )
    args = parser.parse_args()

    # Setup Paths
    raw_scan_path = RAW_SCANS_DIR / args.filename
    if not raw_scan_path.exists():
        print(f"Error: File not found at {raw_scan_path}")
        return

    output_base = DEBUG_FITCH_DIR / raw_scan_path.stem
    if output_base.exists():
        import shutil
        shutil.rmtree(output_base)
    output_base.mkdir(parents=True, exist_ok=True)

    print(f"=== Starting Fitch OCR Debug ===")
    print(f"Input: {raw_scan_path}")
    print(f"Output: {output_base}")

    # 1. Load Image
    img = cv2.imread(str(raw_scan_path))
    
    # 2. Detect Bars & Build Tree
    bars = detect_vertical_bars(img)
    bars = filter_scaffold_bars(bars, img_w=img.shape[1], img_h=img.shape[0])
    print(f"Detected {len(bars)} vertical bars.")
    
    # Draw debug bars
    debug_img = img.copy()
    for i, (x, y, w, h) in enumerate(bars):
        cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(debug_img, str(i+1), (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imwrite(str(output_base / "debug_bars.jpg"), debug_img)

    tree_root = build_proof_tree(bars, img.shape[1], img.shape[0])
    
    # 3. Recursive Process
    library = {}
    print("\n--- Recursive Processing (Bottom-Up) ---")
    process_node_recursive(tree_root, img, output_base, library, client=vlm_client)
    
    # 4. Stitch & Renumber
    print("\n--- Stitching Results ---")
    
    # HEURISTIC: If ROOT is empty/garbage but has children, 
    # use the first child (ID_1) as the main proof.
    # We check if ROOT has any actual content strings or just empty ones.
    root_has_content = any(line.get("wffstr", "").strip() for line in library.get("ROOT", []))
    
    if not root_has_content and "1" in library:
        print("   [Fix] ROOT was empty. Promoting 'ID_1' to top level.")
        top_level_structure = library["1"]
    else:
        top_level_structure = library.get("ROOT", [])

    # A. Stitch: Replace token strings with actual lists
    stitched_structure = recursive_stitch(top_level_structure, library)

    # Unwrap unnecessary nesting 
    # The stitching or VLM results often create layers like [[[ {line1}, ... ]]].
    # We need to peel these layers off until we find the actual list of logic lines.
    while (isinstance(stitched_structure, list) 
           and len(stitched_structure) == 1 
           and isinstance(stitched_structure[0], list)):
        print("   [Structure] Unwrapping nested list layer...")
        stitched_structure = stitched_structure[0]
    
    # NEW: Normalize VLM Output ---
    print("   [Logic] Normalizing symbols and justifications...")
    normalize_proof_structure(stitched_structure)
    remove_empty_lines(stitched_structure)

    # Identify Premises ---
    # We do this BEFORE renumbering or flattening, while structure is clear.
    print(f"   [Logic] Inferring premises from top-level structure...")
    numprems = label_and_count_premises(stitched_structure)
    print(f"   [Logic] Identified {numprems} premises.")
    
    # B. Renumber: Fix the line numbers 1..N
    total_lines = renumber_lines(stitched_structure)
    print(f"   [Renumber] Proof contains {total_lines - 1} lines.")

    infer_missing_rule_heads(stitched_structure)

    # C. Backfill remaining blank justifications using the inferred premise count.
    print("   [Logic] Filling blank justifications with Pr/Hyp defaults...")
    apply_default_justifications(stitched_structure, numprems)
    
    print("\n--- Final JSON Structure ---")
    print(json.dumps(stitched_structure, indent=2, ensure_ascii=False))

    # Save Final
    final_json_path = output_base / "final_structure.json"
    with open(final_json_path, "w") as f:
        json.dump(stitched_structure, f, indent=2)
        
    print(f"Final stitched structure saved to {final_json_path}")

    # 5. Verification
    print("\n--- Proof Checker ---")
    if not proof_checker_available:
        print("   [Verifier] Proof checker unavailable (imports failed).")
        return

    proofline_structure = structure_to_proof_lines(stitched_structure)
    if not proofline_structure:
        print("   [Verifier] No proof lines to verify.")
        return

    flattened = flatten_proof(proofline_structure, [])
    
    # We grab the conclusion from the very last line of the flattened proof
    conc_guess = flattened[-1].wffstr if flattened else ""
    
    print(f"   [Verifier] Inferred premises: {numprems}")
    print(f"   [Verifier] Target conclusion: {conc_guess or '(empty)'}")

    # Pass the explicitly counted 'numprems' to the verifier
    result = check_proof(proofline_structure, numprems=numprems, conc=conc_guess)
    print(f"   [Verifier] Conclusion reached? {'Yes' if result.conc_reached else 'No'}")

    if result.issues:
        print("   [Verifier] Issues:")
        for issue in result.issues:
            print(f"      - {issue}")
    else:
        print("   [Verifier] No issues detected.")
    
if __name__ == "__main__":
    main()
