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

from proofs_ocr.common.stitch import recursive_stitch, renumber_lines
from proofs_ocr.domains.shared import parse_vlm_json_response

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

    # Map Justifications to Verifier Codes (from verifier.py TFL_RULES)
    justification_map = {
        "premise": "Pr",
        "Premise": "Pr",
        "prem": "Pr",
        "Prem": "Pr",
        "pr": "Pr",
        "Pr": "Pr",
        "assumption": "Hyp",
        "assume": "Hyp",
        "hyp": "Hyp",
        "h": "Hyp",
        # Rules often OCR'd poorly
        "ve": "∨E",
        "vi": "∨I",
        "&e": "∧E",
        "&i": "∧I",
        ".e": "∧E",
        ".i": "∧I",
        "~e": "¬E",
        "~i": "¬I",
        "-e": "¬E",
        "-i": "¬I",
        "xe": "⊥E",   # Explosion
        "xi": "⊥I",
        "bot": "⊥",
        "bote": "⊥E",
        "boti": "⊥I",
    }

    contradiction_rule_heads = {"¬E", "⊥I"}
    known_rule_heads = {
        "Pr", "PR", "Hyp", "HYP", "premise", "Premise", "assumption", "Assumption",
        "∨E", "∨I", "∧E", "∧I", "¬E", "¬I", "→E", "→I", "↔E", "↔I",
        "vE", "vI", "&E", "&I", ".E", ".I", "~E", "~I", "-E", "-I",
        "AE", "AI", "MP", "MT", "DS", "DNE", "LEM", "DeM", "DEM", "X", "IP", "R",
    }

    def clean_wff(wff):
        if not wff: return ""
        if "<SUBPROOF_ID_" in str(wff):
            return str(wff)
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
        return compact in known_rule_heads

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
            normalized_head = clean_justification(head)
            normalized_full = clean_justification(justification_text)
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

    def snap_justification_rule_head(jstr):
        """Post-normalization snap: fix rule heads that are still not valid TFL rules."""
        if not jstr:
            return jstr

        valid_rules = {
            "Pr", "Hyp",
            "∧I", "∧E", "∨I", "∨E",
            "→I", "→E", "↔I", "↔E",
            "¬I", "¬E", "⊥I", "⊥E",
        }

        parts = jstr.strip().split(None, 1)
        head = parts[0]
        tail = parts[1] if len(parts) > 1 else ""

        # Already valid — nothing to do
        if head in valid_rules:
            return jstr

        full_text = jstr.strip()

        # --- Pattern: bare line ranges with no rule head ---
        # Two ranges like "3-6 7-10" or "3-6,7-10" → ↔I
        if re.fullmatch(r"\d+\s*-\s*\d+\s*[,;\s]+\s*\d+\s*-\s*\d+", full_text):
            return "↔I " + full_text

        # --- Known OCR confusions for rule heads ---
        head_fixes = {
            # ← read instead of ↔
            "←E": "↔E", "←I": "↔I", "←F": "↔E",
            # ↔F: F misread for E in ↔E
            "↔F": "↔E",
            # ± read instead of ↔I
            "±": "↔I",
            # handwritten ∧ looks like 8
            "8E": "∧E", "8I": "∧I",
            # ∈ in OCR output
            "∈E": "↔E", "∈I": "↔I", "∈S": "∧E",
            # epsilon/fallback
            "εE": "∧E", "εI": "∧I", "εS": "∧E",
            # BE/BI from ∧ → B confusion
            "BE": "∧E", "BI": "∧I",
            # XE/XI for ⊥E/⊥I (explosion)
            "XE": "⊥E", "XI": "⊥I",
        }

        if head in head_fixes:
            fixed = head_fixes[head]
            return (fixed + " " + tail).strip() if tail else fixed

        return jstr

    def clean_justification(jstr):
        if not jstr: return ""
        jstr = jstr.replace("->", "→").replace("<->", "↔").replace("<→", "↔")
        jstr = jstr.replace("←", "↔")  # Common OCR: left-arrow read instead of ↔
        jstr = re.sub(r"([¬∧∨→↔])\s+([IE])\b", r"\1\2", jstr)
        jstr = jstr.replace("&", "∧").replace("~", "¬")
        jstr = re.sub(r"\b([∧∨¬→↔A-Z]{1,3})\s*([IE])(?=\d)", r"\1\2 ", jstr)
        jstr = re.sub(r"\b(X|IP|R|MT|DS|DNE|LEM|DeM|DEM)(?=\d)", r"\1 ", jstr)
        jstr = re.sub(r"\b([∧∨¬→↔A-Z]{1,4})(\d)", r"\1 \2", jstr)
        
        # Split into Rule and Lines (e.g., "vE 2" -> "vE", "2")
        # We assume the Rule is the first "word"
        parts = jstr.strip().split(' ', 1)
        rule = parts[0].lower()
        remainder = parts[1] if len(parts) > 1 else ""

        # Map the rule if known
        if rule in justification_map:
            rule = justification_map[rule]
        else:
            # Try to fix casing for standard rules (e.g., "mp" -> "MP")
            # The verifier is Case Sensitive for rules like 'R', 'X', 'MT'
            rule = rule.upper() 
            # Fix specific case-sensitive ones from verifier.py
            if rule == "AE": rule = "∧E" # Common confusion
            if rule == "AI": rule = "∧I"
        
        return f"{rule} {remainder}".strip()

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
                    item["jstr"] = snap_justification_rule_head(
                        clean_justification(extracted_j)
                    )

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

    leftmost = sorted_boxes[0]
    x, y, w, h = leftmost

    is_left_edge = x <= max(24, int(img_w * 0.12))
    spans_most_height = h >= int(img_h * 0.65)
    is_narrow = w <= max(24, int(img_w * 0.06))

    if is_left_edge and spans_most_height and is_narrow:
        return sorted_boxes[1:]

    # If exactly one tall bar remains at the left edge, it is usually the
    # outer Fitch scaffold rather than a real subproof opener.  A bar that
    # is *not* at the left edge is far more likely to be a genuine subproof
    # bar, so we keep it.
    if len(sorted_boxes) == 1 and is_left_edge and spans_most_height and is_narrow:
        return []

    if len(sorted_boxes) >= 2:
        nx, ny, nw, nh = sorted_boxes[1]
        much_taller_than_next = h >= int(nh * 1.25)
        clearly_left_of_next = (x + w) < nx
        reasonably_close_column = (nx - x) <= max(120, int(img_w * 0.18))
        if spans_most_height and much_taller_than_next and clearly_left_of_next and reasonably_close_column:
            return sorted_boxes[1:]
    return sorted_boxes

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
# 3. VLM Interaction (Mockable)
# ==============================================================================

def build_ocr_prompt() -> str:
    return (
        "Transcribe the Fitch-style proof in this image into a JSON list of objects. "
        "Each object must have exactly these keys: 'line', 'wffstr', and 'jstr'. "
        "Use 'wffstr' only for the formula on that line. Use 'jstr' only for the justification. "
        "Do not place justification text inside 'wffstr'. "
        "If a line is a premise or assumption and no justification is printed, leave 'jstr' as an empty string. "
        "If you see a red box token like <SUBPROOF_ID_X>, copy it exactly as the entire 'wffstr' value for that line and leave 'jstr' empty. "
        "Return JSON only, with no markdown fences or explanation."
    )


def call_vlm(image_path: str, client) -> List[Dict]:
    """
    Real VLM call. expects client to be compatible with typical OpenAI/LocalAI SDKs.
    """
    prompt = build_ocr_prompt()

    print(f"   [VLM] Sending {os.path.basename(image_path)}...")

    # Mock response if no client provided
    if client is None:
        return [{"line": "?", "wffstr": "VLM_NOT_CONFIGURED", "jstr": "Error"}]

    try:
        import base64

        with open(image_path, "rb") as f:
            base64_image = base64.b64encode(f.read()).decode("utf-8")

        # LMStudioQwenClient exposes a simple chat(...) wrapper around the
        # OpenAI-style /chat/completions endpoint, so we mirror its expected payload.
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

        content = client.chat(messages)

        # Extract JSON from markdown blocks if present
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]

        return json.loads(content)

    except Exception as e:
        print(f"   [VLM Error] {e}")
        return []

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

    # 3. Save & OCR
    filename = f"crop_{node.id}.jpg"
    save_path = output_dir / filename
    cv2.imwrite(str(save_path), current_img)
    
    print(f"Processing Node {node.id} (Depth {node.depth})... Saved to {save_path.name}")
    
    # Call VLM
    ocr_result = call_vlm(str(save_path), client)
    
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
