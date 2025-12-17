"""
scripts/debug_fitch_ocr.py

Recursive "Chop-and-Mask" OCR for Fitch-style proofs.
1. Detects vertical subproof bars using CV (logic adapted from bold_fitch_lines.py).
2. Recursively crops and masks inner subproofs with <SUBPROOF_ID_X> tokens.
3. Sends each layer to the VLM (local or remote).
4. Stitches results into the nested format required by app/verifier.py.
"""

from __future__ import annotations

import sys
import os
import cv2
import numpy as np
import json
import re
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union

# Add project root to sys.path to import app modules
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

# Attempt to import project modules
try:
    # 1. Import the class
    from app.vlm_client import LMStudioQwenClient
    from app.verifier import ProofLine, check_proof, flatten_proof

    vlm_client = LMStudioQwenClient()
    print(f"   [Setup] Successfully loaded VLM Client: {type(vlm_client).__name__}")

    # 2. Instantiate the client (create the object)
    # This assumes your class takes no arguments or standard args.
    # If your other files do vlm_client = LMStudioQwenClient(url="..."), copy that matching logic.
    vlm_client = LMStudioQwenClient()
    proof_checker_available = True

except ImportError as e:
    print(f"Warning: Could not import app modules ({e}). Using mocks for VLM/Verifier.")
    vlm_client = None
    proof_checker_available = False

def renumber_lines(proof_structure: list, start_counter: int = 1) -> int:
    """
    Recursively walks the proof structure (lists of dicts/lists).
    Overwrites 'line' numbers in-place with a global, incrementing counter.
    Returns the next available line number.
    """
    current_counter = start_counter
    
    for item in proof_structure:
        if isinstance(item, list):
            # It's a subproof: Recurse into it.
            # The subproof will consume some numbers and return the next one.
            current_counter = renumber_lines(item, current_counter)
        elif isinstance(item, dict):
            # It's a proof line: Assign number and increment.
            item['line'] = current_counter
            current_counter += 1
            
    return current_counter


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
        "prem": "Pr",
        "pr": "Pr",
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

    def clean_wff(wff):
        if not wff: return ""
        # 1. Replace multi-char operators first to avoid breaking them
        wff = wff.replace("->", "→").replace("<->", "↔")
        
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

    def clean_justification(jstr):
        if not jstr: return ""
        
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
                # 1. Clean Formulas
                item["wffstr"] = clean_wff(item.get("wffstr", ""))
                
                # 2. Clean Justification Text
                raw_j = str(item.get("jstr", "")).strip()
                
                # 3. Handle Implicit Assumptions
                # If this is the very first line of a subproof (List), it MUST be an assumption.
                if is_subproof_start and index == 0 and not raw_j:
                    item["jstr"] = "Hyp"
                else:
                    item["jstr"] = clean_justification(raw_j)

    # Start the recursion
    recursive_normalize(structure)
    return structure
    
# ==============================================================================
# 1. Computer Vision Core (Adapted from bold_fitch_lines.py)
# ==============================================================================

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

    # Filter out noise (very short lines)
    final_boxes = [b for b in merged if b[3] > 50]
    return final_boxes

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

def call_vlm(image_path: str, client) -> List[Dict]:
    """
    Real VLM call. expects client to be compatible with typical OpenAI/LocalAI SDKs.
    """
    prompt = (
        "Transcribe the logic proof in this image into a JSON list of objects. "
        "Each object must have 'line' (number or empty), 'wffstr' (formula), and 'jstr' (justification). "
        "IMPORTANT: If you see a red box with a token like <SUBPROOF_ID_X>, "
        "transcribe it EXACTLY as the 'wffstr' for that line. "
        "Do not interpret or correct the token."
    )

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
# 5. Stitching (Inline version of the logic)
# ==============================================================================

def recursive_stitch(current_list, library):
    resolved = []
    token_pattern = re.compile(r"<SUBPROOF_ID_([A-Za-z0-9_]+)>")
    
    for item in current_list:
        # Handle string match (rare)
        if isinstance(item, str):
            match = token_pattern.search(item)
            if match and match.group(1) in library:
                resolved.append(recursive_stitch(library[match.group(1)], library))
            continue
            
        # Handle Dict
        if isinstance(item, dict):
            token_found = False
            # Search values
            for k, v in item.items():
                if isinstance(v, str):
                    match = token_pattern.search(v)
                    if match and match.group(1) in library:
                        # Replace this line with the subproof list
                        child_list = library[match.group(1)]
                        # Important: The subproof list is a LIST of lines. 
                        # We append the LIST itself to make the structure [Line, [Sub], Line]
                        resolved.append(recursive_stitch(child_list, library))
                        token_found = True
                        break
            if token_found:
                continue
                
            # Regular line
            # Convert to Verifier format if needed, here just keeping dict for simplicity
            resolved.append(item)
            
    return resolved

def label_and_count_premises(proof_structure: List[Union[Dict, List]]) -> int:
    """
    Walks the top-level of the stitched proof structure.
    If a line has an empty justification ('jstr') and appears before any 
    subproofs or justified lines, it is labeled as 'Premise'.
    
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
                item["jstr"] = "Premise" # Inject the label
                premise_count += 1
            
            # If it already has a justification (e.g., "Taut Con"), 
            # the premise block is over.
            elif jstr.lower() != "premise": 
                break
                
    return premise_count

# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Debug the Recursive Fitch OCR Pipeline")
    parser.add_argument("filename", help="Name of file in data/raw_scans (e.g., quiz1_studentA_p1.jpg)")
    args = parser.parse_args()

    # Setup Paths
    raw_scan_path = Path(project_root) / "data" / "raw_scans" / args.filename
    if not raw_scan_path.exists():
        print(f"Error: File not found at {raw_scan_path}")
        return

    output_base = Path(project_root) / "data" / "debug_fitch" / raw_scan_path.stem
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

    # Identify Premises ---
    # We do this BEFORE renumbering or flattening, while structure is clear.
    print(f"   [Logic] Inferring premises from top-level structure...")
    numprems = label_and_count_premises(stitched_structure)
    print(f"   [Logic] Identified {numprems} premises.")
    
    # B. Renumber: Fix the line numbers 1..N
    total_lines = renumber_lines(stitched_structure)
    print(f"   [Renumber] Proof contains {total_lines - 1} lines.")

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
