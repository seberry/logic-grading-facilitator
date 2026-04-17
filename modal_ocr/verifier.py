from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import cv2
import numpy as np

from modal_ocr.artifacts import UNARY_VERIFIER_FILENAME
from modal_ocr.shared import SubproofNode

UNARY_PREFIX_CHARS = {"¬", "□", "⋄"}
UNARY_SENSITIVE_RULE_HEADS = {"MC", "DN", "Def⋄"}
SHORT_FORMULA_REREAD_RULE_HEADS = {"MC", "DN", "Def⋄", "□I", "□E"}
UNARY_VERIFIER_RULE_HEADS = {"MC", "DN", "Def⋄", "□I", "□E"}


def leading_unary_prefix(formula_text: str) -> str:
    prefix_chars: List[str] = []
    for char in formula_text:
        if char not in UNARY_PREFIX_CHARS:
            break
        prefix_chars.append(char)
    return "".join(prefix_chars)


def has_top_level_binary_connective(formula_text: str) -> bool:
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


def should_verify_unary_formula(
    entry: Dict[str, Any],
    *,
    normalize_formula_text: Callable[[Any, str], str],
    normalize_justification_text: Callable[[Any], str],
) -> bool:
    scope_role = str(entry.get("scope_role") or "derived_line").strip()
    if scope_role in {"subproof_token", "strict_open"}:
        return False

    formula_text = normalize_formula_text(entry.get("formula_text") or entry.get("wffstr") or "", scope_role)
    if not formula_text:
        return False

    prefix = leading_unary_prefix(formula_text)
    body = formula_text[len(prefix) :].strip()
    if body and has_top_level_binary_connective(body):
        return False

    justification = normalize_justification_text(entry.get("justification_text") or entry.get("jstr") or "")
    rule_head = justification.split(" ", 1)[0] if justification else ""

    if formula_text.startswith(tuple(UNARY_PREFIX_CHARS)):
        return True
    if "□" in formula_text or "⋄" in formula_text:
        return True
    if prefix.count("¬") >= 2:
        return True
    return rule_head in UNARY_VERIFIER_RULE_HEADS


def build_node_lookup(node: SubproofNode) -> Dict[str, SubproofNode]:
    lookup = {node.id: node}
    for child in node.children:
        lookup.update(build_node_lookup(child))
    return lookup


def verifier_node_crop_image(
    *,
    node: SubproofNode,
    full_img: np.ndarray,
    crop_region: Callable[..., np.ndarray],
) -> np.ndarray:
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


def candidate_band_indices(
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
        indices.append(min(max(0, line_order), band_count - 1))
    return indices


def confidence_score(value: Any) -> float:
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
    prefix = leading_unary_prefix(formula_text)
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


def _same_modal_class(left: str, right: str) -> bool:
    left_prefix = leading_unary_prefix(left)
    right_prefix = leading_unary_prefix(right)
    if "□" in left_prefix:
        return "□" in right_prefix
    if "⋄" in left_prefix:
        return "⋄" in right_prefix
    return "□" not in right_prefix and "⋄" not in right_prefix


def _is_structurally_close_unary_formula(
    original: str,
    candidate: str | None,
    *,
    unary_formula_skeleton: Callable[[str], str],
) -> bool:
    if not candidate:
        return False
    return unary_formula_skeleton(original) == unary_formula_skeleton(candidate) and _same_modal_class(original, candidate)


def best_verifier_attempt(
    *,
    original_formula: str,
    expected_line_order: int | None,
    attempts: List[Dict[str, Any]],
    unary_formula_skeleton: Callable[[str], str],
) -> Dict[str, Any]:
    if not attempts:
        return {}

    def score(attempt: Dict[str, Any]) -> Tuple[int, ...]:
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
            1
            if _is_structurally_close_unary_formula(
                original_formula,
                candidate_formula,
                unary_formula_skeleton=unary_formula_skeleton,
            )
            else 0,
            -prefix_distance,
            distance_score,
            max(
                confidence_score(attempt.get("structured_counts", {}).get("confidence")),
                confidence_score(attempt.get("forced_confidence")),
            ),
        )

    return max(attempts, key=score)


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


def merge_margin_candidate_boxes(boxes: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int]]:
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


def margin_anchor_strip_bounds(
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


def build_structured_unary_verifier_prompt(
    entry: Dict[str, Any],
    *,
    normalize_formula_text: Callable[[Any, str], str],
    normalize_justification_text: Callable[[Any], str],
) -> str:
    scope_role = str(entry.get("scope_role") or "derived_line").strip()
    formula_text = normalize_formula_text(entry.get("formula_text") or entry.get("wffstr") or "", scope_role)
    justification = normalize_justification_text(entry.get("justification_text") or entry.get("jstr") or "")
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


def build_forced_choice_unary_verifier_prompt(
    entry: Dict[str, Any],
    candidates: List[str],
    *,
    normalize_formula_text: Callable[[Any, str], str],
    normalize_justification_text: Callable[[Any], str],
) -> str:
    scope_role = str(entry.get("scope_role") or "derived_line").strip()
    formula_text = normalize_formula_text(entry.get("formula_text") or entry.get("wffstr") or "", scope_role)
    justification = normalize_justification_text(entry.get("justification_text") or entry.get("jstr") or "")
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


def _parse_full_image_line_anchors(
    raw: str,
    total_lines: int,
    *,
    parse_unary_recheck_choice: Callable[[str], Dict[str, Any]],
) -> Dict[int, float]:
    parsed = parse_unary_recheck_choice(raw)
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
    verifier_client: Any,
    total_lines: int,
    parse_unary_recheck_choice: Callable[[str], Dict[str, Any]],
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
    return _parse_full_image_line_anchors(
        raw,
        total_lines,
        parse_unary_recheck_choice=parse_unary_recheck_choice,
    )


def _node_crop_origin(node: SubproofNode) -> Tuple[int, int]:
    if node.id == "ROOT":
        return (0, 0)
    x, y, _, _ = node.box
    return (max(0, x - 10), max(0, y - 10))


def _extract_left_margin_strip(
    full_img: np.ndarray,
    *,
    detect_vertical_bars: Callable[[np.ndarray], List[Tuple[int, int, int, int]]],
    right_fraction: float = 1.0,
) -> Tuple[np.ndarray, int]:
    if full_img.size == 0:
        return full_img, 0
    bars = detect_vertical_bars(full_img)
    first_bar_x = bars[0][0] if bars else max(1, full_img.shape[1] // 8)
    scaled_right = int(first_bar_x * max(0.2, min(1.0, right_fraction)))
    if first_bar_x >= 30:
        scaled_right = max(24, scaled_right)
    margin_right = max(1, min(full_img.shape[1], scaled_right))
    return full_img[:, :margin_right].copy(), margin_right


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

    return merge_margin_candidate_boxes(boxes), cleaned


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


def _build_margin_anchor_geometry(
    *,
    full_img: np.ndarray,
    total_lines: int,
    output_dir: Path,
    detect_vertical_bars: Callable[[np.ndarray], List[Tuple[int, int, int, int]]],
    upscale_for_verifier: Callable[[np.ndarray, int], np.ndarray],
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

    candidate_variants: List[
        Tuple[Tuple[int, int, int, int, int, int], float, np.ndarray, int, np.ndarray, List[Tuple[int, int, int, int]], np.ndarray]
    ] = []
    for right_fraction in (0.6, 0.65, 0.75, 0.85, 1.0):
        margin_img, margin_right = _extract_left_margin_strip(
            full_img,
            detect_vertical_bars=detect_vertical_bars,
            right_fraction=right_fraction,
        )
        if margin_img.size == 0:
            continue
        margin_upscaled = upscale_for_verifier(margin_img, 4)
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
        strip_bounds = margin_anchor_strip_bounds(anchor_centers, full_img.shape[0], anchor_heights=anchor_heights)
        gap_values = [right - left for left, right in zip(anchor_centers, anchor_centers[1:])]
        typical_gap = float(np.median(np.array(gap_values, dtype=np.float32))) if gap_values else 0.0
        tail_regularized = bool(gap_values and typical_gap > 0.0 and gap_values[-1] > max(32.0, typical_gap * 1.25))
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


def _derive_formula_from_structured_counts(
    entry: Dict[str, Any],
    parsed: Dict[str, Any],
    *,
    normalize_formula_text: Callable[[Any, str], str],
    formula_components: Callable[[str], Tuple[str, str]],
    int_field: Callable[[Any, int, int, int], int],
) -> str | None:
    scope_role = str(entry.get("scope_role") or "derived_line").strip()
    original_formula = normalize_formula_text(entry.get("formula_text") or entry.get("wffstr") or "", scope_role)
    _, body = formula_components(original_formula)
    if not body:
        return None

    modal_value = str(parsed.get("main_modal_operator") or "").strip().lower()
    if modal_value == "box":
        modal_symbol = "□"
    elif modal_value == "diamond":
        modal_symbol = "⋄"
    else:
        modal_symbol = ""

    before_modal = int_field(parsed.get("negations_before_modal"), 0, 0, 3)
    before_atom = int_field(parsed.get("negations_before_final_atom"), 0, 0, 3)

    if not modal_symbol:
        return f"{'¬' * before_atom}{body}"
    return f"{'¬' * before_modal}{modal_symbol}{'¬' * before_atom}{body}"


def _nearby_verifier_candidates(
    original_formula: str,
    proposed_formula: str | None,
    *,
    formula_components: Callable[[str], Tuple[str, str]],
) -> List[str]:
    prefix, body = formula_components(original_formula)
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
                candidate = f"{'¬' * before_modal}{original_modal}{'¬' * before_atom}{body}"
            else:
                candidate = f"{'¬' * before_atom}{body}"
            if candidate not in candidates:
                candidates.append(candidate)
            if len(candidates) >= 4:
                return candidates
    return candidates[:4]


def _normalize_selected_formula(
    value: Any,
    scope_role: str,
    *,
    normalize_formula_text: Callable[[Any, str], str],
) -> str:
    selected = str(value or "").strip()
    if not selected or selected == "uncertain":
        return "uncertain"
    return normalize_formula_text(selected, scope_role)


def _run_unary_verifier_attempt(
    *,
    entry: Dict[str, Any],
    original_formula: str,
    verifier_client: Any,
    crop_img: np.ndarray,
    verifier_path: Path,
    attempt_metadata: Dict[str, Any],
    normalize_formula_text: Callable[[Any, str], str],
    normalize_justification_text: Callable[[Any], str],
    parse_unary_recheck_choice: Callable[[str], Dict[str, Any]],
    formula_components: Callable[[str], Tuple[str, str]],
    int_field: Callable[[Any, int, int, int], int],
    unary_formula_skeleton: Callable[[str], str],
    upscale_for_verifier: Callable[[np.ndarray, int], np.ndarray],
) -> Dict[str, Any] | None:
    if crop_img.size == 0:
        return None

    line_crop = upscale_for_verifier(crop_img, 4)
    cv2.imwrite(str(verifier_path), line_crop)

    with verifier_path.open("rb") as handle:
        line_base64_image = base64.b64encode(handle.read()).decode("utf-8")

    structured_raw = verifier_client.chat(
        [
            {"role": "system", "content": "You are a careful vision model. Follow the user's instructions exactly."},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": build_structured_unary_verifier_prompt(
                            entry,
                            normalize_formula_text=normalize_formula_text,
                            normalize_justification_text=normalize_justification_text,
                        ),
                    },
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{line_base64_image}"}} ,
                ],
            },
        ]
    )
    structured_parsed = parse_unary_recheck_choice(structured_raw)
    structured_formula = _derive_formula_from_structured_counts(
        entry,
        structured_parsed,
        normalize_formula_text=normalize_formula_text,
        formula_components=formula_components,
        int_field=int_field,
    )

    scope_role = str(entry.get("scope_role") or "derived_line").strip()
    forced_candidates = _nearby_verifier_candidates(
        original_formula,
        structured_formula,
        formula_components=formula_components,
    )
    forced_raw = verifier_client.chat(
        [
            {"role": "system", "content": "You are a careful vision model. Follow the user's instructions exactly."},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": build_forced_choice_unary_verifier_prompt(
                            entry,
                            forced_candidates,
                            normalize_formula_text=normalize_formula_text,
                            normalize_justification_text=normalize_justification_text,
                        ),
                    },
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{line_base64_image}"}} ,
                ],
            },
        ]
    )
    forced_parsed = parse_unary_recheck_choice(forced_raw)
    forced_choice = _normalize_selected_formula(
        forced_parsed.get("selected_formula"),
        scope_role,
        normalize_formula_text=normalize_formula_text,
    )

    proposed_correction = None
    if (
        structured_formula
        and forced_choice == structured_formula
        and structured_formula != original_formula
        and _is_structurally_close_unary_formula(
            original_formula,
            structured_formula,
            unary_formula_skeleton=unary_formula_skeleton,
        )
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


def run_unary_verifier_experiment(
    *,
    proof_structure: List[Any],
    tree_root: SubproofNode,
    full_img: np.ndarray,
    output_dir: Path,
    verifier_client: Any,
    checker_response: Dict[str, Any] | None,
    normalize_formula_text: Callable[[Any, str], str],
    normalize_justification_text: Callable[[Any], str],
    parse_unary_recheck_choice: Callable[[str], Dict[str, Any]],
    unary_formula_skeleton: Callable[[str], str],
    crop_region: Callable[..., np.ndarray],
    detect_vertical_bars: Callable[[np.ndarray], List[Tuple[int, int, int, int]]],
    detect_text_row_bands: Callable[[np.ndarray], List[Tuple[int, int]]],
    extract_band_crop: Callable[..., np.ndarray],
    upscale_for_verifier: Callable[[np.ndarray, int], np.ndarray],
    int_field: Callable[[Any, int, int, int], int],
    formula_components: Callable[[str], Tuple[str, str]],
    iter_proof_line_dicts: Callable[[List[Any]], List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    for path in output_dir.glob("verifier_line_*.jpg"):
        path.unlink(missing_ok=True)
    for path in output_dir.glob("margin_anchor_*"):
        path.unlink(missing_ok=True)
    (output_dir / UNARY_VERIFIER_FILENAME).unlink(missing_ok=True)

    if verifier_client is None or not isinstance(checker_response, dict) or not (checker_response.get("parse_errors") or checker_response.get("check_errors")):
        return []

    node_lookup = build_node_lookup(tree_root)
    all_lines = iter_proof_line_dicts(proof_structure)
    max_line_number = max(
        (int(line.get("line")) for line in all_lines if isinstance(line.get("line"), int)),
        default=0,
    )
    margin_anchor_geometry = _build_margin_anchor_geometry(
        full_img=full_img,
        total_lines=max_line_number,
        output_dir=output_dir,
        detect_vertical_bars=detect_vertical_bars,
        upscale_for_verifier=upscale_for_verifier,
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
            parse_unary_recheck_choice=parse_unary_recheck_choice,
        )
    )

    records: List[Dict[str, Any]] = []
    for entry in all_lines:
        if not should_verify_unary_formula(
            entry,
            normalize_formula_text=normalize_formula_text,
            normalize_justification_text=normalize_justification_text,
        ):
            continue

        node_id = str(entry.get("_ocr_node_id") or "").strip()
        scope_role = str(entry.get("scope_role") or "derived_line").strip()
        original_formula = normalize_formula_text(entry.get("formula_text") or entry.get("wffstr") or "", scope_role)
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
                normalize_formula_text=normalize_formula_text,
                normalize_justification_text=normalize_justification_text,
                parse_unary_recheck_choice=parse_unary_recheck_choice,
                formula_components=formula_components,
                int_field=int_field,
                unary_formula_skeleton=unary_formula_skeleton,
                upscale_for_verifier=upscale_for_verifier,
            )
            if anchor_attempt is not None:
                attempt_records.append(anchor_attempt)

        if node is not None and isinstance(line_order, int):
            node_crop = verifier_node_crop_image(node=node, full_img=full_img, crop_region=crop_region)
            bands = detect_text_row_bands(node_crop)
            _, node_origin_y = _node_crop_origin(node)
            anchor_rel = full_image_line_anchors.get(int(line_number)) if isinstance(line_number, int) else None
            projected_crop_y = None
            if anchor_rel is not None and full_img.shape[0] > 0:
                global_y = (anchor_rel / 1000.0) * float(full_img.shape[0])
                projected_crop_y = global_y - float(node_origin_y)
            for candidate_index in candidate_band_indices(
                line_order=line_order,
                band_count=len(bands),
                bands=bands,
                projected_y=projected_crop_y,
            ):
                line_crop = extract_band_crop(
                    node_crop_img=node_crop,
                    bands=bands,
                    band_index=candidate_index,
                )
                suffix = f"{line_order + 1}"
                if candidate_index != line_order:
                    suffix = f"{suffix}_band{candidate_index + 1}"
                attempt = _run_unary_verifier_attempt(
                    entry=entry,
                    original_formula=original_formula,
                    verifier_client=verifier_client,
                    crop_img=line_crop,
                    verifier_path=output_dir / f"verifier_line_{node_id}_{suffix}.jpg",
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
                    normalize_formula_text=normalize_formula_text,
                    normalize_justification_text=normalize_justification_text,
                    parse_unary_recheck_choice=parse_unary_recheck_choice,
                    formula_components=formula_components,
                    int_field=int_field,
                    unary_formula_skeleton=unary_formula_skeleton,
                    upscale_for_verifier=upscale_for_verifier,
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

        best_attempt = best_verifier_attempt(
            original_formula=original_formula,
            expected_line_order=line_order if isinstance(line_order, int) else None,
            attempts=attempt_records,
            unary_formula_skeleton=unary_formula_skeleton,
        )

        records.append(
            {
                "node_id": node_id,
                "line_label": str(entry.get("line_label") or "").strip(),
                "line_number": line_number,
                "node_line_order": line_order if isinstance(line_order, int) else None,
                "baseline_formula": original_formula,
                "band_count": len(detect_text_row_bands(verifier_node_crop_image(node=node, full_img=full_img, crop_region=crop_region)))
                if node is not None
                else 0,
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


def apply_unary_verifier_corrections(
    proof_structure: List[Any],
    verifier_records: List[Dict[str, Any]],
    *,
    clone_jsonish: Callable[[Any], Any],
    iter_proof_line_dicts: Callable[[List[Any]], List[Dict[str, Any]]],
) -> tuple[List[Any], List[Dict[str, Any]]]:
    updated = clone_jsonish(proof_structure)
    lines_by_number = {
        int(line.get("line")): line
        for line in iter_proof_line_dicts(updated)
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


def run_unary_postprocess(
    *,
    enabled: bool,
    proof_structure: List[Any],
    tree_root: Any,
    full_img: np.ndarray,
    output_dir: Path,
    verifier_client: Any,
    checker_response: Dict[str, Any] | None,
    run_experiment: Callable[..., List[Dict[str, Any]]],
    apply_corrections: Callable[[List[Any], List[Dict[str, Any]]], tuple[List[Any], List[Dict[str, Any]]]],
) -> tuple[List[Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    if not enabled:
        return proof_structure, [], []

    verifier_records = run_experiment(
        proof_structure=proof_structure,
        tree_root=tree_root,
        full_img=full_img,
        output_dir=output_dir,
        verifier_client=verifier_client,
        checker_response=checker_response,
    )
    rewritten_structure, verifier_rewrites = apply_corrections(proof_structure, verifier_records)
    return rewritten_structure, verifier_records, verifier_rewrites
