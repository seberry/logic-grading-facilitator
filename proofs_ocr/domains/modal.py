"""Modal-proof OCR adapter layer.

Owns modal-specific prompts, normalization, and repair policy for the new
clean pipeline. Keep modal heuristics here rather than in common modules.
"""

from __future__ import annotations

import json
import re
from typing import Any, Mapping

import cv2
import numpy as np

from proofs_ocr.common.stitch import extract_subproof_token_id

_WHITESPACE_RE = re.compile(r"\s+")
_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)
_DASH_BETWEEN_DIGITS_RE = re.compile(r"(?<=\d)\s*[-–—]\s*(?=\d)")
_COMMA_BETWEEN_DIGITS_RE = re.compile(r"(?<=\d)\s*,\s*(?=\d)")
VALID_SCOPE_ROLES = {
    "premise",
    "ordinary_assumption",
    "strict_open",
    "derived_line",
    "subproof_token",
}
_IMPLICIT_PREMISE_JUSTIFICATIONS = {
    "",
    "pr",
    "prem",
    "premise",
    "assum",
    "assump",
    "assumption",
}


def _collapse_whitespace(value: Any) -> str:
    return _WHITESPACE_RE.sub(" ", str(value or "")).strip()


def canonical_modal_config(config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Return the minimal config shape needed by the modal baseline."""

    source = dict(config or {})
    system = _collapse_whitespace(source.get("system") or "K") or "K"
    return {
        "system": system,
        "include_derived_rules": bool(source.get("include_derived_rules", False)),
        "conclusion": _collapse_whitespace(source.get("conclusion") or ""),
    }


def normalize_formula_text(value: Any) -> str:
    """Normalize the formula text for the minimal modal baseline."""

    normalized = _collapse_whitespace(value)
    for source, target in (("<->", "↔"), ("->", "→"), ("[]", "□"), ("<>", "⋄")):
        normalized = normalized.replace(source, target)
    return normalized


def normalize_justification_text(value: Any) -> str:
    """Normalize modal justifications without applying repair heuristics."""

    normalized = _collapse_whitespace(value)
    if not normalized:
        return ""
    normalized = normalized.replace("–", "-").replace("—", "-")
    normalized = _DASH_BETWEEN_DIGITS_RE.sub("-", normalized)
    normalized = _COMMA_BETWEEN_DIGITS_RE.sub(",", normalized)
    return normalized


def build_line_ocr_prompt(config: Mapping[str, Any] | None = None) -> str:
    """Build the one-line OCR prompt for the modal baseline."""

    canonical = canonical_modal_config(config)
    lines = [
        "You are transcribing one visible line from a modal Fitch-style proof crop.",
        "Return only JSON. Do not include markdown fences or explanation.",
        'Return exactly one JSON object with these keys: "line_label", "formula_text", "justification_text", "scope_role", "confidence".',
        "Allowed scope_role values are premise, ordinary_assumption, strict_open, derived_line, and subproof_token.",
        "Use subproof_token only if the crop is entirely a red placeholder token such as <SUBPROOF_ID_2>.",
        'If the line is a strict modal opener, use scope_role "strict_open" and formula_text "□".',
        "Prefer Unicode operators: ¬, ∧, ∨, →, ↔, □, ⋄, ⊥.",
        "Preserve repeated unary operators exactly.",
        "Copy the visible line label if present; otherwise return an empty string.",
        "Return an empty justification_text if the crop shows no justification.",
        "Keep formula_text and justification_text separate.",
        f"Modal system: {canonical['system']}",
        f"Include derived rules: {'yes' if canonical['include_derived_rules'] else 'no'}",
    ]
    conclusion = canonical.get("conclusion") or ""
    if conclusion:
        lines.append(f"Target conclusion: {conclusion}")
    return "\n".join(lines)


def build_subproof_token_entry(child_id: str) -> dict[str, str]:
    """Return the synthetic OCR entry used to stand in for a child scope."""

    return {
        "scope_role": "subproof_token",
        "formula_text": f"<SUBPROOF_ID_{child_id}>",
        "justification_text": "",
    }


def has_visible_formula(entry: Mapping[str, Any]) -> bool:
    """Report whether an OCR entry represents a visible proof line."""

    formula_text = normalize_formula_text(entry.get("formula_text") or entry.get("wffstr") or "")
    if not formula_text:
        return False
    return extract_subproof_token_id(formula_text) is None


def resolve_modal_ocr_entry(entry: Any, _index: int) -> dict[str, str] | None:
    """Resolve one OCR entry into the final modal line dictionary."""

    if not isinstance(entry, Mapping):
        return None

    scope_role = _collapse_whitespace(entry.get("scope_role") or "derived_line") or "derived_line"
    formula_text = normalize_formula_text(entry.get("formula_text") or entry.get("wffstr") or "")
    if not formula_text or scope_role == "subproof_token" or extract_subproof_token_id(formula_text):
        return None

    justification_text = normalize_justification_text(entry.get("justification_text") or entry.get("jstr") or "")
    if scope_role in {"premise", "ordinary_assumption", "strict_open"} and (
        not justification_text or justification_text.strip().lower() in _IMPLICIT_PREMISE_JUSTIFICATIONS
    ):
        justification_text = "PR"
    elif not justification_text and _index == 0:
        justification_text = "PR"

    resolved = {
        "wffstr": formula_text,
        "jstr": justification_text,
    }
    line_label = _collapse_whitespace(entry.get("line_label") or "")
    if line_label:
        resolved["line_label"] = line_label
    return resolved


def normalize_scope_role(value: Any) -> str:
    """Map fuzzy model outputs into the baseline scope-role enum."""

    role = _collapse_whitespace(value).lower()
    if role in VALID_SCOPE_ROLES:
        return role

    aliases = {
        "assumption": "ordinary_assumption",
        "ordinary": "ordinary_assumption",
        "open": "ordinary_assumption",
        "strict": "strict_open",
        "box_open": "strict_open",
        "modal_open": "strict_open",
        "line": "derived_line",
        "derived": "derived_line",
    }
    return aliases.get(role, "derived_line")


def parse_json_object_response(raw_response: str) -> dict[str, Any]:
    """Parse one JSON object from a model response."""

    text = _FENCE_RE.sub("", str(raw_response or "").strip()).strip()
    if not text:
        raise ValueError("OCR response was empty")

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end < start:
            raise
        parsed = json.loads(text[start : end + 1])

    if not isinstance(parsed, dict):
        raise ValueError("OCR response was not a JSON object")
    return parsed


def parse_line_ocr_response(raw_response: str) -> dict[str, str]:
    """Resolve a raw vision-model response into one line OCR entry."""

    parsed = parse_json_object_response(raw_response)
    return {
        "line_label": _collapse_whitespace(parsed.get("line_label") or ""),
        "formula_text": normalize_formula_text(parsed.get("formula_text") or ""),
        "justification_text": normalize_justification_text(parsed.get("justification_text") or ""),
        "scope_role": normalize_scope_role(parsed.get("scope_role") or "derived_line"),
        "confidence": _collapse_whitespace(parsed.get("confidence") or ""),
    }


def recover_line_entry_from_crop(
    *,
    line_crop_img: np.ndarray,
    parsed_entry: Mapping[str, Any],
) -> tuple[dict[str, str], str]:
    """Apply narrow modal-domain recovery for tiny strict-opener crops."""

    resolved = {
        "line_label": _collapse_whitespace(parsed_entry.get("line_label") or ""),
        "formula_text": normalize_formula_text(parsed_entry.get("formula_text") or ""),
        "justification_text": normalize_justification_text(parsed_entry.get("justification_text") or ""),
        "scope_role": normalize_scope_role(parsed_entry.get("scope_role") or "derived_line"),
        "confidence": _collapse_whitespace(parsed_entry.get("confidence") or ""),
    }
    if resolved["formula_text"]:
        return resolved, ""
    if resolved["line_label"]:
        return resolved, ""

    if not _looks_like_tiny_strict_open_crop(line_crop_img):
        return resolved, ""

    resolved["formula_text"] = "□"
    resolved["scope_role"] = "strict_open"
    return resolved, "tiny_strict_open_crop"


def recover_first_nonroot_strict_open_from_crop(
    *,
    line_crop_img: np.ndarray,
    parsed_entry: Mapping[str, Any],
) -> tuple[dict[str, str], str]:
    """Recover a first non-root row that visually looks like a strict opener band."""

    resolved = {
        "line_label": _collapse_whitespace(parsed_entry.get("line_label") or ""),
        "formula_text": normalize_formula_text(parsed_entry.get("formula_text") or ""),
        "justification_text": normalize_justification_text(parsed_entry.get("justification_text") or ""),
        "scope_role": normalize_scope_role(parsed_entry.get("scope_role") or "derived_line"),
        "confidence": _collapse_whitespace(parsed_entry.get("confidence") or ""),
    }
    if resolved["scope_role"] == "strict_open" and resolved["formula_text"] == "□":
        resolved["justification_text"] = ""
        return resolved, ""
    if resolved["line_label"]:
        return resolved, ""
    if not _looks_like_first_nonroot_strict_open_band(line_crop_img):
        return resolved, ""

    resolved["formula_text"] = "□"
    resolved["justification_text"] = ""
    resolved["scope_role"] = "strict_open"
    return resolved, "first_nonroot_strict_open_band"


def detect_label_only_crop(line_crop_img: np.ndarray) -> str:
    """Return a skip reason when a wide crop appears to contain only a line label."""

    if line_crop_img.size == 0:
        return "empty_crop"

    gray = cv2.cvtColor(line_crop_img, cv2.COLOR_BGR2GRAY)
    img_h, img_w = gray.shape[:2]
    if img_w < 600 or img_h < 24:
        return ""

    dark = (gray < 185).astype(np.uint8)
    dark_pixels = int(dark.sum())
    if dark_pixels <= 0:
        return "empty_crop"

    coords = cv2.findNonZero(dark)
    if coords is None:
        return "empty_crop"
    left, top, width, height = cv2.boundingRect(coords)
    if left <= max(48, int(img_w * 0.22)):
        return ""
    if left + width >= max(220, int(img_w * 0.42)):
        return ""
    if width > max(110, int(img_w * 0.10)):
        return ""
    if height > max(42, int(img_h * 0.85)):
        return ""
    if dark_pixels > max(260, int(img_w * img_h * 0.01)):
        return ""

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(dark, 8)
    significant_components = 0
    for label_index in range(1, num_labels):
        area = int(stats[label_index, cv2.CC_STAT_AREA])
        if area >= 12:
            significant_components += 1
    if significant_components > 2:
        return ""
    return "label_only_crop"


def _looks_like_tiny_strict_open_crop(line_crop_img: np.ndarray) -> bool:
    if line_crop_img.size == 0:
        return False

    gray = cv2.cvtColor(line_crop_img, cv2.COLOR_BGR2GRAY)
    img_h, img_w = gray.shape[:2]
    left_ignore = min(14, max(8, img_w // 30))
    right_limit = min(img_w, max(80, int(img_w * 0.45)))
    if right_limit <= left_ignore:
        return False

    focused = gray[:, left_ignore:right_limit]
    dark = (focused < 185).astype(np.uint8)
    if int(dark.sum()) < 40:
        return False

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(dark, 8)
    candidates: list[tuple[int, int, int, int, int]] = []
    for label_index in range(1, num_labels):
        area = int(stats[label_index, cv2.CC_STAT_AREA])
        left = int(stats[label_index, cv2.CC_STAT_LEFT])
        top = int(stats[label_index, cv2.CC_STAT_TOP])
        width = int(stats[label_index, cv2.CC_STAT_WIDTH])
        height = int(stats[label_index, cv2.CC_STAT_HEIGHT])
        if area < 25:
            continue
        if left <= 2 and width <= 6 and height >= max(12, int(img_h * 0.60)):
            continue
        candidates.append((area, left, top, width, height))

    if len(candidates) != 1:
        return False

    area, left, top, width, height = candidates[0]
    if top > max(40, int(img_h * 0.60)):
        return False
    if width < 10 or height < 10:
        return False
    if width > max(34, int(img_w * 0.12)) or height > max(34, int(img_h * 0.85)):
        return False
    aspect_ratio = width / max(1, height)
    if not 0.55 <= aspect_ratio <= 1.55:
        return False
    fill_ratio = area / max(1, width * height)
    return 0.22 <= fill_ratio <= 0.72


def _looks_like_first_nonroot_strict_open_band(line_crop_img: np.ndarray) -> bool:
    if line_crop_img.size == 0:
        return False

    gray = cv2.cvtColor(line_crop_img, cv2.COLOR_BGR2GRAY)
    img_h, img_w = gray.shape[:2]
    left_w = min(img_w, max(120, int(img_w * 0.38)))
    focused = gray[:, :left_w]
    dark = (focused < 185).astype(np.uint8)
    if int(dark.sum()) < 60:
        return False

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(dark, 8)
    box_candidate: tuple[int, int, int, int] | None = None
    for label_index in range(1, num_labels):
        area = int(stats[label_index, cv2.CC_STAT_AREA])
        left = int(stats[label_index, cv2.CC_STAT_LEFT])
        top = int(stats[label_index, cv2.CC_STAT_TOP])
        width = int(stats[label_index, cv2.CC_STAT_WIDTH])
        height = int(stats[label_index, cv2.CC_STAT_HEIGHT])
        if area < 25:
            continue
        if left <= 2 and width <= 6 and height >= max(12, int(img_h * 0.60)):
            continue
        if left > max(36, int(left_w * 0.22)) or top > max(20, int(img_h * 0.45)):
            continue
        if width < 10 or height < 10 or width > 40 or height > 40:
            continue
        aspect_ratio = width / max(1, height)
        if not 0.55 <= aspect_ratio <= 1.65:
            continue
        box_candidate = (left, top, width, height)
        break

    if box_candidate is None:
        return False

    kernel_w = min(100, max(20, int(left_w * 0.20)))
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, 1))
    horizontal_mask = cv2.morphologyEx(dark, cv2.MORPH_OPEN, horizontal_kernel)
    num_hlabels, _, hstats, _ = cv2.connectedComponentsWithStats(horizontal_mask, 8)
    underline: tuple[int, int, int, int] | None = None
    box_left, box_top, box_w, box_h = box_candidate
    box_bottom = box_top + box_h - 1
    for label_index in range(1, num_hlabels):
        area = int(hstats[label_index, cv2.CC_STAT_AREA])
        left = int(hstats[label_index, cv2.CC_STAT_LEFT])
        top = int(hstats[label_index, cv2.CC_STAT_TOP])
        width = int(hstats[label_index, cv2.CC_STAT_WIDTH])
        height = int(hstats[label_index, cv2.CC_STAT_HEIGHT])
        if area < 40 or height > 5 or width < 35:
            continue
        if left > max(40, int(left_w * 0.25)):
            continue
        if top < box_bottom - 2 or top > min(img_h - 4, box_bottom + 18):
            continue
        underline = (left, top, width, height)
        break

    if underline is None:
        return False

    _, line_top, _, line_h = underline
    below_y = min(img_h, line_top + line_h + 4)
    if below_y >= img_h:
        return False
    return int(dark[below_y:, :].sum()) >= 15
