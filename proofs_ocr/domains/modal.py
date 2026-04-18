"""Modal-proof OCR adapter layer.

Owns modal-specific prompts, normalization, and repair policy for the new
clean pipeline. Keep modal heuristics here rather than in common modules.
"""

from __future__ import annotations

import json
import re
from typing import Any, Mapping

from proofs_ocr.common.stitch import extract_subproof_token_id

_WHITESPACE_RE = re.compile(r"\s+")
_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)
VALID_SCOPE_ROLES = {
    "premise",
    "ordinary_assumption",
    "strict_open",
    "derived_line",
    "subproof_token",
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

    return _collapse_whitespace(value)


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
    if scope_role in {"premise", "ordinary_assumption", "strict_open"} and not justification_text:
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
