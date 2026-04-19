"""Modal-specific line-zone OCR helpers for formula/justification separation."""

from __future__ import annotations

import re
from typing import Any, Mapping

from modal_ocr.prompts import build_formula_ocr_prompt as _legacy_build_formula_ocr_prompt
from modal_ocr.prompts import build_justification_ocr_prompt as _legacy_build_justification_ocr_prompt

from . import modal

_VISIBLE_LINE_LABEL_RE = re.compile(r"^\d+[.)]?$")


def build_formula_zone_prompt(config: Mapping[str, Any] | None = None) -> str:
    """Return the formula-only OCR prompt for one line zone."""

    return _legacy_build_formula_ocr_prompt(modal.canonical_modal_config(config))


def build_justification_zone_prompt(config: Mapping[str, Any] | None = None) -> str:
    """Return the justification-only OCR prompt for one line zone."""

    return _legacy_build_justification_ocr_prompt(modal.canonical_modal_config(config))


def normalize_formula_zone_result(parsed: Mapping[str, Any]) -> dict[str, str]:
    """Normalize one formula-zone OCR result."""

    return {
        "line_label": _collapse_whitespace(parsed.get("line_label") or ""),
        "formula_text": modal.normalize_formula_text(parsed.get("formula_text") or ""),
        "confidence": _collapse_whitespace(parsed.get("confidence") or ""),
    }


def normalize_justification_zone_result(parsed: Mapping[str, Any]) -> dict[str, str]:
    """Normalize one justification-zone OCR result."""

    raw_text = modal.normalize_justification_text(parsed.get("justification_text") or parsed.get("jstr") or "")
    for source, target in (("[]", "□"), ("<>", "⋄")):
        raw_text = raw_text.replace(source, target)
    return {
        "justification_text": raw_text,
        "confidence": _collapse_whitespace(parsed.get("confidence") or ""),
    }


def merge_line_zone_results(
    *,
    formula_result: Mapping[str, Any],
    justification_result: Mapping[str, Any],
    is_root_node: bool,
    is_first_visible_row_of_nonroot: bool,
) -> dict[str, str]:
    """Merge separate formula/justification OCR results into one baseline line entry."""

    raw_line_label = _collapse_whitespace(formula_result.get("line_label") or "")
    line_label = normalize_visible_line_label(raw_line_label)
    justification_text = modal.normalize_justification_text(justification_result.get("justification_text") or "")
    if raw_line_label and not line_label and not justification_text:
        justification_text = recover_justification_from_formula_label(raw_line_label)

    formula_text = modal.normalize_formula_text(formula_result.get("formula_text") or "")
    scope_role = infer_scope_role(
        formula_text=formula_text,
        justification_text=justification_text,
        is_root_node=is_root_node,
        is_first_visible_row_of_nonroot=is_first_visible_row_of_nonroot,
    )
    return {
        "line_label": line_label,
        "formula_text": formula_text,
        "justification_text": justification_text,
        "scope_role": scope_role,
        "confidence": choose_confidence_label(
            formula_confidence=formula_result.get("confidence") or "",
            justification_confidence=justification_result.get("confidence") or "",
            has_justification=bool(justification_text),
        ),
    }


def infer_scope_role(
    *,
    formula_text: str,
    justification_text: str,
    is_root_node: bool,
    is_first_visible_row_of_nonroot: bool,
) -> str:
    """Infer the baseline scope role from zone-merged content."""

    if formula_text == "□":
        return "strict_open"
    if is_first_visible_row_of_nonroot:
        return "ordinary_assumption"
    if is_root_node and not justification_text:
        return "premise"
    return "derived_line"


def normalize_visible_line_label(raw_value: Any) -> str:
    """Keep only line labels that look like visible proof numbers."""

    raw = _collapse_whitespace(raw_value)
    compact = raw.replace(" ", "")
    return raw if _VISIBLE_LINE_LABEL_RE.fullmatch(compact) else ""


def recover_justification_from_formula_label(raw_value: Any) -> str:
    """Treat a non-numeric line-label zone result as likely justification spill."""

    raw = modal.normalize_justification_text(raw_value)
    if not raw:
        return ""
    if _VISIBLE_LINE_LABEL_RE.fullmatch(raw.replace(" ", "")):
        return ""
    return raw


def choose_confidence_label(
    *,
    formula_confidence: Any,
    justification_confidence: Any,
    has_justification: bool,
) -> str:
    """Collapse per-zone confidence labels into one baseline confidence."""

    formula_score = _confidence_score(formula_confidence)
    justification_score = _confidence_score(justification_confidence)
    combined = formula_score
    if has_justification:
        combined = min(formula_score, justification_score) if justification_score > 0.0 else formula_score
    if combined >= 0.85:
        return "high"
    if combined >= 0.55:
        return "medium"
    if combined > 0.0:
        return "low"
    return ""


def _confidence_score(value: Any) -> float:
    raw = str(value or "").strip().lower()
    if raw in {"high", "0.9", "0.95", "1.0"}:
        return 0.95
    if raw in {"medium", "0.7", "0.75"}:
        return 0.70
    if raw in {"low", "0.4", "0.5"}:
        return 0.45
    if raw in {"uncertain", "", "0"}:
        return 0.0
    try:
        numeric = float(raw)
    except ValueError:
        return 0.0
    return max(0.0, min(1.0, numeric))


def _collapse_whitespace(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()
