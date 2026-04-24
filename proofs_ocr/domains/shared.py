"""Shared domain helpers for Fitch OCR pipelines.

Contains logic-level pieces that are identical (or nearly identical) across
both the TFL and modal pipelines — neutral prompt fragments, VLM response
parsing, and premise/assumption defaulting.

Domain-specific prompt builders (TFL rules, modal scope roles, etc.) live
in ``domains/tfl.py`` and ``domains/modal.py`` and layer their additions
on top of the shared pieces here.
"""

from __future__ import annotations

import json
import re
from typing import Any, Union

from proofs_ocr.common.stitch import extract_subproof_token_id

# ── Neutral prompt fragments ──────────────────────────────────────────

def unicode_operator_instruction() -> str:
    """Instruction to prefer Unicode logic operators in OCR output."""
    return "Prefer Unicode operators: ¬, ∧, ∨, →, ↔, ⊥."


def json_only_instruction() -> str:
    """Instruction to return only JSON without markdown fences."""
    return "Return only JSON. Do not include markdown fences or explanation."


def formula_justification_separation_instruction() -> str:
    """Instruction to keep formula and justification in separate fields."""
    return (
        "Do not place justification text such as ¬E 4,8 into formula_text. "
        "Keep formula_text and justification_text separate."
    )


def contradiction_instruction() -> str:
    """Instruction for recognizing contradiction/falsum symbols."""
    return (
        "A contradiction line may be written as ⊥, _|_, _ | _, or a short "
        "vertical stroke above a horizontal bar; transcribe any of these as ⊥."
    )


def empty_justification_instruction() -> str:
    """Instruction for handling lines with no visible justification."""
    return (
        "Return an empty justification_text if the crop shows no justification."
    )


def build_neutral_prompt_fragments() -> list[str]:
    """Return the list of universal Fitch OCR prompt instructions.

    Domain-specific prompt builders should concatenate these with their
    own additions (e.g., modal scope roles, TFL rule names).
    """
    return [
        json_only_instruction(),
        unicode_operator_instruction(),
        formula_justification_separation_instruction(),
        contradiction_instruction(),
        empty_justification_instruction(),
    ]


# ── VLM response parsing ─────────────────────────────────────────────

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def strip_markdown_fences(raw: str) -> str:
    """Remove markdown code fences from a VLM response string."""
    return _FENCE_RE.sub("", raw.strip()).strip()


def parse_vlm_json_response(raw: str) -> Union[dict, list]:
    """Parse JSON from a raw VLM response, handling fences gracefully.

    Accepts either a JSON object (``{...}``) or JSON array (``[...]``).
    Strips markdown fences first, then falls back to extracting the
    balanced ``{...}`` or ``[...]`` block if ``json.loads`` fails.

    Raises ``ValueError`` if no valid JSON is found.
    """
    text = strip_markdown_fences(raw)
    if not text:
        raise ValueError("VLM response was empty after stripping fences")

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Fallback: extract balanced braces or brackets
    for open_ch, close_ch in [("{", "}"), ("[", "]")]:
        start = text.find(open_ch)
        end = text.rfind(close_ch)
        if start >= 0 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                continue

    raise ValueError(f"Could not parse JSON from VLM response: {text[:200]!r}")


# ── Subproof-token handling ───────────────────────────────────────────

def is_subproof_token(text: str) -> bool:
    """Return True if the text is a subproof placeholder like ``<SUBPROOF_ID_2>``."""
    return extract_subproof_token_id(text) is not None


def build_subproof_token_entry(child_id: str) -> dict[str, str]:
    """Return a synthetic OCR entry that stands in for a child scope.

    Both TFL and modal pipelines use the same ``<SUBPROOF_ID_X>`` convention.
    """
    return {
        "formula_text": f"<SUBPROOF_ID_{child_id}>",
        "justification_text": "",
        "wffstr": f"<SUBPROOF_ID_{child_id}>",
        "jstr": "",
    }


# ── Premise / assumption defaulting ───────────────────────────────────

def default_justification_for_role(
    *,
    justification_text: str,
    scope_role: str = "",
    is_first_in_scope: bool = False,
    premise_label: str = "Pr",
    assumption_label: str = "Hyp",
    premise_aliases: frozenset[str] | None = None,
) -> str:
    """Apply default premise/assumption labeling when justification is blank.

    Both pipelines need this logic: if a line has no justification and is
    a premise or first in its scope, give it the appropriate label.

    Parameters
    ----------
    justification_text:
        Current justification string (may be empty).
    scope_role:
        Role tag from the OCR entry (e.g. "premise", "ordinary_assumption").
    is_first_in_scope:
        Whether this line is the first line in its scope block.
    premise_label:
        Label to use for premises (``"Pr"`` for TFL, ``"PR"`` for modal).
    assumption_label:
        Label to use for assumptions (``"Hyp"`` for TFL, ``"PR"`` for modal).
    premise_aliases:
        Set of lowercase strings that should be treated as premise labels.
    """
    from proofs_ocr.tricks.justification_snaps import PREMISE_ALIAS_HEADS

    if premise_aliases is None:
        premise_aliases = PREMISE_ALIAS_HEADS

    text = justification_text.strip().lower()

    # Already has a meaningful justification
    if text and text not in premise_aliases:
        return justification_text

    # Premise or assumption role with blank/alias justification
    role = scope_role.strip().lower()
    if role in {"premise", "ordinary_assumption"} or text in premise_aliases:
        return premise_label

    # First line in scope with blank justification → assumption
    if is_first_in_scope and not text:
        return assumption_label

    return justification_text


def has_visible_formula(entry: Any) -> bool:
    """Report whether an OCR entry represents a visible proof line.

    Returns False for subproof tokens and entries with empty formulas.
    """
    formula = str(
        entry.get("formula_text", "")
        or entry.get("wffstr", "")
        or ""
    ).strip()
    if not formula:
        return False
    return not is_subproof_token(formula)