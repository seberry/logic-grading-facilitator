"""Symbol normalization tricks for Fitch OCR pipelines.

Provides composable functions for converting ASCII / OCR-artifact symbols
into canonical Unicode logic operators.  These are shared across both the
TFL and modal pipelines because the underlying VLM makes the same kinds
of OCR mistakes regardless of which logic the proof uses.

The functions are intentionally small and pure so they can be composed
or used individually by domain-specific normalization layers.
"""

from __future__ import annotations

import re
from typing import Any

# ── Whitespace ─────────────────────────────────────────────────────────

_WHITESPACE_RE = re.compile(r"\s+")


def collapse_whitespace(value: Any) -> str:
    """Collapse runs of whitespace and strip leading/trailing space."""
    return _WHITESPACE_RE.sub(" ", str(value or "")).strip()


# ── Multi-character operator replacements ──────────────────────────────
# Order matters: replace longer tokens first to avoid partial matches
# (e.g. ``<->`` must be replaced before ``<-`` or ``->``).

# Arrow replacements shared by both TFL and modal proofs.
ARROW_REPLACEMENTS: tuple[tuple[str, str], ...] = (
    ("<->", "↔"),
    ("->", "→"),
    ("=>", "→"),
    ("<→", "↔"),   # OCR artifact: left-arrow + arrow
)

# Symbol replacements for TFL / shared operators.  Applied character by
# character **after** multi-character arrow replacements.
_SYMBOL_MAP: dict[str, str] = {
    "~": "¬",
    "&": "∧",
    ".": "∧",       # VLM often uses dot for AND
    "v": "∨",       # Lowercase v is OR (must run before uppercasing)
    "V": "∨",
    "_|_": "⊥",
}

# Modal-specific symbol extensions.  These are safe to apply on TFL proofs
# too (TFL formulas never contain ``[]`` or ``<>``), but they are kept
# separate so that callers can opt in.
_MODAL_SYMBOL_MAP: dict[str, str] = {
    "[]": "□",
    "<>": "⋄",
    "◇": "⋄",       # Unicode lozenge → canonical ⋄
    "◊": "⋄",       # Another lozenge variant
}


def normalize_arrows(text: str) -> str:
    """Replace ASCII arrow tokens with Unicode equivalents."""
    for src, tgt in ARROW_REPLACEMENTS:
        text = text.replace(src, tgt)
    return text


def normalize_ascii_logic_symbols(text: str) -> str:
    """Replace ASCII logic symbols (~, &, v, .) with Unicode equivalents.

    Multi-character tokens (like ``_|_``) are replaced first, then
    single characters.
    """
    # Handle _|_ (falsum) before single-char replacements
    text = text.replace("_|_", "⊥")
    chars: list[str] = []
    for ch in text:
        chars.append(_SYMBOL_MAP.get(ch, ch))
    return "".join(chars)


def normalize_modal_symbols(text: str) -> str:
    """Replace ASCII bracket / lozenge pairs with canonical modal operators."""
    for src, tgt in _MODAL_SYMBOL_MAP.items():
        text = text.replace(src, tgt)
    return text


def normalize_symbols(
    text: str,
    *,
    include_modal: bool = False,
) -> str:
    """Apply all symbol normalizations in the correct order.

    Parameters
    ----------
    text:
        Raw OCR text (formula or justification).
    include_modal:
        If ``True``, also normalize ``[]`` → ``□`` and ``<>`` / ``◇`` → ``⋄``.
    """
    # 1. Multi-char arrows first
    text = normalize_arrows(text)
    # 2. Modal symbols (safe to apply universally, but opt-in for clarity)
    if include_modal:
        text = normalize_modal_symbols(text)
    # 3. ASCII logic symbols (~ & v . etc.)
    text = normalize_ascii_logic_symbols(text)
    return text


# ── Per-character formula cleaner for TFL ──────────────────────────────
# The TFL pipeline uppercases atoms because TFL requires P, not p.
# Modal proofs may use lowercase atoms so this is TFL-specific.

def uppercase_atoms(text: str) -> str:
    """Uppercase alphabetic characters in a formula string.

    This is appropriate for TFL proofs where atoms are always uppercase.
    Skip this for modal proofs that may use lowercase atoms.
    """
    # Preserve already-normalized Unicode operators
    chars: list[str] = []
    for ch in text:
        if ch.isalpha():
            chars.append(ch.upper())
        else:
            chars.append(ch)
    return "".join(chars)