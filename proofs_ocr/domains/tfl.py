"""TFL-proof OCR adapter layer.

Owns TFL-specific prompts, normalization, and comparison policy for the Fitch
OCR pipeline.  This started as a placeholder and now provides the canonical
comparison helpers needed by the TFL gold-regression runner.
"""

from __future__ import annotations

import re
from typing import Any

_WHITESPACE_RE = re.compile(r"\s+")

# ── Symbol normalisation ─────────────────────────────────────────────

# Order matters: replace multi-character tokens before single characters.
_SYMBOL_REPLACEMENTS = (
    ("<->", "↔"),
    ("->", "→"),
    ("=>", "→"),
    ("<→", "↔"),
    ("~", "¬"),
    ("&", "∧"),
    ("v", "∨"),
    ("V", "∨"),
    ("_|_", "⊥"),
)

# ── Whitespace / cosmetic helpers ────────────────────────────────────

def _collapse_whitespace(value: Any) -> str:
    return _WHITESPACE_RE.sub(" ", str(value or "")).strip()


# ── Canonical comparison helpers ─────────────────────────────────────

def canonicalize_formula_for_comparison(value: Any) -> str:
    """Canonicalize a TFL formula for equality checks.

    Collapses whitespace, normalises ASCII logic symbols to Unicode, and
    strips remaining spaces so that e.g. ``P ∧ Q`` matches ``P∧Q``.
    """

    normalized = _collapse_whitespace(value)
    if not normalized:
        return ""

    # Phase 0: Blanket OCR-artifact repair for TFL formulas.
    # TFL formulas never contain digits, epsilons, or element-of symbols.
    # Must match the pipeline's clean_wff Phase 0 replacements exactly.
    normalized = normalized.replace("∈", "E").replace("ε", "E")   # epsilon → E
    normalized = normalized.replace("8", "∧")                       # handwritten ∧ misread as 8
    normalized = normalized.replace("5", "S")                        # handwritten S misread as 5

    # Multi-char operators first to avoid partial replacements
    for src, tgt in _SYMBOL_REPLACEMENTS:
        normalized = normalized.replace(src, tgt)
    # Strip all remaining spaces for comparison
    normalized = _WHITESPACE_RE.sub("", normalized)
    return normalized


# ── Post-OCR repair patterns (Phase 2: handwritten ∧→8 confusion) ────

# Justification head: "8 E 3" → "∧E 3", "8 I 4,5" → "∧I 4,5".
# Only fires when '8' starts the string and is followed by I or E (TFL rules).
_EIGHT_BEFORE_EI_RE = re.compile(r"^8\s+(?=[IE]\b)")

# Formula: "K8W" or "K 8 W" → "K∧W".  Only fires between capital letters
# where '8' cannot be a legitimate formula token.
_EIGHT_BETWEEN_CAPS_RE = re.compile(r"(?<=[A-Z])\s*8\s*(?=[A-Z])")

_DASH_BETWEEN_DIGITS_RE = re.compile(r"(?<=\d)\s*[-–—]\s*(?=\d)")
_COMMA_BETWEEN_DIGITS_RE = re.compile(r"(?<=\d)\s*,\s*(?=\d)")
_SPACE_BETWEEN_DIGITS_RE = re.compile(r"(?<=\d)\s+(?=\d)")

# Justification head aliases (lowercase → canonical head)
_JUSTIFICATION_HEAD_ALIASES: dict[str, str] = {
    "pr": "Pr",
    "premise": "Pr",
    "prem": "Pr",
    "hyp": "Hyp",
    "assumption": "Hyp",
    "assume": "Hyp",
    "h": "Hyp",
    # Standard TFL rules
    "∧i": "∧I",
    "∧e": "∧E",
    "&i": "∧I",
    "&e": "∧E",
    "∨i": "∨I",
    "∨e": "∨E",
    "vi": "∨I",
    "ve": "∨E",
    "→i": "→I",
    "→e": "→E",
    "↔i": "↔I",
    "↔e": "↔E",
    "¬i": "¬I",
    "¬e": "¬E",
    "~i": "¬I",
    "~e": "¬E",
    "-i": "¬I",
    "-e": "¬E",
    # Explosion / falsum
    "⊥i": "⊥I",
    "⊥e": "⊥E",
    "xi": "⊥I",
    "xe": "⊥E",
    "boti": "⊥I",
    "bote": "⊥E",
    # Derived / shorthand
    "ip": "IP",
    "r": "R",
    "mt": "MT",
    "ds": "DS",
    "dne": "DNE",
    "dem": "DeM",
    "lem": "LEM",
}


def canonicalize_justification_for_comparison(value: Any) -> str:
    """Canonicalize a TFL justification for equality checks.

    Normalises the rule head, standardises dash/comma formatting in
    citations, and collapses whitespace.
    """

    normalized = _collapse_whitespace(value)
    if not normalized:
        return ""

    # Normalise arrows / symbols that may appear in rule heads
    for src, tgt in _SYMBOL_REPLACEMENTS:
        normalized = normalized.replace(src, tgt)
    normalized = normalized.replace("×", "X")
    normalized = re.sub(r"\b(X)(?=\d)", r"\1 ", normalized)

    # Phase 2a: repair epsilon/greek-epsilon misread as OCR artifact for E.
    # ∈ (U+2208) and ε (U+03B5) never appear in TFL justifications;
    # they are VLM misreads of the rule letter E (Elimination).
    # Must run before 8→∧ so that "8 ∈ S" → "8 E S" → "∧E S".
    normalized = normalized.replace("∈", "E").replace("ε", "E")

    # Phase 2b: repair handwritten ∧ misread as 8 in justification heads
    # e.g. "8 E 3" → "∧E 3", "8 I 4,5" → "∧I 4,5"
    normalized = _EIGHT_BEFORE_EI_RE.sub("∧", normalized)

    # Normalise dash/comma between digit citations
    normalized = normalized.replace("–", "-").replace("—", "-")
    normalized = _DASH_BETWEEN_DIGITS_RE.sub("-", normalized)
    normalized = _COMMA_BETWEEN_DIGITS_RE.sub(",", normalized)

    # Split into rule head + tail
    parts = normalized.split(" ", 1)
    head = parts[0].strip()
    tail = parts[1].strip() if len(parts) > 1 else ""

    # Convert bare spaces between digits in the tail to commas.
    # Handles OCR output like "1, 3-4 5-6" → "1,3-4,5-6" where the VLM
    # omits a comma between citation groups.
    if tail:
        tail = _SPACE_BETWEEN_DIGITS_RE.sub(",", tail)

    # Alias lookup (case-insensitive)
    alias = _JUSTIFICATION_HEAD_ALIASES.get(head.lower())
    if alias:
        head = alias

    return f"{head} {tail}".strip()
