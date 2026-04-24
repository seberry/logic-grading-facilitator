"""Justification rule-head snapping tricks for Fitch OCR pipelines.

VLMs systematically misread handwritten justification rule heads. This
module collects the known OCR confusions as composable repair functions
that both TFL and modal pipelines can reuse.

The primary entry point is :func:`snap_justification_rule_head`, which
applies a table of known OCR misread → canonical rule corrections.

Additional helpers handle specific confusions (8 for ∧, epsilon for E,
etc.) that can be applied independently or as part of a normalization
stack.
"""

from __future__ import annotations

import re
from typing import Any

# ── Shared regex patterns ──────────────────────────────────────────────

_WHITESPACE_RE = re.compile(r"\s+")

# Bare line numbers or numeric noise in the justification column that
# should be treated as empty (the real justification was lost).
_BARE_NUMERIC_RE = re.compile(r"^[\d,.\-]+$")

# Split a justification into (rule_head, tail) preserving spacing.
_HEAD_TAIL_RE = re.compile(r"\s+")


# ── OCR confusion tables ──────────────────────────────────────────────

# Handwritten ∧ misread as 8 — only valid before I/E rule suffixes.
_EIGHT_BEFORE_EI_RE = re.compile(r"^8\s+(?=[IE]\b)")

# Handwritten ∧ misread as 8 between capital atoms in formulas.
# e.g. "K8W" or "K 8 W" → "K∧W"
_EIGHT_BETWEEN_CAPS_RE = re.compile(r"(?<=[A-Z])\s*8\s*(?=[A-Z])")


def snap_8_before_ei(text: str) -> str:
    """Repair handwritten ∧ misread as ``8`` in justification heads.

    Handles ``8 E 3`` → ``∧E 3`` and ``8 I 4,5`` → ``∧I 4,5``.
    Only fires when ``8`` starts the string and is followed by I or E.
    """
    return _EIGHT_BEFORE_EI_RE.sub("∧", text)


def snap_8_between_caps(text: str) -> str:
    """Repair handwritten ∧ misread as ``8`` between capital atoms.

    Handles ``K8W`` → ``K∧W``. Only fires between uppercase letters
    where ``8`` cannot be a legitimate formula token.
    """
    return _EIGHT_BETWEEN_CAPS_RE.sub("∧", text)


def snap_epsilon_to_e(text: str) -> str:
    """Replace OCR-artifact epsilon/greek-epsilon with plain E.

    ``∈`` (U+2208) and ``ε`` (U+03B5) never appear in TFL or modal
    justifications; they are VLM misreads of the rule letter E
    (Elimination).
    """
    return text.replace("∈", "E").replace("ε", "E")


# ── Full snap table from tfl_fitch.py ─────────────────────────────────
# Maps OCR-misread rule heads to their canonical equivalents.

#: Shared OCR confusions for rule heads — valid for both TFL and modal.
SHARED_HEAD_FIXES: dict[str, str] = {
    # ← read instead of ↔
    "←E": "↔E",
    "←I": "↔I",
    "←F": "↔E",
    # ↔F: F misread for E in ↔E
    "↔F": "↔E",
    # ± read instead of ↔I
    "±": "↔I",
    # Handwritten ∧ looks like 8
    "8E": "∧E",
    "8I": "∧I",
    # ∈ in OCR output
    "∈E": "↔E",
    "∈I": "↔I",
    "∈S": "∧E",
    # epsilon/fallback
    "εE": "∧E",
    "εI": "∧I",
    "εS": "∧E",
    # BE/BI from ∧ → B confusion
    "BE": "∧E",
    "BI": "∧I",
    # XE/XI for ⊥E/⊥I (explosion)
    "XE": "⊥E",
    "XI": "⊥I",
}

#: Modal-specific OCR confusions for rule heads.
MODAL_HEAD_FIXES: dict[str, str] = {
    # These may be added as modal regression testing reveals systematic
    # misreads of □I, □E, R4, R5, etc.
}


def snap_justification_rule_head(
    jstr: str,
    *,
    extra_fixes: dict[str, str] | None = None,
) -> str:
    """Post-normalization snap: fix rule heads that are not yet canonical.

    Splits the justification into (head, tail), looks up the head in the
    fix table, and reconstructs.  If the head is already valid or not in
    the table, returns the input unchanged.

    Parameters
    ----------
    jstr:
        Justification string (already symbol-normalized).
    extra_fixes:
        Optional additional head → canonical mappings to merge with the
        shared table (e.g. modal-specific fixes).
    """
    if not jstr:
        return jstr

    fixes = dict(SHARED_HEAD_FIXES)
    if extra_fixes:
        fixes.update(extra_fixes)

    parts = jstr.strip().split(None, 1)
    head = parts[0]
    tail = parts[1] if len(parts) > 1 else ""

    if head in fixes:
        fixed = fixes[head]
        return (fixed + " " + tail).strip() if tail else fixed

    return jstr


# ── Bare-line-number detection ────────────────────────────────────────

def is_bare_numeric(jstr: str) -> bool:
    """Return True if justification is just digits, commas, dots, dashes.

    This indicates OCR noise — the real justification was lost and only
    line-number fragments ended up in the justification column.
    """
    return bool(_BARE_NUMERIC_RE.fullmatch(jstr.strip()))


# ── Premise / assumption label aliases ────────────────────────────────
# Shared across both pipelines — both need to recognize the same set of
# VLM outputs for premise/assumption labels.

PREMISE_ALIAS_HEADS: frozenset[str] = frozenset({
    "pr",
    "prem",
    "premise",
    "hyp",
    "assumption",
    "assume",
    "assum",
    "assump",
})

#: Canonical TFL justification head aliases (lowercase → canonical).
TFL_JUSTIFICATION_ALIASES: dict[str, str] = {
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

#: Canonical modal justification head aliases (lowercase → canonical).
#: Extends TFL aliases with modal-specific rules.
MODAL_JUSTIFICATION_ALIASES: dict[str, str] = {
    **TFL_JUSTIFICATION_ALIASES,
    "andi": "∧I",
    "ande": "∧E",
    "ai": "∧I",
    "ae": "∧E",
    "ori": "∨I",
    "ore": "∨E",
    "impi": "→I",
    "impe": "→E",
    "ci": "→I",
    "ce": "→E",
    "bici": "↔I",
    "bice": "↔E",
    "noti": "¬I",
    "note": "¬E",
    "ni": "¬I",
    "ne": "¬E",
    "dn": "DN",
    "mc": "MC",
    "boxi": "□I",
    "boxe": "□E",
    "oi": "□I",
    "de": "□E",
    "r4": "R4",
    "rfour": "R4",
    "r5": "R5",
    "rfive": "R5",
    "rt": "RT",
    "defdiamond": "Def⋄",
    "def⋄": "Def⋄",
    "def◇": "Def⋄",
    "def◊": "Def⋄",
}


def resolve_justification_head(
    head: str,
    *,
    aliases: dict[str, str] | None = None,
) -> str:
    """Resolve a justification rule head via alias lookup.

    Parameters
    ----------
    head:
        Raw rule head text (e.g. ``"ande"``, ``"~E"``).
    aliases:
        Alias table to use.  Defaults to :data:`TFL_JUSTIFICATION_ALIASES`.
    """
    if aliases is None:
        aliases = TFL_JUSTIFICATION_ALIASES
    return aliases.get(head.lower(), head)