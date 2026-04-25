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

# Bare double line-range pattern — "3-6 7-10" or "3-6,7-10"
_BARE_DOUBLE_RANGE_RE = re.compile(r"\d+\s*-\s*\d+\s*[,;\s]+\s*\d+\s*-\s*\d+")


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
    valid_rules: frozenset[str] | None = None,
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
    valid_rules:
        Optional set of already-canonical rule heads.  If the extracted
        head is in this set, the string is returned unchanged without
        consulting the fix table.
    """
    if not jstr:
        return jstr

    fixes = dict(SHARED_HEAD_FIXES)
    if extra_fixes:
        fixes.update(extra_fixes)

    parts = jstr.strip().split(None, 1)
    head = parts[0]
    tail = parts[1] if len(parts) > 1 else ""

    # Already valid — nothing to do
    if valid_rules and head in valid_rules:
        return jstr

    if head in fixes:
        fixed = fixes[head]
        return (fixed + " " + tail).strip() if tail else fixed

    # Bare double line-range with no rule head: "3-6 7-10" → "↔I 3-6 7-10"
    full_text = jstr.strip()
    if _BARE_DOUBLE_RANGE_RE.fullmatch(full_text):
        return "↔I " + full_text

    return jstr


# ── Fuzzy justification repair via combinatorial OCR substitution ──────

#: Known OCR character confusions for rule heads.
#: Each tuple is (ocr_char, true_char).  Applied combinatorially when
#: the post-normalization rule head is not in *valid_rules*.
_OCR_HEAD_CONFUSIONS: list[tuple[str, str]] = [
    ("8", "∧"),
    ("∈", "E"),
    ("ε", "E"),
    ("B", "∧"),
    ("X", "⊥"),
    ("±", "↔"),
    ("S", "5"),
    ("O", "0"),
]

#: OCR confusions that can appear in citation tails (line-number refs).
#: Only digit ↔ letter confusions belong here.
_OCR_TAIL_CONFUSIONS: list[tuple[str, str]] = [
    ("S", "5"),
    ("O", "0"),
]


def fuzzy_snap_justification(
    jstr: str,
    *,
    valid_rules: frozenset[str] | None = None,
) -> str:
    """Attempt combinatorial OCR-substitution repair on an invalid justification.

    After existing normalization (symbol snaps, alias lookup), if the rule
    head is still not a recognized justification, this function tries all
    combinations of known OCR character substitutions on the head.  If exactly
    one substitution combination yields a valid rule, it is accepted.

    Also applies digit-like substitutions (``S→5``, ``O→0``) on the
    citation tail.

    Parameters
    ----------
    jstr:
        Already-normalized justification string.
    valid_rules:
        Set of valid canonical rule heads.  Defaults to
        :data:`TFL_VALID_RULE_HEADS`.

    Returns
    -------
    str
        The repaired justification, or the original if no unique repair
        was found.
    """
    if valid_rules is None:
        valid_rules = TFL_VALID_RULE_HEADS

    if not jstr:
        return jstr

    parts = jstr.strip().split(None, 1)
    head = parts[0]
    tail = parts[1] if len(parts) > 1 else ""

    # --- Tail digit rescue: S→5, O→0 in line-number references ---
    # This runs regardless of whether the head is already valid.
    if tail:
        for ocr_char, true_char in _OCR_TAIL_CONFUSIONS:
            tail = tail.replace(ocr_char, true_char)

    # Already valid — nothing more to do (return with fixed tail)
    if head in valid_rules:
        return f"{head} {tail}".strip() if tail else head

    # --- Combinatorial head repair ---
    # Collect which characters in the head appear in the confusion table
    confusion_positions: list[tuple[int, list[str]]] = []
    for i, ch in enumerate(head):
        options = {ch}
        for ocr_char, true_char in _OCR_HEAD_CONFUSIONS:
            if ch == ocr_char:
                options.add(true_char)
            elif ch == true_char:
                options.add(ocr_char)
        if len(options) > 1:
            confusion_positions.append((i, sorted(options)))

    if not confusion_positions:
        # No applicable confusions — return with tail fixed
        return f"{head} {tail}".strip() if tail else head

    # Generate all substitution combinations
    candidates: list[str] = []
    head_chars = list(head)

    def _generate(pos_index: int) -> None:
        if pos_index >= len(confusion_positions):
            candidates.append("".join(head_chars))
            return
        str_pos, options = confusion_positions[pos_index]
        original = head_chars[str_pos]
        for opt in options:
            head_chars[str_pos] = opt
            _generate(pos_index + 1)
        head_chars[str_pos] = original

    _generate(0)

    # Filter to valid rule heads
    valid_candidates = [c for c in candidates if c in valid_rules]

    if len(valid_candidates) == 1:
        fixed_head = valid_candidates[0]
        return f"{fixed_head} {tail}".strip() if tail else fixed_head

    # 0 or 2+ matches — return with tail fixed but head unchanged
    return f"{head} {tail}".strip() if tail else head


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
    "bot": "⊥",
    ".e": "∧E",
    ".i": "∧I",
    # Derived / shorthand
    "ip": "IP",
    "r": "R",
    "mt": "MT",
    "ds": "DS",
    "dne": "DNE",
    "dem": "DeM",
    "lem": "LEM",
}

#: Valid post-normalization TFL rule heads.
TFL_VALID_RULE_HEADS: frozenset[str] = frozenset({
    "Pr", "Hyp",
    "∧I", "∧E", "∨I", "∨E",
    "→I", "→E", "↔I", "↔E",
    "¬I", "¬E", "⊥I", "⊥E",
})

#: Broader set of known rule heads for pre-normalization matching.
#: Includes ASCII/loose forms used by VLM before normalization.
TFL_KNOWN_RULE_HEADS: frozenset[str] = frozenset({
    "Pr", "PR", "Hyp", "HYP", "premise", "Premise", "assumption", "Assumption",
    "∨E", "∨I", "∧E", "∧I", "¬E", "¬I", "→E", "→I", "↔E", "↔I",
    "vE", "vI", "&E", "&I", ".E", ".I", "~E", "~I", "-E", "-I",
    "AE", "AI", "MP", "MT", "DS", "DNE", "LEM", "DeM", "DEM", "X", "IP", "R",
})

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


# ── TFL justification normalization ───────────────────────────────────

def clean_justification_for_tfl(jstr: str) -> str:
    """Full TFL justification normalization: symbols, spaces, alias lookup.

    Applies in order:
    1. Symbol normalization (→, ↔, ←→↔, ∧, ¬)
    2. Space collapsing around operator+I/E patterns
    3. snap_8_before_ei (8 E → ∧E, 8 I → ∧I)
    4. snap_epsilon_to_e (∈→E, ε→E)
    5. Alias lookup via TFL_JUSTIFICATION_ALIASES (lowercase key)
    6. Uppercase fallback for unrecognized rules
    7. AE→∧E, AI→∧I correction (common ∧↔A confusion)
    """
    if not jstr:
        return ""
    jstr = jstr.replace("->", "→").replace("<->", "↔").replace("<→", "↔")
    jstr = jstr.replace("←", "↔")
    # Collapse spaces: "8 E" -> "8E", "8 I" -> "8I" (handwritten ∧ looks like 8)
    jstr = re.sub(r"\b8\s+([IE])\b", r"8\1", jstr)
    jstr = re.sub(r"([¬∧∨→↔])\s+([IE])\b", r"\1\2", jstr)
    jstr = jstr.replace("&", "∧").replace("~", "¬")
    jstr = re.sub(r"\b([∧∨¬→↔A-Z]{1,3})\s*([IE])(?=\d)", r"\1\2 ", jstr)
    jstr = re.sub(r"\b(X|IP|R|MT|DS|DNE|LEM|DeM|DEM)(?=\d)", r"\1 ", jstr)
    jstr = re.sub(r"\b([∧∨¬→↔A-Z]{1,4})(\d)", r"\1 \2", jstr)

    # Split into Rule and Lines (e.g., "vE 2" -> "vE", "2")
    parts = jstr.strip().split(" ", 1)
    rule = parts[0].lower()
    remainder = parts[1] if len(parts) > 1 else ""

    # Map the rule if known
    if rule in TFL_JUSTIFICATION_ALIASES:
        rule = TFL_JUSTIFICATION_ALIASES[rule]
    else:
        # Try to fix casing for standard rules (e.g., "mp" -> "MP")
        rule = rule.upper()
        # Fix specific case-sensitive ones from verifier.py
        if rule == "AE":
            rule = "∧E"
        if rule == "AI":
            rule = "∧I"

    return f"{rule} {remainder}".strip()
