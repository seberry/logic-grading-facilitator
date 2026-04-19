"""Helpers for deriving curated modal gold structures from typeset source.tex."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, List


_WHITESPACE_RE = re.compile(r"\s+")
_METAV_RE = re.compile(r"\\metav\{([^}]+)\}")
_HYPO_RE = re.compile(r"\\hypo\{(?P<label>[^}]+)\}\{(?P<formula>.+)\}\s*$")
_HAVE_RE = re.compile(r"\\have\{(?P<label>[^}]+)\}\{(?P<formula>.+?)\}(?P<rest>.*)$")
_BY_RE = re.compile(r"\\by\{(?P<rule>.+?)\}\{(?P<refs>[^}]+)\}")


def _collapse_whitespace(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text).strip()


def _normalize_formula(text: str) -> str:
    normalized = _METAV_RE.sub(r"\1", str(text or ""))
    for source, target in (
        ("\\ebox", "□"),
        ("\\ediamond", "◇"),
        ("\\enot", "¬"),
        ("\\eif", "→"),
        ("\\ered", "⊥"),
    ):
        normalized = normalized.replace(source, target)
    normalized = _collapse_whitespace(normalized)
    normalized = re.sub(r"([□◇¬])\s+", r"\1", normalized)
    normalized = normalized.replace("( ", "(").replace(" )", ")")
    return normalized


def _normalize_rule_label(text: str) -> str:
    return _normalize_formula(text)


def _parse_justification(rest: str) -> str:
    text = _collapse_whitespace(rest)
    if not text:
        return ""

    for pattern, label in (
        (r"\\boxe\{([^}]+)\}", "□E"),
        (r"\\boxi\{([^}]+)\}", "□I"),
        (r"\\ci\{([^}]+)\}", "→I"),
        (r"\\ce\{([^}]+)\}", "→E"),
        (r"\\ne\{([^}]+)\}", "¬E"),
        (r"\\rfour\{([^}]+)\}", "R4"),
        (r"\\rfive\{([^}]+)\}", "R5"),
    ):
        match = re.search(pattern, text)
        if match:
            return f"{label} {match.group(1)}"

    by_match = _BY_RE.search(text)
    if by_match:
        rule = _normalize_rule_label(by_match.group("rule"))
        refs = _collapse_whitespace(by_match.group("refs"))
        return f"{rule} {refs}".strip()

    return _normalize_rule_label(text)


def parse_modal_typeset_source(tex_text: str) -> List[Any]:
    """Parse the small teachlogic source.tex fixtures into curated gold JSON."""

    root: List[Any] = []
    stack: List[List[Any]] = [root]
    pending_opens = 0

    for raw_line in str(tex_text or "").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("%"):
            continue
        if r"\begin{nd}" in line or r"\end{nd}" in line:
            continue

        if line == r"\open":
            pending_opens += 1
            continue

        if "\\close" in line and line.replace(r"\close", "").strip() == "":
            close_count = line.count(r"\close")
            for _ in range(close_count):
                if len(stack) <= 1:
                    raise ValueError("Unbalanced \\close in source.tex")
                stack.pop()
            continue

        while pending_opens > 0:
            nested: List[Any] = []
            stack[-1].append(nested)
            stack.append(nested)
            pending_opens -= 1

        hypo_match = _HYPO_RE.match(line)
        if hypo_match:
            label = _collapse_whitespace(hypo_match.group("label"))
            stack[-1].append(
                {
                    "line_label": label,
                    "wffstr": _normalize_formula(hypo_match.group("formula")),
                    "jstr": "PR",
                    "line": int(label),
                }
            )
            continue

        have_match = _HAVE_RE.match(line)
        if have_match:
            label = _collapse_whitespace(have_match.group("label"))
            stack[-1].append(
                {
                    "line_label": label,
                    "wffstr": _normalize_formula(have_match.group("formula")),
                    "jstr": _parse_justification(have_match.group("rest")),
                    "line": int(label),
                }
            )
            continue

    if pending_opens:
        raise ValueError("Unbalanced \\open in source.tex")
    if len(stack) != 1:
        raise ValueError("Unbalanced subproof nesting in source.tex")
    return root


def load_modal_typeset_gold_from_source(path: Path) -> List[Any]:
    return parse_modal_typeset_source(path.read_text(encoding="utf-8"))
