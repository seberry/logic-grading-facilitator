"""Candidate classification helpers for verifier-guided TFL row rereads."""

from __future__ import annotations

import re
from collections.abc import Mapping as MappingABC
from typing import Any, Mapping


def normalize_candidate_text(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"\s+", "", text)
    text = text.replace("->", "→").replace("&", "∧")
    return text


def parsed_line_number(parsed: Any) -> int | None:
    if not isinstance(parsed, MappingABC):
        return None
    raw = parsed.get("line")
    if isinstance(raw, int):
        return raw
    text = str(raw or "").strip()
    if text.isdigit():
        return int(text)
    return None


def candidate_materially_changed(
    suspicion: Mapping[str, Any],
    parsed: Any,
    *,
    line_matches_target: bool,
) -> bool:
    if not isinstance(parsed, MappingABC):
        return False
    if not line_matches_target:
        return False
    old_wff = normalize_candidate_text(suspicion.get("wffstr"))
    old_j = normalize_candidate_text(suspicion.get("jstr"))
    new_wff = normalize_candidate_text(parsed.get("wffstr"))
    new_j = normalize_candidate_text(parsed.get("jstr"))
    return bool(new_wff or new_j) and (old_wff != new_wff or old_j != new_j)


def parse_citation_line_numbers(jstr: str) -> list[int]:
    """Extract cited line numbers from a TFL justification string."""
    numbers: set[int] = set()
    for start, end in re.findall(r"(\d+)\s*[-–]\s*(\d+)", jstr):
        a = int(start)
        b = int(end)
        lo, hi = sorted((a, b))
        numbers.update(range(lo, hi + 1))
    without_ranges = re.sub(r"\d+\s*[-–]\s*\d+", " ", jstr)
    for num in re.findall(r"\d+", without_ranges):
        numbers.add(int(num))
    return sorted(numbers)
