"""Citation cleanup tricks for Fitch OCR pipelines.

Handles standardization of line-number citations in justification text.
Both TFL and modal pipelines need the same dash/comma/space normalization
for citation ranges like ``3-5``, ``4,7``, ``3 5`` (missing comma), so
these are extracted into a shared module.
"""

from __future__ import annotations

import re

# ── Regex patterns ─────────────────────────────────────────────────────

_DASH_BETWEEN_DIGITS_RE = re.compile(r"(?<=\d)\s*[-–—]\s*(?=\d)")
_COMMA_BETWEEN_DIGITS_RE = re.compile(r"(?<=\d)\s*,\s*(?=\d)")
_SPACE_BETWEEN_DIGITS_RE = re.compile(r"(?<=\d)\s+(?=\d)")


def normalize_citation_dashes(text: str) -> str:
    """Standardize citation range dashes to plain ASCII hyphens.

    OCR often produces en-dash (–) or em-dash (—) instead of ``-``.
    """
    text = text.replace("–", "-").replace("—", "-")
    return _DASH_BETWEEN_DIGITS_RE.sub("-", text)


def normalize_citation_commas(text: str) -> str:
    """Normalize spacing around commas between digit citations.

    Ensures ``4 , 7`` becomes ``4,7`` (no space before, no space after
    within digit sequences).
    """
    return _COMMA_BETWEEN_DIGITS_RE.sub(",", text)


def normalize_citation_spaces(text: str) -> str:
    """Convert bare spaces between digit groups to commas.

    Handles OCR output like ``1, 3-4 5-6`` → ``1,3-4,5-6`` where the
    VLM omits a comma between citation groups.
    """
    return _SPACE_BETWEEN_DIGITS_RE.sub(",", text)


def normalize_citations(text: str) -> str:
    """Apply all citation cleanup normalizations in order.

    Pipeline: dashes → commas → spaces.
    """
    text = normalize_citation_dashes(text)
    text = normalize_citation_commas(text)
    text = normalize_citation_spaces(text)
    return text