"""Shared placeholder expansion and proof stitching helpers.

These helpers cover the structural part of recursive proof OCR:

1. detect a synthetic child placeholder such as ``<SUBPROOF_ID_2>``
2. replace that placeholder with the already-resolved child block
3. recurse until the full nested proof structure is reconstructed

They intentionally do not own domain-specific line normalization or modal-only
fallback insertion heuristics.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Callable, Iterable, List, Mapping, Sequence

TOKEN_PATTERN = re.compile(r"<SUBPROOF_ID_([A-Za-z0-9_]+)>")

TokenResolver = Callable[[Any, int], Any | None]

_logger = logging.getLogger(__name__)


def extract_subproof_token_id(item: Any) -> str | None:
    """Return the referenced subproof id if an item contains a placeholder."""

    if isinstance(item, str):
        match = TOKEN_PATTERN.search(item)
        return match.group(1) if match else None

    if isinstance(item, Mapping):
        for value in item.values():
            if not isinstance(value, str):
                continue
            match = TOKEN_PATTERN.search(value)
            if match:
                return match.group(1)

    return None


def recursive_stitch(
    items: Sequence[Any],
    library: Mapping[str, Sequence[Any]],
    *,
    resolve_item: TokenResolver | None = None,
    _visited: frozenset[str] | None = None,
) -> List[Any]:
    """Expand child placeholders into a nested proof structure.

    ``items`` should contain either line-like records or placeholder-bearing
    records that refer to previously OCR'd child scopes stored in ``library``.
    The optional ``resolve_item`` callback lets a pipeline convert intermediate
    OCR entries into final line dictionaries while keeping the placeholder
    expansion logic shared.

    Cycle protection: if a placeholder references an ID that is already on the
    current expansion path (tracked via ``_visited``), the placeholder is treated
    as a plain line instead of being expanded — preventing infinite recursion
    caused by VLM hallucinated back-references.
    """
    if _visited is None:
        _visited = frozenset()

    resolved: List[Any] = []
    for index, item in enumerate(items):
        child_id = extract_subproof_token_id(item)
        if child_id and child_id in library:
            if child_id in _visited:
                # Cycle detected — treat as a plain line to avoid infinite recursion
                _logger.warning(
                    "Cycle detected: <SUBPROOF_ID_%s> references an ancestor "
                    "already on the expansion path; treating as plain line.",
                    child_id,
                )
                normalized = (
                    resolve_item(item, index) if resolve_item is not None else item
                )
                if normalized is not None:
                    resolved.append(normalized)
                continue

            resolved.append(
                recursive_stitch(
                    library[child_id],
                    library,
                    resolve_item=resolve_item,
                    _visited=_visited | {child_id},
                )
            )
            continue

        if isinstance(item, list):
            resolved.append(
                recursive_stitch(
                    item,
                    library,
                    resolve_item=resolve_item,
                    _visited=_visited,
                )
            )
            continue

        normalized = resolve_item(item, index) if resolve_item is not None else item
        if normalized is None:
            continue
        resolved.append(normalized)
    return resolved


def renumber_lines(proof_structure: Iterable[Any], start_counter: int = 1) -> int:
    """Renumber nested proof lines in depth-first order."""

    current_counter = start_counter
    for item in proof_structure:
        if isinstance(item, list):
            current_counter = renumber_lines(item, current_counter)
        elif isinstance(item, dict):
            item["line"] = current_counter
            current_counter += 1
    return current_counter