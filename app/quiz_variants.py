from __future__ import annotations

import re
from typing import Any, Dict, List, Optional


def normalize_quiz_version(value: Any) -> Optional[str]:
    """Normalize common model/user spellings such as ``Version A`` to ``A``."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    text = re.sub(r"^(?:quiz\s+)?(?:version|ver\.?|v)\s*[-:#]?\s*", "", text, flags=re.I)
    text = re.sub(r"[^A-Za-z0-9_-]+", "", text).upper()
    return text[:16] or None


def problem_variant(spec: Any, quiz_version: Optional[str]) -> Any:
    """Return the matching version override from a ProblemSpec, if present."""
    normalized = normalize_quiz_version(quiz_version)
    if not normalized:
        return None
    variants = getattr(spec, "variants", {}) or {}
    for key, value in variants.items():
        if normalize_quiz_version(key) == normalized:
            return value
    return None


def resolved_problem_context(spec: Any, quiz_version: Optional[str]) -> Dict[str, Any]:
    """Resolve display fields, falling back from the version override to the base spec."""
    override = problem_variant(spec, quiz_version)
    fields = (
        "label",
        "prompt",
        "question_image",
        "answer_key",
        "rubric_notes",
        "segmentation_hint",
    )
    result: Dict[str, Any] = {"quiz_version": normalize_quiz_version(quiz_version)}
    for field in fields:
        override_value = getattr(override, field, None) if override is not None else None
        result[field] = override_value if override_value is not None else getattr(spec, field, None)
    result["has_override"] = override is not None
    return result


def available_quiz_versions(grading_key: Any) -> List[str]:
    """Return normalized version keys in stable natural order."""
    found: set[str] = set()
    for spec in getattr(grading_key, "problems", []) or []:
        for key in (getattr(spec, "variants", {}) or {}).keys():
            normalized = normalize_quiz_version(key)
            if normalized:
                found.add(normalized)
    return sorted(found, key=lambda value: (len(value), value))
