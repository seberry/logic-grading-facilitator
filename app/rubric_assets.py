from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Optional
from urllib.parse import quote

from .quiz_variants import available_quiz_versions, normalize_quiz_version, resolved_problem_context
from .storage import RUBRIC_ASSETS_DIR


def safe_asset_component(value: object, fallback: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "").strip()).strip("._")
    return cleaned or fallback


def conventional_question_image_ref(quiz_id: str, problem_id: str, quiz_version: str) -> str:
    """Return the portable conventional path used by the extraction script."""
    return "/".join(
        (
            safe_asset_component(quiz_id, "quiz"),
            f"{safe_asset_component(problem_id, 'problem')}_{safe_asset_component(quiz_version, 'UNKNOWN')}.png",
        )
    )


def resolve_rubric_asset(reference: object, root: Path = RUBRIC_ASSETS_DIR) -> Optional[Path]:
    """Resolve a relative rubric asset while rejecting absolute/traversal paths."""
    text = str(reference or "").strip().replace("\\", "/")
    if not text:
        return None
    relative = Path(text)
    if relative.is_absolute() or ".." in relative.parts:
        return None
    root = root.resolve()
    candidate = (root / relative).resolve()
    if candidate != root and root not in candidate.parents:
        return None
    return candidate


def question_image_reference(grading_key, spec, quiz_version: Optional[str]) -> Optional[str]:
    """Use an explicit rubric path, then fall back to the conventional filename."""
    version = normalize_quiz_version(quiz_version)
    context = resolved_problem_context(spec, version)
    explicit = str(context.get("question_image") or "").strip().replace("\\", "/")
    if explicit:
        return explicit
    if version:
        return conventional_question_image_ref(grading_key.quiz_id, spec.id, version)
    return None


def question_image_url(grading_key, spec, quiz_version: Optional[str]) -> Optional[str]:
    reference = question_image_reference(grading_key, spec, quiz_version)
    path = resolve_rubric_asset(reference)
    if not reference or path is None or not path.is_file():
        return None
    return f"/rubric_assets/{quote(reference, safe='/')}"


def existing_question_assets(grading_key, root: Path = RUBRIC_ASSETS_DIR) -> Dict[str, Path]:
    """Collect declared/conventional blank-quiz crops for a portable TF bundle."""
    assets: Dict[str, Path] = {}
    versions = available_quiz_versions(grading_key)
    for spec in grading_key.problems:
        candidates = versions or [None]
        for version in candidates:
            reference = question_image_reference(grading_key, spec, version)
            path = resolve_rubric_asset(reference, root=root)
            if reference and path is not None and path.is_file():
                assets[reference] = path
    return assets
