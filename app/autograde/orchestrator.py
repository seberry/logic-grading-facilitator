from __future__ import annotations

from typing import Any, Dict

from .base import AutoGradeResult
from .registry import get_engine


def autograde_submission(
    problem_type: str | None,
    answer_text: str,
    autograde_config: Dict[str, Any] | None = None,
    runtime_options: Dict[str, Any] | None = None,
) -> AutoGradeResult:
    normalized_problem_type = str(problem_type or "").strip()
    config = autograde_config or {}
    runtime = runtime_options or {}

    if not normalized_problem_type:
        return AutoGradeResult(
            problem_type="",
            status="skipped",
            passed=None,
            summary="No problem_type is configured for this problem.",
            draft_comment="Autograder skipped: no problem type is configured.",
        )

    engine = get_engine(normalized_problem_type)
    if engine is None:
        return AutoGradeResult(
            problem_type=normalized_problem_type,
            status="unsupported",
            passed=None,
            summary=f"No autograde engine is registered for problem type '{normalized_problem_type}'.",
            draft_comment=(
                f"Autograder skipped: no engine is registered for problem type '{normalized_problem_type}'."
            ),
        )

    return engine(answer_text, config, runtime)
