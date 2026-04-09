from __future__ import annotations

from typing import Any, Dict

from .base import AutoGradeResult


def autograde(
    answer_text: str,
    autograde_config: Dict[str, Any],
    runtime_options: Dict[str, Any] | None = None,
) -> AutoGradeResult:
    return AutoGradeResult(
        problem_type="fitch_proof",
        status="not_implemented",
        passed=None,
        summary="Fitch proof autograding scaffold exists, but this engine is not wired yet.",
        draft_comment=(
            "Autograder status: Fitch proof engine scaffold only.\n\n"
            "Use the debug OCR/verifier workflow for now; strip-view autograding is not wired yet."
        ),
        extracted_answer={"raw_answer": answer_text},
        metadata={"autograde_config": autograde_config},
    )
