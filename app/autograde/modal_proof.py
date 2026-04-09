from __future__ import annotations

from typing import Any, Dict

from .base import AutoGradeResult


def autograde(answer_text: str, autograde_config: Dict[str, Any]) -> AutoGradeResult:
    return AutoGradeResult(
        problem_type="modal_proof",
        status="not_implemented",
        passed=None,
        summary="Modal proof autograding scaffold exists, but this engine is not wired yet.",
        draft_comment=(
            "Autograder status: modal proof engine scaffold only.\n\n"
            "The deterministic checker integration has not been added yet."
        ),
        extracted_answer={"raw_answer": answer_text},
        metadata={"autograde_config": autograde_config},
    )
