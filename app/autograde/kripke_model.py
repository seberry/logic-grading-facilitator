from __future__ import annotations

from typing import Any, Dict

from .base import AutoGradeResult


def autograde(answer_text: str, autograde_config: Dict[str, Any]) -> AutoGradeResult:
    return AutoGradeResult(
        problem_type="kripke_model",
        status="not_implemented",
        passed=None,
        summary="Kripke model autograding scaffold exists, but this engine is not wired yet.",
        draft_comment=(
            "Autograder status: Kripke model engine scaffold only.\n\n"
            "The shared autograde contract is in place, but the checker/debug pipeline is still under construction."
        ),
        extracted_answer={"raw_answer": answer_text},
        metadata={"autograde_config": autograde_config},
    )
