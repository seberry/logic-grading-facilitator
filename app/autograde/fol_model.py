from __future__ import annotations

from typing import Any, Dict, List

from app.fol_model_checker import FolModelParseError, check_fol_task

from .base import AutoGradeResult, AutoGradeTraceBlock


def autograde(answer_text: str, autograde_config: Dict[str, Any]) -> AutoGradeResult:
    try:
        task_result = check_fol_task(answer_text, autograde_config, trace_mode="terse")
    except FolModelParseError as exc:
        return AutoGradeResult(
            problem_type="fol_model",
            status="invalid_input",
            passed=None,
            summary=str(exc),
            draft_comment=f"Autograder could not parse this FOL model setup.\n\nError: {exc}",
            errors=[str(exc)],
        )

    trace_blocks: List[AutoGradeTraceBlock] = [
        AutoGradeTraceBlock(
            title=block.get("title", ""),
            result=block.get("result", ""),
            body=block.get("body", ""),
        )
        for block in task_result.trace_blocks
    ]

    return AutoGradeResult(
        problem_type="fol_model",
        status="scored",
        passed=task_result.passed,
        summary=task_result.summary,
        draft_comment=task_result.draft_comment,
        score=_score_from_config(task_result.passed, autograde_config),
        trace_blocks=trace_blocks,
        extracted_answer={"model_text": answer_text},
        metadata={
            "task_kind": task_result.task_kind,
            "model_summary": task_result.model_summary,
        },
    )


def _score_from_config(passed: bool, autograde_config: Dict[str, Any]) -> float | None:
    if passed:
        value = autograde_config.get("default_score_correct")
    else:
        value = autograde_config.get("default_score_incorrect")
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
