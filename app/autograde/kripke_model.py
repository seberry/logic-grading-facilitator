from __future__ import annotations

from typing import Any, Dict

from app.kripke_model_checker import KripkeModelParseError, check_kripke_task

from .base import AutoGradeResult, AutoGradeTraceBlock


def autograde(
    answer_text: str,
    autograde_config: Dict[str, Any],
    runtime_options: Dict[str, Any] | None = None,
) -> AutoGradeResult:
    runtime = runtime_options or {}
    trace_mode = str(runtime.get("trace_mode") or "terse")

    try:
        task_result = check_kripke_task(answer_text, autograde_config, trace_mode=trace_mode)
    except KripkeModelParseError as exc:
        return AutoGradeResult(
            problem_type="kripke_model",
            status="invalid_input",
            passed=None,
            summary=str(exc),
            draft_comment=f"Autograder could not parse this Kripke model setup.\n\nError: {exc}",
            errors=[str(exc)],
        )

    return AutoGradeResult(
        problem_type="kripke_model",
        status="scored",
        passed=task_result.passed,
        summary=task_result.summary,
        draft_comment=task_result.draft_comment,
        score=_score_from_config(task_result.passed, autograde_config),
        trace_blocks=[
            AutoGradeTraceBlock(
                title=block.get("title", ""),
                result=block.get("result", ""),
                body=block.get("body", ""),
            )
            for block in task_result.trace_blocks
        ],
        extracted_answer={"model_text": answer_text},
        metadata={
            "task_kind": task_result.task_kind,
            "model_summary": task_result.model_summary,
            "trace_mode": trace_mode,
        },
    )


def _score_from_config(passed: bool, autograde_config: Dict[str, Any]) -> float | None:
    value = autograde_config.get("default_score_correct") if passed else autograde_config.get("default_score_incorrect")
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
