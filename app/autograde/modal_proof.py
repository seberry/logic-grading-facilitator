from __future__ import annotations

from typing import Any, Dict

from app.modal_proof_deduct import (
    ModalProofCheckerProtocolError,
    ModalProofCheckerUnavailable,
    ModalProofConfigError,
    run_modal_proof_checker,
)

from .base import AutoGradeResult, AutoGradeTraceBlock


def autograde(
    answer_text: str,
    autograde_config: Dict[str, Any],
    runtime_options: Dict[str, Any] | None = None,
) -> AutoGradeResult:
    system = str(autograde_config.get("system", "")).strip()

    try:
        result = run_modal_proof_checker(
            answer_text=answer_text,
            autograde_config=autograde_config,
            runtime_options=runtime_options,
        )
    except ModalProofConfigError as exc:
        return AutoGradeResult(
            problem_type="modal_proof",
            status="invalid_input",
            passed=None,
            summary=str(exc),
            draft_comment=f"Modal proof checker could not run: {exc}",
            extracted_answer={"raw_answer": answer_text},
            metadata={"autograde_config": autograde_config},
            errors=[str(exc)],
        )
    except (ModalProofCheckerUnavailable, ModalProofCheckerProtocolError) as exc:
        return AutoGradeResult(
            problem_type="modal_proof",
            status="not_implemented",
            passed=None,
            summary=str(exc),
            draft_comment=f"Modal proof checker unavailable: {exc}",
            extracted_answer={"raw_answer": answer_text},
            metadata={"autograde_config": autograde_config},
            errors=[str(exc)],
        )

    parse_errors = result.parse_errors
    check_errors = result.check_errors

    passed = not parse_errors and not check_errors
    if result.conclusion_reached is not None:
        passed = passed and result.conclusion_reached

    summary = f"Modal proof checked under system {system or '(unknown)'}."
    if parse_errors:
        summary = f"Modal proof checked under system {system}; {len(parse_errors)} parse error(s) found."
    elif check_errors:
        summary = f"Modal proof checked under system {system}; {len(check_errors)} rule error(s) found."
    elif result.conclusion_reached is True:
        summary = f"Modal proof checked under system {system}; conclusion reached."
    elif result.conclusion_reached is False:
        summary = f"Modal proof checked under system {system}; conclusion not reached."

    comment_lines = [f"Modal proof checker system: {system or '(unknown)'}"]
    if result.conclusion_reached is not None:
        comment_lines.append(
            f"Conclusion reached: {'yes' if result.conclusion_reached else 'no'}"
        )
    if parse_errors:
        comment_lines.append(f"Parse errors: {len(parse_errors)}")
        for issue in parse_errors:
            comment_lines.append(f"- line {issue.line}: {issue.message}")
    if check_errors:
        comment_lines.append(f"Rule errors: {len(check_errors)}")
        for issue in check_errors:
            comment_lines.append(f"- line {issue.line}: {issue.message}")
    if not parse_errors and not check_errors:
        comment_lines.append("No parse or rule errors detected.")

    trace_blocks = []
    if parse_errors:
        trace_blocks.append(
            AutoGradeTraceBlock(
                title="Parse errors",
                result=f"{len(parse_errors)} issue(s)",
                body="\n".join(f"line {issue.line}: {issue.code} - {issue.message}" for issue in parse_errors),
            )
        )
    if check_errors:
        trace_blocks.append(
            AutoGradeTraceBlock(
                title="Rule errors",
                result=f"{len(check_errors)} issue(s)",
                body="\n".join(f"line {issue.line}: {issue.code} - {issue.message}" for issue in check_errors),
            )
        )
    if not parse_errors and not check_errors:
        trace_blocks.append(
            AutoGradeTraceBlock(
                title="Checker result",
                result="No errors",
                body=(
                    "Conclusion reached."
                    if result.conclusion_reached is True
                    else "Conclusion not requested."
                    if result.conclusion_reached is None
                    else "Conclusion not reached."
                ),
            )
        )

    score = None
    if passed and autograde_config.get("default_score_correct") is not None:
        score = float(autograde_config["default_score_correct"])
    elif not passed and autograde_config.get("default_score_incorrect") is not None:
        score = float(autograde_config["default_score_incorrect"])

    return AutoGradeResult(
        problem_type="modal_proof",
        status="scored",
        passed=passed,
        summary=summary,
        draft_comment="\n".join(comment_lines),
        score=score,
        trace_blocks=trace_blocks,
        extracted_answer={
            "raw_answer": answer_text,
            "normalized_lines": [
                {
                    "depth": line.depth,
                    "sentence": line.sentence,
                    "citation": line.citation,
                }
                for line in result.normalized_lines
            ],
        },
        metadata={
            "autograde_config": autograde_config,
            "checker_metadata": result.metadata,
        },
    )
