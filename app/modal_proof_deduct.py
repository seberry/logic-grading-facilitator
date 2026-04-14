from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


DEDUCT_CHECKER_BIN_ENV = "DEDUCT_CHECKER_BIN"
SUPPORTED_MODAL_SYSTEMS = {"K", "T", "S4", "S5"}


class ModalProofConfigError(ValueError):
    """Raised when modal proof config or proof input is malformed."""


class ModalProofCheckerUnavailable(RuntimeError):
    """Raised when the external deduct checker cannot be executed."""


class ModalProofCheckerProtocolError(RuntimeError):
    """Raised when the external deduct checker returns malformed output."""


@dataclass
class ModalCheckerIssue:
    line: int
    code: str
    message: str


@dataclass
class ModalCheckerLine:
    depth: int
    sentence: str
    citation: str


@dataclass
class ModalCheckerResult:
    ok: bool
    parse_errors: List[ModalCheckerIssue]
    check_errors: List[ModalCheckerIssue]
    conclusion_reached: Optional[bool]
    normalized_lines: List[ModalCheckerLine]
    metadata: Dict[str, Any]
    request_payload: Dict[str, Any]
    response_payload: Dict[str, Any]


def _normalize_modal_payload(payload: Any) -> Dict[str, Any]:
    if isinstance(payload, list):
        return {"lines": payload, "meta": {}}
    if isinstance(payload, dict):
        lines = payload.get("lines")
        if isinstance(lines, list):
            return {"lines": lines, "meta": payload.get("meta", {})}
        structure = payload.get("structure")
        if isinstance(structure, list):
            return {"lines": structure, "meta": payload.get("meta", {})}
    raise ModalProofConfigError(
        "Modal proof answer_text must be JSON containing a list or an object with a 'lines' list."
    )


def _line_from_dict(item: Dict[str, Any], current_depth: int) -> ModalCheckerLine:
    location = item.get("location")
    depth = current_depth
    if isinstance(location, list) and location:
        depth = max(0, len(location) - 1)

    sentence = item.get("sentence", item.get("wffstr", ""))
    citation = item.get("citation", item.get("jstr", ""))

    return ModalCheckerLine(
        depth=depth,
        sentence=str(sentence).strip(),
        citation=str(citation).strip(),
    )


def _flatten_modal_lines(structure: List[Any], depth: int = 0) -> List[ModalCheckerLine]:
    flattened: List[ModalCheckerLine] = []
    for item in structure:
        if isinstance(item, list):
            flattened.extend(_flatten_modal_lines(item, depth + 1))
            continue
        if isinstance(item, dict):
            flattened.append(_line_from_dict(item, depth))
            continue
        raise ModalProofConfigError("Each proof line must be a JSON object or nested list.")
    return flattened


def build_modal_proof_request(answer_text: str, autograde_config: Dict[str, Any]) -> Dict[str, Any]:
    system = str(autograde_config.get("system", "")).strip()
    if system not in SUPPORTED_MODAL_SYSTEMS:
        raise ModalProofConfigError(
            f"modal_proof requires autograde.system to be one of {sorted(SUPPORTED_MODAL_SYSTEMS)}."
        )

    if not str(answer_text or "").strip():
        raise ModalProofConfigError("Modal proof answer_text is empty.")

    try:
        payload = json.loads(answer_text)
    except json.JSONDecodeError as exc:
        raise ModalProofConfigError(f"Modal proof answer_text is not valid JSON: {exc}") from exc

    normalized = _normalize_modal_payload(payload)
    flattened = _flatten_modal_lines(normalized["lines"])
    if not flattened:
        raise ModalProofConfigError("No modal proof lines were found in answer_text.")

    conclusion = autograde_config.get("conclusion")
    conclusion_value = str(conclusion).strip() if conclusion is not None else ""

    return {
        "system": system,
        "include_derived_rules": bool(autograde_config.get("include_derived_rules", False)),
        "conclusion": conclusion_value or None,
        "lines": [
            {
                "depth": line.depth,
                "sentence": line.sentence,
                "citation": line.citation,
            }
            for line in flattened
        ],
    }


def _resolve_checker_binary(runtime_options: Dict[str, Any] | None = None) -> str:
    runtime = runtime_options or {}
    configured = runtime.get("binary_path")
    candidate = str(configured).strip() if configured else os.getenv(DEDUCT_CHECKER_BIN_ENV, "").strip()
    if not candidate:
        raise ModalProofCheckerUnavailable(
            f"Modal proof checker is unavailable: set {DEDUCT_CHECKER_BIN_ENV} to the deduct CLI wrapper binary."
        )

    path = Path(candidate)
    if path.exists():
        return str(path)

    raise ModalProofCheckerUnavailable(
        f"Modal proof checker binary was not found at '{candidate}'."
    )


def _parse_issue_list(raw_issues: Any) -> List[ModalCheckerIssue]:
    issues: List[ModalCheckerIssue] = []
    if not isinstance(raw_issues, list):
        return issues
    for item in raw_issues:
        if not isinstance(item, dict):
            continue
        try:
            line = int(item.get("line", 0))
        except (TypeError, ValueError):
            line = 0
        issues.append(
            ModalCheckerIssue(
                line=line,
                code=str(item.get("code", "")).strip(),
                message=str(item.get("message", "")).strip(),
            )
        )
    return issues


def _parse_normalized_lines(raw_lines: Any) -> List[ModalCheckerLine]:
    lines: List[ModalCheckerLine] = []
    if not isinstance(raw_lines, list):
        return lines
    for item in raw_lines:
        if not isinstance(item, dict):
            continue
        try:
            depth = int(item.get("depth", 0))
        except (TypeError, ValueError):
            depth = 0
        lines.append(
            ModalCheckerLine(
                depth=depth,
                sentence=str(item.get("sentence", "")).strip(),
                citation=str(item.get("citation", "")).strip(),
            )
        )
    return lines


def run_modal_proof_checker(
    answer_text: str,
    autograde_config: Dict[str, Any],
    runtime_options: Dict[str, Any] | None = None,
) -> ModalCheckerResult:
    request_payload = build_modal_proof_request(answer_text, autograde_config)
    binary = _resolve_checker_binary(runtime_options)

    completed = subprocess.run(
        [binary],
        input=json.dumps(request_payload, ensure_ascii=False),
        capture_output=True,
        text=True,
        check=False,
    )

    if completed.returncode != 0:
        stderr = completed.stderr.strip() or completed.stdout.strip() or "unknown deduct wrapper failure"
        raise ModalProofCheckerUnavailable(
            f"Modal proof checker wrapper failed with exit code {completed.returncode}: {stderr}"
        )

    try:
        response = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise ModalProofCheckerProtocolError(
            f"Modal proof checker returned invalid JSON: {exc}"
        ) from exc

    if not isinstance(response, dict):
        raise ModalProofCheckerProtocolError("Modal proof checker returned a non-object JSON response.")

    conclusion_reached_raw = response.get("conclusion_reached")
    conclusion_reached = conclusion_reached_raw if isinstance(conclusion_reached_raw, bool) else None

    metadata = response.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}

    return ModalCheckerResult(
        ok=bool(response.get("ok", False)),
        parse_errors=_parse_issue_list(response.get("parse_errors")),
        check_errors=_parse_issue_list(response.get("check_errors")),
        conclusion_reached=conclusion_reached,
        normalized_lines=_parse_normalized_lines(response.get("normalized_lines")),
        metadata=metadata,
        request_payload=request_payload,
        response_payload=response,
    )
