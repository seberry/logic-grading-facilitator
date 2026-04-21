from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import cv2

from proofs_ocr.domains import modal
from proofs_ocr.pipelines.modal_v2_formula_rereads import (
    build_boxed_formula_context_crop,
    build_formula_crop,
    upscale_for_reread,
)

from . import modal_proof_debug_artifacts as artifact_helpers
from . import modal_proof_debug_page as page_helpers

FORMULA_REVIEW_FILENAME = "checker_formula_rereads.json"
_LINE_REF_RE = re.compile(r"(?<!\d)(\d+)(?:\s*[-–—]\s*(\d+))?")


def run_checker_guided_formula_review(
    *,
    example_dir: Path,
    state: Mapping[str, Any],
    client: Any,
    modal_config: Mapping[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    """Generate checker-guided formula reread candidates from saved modal_v2 line crops."""

    proof_lines = page_helpers.proof_lines_from_text(str(state.get("proof_json_text") or ""))
    checker_response = state.get("checker_response")
    target_reasons = _collect_target_reasons(checker_response=checker_response, proof_lines=proof_lines)
    source_lookup = artifact_helpers.build_line_number_source_lookup(example_dir)
    records: List[Dict[str, Any]] = []

    for line_number in _ordered_target_line_numbers(target_reasons):
        source = source_lookup.get(line_number)
        current_line = proof_lines.get(line_number)
        if not source or not isinstance(current_line, dict):
            continue
        reasons = list(target_reasons.get(line_number) or [])
        if not _should_review_line(
            line_number=line_number,
            target_reasons=reasons,
            source_record=source,
        ):
            continue
        current_formula = modal.normalize_formula_text(current_line.get("formula") or "")
        if not current_formula:
            continue

        line_crop_path = Path(str(source.get("line_crop_path") or ""))
        line_crop_img = cv2.imread(str(line_crop_path))
        if line_crop_img is None or line_crop_img.size == 0:
            continue

        attempts = _run_formula_attempts(
            client=client,
            example_dir=example_dir,
            line_number=line_number,
            line_crop_img=line_crop_img,
            modal_config=modal_config,
        )
        if not attempts:
            continue

        structured_formula = str(attempts[0].get("structured_formula") or "").strip()
        forced_choice = str(attempts[1].get("structured_formula") or "").strip() if len(attempts) > 1 else ""
        proposed_correction = _select_formula_proposal(
            baseline_formula=current_formula,
            attempt_formulas=[str(item.get("structured_formula") or "").strip() for item in attempts],
            target_reasons=reasons,
        )
        records.append(
            {
                "line_number": int(line_number),
                "line_label": str(line_number),
                "baseline_formula": current_formula,
                "structured_formula": structured_formula,
                "forced_choice": forced_choice,
                "proposed_correction": proposed_correction,
                "attempt_kind": "checker_guided_formula",
                "target_reasons": reasons,
                "line_crop_path": str(line_crop_path),
                "candidate_attempts": attempts,
            }
        )

    write_formula_review_results(example_dir=example_dir, records=records)
    return records


def write_formula_review_results(*, example_dir: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path = example_dir / FORMULA_REVIEW_FILENAME
    path.write_text(json.dumps(list(records), indent=2, ensure_ascii=False), encoding="utf-8")


def _collect_target_reasons(
    *,
    checker_response: Any,
    proof_lines: Mapping[int, Mapping[str, Any]],
) -> Dict[int, List[str]]:
    targets: Dict[int, List[str]] = {}
    issue_map = page_helpers.checker_issues_by_line(checker_response)
    valid_lines = {int(line_number) for line_number in proof_lines if int(line_number) > 0}
    for raw_line_number, issues in issue_map.items():
        line_number = int(raw_line_number)
        if line_number <= 0:
            continue
        issue_kinds = {
            str(issue.get("kind") or "").strip()
            for issue in issues
            if isinstance(issue, dict)
        }
        if "parse" in issue_kinds:
            _append_target_reason(targets, line_number, "parse_error")
        if "rule" in issue_kinds:
            _append_target_reason(targets, line_number, "check_error")
        citation = str((proof_lines.get(line_number) or {}).get("citation") or "")
        for ref in _parse_line_references(citation):
            if ref in valid_lines:
                _append_target_reason(targets, ref, f"cited_by_{line_number}")
        if "parse" in issue_kinds:
            for neighbor in (line_number - 1, line_number + 1):
                if neighbor in valid_lines:
                    _append_target_reason(targets, neighbor, f"adjacent_to_{line_number}")
    return targets


def _append_target_reason(targets: Dict[int, List[str]], line_number: int, reason: str) -> None:
    if line_number <= 0 or not reason:
        return
    reasons = targets.setdefault(line_number, [])
    if reason not in reasons:
        reasons.append(reason)


def _ordered_target_line_numbers(target_reasons: Mapping[int, Sequence[str]]) -> List[int]:
    def priority(line_number: int) -> tuple[int, int]:
        reasons = list(target_reasons.get(line_number) or [])
        if any(reason == "parse_error" for reason in reasons):
            return (0, line_number)
        if any(reason == "check_error" for reason in reasons):
            return (1, line_number)
        if any(reason.startswith("cited_by_") for reason in reasons):
            return (2, line_number)
        return (3, line_number)

    return sorted((int(line_number) for line_number in target_reasons if int(line_number) > 0), key=priority)


def _should_review_line(
    *,
    line_number: int,
    target_reasons: Sequence[str],
    source_record: Mapping[str, Any],
) -> bool:
    del line_number
    scope_role = str(source_record.get("scope_role") or "").strip()
    if scope_role != "strict_open":
        return True
    return any(reason in {"parse_error", "check_error"} for reason in target_reasons)


def _parse_line_references(citation: str) -> set[int]:
    refs: set[int] = set()
    for match in _LINE_REF_RE.finditer(str(citation or "")):
        start = int(match.group(1))
        end_text = match.group(2)
        refs.add(start)
        if not end_text:
            continue
        end = int(end_text)
        step = 1 if end >= start else -1
        span = abs(end - start)
        if span <= 8:
            for candidate in range(start, end + step, step):
                refs.add(candidate)
        else:
            refs.add(end)
    return refs


def _run_formula_attempts(
    *,
    client: Any,
    example_dir: Path,
    line_number: int,
    line_crop_img,
    modal_config: Mapping[str, Any] | None,
) -> List[Dict[str, Any]]:
    attempt_specs = [
        ("formula_crop", build_formula_crop(line_crop_img), modal.build_targeted_formula_ocr_prompt(modal_config, focus_mode="formula_crop")),
        ("boxed_formula", build_boxed_formula_context_crop(line_crop_img), modal.build_targeted_formula_ocr_prompt(modal_config, focus_mode="boxed_formula")),
    ]
    attempts: List[Dict[str, Any]] = []
    for attempt_kind, crop_img, prompt in attempt_specs:
        if crop_img is None or crop_img.size == 0:
            continue
        scaled = upscale_for_reread(crop_img)
        crop_path = example_dir / f"checker_formula_line_{line_number}_{attempt_kind}.jpg"
        cv2.imwrite(str(crop_path), scaled)
        raw_response = client.chat_with_image_prompt(crop_path, prompt)
        parse_error = ""
        structured_formula = ""
        confidence = ""
        try:
            parsed = modal.parse_json_object_response(raw_response)
            structured_formula = modal.normalize_formula_text(parsed.get("formula_text") or "")
            confidence = str(parsed.get("confidence") or "").strip()
        except (ValueError, json.JSONDecodeError) as exc:
            parse_error = str(exc)
        attempts.append(
            {
                "attempt_kind": attempt_kind,
                "structured_formula": structured_formula,
                "forced_choice": "",
                "confidence": confidence,
                "raw_response": raw_response,
                "parse_error": parse_error,
                "line_crop_path": str(crop_path),
            }
        )
    return attempts


def _select_formula_proposal(
    *,
    baseline_formula: str,
    attempt_formulas: Sequence[str],
    target_reasons: Sequence[str] | None = None,
) -> str:
    normalized_baseline = modal.canonicalize_formula_for_comparison(baseline_formula)
    changed_pairs: list[tuple[str, str]] = []
    for value in attempt_formulas:
        display_formula = modal.normalize_formula_text(value)
        canonical_formula = modal.canonicalize_formula_for_comparison(value)
        if not display_formula or not canonical_formula or canonical_formula == normalized_baseline:
            continue
        changed_pairs.append((canonical_formula, display_formula))
    if not changed_pairs:
        return ""
    counts = Counter(canonical for canonical, _display in changed_pairs)
    if len(counts) != 1:
        return ""
    canonical_value = next(iter(counts))
    if max(counts.values()) < 2 and normalized_baseline:
        return ""
    direct_issue = any(reason in {"parse_error", "check_error"} for reason in (target_reasons or []))
    if normalized_baseline and not direct_issue and not _looks_like_minor_formula_change(normalized_baseline, canonical_value):
        return ""
    for canonical_formula, display_formula in changed_pairs:
        if canonical_formula == canonical_value:
            return display_formula
    return ""


def _looks_like_minor_formula_change(baseline_formula: str, candidate_formula: str) -> bool:
    if not baseline_formula or not candidate_formula:
        return False
    if baseline_formula in candidate_formula or candidate_formula in baseline_formula:
        return True
    return False
