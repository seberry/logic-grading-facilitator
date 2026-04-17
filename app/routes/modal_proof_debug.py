from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote

from fastapi import APIRouter, File, Form, Request, Response, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from app.autograde import AutoGradeResult, autograde_submission
from app.storage import RAW_SCANS_DIR
from app.storage import DEBUG_MODAL_PROOF_DIR
from scripts.debug_modal_proof_ocr import (
    build_vlm_client,
    default_unary_verifier_model,
    transcribe_image_to_modal_debug_state,
)

router = APIRouter()

TEMPLATES_DIR = Path(__file__).resolve().parents[1] / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

STATE_FILENAME = "state.json"
OPTIONAL_TEXT_ARTIFACTS = ("notes.md", "source.txt")
OCR_TEXT_ARTIFACTS = (
    "ocr_prompt.txt",
    "intermediate_ROOT.json",
    "intermediate_1.json",
    "intermediate_2.json",
    "response_ROOT.json",
    "response_1.json",
    "response_2.json",
    "final_structure_baseline.json",
    "final_structure.json",
    "unary_verifier.json",
    "verifier_rewrites.json",
)
IMAGE_ARTIFACTS = (
    "source_image",
    "debug_bars.jpg",
    "crop_ROOT.jpg",
    "crop_1.jpg",
    "crop_2.jpg",
    "margin_anchor_margin_base.jpg",
    "margin_anchor_margin_upscaled_4x.jpg",
    "margin_anchor_binary.png",
    "margin_anchor_candidate_boxes.jpg",
    "margin_anchor_strip_overlay.jpg",
)
SYSTEM_OPTIONS = ["K", "T", "S4", "S5"]
STARTER_EXAMPLE = "starter"

DEFAULT_PROOF_JSON_TEXT = json.dumps(
    {
        "lines": [
            {"wffstr": "A", "jstr": "PR"},
        ],
        "meta": {},
    },
    indent=2,
    ensure_ascii=False,
)


def _clean_example_slug(raw: Optional[str]) -> str:
    text = str(raw or "").strip().lower()
    text = re.sub(r"[\s\-–]+", "_", text)
    text = re.sub(r"[^a-z0-9_]", "", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or STARTER_EXAMPLE


def list_modal_proof_examples() -> List[str]:
    if not DEBUG_MODAL_PROOF_DIR.exists():
        return []
    return sorted([p.name for p in DEBUG_MODAL_PROOF_DIR.iterdir() if p.is_dir()], key=str.lower)


def _state_path(example_slug: str) -> Path:
    return DEBUG_MODAL_PROOF_DIR / example_slug / STATE_FILENAME


def _artifact_text(example_slug: str, filename: str) -> str:
    path = DEBUG_MODAL_PROOF_DIR / example_slug / filename
    if not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _default_page_state() -> Dict[str, Any]:
    return {
        "system": "K",
        "include_derived_rules": False,
        "conclusion": "A",
        "vision_base_url": "",
        "vision_model": "",
        "verifier_vision_model": "",
        "source_image_name": "",
        "proof_json_text": DEFAULT_PROOF_JSON_TEXT,
        "original_proof_json_text": DEFAULT_PROOF_JSON_TEXT,
        "verifier_rewrite_summary": [],
        "review_decisions": {},
        "draft_comment": "",
        "draft_grade": "",
        "checker_request": None,
        "checker_response": None,
        "run_summary": "Edit the starter proof JSON and run the modal checker.",
        "status": "",
        "message": "",
        "checker_error": "",
    }


def _load_example_state(example_slug: Optional[str]) -> Dict[str, Any]:
    state = _default_page_state()
    if not example_slug:
        return state

    path = _state_path(example_slug)
    if not path.exists():
        return state

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return state

    for key in (
        "system",
        "include_derived_rules",
        "conclusion",
        "proof_json_text",
        "original_proof_json_text",
        "vision_base_url",
        "vision_model",
        "verifier_vision_model",
        "source_image_name",
        "verifier_rewrite_summary",
        "review_decisions",
        "draft_comment",
        "draft_grade",
        "checker_request",
        "checker_response",
        "run_summary",
    ):
        if key in raw:
            state[key] = raw[key]
    return state


def _save_example_state(example_slug: str, state: Dict[str, Any]) -> None:
    path = _state_path(example_slug)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "system": state.get("system", "K"),
                "include_derived_rules": bool(state.get("include_derived_rules", False)),
                "conclusion": state.get("conclusion", ""),
                "vision_base_url": state.get("vision_base_url", ""),
                "vision_model": state.get("vision_model", ""),
                "verifier_vision_model": state.get("verifier_vision_model", ""),
                "source_image_name": state.get("source_image_name", ""),
                "proof_json_text": state.get("proof_json_text", DEFAULT_PROOF_JSON_TEXT),
                "original_proof_json_text": state.get("original_proof_json_text", DEFAULT_PROOF_JSON_TEXT),
                "verifier_rewrite_summary": state.get("verifier_rewrite_summary") or [],
                "review_decisions": state.get("review_decisions") or {},
                "draft_comment": state.get("draft_comment", ""),
                "draft_grade": state.get("draft_grade", ""),
                "checker_request": state.get("checker_request"),
                "checker_response": state.get("checker_response"),
                "run_summary": state.get("run_summary", ""),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def _checker_issues(checker_response: Any, key: str) -> List[Dict[str, Any]]:
    if not isinstance(checker_response, dict):
        return []
    issues = checker_response.get(key)
    return issues if isinstance(issues, list) else []


def _checker_lines(checker_response: Any) -> List[Dict[str, Any]]:
    if not isinstance(checker_response, dict):
        return []
    lines = checker_response.get("normalized_lines")
    return lines if isinstance(lines, list) else []


def _checker_request_lines(checker_request: Any) -> List[Dict[str, Any]]:
    if not isinstance(checker_request, dict):
        return []
    lines = checker_request.get("lines")
    return lines if isinstance(lines, list) else []


def _ascii_scope_prefix(depth: Any, next_depth: Any = 0) -> str:
    try:
        current = max(0, int(depth))
    except (TypeError, ValueError):
        current = 0
    try:
        following = max(0, int(next_depth))
    except (TypeError, ValueError):
        following = 0

    if current <= 0:
        return ""

    close_from = following + 1 if current > following else 0
    parts: List[str] = []
    for level in range(1, current + 1):
        if close_from and level >= close_from:
            parts.append("└ " if level < current else "└─")
        else:
            parts.append("│ ")
    return "".join(parts)


def _build_ascii_proof_records(lines: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    prepared: List[Dict[str, Any]] = []
    for index, item in enumerate(lines, start=1):
        if not isinstance(item, dict):
            continue
        try:
            depth = max(0, int(item.get("depth", 0)))
        except (TypeError, ValueError):
            depth = 0
        prepared.append(
            {
                "line_number": int(item.get("line_number") or index),
                "depth": depth,
                "sentence": str(item.get("sentence") or item.get("formula") or "").strip(),
                "citation": str(item.get("citation") or "").strip(),
                "note_text": str(item.get("note_text") or "").strip(),
                "status": str(item.get("status") or "ok").strip() or "ok",
            }
        )

    records: List[Dict[str, str]] = []
    for index, item in enumerate(prepared):
        next_depth = prepared[index + 1]["depth"] if index + 1 < len(prepared) else 0
        prefix = _ascii_scope_prefix(item["depth"], next_depth)
        line_text = f'{item["line_number"]:>2}. {prefix}{item["sentence"]}'
        if item["citation"]:
            line_text += f'    {item["citation"]}'
        records.append(
            {
                "line_text": line_text,
                "note_text": item["note_text"],
                "status": item["status"],
            }
        )
    return records


def _parse_review_decisions(raw: Any) -> Dict[str, Dict[str, str]]:
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return {}
        try:
            raw = json.loads(text)
        except json.JSONDecodeError:
            return {}

    if not isinstance(raw, dict):
        return {}

    parsed: Dict[str, Dict[str, str]] = {}
    for key, value in raw.items():
        line_key = str(key or "").strip()
        if not line_key or not isinstance(value, dict):
            continue
        mode = str(value.get("mode") or "").strip()
        if mode not in {"keep_current", "candidate", "manual"}:
            continue
        decision = {
            "mode": mode,
            "formula": str(value.get("formula") or "").strip(),
            "source_key": str(value.get("source_key") or "").strip(),
        }
        parsed[line_key] = decision
    return parsed


def _proof_lines_from_text(text: str) -> Dict[int, Dict[str, Any]]:
    if not str(text or "").strip():
        return {}
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {}

    if isinstance(parsed, list):
        structure = parsed
    elif isinstance(parsed, dict):
        if isinstance(parsed.get("lines"), list):
            structure = parsed.get("lines")
        elif isinstance(parsed.get("structure"), list):
            structure = parsed.get("structure")
        else:
            return {}
    else:
        return {}

    lines: Dict[int, Dict[str, Any]] = {}
    auto_line_number = 1

    def visit(node: Any, depth: int = 0) -> None:
        nonlocal auto_line_number
        if isinstance(node, list):
            for child in node:
                visit(child, depth + 1)
            return
        if not isinstance(node, dict):
            return

        raw_line = node.get("line")
        if isinstance(raw_line, int):
            line_number = raw_line
        else:
            line_number = auto_line_number
        formula = str(node.get("wffstr") or node.get("sentence") or "").strip()
        citation = str(node.get("jstr") or node.get("citation") or "").strip()
        lines[line_number] = {
            "line_number": line_number,
            "formula": formula,
            "citation": citation,
            "depth": max(0, depth - 1),
        }
        auto_line_number = max(auto_line_number + 1, line_number + 1)

    visit(structure, 0)
    return lines


def _checker_issues_by_line(checker_response: Any) -> Dict[int, List[Dict[str, str]]]:
    issues_by_line: Dict[int, List[Dict[str, str]]] = {}
    if not isinstance(checker_response, dict):
        return issues_by_line

    for issue_type, key in (("parse", "parse_errors"), ("rule", "check_errors")):
        for issue in _checker_issues(checker_response, key):
            if not isinstance(issue, dict):
                continue
            line_number = issue.get("line")
            if not isinstance(line_number, int):
                continue
            issues_by_line.setdefault(line_number, []).append(
                {
                    "kind": issue_type,
                    "code": str(issue.get("code") or "").strip(),
                    "message": str(issue.get("message") or "").strip(),
                }
            )
    return issues_by_line


def _attempt_source_label(attempt: Dict[str, Any], result_kind: str) -> str:
    attempt_kind = str(attempt.get("attempt_kind") or "local_band").strip()
    if attempt_kind == "margin_anchor":
        prefix = "Margin-anchor"
    else:
        band_index = attempt.get("band_index")
        if isinstance(band_index, int):
            prefix = f"Local-band {band_index + 1}"
        else:
            prefix = "Local-band"
    return f"{prefix} {result_kind}"


def _build_review_panel(
    *,
    state: Dict[str, Any],
    unary_verifier_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    current_lines = _proof_lines_from_text(str(state.get("proof_json_text") or ""))
    original_lines = _proof_lines_from_text(str(state.get("original_proof_json_text") or ""))
    checker_issues = _checker_issues_by_line(state.get("checker_response"))
    verifier_by_line: Dict[int, Dict[str, Any]] = {}
    for item in unary_verifier_results:
        if not isinstance(item, dict):
            continue
        line_number = item.get("line_number")
        if not isinstance(line_number, int):
            raw_label = str(item.get("line_label") or "").strip()
            if raw_label.isdigit():
                line_number = int(raw_label)
        if isinstance(line_number, int):
            verifier_by_line[line_number] = item
    saved_decisions = _parse_review_decisions(state.get("review_decisions"))

    flagged_lines = sorted(set(current_lines.keys()) | set(checker_issues.keys()) | set(verifier_by_line.keys()))
    review_items: List[Dict[str, Any]] = []
    resolved_count = 0

    for line_number in flagged_lines:
        current_line = current_lines.get(line_number, {})
        original_line = original_lines.get(line_number, {})
        verifier = verifier_by_line.get(line_number, {})
        issues = checker_issues.get(line_number, [])

        candidates_by_formula: Dict[str, Dict[str, Any]] = {}

        def add_candidate(formula: str, source: str) -> None:
            formula = str(formula or "").strip()
            if not formula:
                return
            source_key = f"formula::{formula}"
            record = candidates_by_formula.setdefault(
                formula,
                {
                    "source_key": source_key,
                    "formula": formula,
                    "sources": [],
                },
            )
            if source not in record["sources"]:
                record["sources"].append(source)

        current_formula = str(current_line.get("formula") or "").strip()
        current_citation = str(current_line.get("citation") or "").strip()
        original_formula = str(original_line.get("formula") or "").strip()

        add_candidate(current_formula, "Current proof JSON")
        if original_formula:
            add_candidate(original_formula, "Original OCR JSON")

        baseline_formula = str(verifier.get("baseline_formula") or "").strip()
        structured_formula = str(verifier.get("structured_formula") or "").strip()
        forced_choice = str(verifier.get("forced_choice") or "").strip()
        proposed_correction = str(verifier.get("proposed_correction") or "").strip()
        attempt_kind = str(verifier.get("attempt_kind") or "").strip()

        if baseline_formula:
            add_candidate(baseline_formula, "Verifier baseline")
        if structured_formula:
            label = "Selected structured 8b"
            if attempt_kind:
                label += f" ({attempt_kind.replace('_', ' ')})"
            add_candidate(structured_formula, label)
        if forced_choice and forced_choice != "uncertain":
            label = "Selected forced-choice 8b"
            if attempt_kind:
                label += f" ({attempt_kind.replace('_', ' ')})"
            add_candidate(forced_choice, label)
        if proposed_correction:
            add_candidate(proposed_correction, "Verifier proposed correction")

        for attempt in verifier.get("candidate_attempts") or []:
            if not isinstance(attempt, dict):
                continue
            attempt_structured = str(attempt.get("structured_formula") or "").strip()
            attempt_forced = str(attempt.get("forced_choice") or "").strip()
            if attempt_structured:
                add_candidate(attempt_structured, _attempt_source_label(attempt, "structured"))
            if attempt_forced and attempt_forced != "uncertain":
                add_candidate(attempt_forced, _attempt_source_label(attempt, "forced"))

        candidate_list = sorted(
            candidates_by_formula.values(),
            key=lambda item: (
                0 if item["formula"] == current_formula else 1,
                item["formula"],
            ),
        )
        candidate_lookup = {item["source_key"]: item for item in candidate_list}

        flag_reasons: List[str] = []
        if issues:
            flag_reasons.append("checker issue")
        if proposed_correction:
            flag_reasons.append("proposed correction")
        if forced_choice == "uncertain":
            flag_reasons.append("forced choice uncertain")
        if structured_formula and forced_choice and structured_formula != forced_choice:
            flag_reasons.append("structured / forced disagree")
        if len(candidate_list) > 1:
            flag_reasons.append("multiple candidate formulas")

        if not flag_reasons:
            continue

        recommended_source_key = f"formula::{proposed_correction}" if proposed_correction else ""

        saved = saved_decisions.get(str(line_number))
        selected_decision: Optional[Dict[str, str]] = None
        if saved:
            mode = saved.get("mode", "")
            if mode == "keep_current":
                selected_decision = {
                    "mode": mode,
                    "formula": current_formula,
                    "source_key": "",
                }
            elif mode == "candidate":
                source_key = saved.get("source_key", "")
                candidate = candidate_lookup.get(source_key)
                if candidate:
                    selected_decision = {
                        "mode": mode,
                        "formula": candidate["formula"],
                        "source_key": candidate["source_key"],
                    }
            elif mode == "manual":
                formula = saved.get("formula", "").strip()
                if formula:
                    selected_decision = {
                        "mode": mode,
                        "formula": formula,
                        "source_key": "",
                    }

        if selected_decision:
            resolved_count += 1

        review_items.append(
            {
                "line_number": line_number,
                "citation": current_citation,
                "current_formula": current_formula,
                "original_formula": original_formula,
                "baseline_formula": baseline_formula,
                "structured_formula": structured_formula,
                "forced_choice": forced_choice,
                "proposed_correction": proposed_correction,
                "flag_reasons": flag_reasons,
                "issues": issues,
                "candidates": candidate_list,
                "recommended_source_key": recommended_source_key,
                "selected_decision": selected_decision,
                "line_crop_filename": verifier.get("line_crop_filename"),
                "line_crop_url": verifier.get("line_crop_url"),
                "strip_y1": verifier.get("strip_y1"),
                "strip_y2": verifier.get("strip_y2"),
                "attempt_kind": verifier.get("attempt_kind"),
                "candidate_attempts": verifier.get("candidate_attempts") or [],
            }
        )

    return {
        "items": review_items,
        "flagged_count": len(review_items),
        "resolved_count": resolved_count,
        "unresolved_count": max(0, len(review_items) - resolved_count),
    }


def _load_unary_verifier_results(example_slug: str) -> List[Dict[str, Any]]:
    if not example_slug or example_slug == STARTER_EXAMPLE:
        return []
    path = DEBUG_MODAL_PROOF_DIR / example_slug / "unary_verifier.json"
    if not path.exists():
        return []
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(parsed, list):
        return []

    annotated: List[Dict[str, Any]] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        record = dict(item)
        line_crop_path = str(record.get("line_crop_path") or "").strip()
        if line_crop_path:
            filename = Path(line_crop_path).name
            record["line_crop_filename"] = filename
            record["line_crop_url"] = f"/debug_modal_proof_assets/{example_slug}/{quote(filename)}"

        attempts = record.get("candidate_attempts")
        if isinstance(attempts, list):
            annotated_attempts: List[Dict[str, Any]] = []
            for attempt in attempts:
                if not isinstance(attempt, dict):
                    continue
                attempt_record = dict(attempt)
                attempt_crop_path = str(attempt_record.get("line_crop_path") or "").strip()
                if attempt_crop_path:
                    filename = Path(attempt_crop_path).name
                    attempt_record["line_crop_filename"] = filename
                    attempt_record["line_crop_url"] = f"/debug_modal_proof_assets/{example_slug}/{quote(filename)}"
                annotated_attempts.append(attempt_record)
            record["candidate_attempts"] = annotated_attempts
        annotated.append(record)
    return annotated


def _render_checker_request_preview(checker_request: Any) -> str:
    lines = _checker_request_lines(checker_request)
    if not lines:
        return ""

    rendered: List[str] = []
    for idx, item in enumerate(lines, start=1):
        if not isinstance(item, dict):
            continue
        try:
            depth = max(0, int(item.get("depth", 0)))
        except (TypeError, ValueError):
            depth = 0
        sentence = str(item.get("sentence", "")).strip()
        citation = str(item.get("citation", "")).strip()
        indent = "  " * depth
        line = f"{idx:>2}. {indent}{sentence}"
        if citation:
            line += f"    {citation}"
        rendered.append(line)
    return "\n".join(rendered)


def _render_page(
    request: Request,
    *,
    example: Optional[str],
    state: Dict[str, Any],
) -> Response:
    examples = list_modal_proof_examples()
    if example and (example == STARTER_EXAMPLE or example in examples):
        selected = example
    elif examples:
        selected = examples[0]
    else:
        selected = STARTER_EXAMPLE

    artifact_texts = {}
    image_artifacts: List[Dict[str, str]] = []
    unary_verifier_results: List[Dict[str, Any]] = []
    if selected and selected != STARTER_EXAMPLE:
        for filename in (*OPTIONAL_TEXT_ARTIFACTS, *OCR_TEXT_ARTIFACTS):
            text = _artifact_text(selected, filename)
            if text.strip():
                artifact_texts[filename] = text
        example_dir = DEBUG_MODAL_PROOF_DIR / selected
        source_image_name = str(state.get("source_image_name") or "").strip()
        image_candidates = list(IMAGE_ARTIFACTS)
        if source_image_name:
            image_candidates.insert(0, source_image_name)
        seen = set()
        for filename in image_candidates:
            if not filename or filename in seen:
                continue
            seen.add(filename)
            path = example_dir / filename
            if path.exists() and path.is_file():
                image_artifacts.append(
                    {
                        "filename": filename,
                        "url": f"/debug_modal_proof_assets/{selected}/{quote(filename)}",
                    }
                )
        for path in sorted(example_dir.glob("verifier_line_*.jpg"), key=lambda p: p.name.lower()):
            if path.name in seen:
                continue
            seen.add(path.name)
            image_artifacts.append(
                {
                    "filename": path.name,
                    "url": f"/debug_modal_proof_assets/{selected}/{quote(path.name)}",
                }
            )
        unary_verifier_results = _load_unary_verifier_results(selected)

    review_panel = _build_review_panel(
        state=state,
        unary_verifier_results=unary_verifier_results,
    )
    review_by_line = {
        int(item.get("line_number")): item
        for item in (review_panel.get("items") or [])
        if isinstance(item, dict) and isinstance(item.get("line_number"), int)
    }
    normalized_lines = _checker_lines(state.get("checker_response"))
    if normalized_lines:
        checker_ascii_base = [
            {
                "line_number": idx,
                "depth": line.get("depth", 0),
                "sentence": line.get("sentence", ""),
                "citation": line.get("citation", ""),
            }
            for idx, line in enumerate(normalized_lines, start=1)
            if isinstance(line, dict)
        ]
    else:
        checker_ascii_base = [
            {
                "line_number": line_number,
                "depth": line.get("depth", 0),
                "sentence": line.get("formula", ""),
                "citation": line.get("citation", ""),
            }
            for line_number, line in sorted(_proof_lines_from_text(str(state.get("proof_json_text") or "")).items())
            if isinstance(line, dict)
        ]
    checker_ascii_source: List[Dict[str, Any]] = []
    for line in checker_ascii_base:
        idx = int(line.get("line_number") or 0)
        review_item = review_by_line.get(idx, {})
        notes: List[str] = []
        issues = review_item.get("issues") or []
        flag_reasons = [
            str(reason).strip()
            for reason in (review_item.get("flag_reasons") or [])
            if str(reason).strip() and str(reason).strip() != "checker issue"
        ]
        if issues:
            messages = [str(issue.get("message") or "").strip() for issue in issues if str(issue.get("message") or "").strip()]
            if messages:
                notes.append(f"checker: {'; '.join(messages)}")
        if flag_reasons:
            notes.append(f"ocr: {', '.join(flag_reasons)}")
        checker_ascii_source.append(
            {
                "line_number": idx,
                "depth": line.get("depth", 0),
                "sentence": line.get("sentence", ""),
                "citation": line.get("citation", ""),
                "note_text": " | ".join(notes),
                "status": "both" if issues and flag_reasons else "checker" if issues else "disagreement" if flag_reasons else "ok",
            }
        )
    checker_ascii_records = _build_ascii_proof_records(checker_ascii_source)

    return templates.TemplateResponse(
        request,
        "debug_modal_proof.html",
        {
            "available_examples": examples,
            "selected_example": selected,
            "system_options": SYSTEM_OPTIONS,
            "parse_errors": _checker_issues(state.get("checker_response"), "parse_errors"),
            "check_errors": _checker_issues(state.get("checker_response"), "check_errors"),
            "normalized_lines": normalized_lines,
            "checker_ascii_records": checker_ascii_records,
            "checker_request_preview": _render_checker_request_preview(state.get("checker_request")),
            "optional_artifacts": artifact_texts,
            "image_artifacts": image_artifacts,
            "unary_verifier_results": unary_verifier_results,
            "review_panel": review_panel,
            **state,
        },
    )


def _state_from_form(
    *,
    example: Optional[str],
    system: str,
    include_derived_rules: bool,
    conclusion: str,
    vision_base_url: str,
    vision_model: str,
    verifier_vision_model: str,
    proof_json_text: str,
    review_decisions_json: str,
    draft_comment: str,
    draft_grade: str,
) -> tuple[str, Dict[str, Any]]:
    example_slug = _clean_example_slug(example)
    state = _load_example_state(example_slug)
    state.update(
        {
            "system": str(system or "K").strip(),
            "include_derived_rules": bool(include_derived_rules),
            "conclusion": str(conclusion or "").strip(),
            "vision_base_url": str(vision_base_url or "").strip(),
            "vision_model": str(vision_model or "").strip(),
            "verifier_vision_model": str(verifier_vision_model or "").strip(),
            "proof_json_text": proof_json_text,
            "review_decisions": _parse_review_decisions(review_decisions_json),
            "draft_comment": draft_comment,
            "draft_grade": draft_grade,
            "status": "",
            "message": "",
            "checker_error": "",
        }
    )
    return example_slug, state


def _build_autograde_config(state: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "system": state.get("system", "K"),
        "include_derived_rules": bool(state.get("include_derived_rules", False)),
        "conclusion": state.get("conclusion") or None,
    }


def _find_saved_source_image(example_slug: str, state: Dict[str, Any]) -> Optional[Path]:
    source_image_name = str(state.get("source_image_name") or "").strip()
    example_dir = DEBUG_MODAL_PROOF_DIR / example_slug
    if source_image_name:
        candidate = example_dir / source_image_name
        if candidate.exists() and candidate.is_file():
            return candidate
    for candidate in sorted(example_dir.iterdir(), key=lambda p: p.name.lower()) if example_dir.exists() else []:
        if candidate.is_file() and candidate.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
            return candidate
    return None


def _resolve_source_image_path(example_slug: str, raw_path: str) -> Optional[Path]:
    text = str(raw_path or "").strip()
    if not text:
        return None
    candidate = Path(text)
    if candidate.is_absolute():
        return candidate

    example_candidate = (DEBUG_MODAL_PROOF_DIR / example_slug / candidate).resolve()
    if example_candidate.exists():
        return example_candidate

    raw_scans_candidate = (RAW_SCANS_DIR / candidate).resolve()
    if raw_scans_candidate.exists():
        return raw_scans_candidate

    return raw_scans_candidate


async def _persist_uploaded_source_image(
    example_slug: str,
    proof_image: UploadFile | None,
) -> Optional[Path]:
    if proof_image is None or not getattr(proof_image, "filename", ""):
        return None

    suffix = Path(proof_image.filename or "").suffix.lower() or ".png"
    if suffix not in {".png", ".jpg", ".jpeg", ".webp"}:
        raise ValueError("OCR image must be PNG, JPG, JPEG, or WEBP")

    example_dir = DEBUG_MODAL_PROOF_DIR / example_slug
    example_dir.mkdir(parents=True, exist_ok=True)
    target_path = example_dir / f"source_image{suffix}"
    with target_path.open("wb") as handle:
        shutil.copyfileobj(proof_image.file, handle)
    return target_path


@router.get("/debug/modal-proof", response_class=HTMLResponse)
async def modal_proof_debug_page(
    request: Request,
    example: Optional[str] = None,
) -> Response:
    example_slug = _clean_example_slug(example)
    state = _load_example_state(example_slug if example else None)
    status = request.query_params.get("status", "")
    message = request.query_params.get("message", "")
    if status or message:
        state["status"] = status
        state["message"] = message
    return _render_page(request, example=example_slug if example else None, state=state)


@router.get("/debug/modal-proof/save", response_class=HTMLResponse)
@router.get("/debug/modal-proof/run_checker", response_class=HTMLResponse)
@router.get("/debug/modal-proof/run_ocr", response_class=HTMLResponse)
async def modal_proof_debug_action_redirect(
    example: Optional[str] = None,
) -> Response:
    example_slug = _clean_example_slug(example)
    target = "/debug/modal-proof"
    if example_slug:
        target += f"?example={quote(example_slug)}"
    return RedirectResponse(url=target, status_code=303)


@router.post("/debug/modal-proof/save", response_class=HTMLResponse)
async def save_modal_proof_state(
    request: Request,
    example: str = Form(STARTER_EXAMPLE),
    system: str = Form("K"),
    include_derived_rules: bool = Form(False),
    conclusion: str = Form(""),
    vision_base_url: str = Form(""),
    vision_model: str = Form(""),
    verifier_vision_model: str = Form(""),
    proof_json_text: str = Form(...),
    review_decisions_json: str = Form(""),
    draft_comment: str = Form(""),
    draft_grade: str = Form(""),
) -> Response:
    example_slug, state = _state_from_form(
        example=example,
        system=system,
        include_derived_rules=include_derived_rules,
        conclusion=conclusion,
        vision_base_url=vision_base_url,
        vision_model=vision_model,
        verifier_vision_model=verifier_vision_model,
        proof_json_text=proof_json_text,
        review_decisions_json=review_decisions_json,
        draft_comment=draft_comment,
        draft_grade=draft_grade,
    )
    state["status"] = "success"
    state["message"] = "Saved current modal proof debug state."
    _save_example_state(example_slug, state)

    return RedirectResponse(
        url=f"/debug/modal-proof?example={quote(example_slug)}&status=success&message={quote(state['message'])}",
        status_code=303,
    )


@router.post("/debug/modal-proof/run_checker", response_class=HTMLResponse)
async def run_modal_proof_debug_checker(
    example: str = Form(STARTER_EXAMPLE),
    system: str = Form("K"),
    include_derived_rules: bool = Form(False),
    conclusion: str = Form(""),
    vision_base_url: str = Form(""),
    vision_model: str = Form(""),
    verifier_vision_model: str = Form(""),
    proof_json_text: str = Form(...),
    review_decisions_json: str = Form(""),
    draft_comment: str = Form(""),
    draft_grade: str = Form(""),
) -> Response:
    example_slug, state = _state_from_form(
        example=example,
        system=system,
        include_derived_rules=include_derived_rules,
        conclusion=conclusion,
        vision_base_url=vision_base_url,
        vision_model=vision_model,
        verifier_vision_model=verifier_vision_model,
        proof_json_text=proof_json_text,
        review_decisions_json=review_decisions_json,
        draft_comment=draft_comment,
        draft_grade=draft_grade,
    )

    result: AutoGradeResult = autograde_submission(
        problem_type="modal_proof",
        answer_text=proof_json_text,
        autograde_config=_build_autograde_config(state),
    )

    state["run_summary"] = result.summary
    state["draft_comment"] = result.draft_comment or draft_comment
    if result.score is not None:
        state["draft_grade"] = str(result.score)
    state["checker_request"] = result.metadata.get("checker_request")
    state["checker_response"] = result.metadata.get("checker_response")

    if result.status == "scored":
        state["status"] = "success"
        state["message"] = "Checker ran successfully through the shared autograde layer."
    else:
        state["status"] = "error"
        state["message"] = "Checker could not run."
        state["checker_error"] = "; ".join(result.errors) or result.summary

    _save_example_state(example_slug, state)
    return RedirectResponse(
        url=f"/debug/modal-proof?example={quote(example_slug)}&status={quote(str(state.get('status') or ''))}&message={quote(str(state.get('message') or ''))}",
        status_code=303,
    )


@router.post("/debug/modal-proof/run_ocr", response_class=HTMLResponse)
async def run_modal_proof_debug_ocr(
    example: str = Form(STARTER_EXAMPLE),
    system: str = Form("K"),
    include_derived_rules: bool = Form(False),
    conclusion: str = Form(""),
    vision_base_url: str = Form(""),
    vision_model: str = Form(""),
    verifier_vision_model: str = Form(""),
    proof_json_text: str = Form(DEFAULT_PROOF_JSON_TEXT),
    review_decisions_json: str = Form(""),
    draft_comment: str = Form(""),
    draft_grade: str = Form(""),
    proof_image_path: str = Form(""),
    proof_image: UploadFile | None = File(None),
) -> Response:
    example_slug, state = _state_from_form(
        example=example,
        system=system,
        include_derived_rules=include_derived_rules,
        conclusion=conclusion,
        vision_base_url=vision_base_url,
        vision_model=vision_model,
        verifier_vision_model=verifier_vision_model,
        proof_json_text=proof_json_text,
        review_decisions_json=review_decisions_json,
        draft_comment=draft_comment,
        draft_grade=draft_grade,
    )

    example_dir = DEBUG_MODAL_PROOF_DIR / example_slug
    example_dir.mkdir(parents=True, exist_ok=True)

    try:
        uploaded_image_path = await _persist_uploaded_source_image(example_slug, proof_image)
        source_image_path = uploaded_image_path or _resolve_source_image_path(example_slug, proof_image_path) or _find_saved_source_image(example_slug, state)
        if source_image_path is None:
            raise ValueError("Provide a proof image upload or an existing image path before running OCR.")
        if not source_image_path.exists() or not source_image_path.is_file():
            raise ValueError(f"OCR image not found: {source_image_path}")

        client = build_vlm_client(
            base_url=state.get("vision_base_url") or None,
            model=state.get("vision_model") or None,
        )
        verifier_client = build_vlm_client(
            base_url=state.get("vision_base_url") or None,
            model=state.get("verifier_vision_model") or default_unary_verifier_model(),
        )
        ocr_state = transcribe_image_to_modal_debug_state(
            image_path=source_image_path,
            output_dir=example_dir,
            modal_config=_build_autograde_config(state),
            client=client,
            verifier_client=verifier_client,
            vision_base_url=state.get("vision_base_url") or None,
            vision_model=state.get("vision_model") or None,
            verifier_vision_model=state.get("verifier_vision_model") or None,
        )
        ocr_state["draft_comment"] = draft_comment
        ocr_state["draft_grade"] = draft_grade
        ocr_state["review_decisions"] = {}
        ocr_state["status"] = "success"
        ocr_state["message"] = "OCR ran and updated the modal proof state."
        state = ocr_state
    except Exception as exc:
        state["status"] = "error"
        state["message"] = f"Modal OCR failed: {exc}"

    _save_example_state(example_slug, state)
    return RedirectResponse(
        url=f"/debug/modal-proof?example={quote(example_slug)}&status={quote(str(state.get('status') or ''))}&message={quote(str(state.get('message') or ''))}",
        status_code=303,
    )
