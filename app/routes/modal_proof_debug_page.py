from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from fastapi import Request, Response
from fastapi.templating import Jinja2Templates


@dataclass(frozen=True)
class RenderDeps:
    templates: Jinja2Templates
    debug_dir: Path
    starter_example: str
    optional_text_artifacts: Sequence[str]
    ocr_text_artifacts: Sequence[str]
    image_artifacts: Sequence[str]
    strict_opener_stage_specs: Sequence[Tuple[str, str, str, str]]
    line_number_debug_image_names: set[str]
    system_options: Sequence[str]
    ocr_pipeline_options: Sequence[Tuple[str, str]]
    ocr_pipeline_labels: Mapping[str, str]
    list_modal_proof_examples: Callable[[], List[str]]
    load_modal_regression_fixtures: Callable[[], List[Dict[str, str]]]
    fixture_option_label: Callable[[Dict[str, str]], str]
    artifact_text: Callable[[str, str], str]
    artifact_url: Callable[[str, str], str]
    load_unary_verifier_results: Callable[[str], List[Dict[str, Any]]]
    checker_issues: Callable[[Any, str], List[Dict[str, Any]]]
    checker_lines: Callable[[Any], List[Dict[str, Any]]]
    build_review_panel: Callable[..., Dict[str, Any]]
    render_checker_request_preview: Callable[[Any], str]
    build_ascii_proof_records: Callable[[List[Dict[str, Any]]], List[Dict[str, str]]]
    proof_lines_from_text: Callable[[str], Dict[int, Dict[str, Any]]]
    read_json_artifact: Callable[[Path], Any]
    load_node_response: Callable[[Path, str], Dict[str, Any]]
    node_response_summary: Callable[[Dict[str, Any]], Dict[str, Any]]
    build_stitched_line_lookup: Callable[[Path], Dict[Tuple[str, int], Dict[str, Any]]]
    find_line_artifact: Callable[[Dict[str, Any], int], Optional[Dict[str, Any]]]
    row_order_to_visible_line_order: Callable[[Dict[str, Any], int], Optional[int]]


def node_id_sort_key(node_id: str) -> tuple[int, int, str]:
    text = str(node_id or "").strip()
    if text == "ROOT":
        return (0, 0, "")
    if text.isdigit():
        return (1, int(text), "")
    return (2, 0, text.lower())


def node_crop_stage_title(node_id: str) -> str:
    text = str(node_id or "").strip()
    if text == "ROOT":
        return "Whole proof crop"
    if text:
        return f"Subproof crop {text}"
    return "Proof crop"


def checker_issues(checker_response: Any, key: str) -> List[Dict[str, Any]]:
    if not isinstance(checker_response, dict):
        return []
    issues = checker_response.get(key)
    return issues if isinstance(issues, list) else []


def checker_lines(checker_response: Any) -> List[Dict[str, Any]]:
    if not isinstance(checker_response, dict):
        return []
    lines = checker_response.get("normalized_lines")
    return lines if isinstance(lines, list) else []


def checker_request_lines(checker_request: Any) -> List[Dict[str, Any]]:
    if not isinstance(checker_request, dict):
        return []
    lines = checker_request.get("lines")
    return lines if isinstance(lines, list) else []


def ascii_scope_prefix(depth: Any, next_depth: Any = 0) -> str:
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


def build_ascii_proof_records(lines: List[Dict[str, Any]]) -> List[Dict[str, str]]:
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
        prefix = ascii_scope_prefix(item["depth"], next_depth)
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


def proof_lines_from_text(text: str) -> Dict[int, Dict[str, Any]]:
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


def checker_issues_by_line(checker_response: Any) -> Dict[int, List[Dict[str, str]]]:
    issues_by_line: Dict[int, List[Dict[str, str]]] = {}
    if not isinstance(checker_response, dict):
        return issues_by_line

    for issue_type, key in (("parse", "parse_errors"), ("rule", "check_errors")):
        for issue in checker_issues(checker_response, key):
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


def attempt_source_label(attempt: Dict[str, Any], result_kind: str) -> str:
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


def build_review_panel(
    *,
    state: Dict[str, Any],
    unary_verifier_results: List[Dict[str, Any]],
    parse_review_decisions,
) -> Dict[str, Any]:
    current_lines = proof_lines_from_text(str(state.get("proof_json_text") or ""))
    original_lines = proof_lines_from_text(str(state.get("original_proof_json_text") or ""))
    checker_issue_map = checker_issues_by_line(state.get("checker_response"))
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
    saved_decisions = parse_review_decisions(state.get("review_decisions"))

    flagged_lines = sorted(set(current_lines.keys()) | set(checker_issue_map.keys()) | set(verifier_by_line.keys()))
    review_items: List[Dict[str, Any]] = []
    resolved_count = 0

    for line_number in flagged_lines:
        current_line = current_lines.get(line_number, {})
        original_line = original_lines.get(line_number, {})
        verifier = verifier_by_line.get(line_number, {})
        issues = checker_issue_map.get(line_number, [])

        candidates_by_formula: Dict[str, Dict[str, Any]] = {}

        def add_candidate(formula: str, source: str) -> None:
            formula_text = str(formula or "").strip()
            if not formula_text:
                return
            source_key = f"formula::{formula_text}"
            record = candidates_by_formula.setdefault(
                formula_text,
                {
                    "source_key": source_key,
                    "formula": formula_text,
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
                add_candidate(attempt_structured, attempt_source_label(attempt, "structured"))
            if attempt_forced and attempt_forced != "uncertain":
                add_candidate(attempt_forced, attempt_source_label(attempt, "forced"))

        candidate_list = sorted(
            candidates_by_formula.values(),
            key=lambda item: (0 if item["formula"] == current_formula else 1, item["formula"]),
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


def render_checker_request_preview(checker_request: Any) -> str:
    lines = checker_request_lines(checker_request)
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


def render_page(
    request: Request,
    *,
    example: Optional[str],
    state: Dict[str, Any],
    deps: RenderDeps,
) -> Response:
    examples = deps.list_modal_proof_examples()
    regression_fixtures = deps.load_modal_regression_fixtures()
    fixture_options = [{**fixture, "label": deps.fixture_option_label(fixture)} for fixture in regression_fixtures]
    if example and (example == deps.starter_example or example in examples):
        selected = example
    elif examples:
        selected = examples[0]
    else:
        selected = deps.starter_example

    artifact_texts: Dict[str, str] = {}
    image_artifacts: List[Dict[str, str]] = []
    primary_image_artifact: Optional[Dict[str, str]] = None
    baseline_ocr_stages: List[Dict[str, Any]] = []
    strict_opener_stages: List[Dict[str, Any]] = []
    node_row_stages: List[Dict[str, Any]] = []
    line_ocr_stages: List[Dict[str, Any]] = []
    line_number_debug_artifacts: List[Dict[str, str]] = []
    auxiliary_image_artifacts: List[Dict[str, str]] = []
    unary_verifier_results: List[Dict[str, Any]] = []
    current_pipeline_id = str(state.get("ocr_pipeline") or "legacy_modal_debug").strip() or "legacy_modal_debug"
    current_pipeline_label = deps.ocr_pipeline_labels.get(current_pipeline_id, current_pipeline_id)
    selected_fixture_id = str(state.get("fixture_id") or "").strip()
    baseline_quick_view: Dict[str, Any] = {
        "enabled": False,
        "pipeline_label": current_pipeline_label,
        "run_summary": str(state.get("run_summary") or "").strip(),
        "source_image_url": "",
        "source_image_filename": "",
        "debug_bars_url": "",
        "debug_bars_filename": "",
        "baseline_structure_text": "",
        "has_artifacts": False,
    }
    provenance = {
        "selected_example": selected,
        "fixture_id": selected_fixture_id,
        "artifact_dir": str((deps.debug_dir / selected).resolve()) if selected else "",
        "source_image_name": str(state.get("source_image_name") or "").strip(),
        "source_image_path": str(state.get("source_image_path") or "").strip(),
        "ocr_pipeline_label": current_pipeline_label,
        "vision_model": str(state.get("vision_model") or "").strip(),
        "last_run_at": str(state.get("last_run_at") or "").strip(),
    }
    if selected:
        for filename in (*deps.optional_text_artifacts, *deps.ocr_text_artifacts):
            text = deps.artifact_text(selected, filename)
            if text.strip():
                artifact_texts[filename] = text
        example_dir = deps.debug_dir / selected
        source_image_name = str(state.get("source_image_name") or "").strip()
        image_candidates = list(deps.image_artifacts)
        if source_image_name:
            image_candidates.insert(0, source_image_name)
        seen = set()
        for filename in image_candidates:
            if not filename or filename in seen:
                continue
            seen.add(filename)
            path = example_dir / filename
            if path.exists() and path.is_file():
                image_artifacts.append({"filename": filename, "url": deps.artifact_url(selected, filename)})
        for path in sorted(example_dir.glob("verifier_line_*.jpg"), key=lambda item: item.name.lower()):
            if path.name in seen:
                continue
            seen.add(path.name)
            image_artifacts.append({"filename": path.name, "url": deps.artifact_url(selected, path.name)})
        unary_verifier_results = deps.load_unary_verifier_results(selected)

        source_candidate = next((item for item in image_artifacts if item["filename"] == source_image_name), None)
        if source_candidate:
            primary_image_artifact = source_candidate
        elif image_artifacts:
            primary_image_artifact = image_artifacts[0]

        baseline_quick_view = {
            "enabled": current_pipeline_id == "modal_v2_baseline",
            "pipeline_label": current_pipeline_label,
            "run_summary": str(state.get("run_summary") or "").strip(),
            "source_image_url": primary_image_artifact["url"] if primary_image_artifact else "",
            "source_image_filename": primary_image_artifact["filename"] if primary_image_artifact else "",
            "debug_bars_url": deps.artifact_url(selected, "debug_bars.jpg") if (example_dir / "debug_bars.jpg").exists() else "",
            "debug_bars_filename": "debug_bars.jpg" if (example_dir / "debug_bars.jpg").exists() else "",
            "baseline_structure_text": artifact_texts.get("final_structure_baseline.json", ""),
            "has_artifacts": bool(
                (example_dir / "debug_bars.jpg").exists() or artifact_texts.get("final_structure_baseline.json", "").strip()
            ),
        }

        baseline_stage_ids = set()
        for path in example_dir.glob("crop_*.jpg"):
            baseline_stage_ids.add(path.stem.removeprefix("crop_"))
        for path in example_dir.glob("intermediate_*.json"):
            baseline_stage_ids.add(path.stem.removeprefix("intermediate_"))
        for path in example_dir.glob("response_*.json"):
            baseline_stage_ids.add(path.stem.removeprefix("response_"))
        for stage_id in sorted(baseline_stage_ids, key=node_id_sort_key):
            title = node_crop_stage_title(stage_id)
            image_name = f"crop_{stage_id}.jpg"
            intermediate_name = f"intermediate_{stage_id}.json"
            response_name = f"response_{stage_id}.json"
            image_path = example_dir / image_name
            intermediate_text = artifact_texts.get(intermediate_name, "")
            response_text = artifact_texts.get(response_name, "")
            if not image_path.exists() and not intermediate_text.strip() and not response_text.strip():
                continue
            baseline_ocr_stages.append(
                {
                    "stage_id": stage_id,
                    "title": title,
                    "image_filename": image_name if image_path.exists() else "",
                    "image_url": deps.artifact_url(selected, image_name) if image_path.exists() else "",
                    "intermediate_name": intermediate_name,
                    "intermediate_text": intermediate_text,
                    "response_name": response_name,
                    "response_text": response_text,
                }
            )

        for stage_id, title, image_name, response_name in deps.strict_opener_stage_specs:
            image_path = example_dir / image_name
            response_text = artifact_texts.get(response_name, "")
            if not image_path.exists() and not response_text.strip():
                continue
            strict_opener_stages.append(
                {
                    "stage_id": stage_id,
                    "title": title,
                    "image_filename": image_name if image_path.exists() else "",
                    "image_url": deps.artifact_url(selected, image_name) if image_path.exists() else "",
                    "response_name": response_name,
                    "response_text": response_text,
                }
            )

        for rows_path in sorted(example_dir.glob("node_rows_*.json"), key=lambda item: item.name.lower()):
            node_suffix = rows_path.stem.removeprefix("node_rows_")
            overlay_name = f"node_rows_overlay_{node_suffix}.jpg"
            entries_name = f"node_entries_{node_suffix}.json"
            bands_name = f"node_row_bands_{node_suffix}.json"
            overlay_path = example_dir / overlay_name
            entries_text = deps.artifact_text(selected, entries_name)
            bands_text = deps.artifact_text(selected, bands_name)
            node_response = deps.load_node_response(example_dir, node_suffix)
            node_row_stages.append(
                {
                    "node_id": node_suffix,
                    "title": f"Node {node_suffix} row segmentation",
                    "overlay_filename": overlay_name if overlay_path.exists() else "",
                    "overlay_url": deps.artifact_url(selected, overlay_name) if overlay_path.exists() else "",
                    "rows_name": rows_path.name,
                    "rows_text": rows_path.read_text(encoding="utf-8"),
                    "entries_name": entries_name,
                    "entries_text": entries_text,
                    "bands_name": bands_name,
                    "bands_text": bands_text,
                    "assembly_summary": deps.node_response_summary(node_response),
                }
            )

        line_pattern = re.compile(r"node_line_(?P<node_id>[^_]+)_(?P<row_order>\d+)\.jpg$")
        for image_path in sorted(example_dir.glob("node_line_*.jpg"), key=lambda item: item.name.lower()):
            match = line_pattern.match(image_path.name)
            if not match:
                continue
            node_id = match.group("node_id")
            row_order = int(match.group("row_order"))
            response_name = f"node_line_{node_id}_{row_order}.json"
            response_text = deps.artifact_text(selected, response_name)
            line_ocr_stages.append(
                {
                    "node_id": node_id,
                    "row_order": row_order,
                    "title": f"Node {node_id} row {row_order}",
                    "image_filename": image_path.name,
                    "image_url": deps.artifact_url(selected, image_path.name),
                    "response_name": response_name,
                    "response_text": response_text,
                }
            )

        for artifact in image_artifacts:
            filename = artifact["filename"]
            if primary_image_artifact and filename == primary_image_artifact["filename"]:
                continue
            if filename in deps.line_number_debug_image_names:
                line_number_debug_artifacts.append(artifact)
                continue
            if any(filename == stage["image_filename"] for stage in baseline_ocr_stages if stage.get("image_filename")):
                continue
            auxiliary_image_artifacts.append(artifact)

    review_panel = deps.build_review_panel(state=state, unary_verifier_results=unary_verifier_results)
    review_by_line = {
        int(item.get("line_number")): item
        for item in (review_panel.get("items") or [])
        if isinstance(item, dict) and isinstance(item.get("line_number"), int)
    }
    normalized_lines = deps.checker_lines(state.get("checker_response"))
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
            for line_number, line in sorted(deps.proof_lines_from_text(str(state.get("proof_json_text") or "")).items())
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
    checker_ascii_records = deps.build_ascii_proof_records(checker_ascii_source)
    checker_line_lookup = {int(line.get("line_number") or 0): line for line in checker_ascii_base}

    if selected:
        example_dir = deps.debug_dir / selected
        stitched_lookup = deps.build_stitched_line_lookup(example_dir)
        for stage in line_ocr_stages:
            node_id = str(stage.get("node_id") or "").strip()
            row_order = stage.get("row_order")
            if not node_id or not isinstance(row_order, int):
                continue
            parsed_response = deps.read_json_artifact(example_dir / stage["response_name"])
            if isinstance(parsed_response, dict):
                stage["parsed_response"] = parsed_response
            node_response = deps.load_node_response(example_dir, node_id)
            artifact_record = deps.find_line_artifact(node_response, row_order)
            if artifact_record:
                artifact_kind = str(artifact_record.get("kind") or "").strip()
                if artifact_kind:
                    stage["artifact_kind"] = artifact_kind
                if artifact_record.get("skipped_reason"):
                    stage["skipped_reason"] = str(artifact_record.get("skipped_reason") or "").strip()
                resolved = artifact_record.get("resolved")
                if isinstance(resolved, dict):
                    stage["resolved_scope_role"] = str(resolved.get("scope_role") or "").strip()
            visible_line_order = deps.row_order_to_visible_line_order(node_response, row_order)
            if visible_line_order is not None:
                stage["visible_line_order"] = visible_line_order
                stitched_line = stitched_lookup.get((node_id, visible_line_order))
                if stitched_line:
                    stage["stitched_line_number"] = stitched_line["line_number"]
                    stage["stitched_formula"] = stitched_line["formula"]
                    stage["stitched_citation"] = stitched_line["citation"]
                    stage["stitched_scope_role"] = stitched_line["scope_role"]
                    checker_line = checker_line_lookup.get(stitched_line["line_number"])
                    if checker_line:
                        stage["stitched_depth"] = checker_line.get("depth", 0)
                        sentence = str(checker_line.get("sentence") or stitched_line["formula"]).strip()
                        citation = str(checker_line.get("citation") or stitched_line["citation"]).strip()
                        stage["stitched_line_text"] = f"{sentence}{'    ' + citation if citation else ''}"
            else:
                stage["assembly_status"] = "skipped"
            if not stage.get("assembly_status"):
                artifact_kind = str(stage.get("artifact_kind") or "").strip()
                if artifact_kind == "token_overlap_skipped":
                    stage["assembly_status"] = "skipped"
                elif stage.get("stitched_line_number"):
                    stage["assembly_status"] = "stitched"
                else:
                    stage["assembly_status"] = "node_only"

    return deps.templates.TemplateResponse(
        request,
        "debug_modal_proof.html",
        {
            "available_examples": examples,
            "fixture_options": fixture_options,
            "selected_example": selected,
            "selected_fixture_id": selected_fixture_id,
            "system_options": deps.system_options,
            "ocr_pipeline_options": deps.ocr_pipeline_options,
            "current_pipeline_id": current_pipeline_id,
            "current_pipeline_label": current_pipeline_label,
            "provenance": provenance,
            "parse_errors": deps.checker_issues(state.get("checker_response"), "parse_errors"),
            "check_errors": deps.checker_issues(state.get("checker_response"), "check_errors"),
            "normalized_lines": normalized_lines,
            "checker_ascii_records": checker_ascii_records,
            "checker_request_preview": deps.render_checker_request_preview(state.get("checker_request")),
            "optional_artifacts": artifact_texts,
            "image_artifacts": image_artifacts,
            "primary_image_artifact": primary_image_artifact,
            "baseline_ocr_stages": baseline_ocr_stages,
            "strict_opener_stages": strict_opener_stages,
            "node_row_stages": node_row_stages,
            "line_ocr_stages": line_ocr_stages,
            "line_number_debug_artifacts": line_number_debug_artifacts,
            "auxiliary_image_artifacts": auxiliary_image_artifacts,
            "baseline_quick_view": baseline_quick_view,
            "unary_verifier_results": unary_verifier_results,
            "review_panel": review_panel,
            **state,
        },
    )
