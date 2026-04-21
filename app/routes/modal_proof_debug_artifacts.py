from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote

from proofs_ocr.common.stitch import extract_subproof_token_id, recursive_stitch, renumber_lines


def artifact_text(debug_dir: Path, example_slug: str, filename: str) -> str:
    path = debug_dir / example_slug / filename
    if not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def artifact_url(debug_dir: Path, example_slug: str, filename: str) -> str:
    path = debug_dir / example_slug / filename
    base = f"/debug_modal_proof_assets/{example_slug}/{quote(filename)}"
    try:
        version = path.stat().st_mtime_ns
    except OSError:
        return base
    return f"{base}?v={version}"


def read_json_artifact(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def load_node_entry_library(example_dir: Path) -> Dict[str, List[Dict[str, Any]]]:
    library: Dict[str, List[Dict[str, Any]]] = {}
    preferred_paths = sorted(example_dir.glob("node_entries_*.json"), key=lambda path: path.name.lower())
    fallback_paths = sorted(example_dir.glob("intermediate_*.json"), key=lambda path: path.name.lower())
    source_paths = preferred_paths or fallback_paths
    prefix = "node_entries_" if preferred_paths else "intermediate_"
    for path in source_paths:
        node_id = path.stem.removeprefix(prefix)
        parsed = read_json_artifact(path)
        if isinstance(parsed, list):
            library[node_id] = [item for item in parsed if isinstance(item, dict)]
    return library


def _prepare_debug_stitch_library(
    library: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, List[Dict[str, Any]]]:
    prepared: Dict[str, List[Dict[str, Any]]] = {}
    for node_id, entries in library.items():
        visible_line_order = 0
        prepared_entries: List[Dict[str, Any]] = []
        for item in entries:
            entry = dict(item)
            entry.setdefault("_debug_node_id", node_id)
            formula_text = str(entry.get("formula_text") or entry.get("wffstr") or "").strip()
            scope_role = str(entry.get("scope_role") or "").strip()
            if scope_role == "subproof_token" or extract_subproof_token_id(formula_text):
                prepared_entries.append(entry)
                continue
            entry.setdefault("_debug_visible_line_order", visible_line_order)
            prepared_entries.append(entry)
            visible_line_order += 1
        prepared[node_id] = prepared_entries
    return prepared


def flatten_stitched_lines(structure: Any) -> List[Dict[str, Any]]:
    flattened: List[Dict[str, Any]] = []

    def visit(node: Any) -> None:
        if isinstance(node, list):
            for child in node:
                visit(child)
            return
        if isinstance(node, dict):
            flattened.append(node)

    visit(structure)
    return flattened


def _resolve_debug_stitch_item(item: Any, index: int) -> Optional[Dict[str, Any]]:
    if not isinstance(item, dict):
        return None

    formula_text = str(item.get("formula_text") or item.get("wffstr") or "").strip()
    if extract_subproof_token_id(formula_text):
        return None

    scope_role = str(item.get("scope_role") or "").strip()
    citation = str(item.get("justification_text") or item.get("jstr") or "").strip()
    if scope_role in {"premise", "ordinary_assumption", "strict_open"}:
        citation = "PR"

    line: Dict[str, Any] = {
        "wffstr": formula_text,
        "jstr": citation,
        "scope_role": scope_role,
    }
    node_id = str(item.get("_ocr_node_id") or "").strip()
    if not node_id:
        node_id = str(item.get("_debug_node_id") or "").strip()
    if node_id:
        line["_ocr_node_id"] = node_id
    row_order = item.get("_ocr_row_order")
    if isinstance(row_order, int):
        line["_ocr_row_order"] = row_order
    visible_line_order = item.get("_debug_visible_line_order")
    line["_ocr_line_order"] = int(visible_line_order) if isinstance(visible_line_order, int) else index
    return line


def _visible_line_orders(entries: List[Dict[str, Any]]) -> Dict[int, int]:
    orders: Dict[int, int] = {}
    visible_line_order = 0
    for entry in entries:
        formula_text = str(entry.get("formula_text") or entry.get("wffstr") or "").strip()
        scope_role = str(entry.get("scope_role") or "").strip()
        if scope_role == "subproof_token" or extract_subproof_token_id(formula_text):
            continue
        row_order = entry.get("_ocr_row_order")
        if isinstance(row_order, int):
            orders[row_order] = visible_line_order
        visible_line_order += 1
    return orders


def build_stitched_line_lookup(example_dir: Path) -> Dict[Tuple[str, int], Dict[str, Any]]:
    library = _prepare_debug_stitch_library(load_node_entry_library(example_dir))
    if not library:
        return {}

    top_level_entries = library.get("ROOT") or library.get("1") or []
    if not top_level_entries:
        return {}

    stitched = recursive_stitch(top_level_entries, library, resolve_item=_resolve_debug_stitch_item)
    renumber_lines(stitched)
    visible_orders_by_node = {node_id: _visible_line_orders(entries) for node_id, entries in library.items()}

    lookup: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for line in flatten_stitched_lines(stitched):
        node_id = str(line.get("_ocr_node_id") or "").strip()
        row_order = line.get("_ocr_row_order")
        line_order = (
            visible_orders_by_node.get(node_id, {}).get(row_order)
            if isinstance(row_order, int)
            else line.get("_ocr_line_order")
        )
        if not node_id or not isinstance(line_order, int):
            continue
        lookup[(node_id, line_order)] = {
            "line_number": int(line.get("line") or 0),
            "formula": str(line.get("wffstr") or "").strip(),
            "citation": str(line.get("jstr") or "").strip(),
            "scope_role": str(line.get("scope_role") or "").strip(),
            "source_node_id": node_id,
            "source_row_order": int(row_order) if isinstance(row_order, int) else 0,
            "visible_line_order": int(line_order),
        }
    return lookup


def build_line_number_source_lookup(example_dir: Path) -> Dict[int, Dict[str, Any]]:
    stitched_lookup = build_stitched_line_lookup(example_dir)
    result: Dict[int, Dict[str, Any]] = {}
    for stitched in stitched_lookup.values():
        line_number = int(stitched.get("line_number") or 0)
        node_id = str(stitched.get("source_node_id") or "").strip()
        row_order = int(stitched.get("source_row_order") or 0)
        if line_number <= 0 or not node_id or row_order <= 0:
            continue
        result[line_number] = {
            "line_number": line_number,
            "node_id": node_id,
            "row_order": row_order,
            "formula": str(stitched.get("formula") or "").strip(),
            "citation": str(stitched.get("citation") or "").strip(),
            "scope_role": str(stitched.get("scope_role") or "").strip(),
            "line_crop_path": str((example_dir / f"node_line_{node_id}_{row_order}.jpg").resolve()),
            "formula_crop_path": str((example_dir / f"node_line_formula_{node_id}_{row_order}.jpg").resolve()),
        }
    return result


def load_node_response(example_dir: Path, node_id: str) -> Dict[str, Any]:
    parsed = read_json_artifact(example_dir / f"response_{node_id}.json")
    if isinstance(parsed, dict):
        return parsed
    if isinstance(parsed, list):
        return {
            "entries": [item for item in parsed if isinstance(item, dict)],
            "legacy_response_format": True,
        }
    return {}


def find_line_artifact(node_response: Dict[str, Any], row_order: int) -> Optional[Dict[str, Any]]:
    for item in node_response.get("line_artifacts") or []:
        if not isinstance(item, dict):
            continue
        if item.get("row_order") == row_order:
            return item
    for entry in node_response.get("entries") or []:
        if not isinstance(entry, dict):
            continue
        if entry.get("_ocr_row_order") != row_order:
            continue
        return {
            "kind": "line",
            "row_order": row_order,
            "resolved": {
                "scope_role": str(entry.get("scope_role") or "").strip(),
                "formula_text": str(entry.get("formula_text") or entry.get("wffstr") or "").strip(),
                "justification_text": str(entry.get("justification_text") or entry.get("jstr") or "").strip(),
            },
        }
    return None


def node_response_summary(node_response: Dict[str, Any]) -> Dict[str, Any]:
    summary = {
        "row_count": 0,
        "line_row_count": 0,
        "token_row_count": 0,
        "skipped_count": 0,
        "entry_count": 0,
        "opener_kind": "",
        "skipped_rows": [],
    }
    row_records = [item for item in (node_response.get("row_records") or []) if isinstance(item, dict)]
    summary["row_count"] = len(row_records)
    summary["line_row_count"] = sum(1 for item in row_records if str(item.get("kind") or "line") == "line")
    summary["token_row_count"] = sum(1 for item in row_records if str(item.get("kind") or "") == "subproof_token")
    line_artifacts = [item for item in (node_response.get("line_artifacts") or []) if isinstance(item, dict)]
    skipped = [
        item
        for item in line_artifacts
        if str(item.get("kind") or "").strip() in {"token_overlap_skipped", "strict_open_consumed"}
    ]
    summary["skipped_count"] = len(skipped)
    summary["skipped_rows"] = [int(item.get("row_order") or 0) for item in skipped if isinstance(item.get("row_order"), int)]
    summary["entry_count"] = len([item for item in (node_response.get("entries") or []) if isinstance(item, dict)])
    opener = node_response.get("opener_result")
    if isinstance(opener, dict):
        summary["opener_kind"] = str(opener.get("kind") or "").strip()
    return summary


def row_order_to_visible_line_order(node_response: Dict[str, Any], row_order: int) -> Optional[int]:
    visible_line_order = 0
    line_artifacts = node_response.get("line_artifacts") or []
    if line_artifacts:
        for item in line_artifacts:
            if not isinstance(item, dict):
                continue
            kind = str(item.get("kind") or "").strip()
            if kind in {"subproof_token", "token_overlap_skipped"}:
                continue
            candidate = item.get("row_order")
            if isinstance(candidate, int) and candidate == row_order:
                return visible_line_order
            visible_line_order += 1
        return None

    for entry in node_response.get("entries") or []:
        if not isinstance(entry, dict):
            continue
        formula_text = str(entry.get("formula_text") or entry.get("wffstr") or "").strip()
        scope_role = str(entry.get("scope_role") or "").strip()
        if scope_role == "subproof_token" or extract_subproof_token_id(formula_text):
            continue
        candidate = entry.get("_ocr_row_order")
        if isinstance(candidate, int) and candidate == row_order:
            return visible_line_order
        visible_line_order += 1
    return None


def _annotate_review_result_assets(parsed: Any, *, example_slug: str) -> List[Dict[str, Any]]:
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


def load_unary_verifier_results(debug_dir: Path, example_slug: str, starter_example: str) -> List[Dict[str, Any]]:
    if not example_slug or example_slug == starter_example:
        return []
    path = debug_dir / example_slug / "unary_verifier.json"
    if not path.exists():
        return []
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return _annotate_review_result_assets(parsed, example_slug=example_slug)


def load_formula_review_results(debug_dir: Path, example_slug: str, starter_example: str) -> List[Dict[str, Any]]:
    if not example_slug or example_slug == starter_example:
        return []
    path = debug_dir / example_slug / "checker_formula_rereads.json"
    if not path.exists():
        return []
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return _annotate_review_result_assets(parsed, example_slug=example_slug)
