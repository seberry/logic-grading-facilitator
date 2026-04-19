from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote

from scripts.debug_modal_proof_ocr import recursive_stitch, renumber_lines


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


def build_stitched_line_lookup(example_dir: Path) -> Dict[Tuple[str, int], Dict[str, Any]]:
    library = load_node_entry_library(example_dir)
    if not library:
        return {}

    root_entries = library.get("ROOT", [])
    root_has_content = any(str(item.get("formula_text") or item.get("wffstr") or "").strip() for item in root_entries)
    top_level_entries = library.get("1", []) if not root_has_content and "1" in library else root_entries
    if not top_level_entries:
        return {}

    stitched = recursive_stitch(top_level_entries, library)
    while isinstance(stitched, list) and len(stitched) == 1 and isinstance(stitched[0], list):
        stitched = stitched[0]
    renumber_lines(stitched)

    lookup: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for line in flatten_stitched_lines(stitched):
        node_id = str(line.get("_ocr_node_id") or "").strip()
        line_order = line.get("_ocr_line_order")
        if not node_id or not isinstance(line_order, int):
            continue
        lookup[(node_id, line_order)] = {
            "line_number": int(line.get("line") or 0),
            "formula": str(line.get("wffstr") or "").strip(),
            "citation": str(line.get("jstr") or "").strip(),
            "scope_role": str(line.get("scope_role") or "").strip(),
        }
    return lookup


def load_node_response(example_dir: Path, node_id: str) -> Dict[str, Any]:
    parsed = read_json_artifact(example_dir / f"response_{node_id}.json")
    return parsed if isinstance(parsed, dict) else {}


def find_line_artifact(node_response: Dict[str, Any], row_order: int) -> Optional[Dict[str, Any]]:
    for item in node_response.get("line_artifacts") or []:
        if not isinstance(item, dict):
            continue
        if item.get("row_order") == row_order:
            return item
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
    for item in node_response.get("line_artifacts") or []:
        if not isinstance(item, dict):
            continue
        kind = str(item.get("kind") or "").strip()
        if kind in {"subproof_token", "strict_open_consumed", "token_overlap_skipped"}:
            continue
        candidate = item.get("row_order")
        if isinstance(candidate, int) and candidate == row_order:
            return visible_line_order
        visible_line_order += 1
    return None


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
