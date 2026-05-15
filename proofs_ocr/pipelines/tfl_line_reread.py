"""Verifier-guided single-line reread helpers for TFL/Fitch OCR.

The main OCR pipeline intentionally avoids semantic auto-repair: a verifier
error might be a real student mistake.  This module provides an opt-in probe
path that uses verifier issues only to choose which visual rows to reread.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, List, Mapping

import numpy as np

from proofs_ocr.pipelines.tfl_line_reread_candidates import (
    candidate_materially_changed as _candidate_materially_changed,
    parse_citation_line_numbers,
    parsed_line_number as _parsed_line_number,
)
from proofs_ocr.pipelines.tfl_line_reread_geometry import (
    detect_ocr_row_bounds,
    detect_visible_gutter_line_bounds,
    extract_single_line_crop,
    locate_or_derive_full_width_crop,
)
from proofs_ocr.pipelines.tfl_line_reread_vlm import (
    build_single_line_reread_prompt,
    call_single_line_reread,
    env_flag as _env_flag,
    env_int as _env_int,
    get_27b_extra_payload as _get_27b_extra_payload,
    get_27b_model as _get_27b_model,
)

try:
    from app.verifier import ProofLine, check_proof
except ImportError:
    ProofLine = None  # type: ignore[assignment]
    check_proof = None  # type: ignore[assignment]


OCR_META_PREFIX = "_ocr_"


def tag_ocr_result_lines(
    ocr_result: object,
    *,
    node_id: str,
    crop_path: str,
    expected_lines: int | None,
    is_leaf: bool,
    node_box: tuple[int, int, int, int] | None = None,
    parent_id: str | None = None,
    row_bounds: list[tuple[int, int]] | None = None,
) -> None:
    """Attach transient OCR provenance to line dictionaries in a node result."""
    if not isinstance(ocr_result, list):
        return

    local_index = 0
    for item in ocr_result:
        if not isinstance(item, dict):
            continue
        local_index += 1
        item["_ocr_node_id"] = str(node_id)
        item["_ocr_local_index"] = local_index
        item["_ocr_crop_path"] = str(crop_path)
        item["_ocr_expected_lines"] = expected_lines
        item["_ocr_is_leaf"] = bool(is_leaf)
        item["_ocr_node_box"] = list(node_box) if node_box is not None else None
        item["_ocr_parent_id"] = str(parent_id) if parent_id is not None else None
        if row_bounds and 1 <= local_index <= len(row_bounds):
            y1, y2 = row_bounds[local_index - 1]
            item["_ocr_row_bounds"] = [int(y1), int(y2)]
        else:
            item["_ocr_row_bounds"] = None


def strip_ocr_metadata(proof_structure: object) -> None:
    """Remove transient OCR provenance keys before writing public JSON."""
    if isinstance(proof_structure, list):
        for item in proof_structure:
            strip_ocr_metadata(item)
        return
    if isinstance(proof_structure, dict):
        for key in list(proof_structure.keys()):
            if key.startswith(OCR_META_PREFIX):
                proof_structure.pop(key, None)


def _structure_to_proof_lines(structure: list) -> List[Any]:
    if ProofLine is None:
        raise RuntimeError("TFL verifier is unavailable")

    proof_lines: List[Any] = []
    for item in structure:
        if isinstance(item, list):
            proof_lines.append(_structure_to_proof_lines(item))
        elif isinstance(item, dict):
            proof_lines.append(
                ProofLine(
                    wffstr=str(item.get("wffstr", "")).strip(),
                    jstr=str(item.get("jstr", "")).strip(),
                    location=[],
                )
            )
    return proof_lines


def _flatten_line_dicts(structure: list) -> list[dict[str, Any]]:
    flattened: list[dict[str, Any]] = []

    def walk(node: object) -> None:
        if isinstance(node, list):
            for child in node:
                walk(child)
        elif isinstance(node, dict):
            flattened.append(node)

    walk(structure)
    return flattened


def find_verifier_suspicions(
    proof_structure: list,
    *,
    numprems: int,
    conclusion: str | None = None,
) -> list[dict[str, Any]]:
    """Return verifier issues grouped by final proof line number."""
    if check_proof is None:
        return []

    proof_lines = _structure_to_proof_lines(proof_structure)
    flat_dicts = _flatten_line_dicts(proof_structure)
    if not proof_lines or not flat_dicts:
        return []

    target_conclusion = conclusion
    if target_conclusion is None:
        target_conclusion = str(flat_dicts[-1].get("wffstr", "")).strip()

    result = check_proof(proof_lines, numprems=numprems, conc=target_conclusion)
    grouped: dict[int, list[str]] = {}
    for issue in result.issues:
        match = re.match(r"Line\s+(\d+):\s*(.*)", issue)
        if not match:
            continue
        line_no = int(match.group(1))
        grouped.setdefault(line_no, []).append(match.group(2).strip())

    suspicions: list[dict[str, Any]] = []
    for line_no in sorted(grouped):
        if 1 <= line_no <= len(flat_dicts):
            line_dict = flat_dicts[line_no - 1]
            suspicions.append(
                {
                    "line": line_no,
                    "wffstr": str(line_dict.get("wffstr", "")).strip(),
                    "jstr": str(line_dict.get("jstr", "")).strip(),
                    "issues": grouped[line_no],
                    "ocr": {
                        "node_id": line_dict.get("_ocr_node_id"),
                        "local_index": line_dict.get("_ocr_local_index"),
                        "crop_path": line_dict.get("_ocr_crop_path"),
                        "expected_lines": line_dict.get("_ocr_expected_lines"),
                        "is_leaf": line_dict.get("_ocr_is_leaf"),
                        "node_box": line_dict.get("_ocr_node_box"),
                        "parent_id": line_dict.get("_ocr_parent_id"),
                        "row_bounds": line_dict.get("_ocr_row_bounds"),
                    },
                }
            )
    return suspicions


def collect_verifier_reread_candidates(
    proof_structure: list,
    *,
    output_dir: Path,
    client: Any,
    numprems: int,
    model: str | None = None,
    timeout: int = 120,
    source_image: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    """Reread verifier-suspicious rows and return candidate records."""
    suspicions = find_verifier_suspicions(proof_structure, numprems=numprems)
    flat_lines = _flatten_line_dicts(proof_structure)
    node_expected_lines: dict[str, int] = {}
    node_line_ranges: dict[str, tuple[int, int]] = {}
    for line_dict in flat_lines:
        node_id = line_dict.get("_ocr_node_id")
        local_index = line_dict.get("_ocr_local_index")
        if node_id is None or not isinstance(local_index, int):
            continue
        key = str(node_id)
        node_expected_lines[key] = max(node_expected_lines.get(key, 0), local_index)
        line_no = line_dict.get("line")
        if isinstance(line_no, int):
            current = node_line_ranges.get(key)
            if current is None:
                node_line_ranges[key] = (line_no, line_no)
            else:
                node_line_ranges[key] = (min(current[0], line_no), max(current[1], line_no))
    records: list[dict[str, Any]] = []
    reread_dir = output_dir / "verifier_line_rereads_27b"
    reread_model = _get_27b_model(model)
    max_tokens = _env_int("TFL_FITCH_27B_MAX_TOKENS", 256) or 256
    extra_payload = _get_27b_extra_payload()

    def reread_one(target: Mapping[str, Any], *, target_kind: str, source_line: int | None = None) -> dict[str, Any]:
        ocr = target.get("ocr") if isinstance(target.get("ocr"), dict) else {}
        crop_path = Path(str(ocr.get("crop_path") or ""))
        local_index = ocr.get("local_index")
        expected_lines = ocr.get("expected_lines")
        if not isinstance(expected_lines, int):
            expected_lines = node_expected_lines.get(str(ocr.get("node_id") or ""))

        record: dict[str, Any] = {
            "line": target.get("line"),
            "target_kind": target_kind,
            "source_rejected_line": source_line,
            "suspicion": target,
            "status": "skipped",
        }
        if not crop_path.exists() and source_image is None:
            record["reason"] = "missing crop provenance"
            return record
        if not isinstance(local_index, int) or not isinstance(expected_lines, int):
            record["reason"] = "missing row provenance"
            return record

        full_crop_path, full_crop_source = locate_or_derive_full_width_crop(
            ocr=ocr,
            source_image=source_image,
            output_dir=output_dir,
        )
        if full_crop_path is None:
            record["reason"] = full_crop_source
            return record

        row_path = reread_dir / (
            f"{target_kind}_line_{int(target['line']):02d}_"
            f"node_{ocr.get('node_id')}_row_{local_index}.jpg"
        )
        try:
            row_bounds_value = ocr.get("row_bounds")
            row_bounds_tuple = None
            target_line = target.get("line")
            node_line_range = node_line_ranges.get(str(ocr.get("node_id") or ""))
            if (
                not bool(ocr.get("is_leaf"))
                and isinstance(target_line, int)
                and node_line_range is not None
            ):
                gutter_bounds = detect_visible_gutter_line_bounds(
                    full_crop_path,
                    target_line=target_line,
                    first_line=node_line_range[0],
                    last_line=node_line_range[1],
                    node_box=ocr.get("node_box") if isinstance(ocr.get("node_box"), list) else None,
                )
                if gutter_bounds is not None:
                    row_bounds_tuple = gutter_bounds
            if (
                row_bounds_tuple is None
                and
                isinstance(row_bounds_value, list)
                and len(row_bounds_value) == 2
                and all(isinstance(value, (int, float)) for value in row_bounds_value)
            ):
                row_bounds_tuple = (int(row_bounds_value[0]), int(row_bounds_value[1]))
            extract_single_line_crop(
                full_crop_path,
                local_index=local_index,
                expected_lines=expected_lines,
                output_path=row_path,
                row_bounds=row_bounds_tuple,
                row_bounds_node_box=ocr.get("node_box")
                if (
                    not bool(ocr.get("is_leaf"))
                    and isinstance(ocr.get("node_box"), list)
                )
                else None,
            )
            prompt = build_single_line_reread_prompt(target, target_kind=target_kind)
            if _env_flag("TFL_FITCH_27B_NO_THINK"):
                prompt = f"/no_think\n{prompt}"
            raw, parsed = call_single_line_reread(
                client=client,
                row_path=row_path,
                prompt=prompt,
                model=reread_model,
                timeout=timeout,
                max_tokens=max_tokens,
                extra_payload=extra_payload,
            )
            parsed_line = _parsed_line_number(parsed)
            target_line = target.get("line")
            line_matches_target = parsed_line is None or parsed_line == target_line
            material_change = _candidate_materially_changed(
                target,
                parsed,
                line_matches_target=line_matches_target,
            )
            record.update(
                {
                    "status": "candidate",
                    "row_crop": str(row_path),
                    "full_width_crop": str(full_crop_path),
                    "full_width_crop_source": full_crop_source,
                    "prompt": prompt,
                    "raw": raw,
                    "parsed": parsed,
                    "parsed_line": parsed_line,
                    "line_matches_target": line_matches_target,
                    "material_change": material_change,
                    "metadata": getattr(client, "last_response_metadata", {}) or {},
                }
            )
        except Exception as exc:  # noqa: BLE001 - probe records failures.
            record.update({"status": "error", "error": str(exc)})
        return record

    for suspicion in suspicions:
        record = reread_one(suspicion, target_kind="rejected")
        records.append(record)

        if record.get("status") == "candidate" and record.get("material_change"):
            continue

        for cited_line in parse_citation_line_numbers(str(suspicion.get("jstr") or "")):
            if cited_line == suspicion.get("line") or cited_line < 1 or cited_line > len(flat_lines):
                continue
            line_dict = flat_lines[cited_line - 1]
            if line_dict.get("_ocr_local_index") is None:
                continue
            cited_target = {
                "line": cited_line,
                "wffstr": str(line_dict.get("wffstr", "")).strip(),
                "jstr": str(line_dict.get("jstr", "")).strip(),
                "issues": [
                    f"Secondary evidence for rejected line {suspicion.get('line')}",
                ],
                "ocr": {
                    "node_id": line_dict.get("_ocr_node_id"),
                    "local_index": line_dict.get("_ocr_local_index"),
                    "crop_path": line_dict.get("_ocr_crop_path"),
                    "expected_lines": line_dict.get("_ocr_expected_lines"),
                    "is_leaf": line_dict.get("_ocr_is_leaf"),
                    "node_box": line_dict.get("_ocr_node_box"),
                    "parent_id": line_dict.get("_ocr_parent_id"),
                    "row_bounds": line_dict.get("_ocr_row_bounds"),
                },
            }
            records.append(
                reread_one(
                    cited_target,
                    target_kind="cited",
                    source_line=int(suspicion["line"]),
                )
            )

    return records
