"""Verifier-guided single-line reread helpers for TFL/Fitch OCR.

The main OCR pipeline intentionally avoids semantic auto-repair: a verifier
error might be a real student mistake.  This module provides an opt-in probe
path that uses verifier issues only to choose which visual rows to reread.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping as MappingABC
from pathlib import Path
from typing import Any, Dict, List, Mapping

import cv2
import numpy as np

from proofs_ocr.common.rows import band_height, detect_text_row_bands

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


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int | None = None) -> int | None:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _get_27b_model(model: str | None = None) -> str | None:
    explicit = (model or "").strip()
    if explicit:
        return explicit
    env_model = os.getenv("TFL_FITCH_VERIFIER_REREAD_MODEL") or os.getenv("LMSTUDIO_MODEL_27B")
    if env_model is None:
        env_model = "qwen/qwen3.6-27b"
    env_model = env_model.strip()
    return env_model or None


def _get_27b_extra_payload() -> dict[str, object]:
    extra_payload: dict[str, object] = {}
    if _env_flag("TFL_FITCH_27B_ENABLE_THINKING_FALSE"):
        extra_payload["chat_template_kwargs"] = {"enable_thinking": False}
    return extra_payload


def _normalize_candidate_text(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"\s+", "", text)
    text = text.replace("->", "→").replace("&", "∧")
    return text


def _parsed_line_number(parsed: Any) -> int | None:
    if not isinstance(parsed, MappingABC):
        return None
    raw = parsed.get("line")
    if isinstance(raw, int):
        return raw
    text = str(raw or "").strip()
    if text.isdigit():
        return int(text)
    return None


def _candidate_materially_changed(
    suspicion: Mapping[str, Any],
    parsed: Any,
    *,
    line_matches_target: bool,
) -> bool:
    if not isinstance(parsed, MappingABC):
        return False
    if not line_matches_target:
        return False
    old_wff = _normalize_candidate_text(suspicion.get("wffstr"))
    old_j = _normalize_candidate_text(suspicion.get("jstr"))
    new_wff = _normalize_candidate_text(parsed.get("wffstr"))
    new_j = _normalize_candidate_text(parsed.get("jstr"))
    return bool(new_wff or new_j) and (old_wff != new_wff or old_j != new_j)


def parse_citation_line_numbers(jstr: str) -> list[int]:
    """Extract cited line numbers from a TFL justification string."""
    numbers: set[int] = set()
    for start, end in re.findall(r"(\d+)\s*[-–]\s*(\d+)", jstr):
        a = int(start)
        b = int(end)
        lo, hi = sorted((a, b))
        numbers.update(range(lo, hi + 1))
    without_ranges = re.sub(r"\d+\s*[-–]\s*\d+", " ", jstr)
    for num in re.findall(r"\d+", without_ranges):
        numbers.add(int(num))
    return sorted(numbers)


def _fullwidth_sibling_candidates(crop_path: Path, node_id: Any) -> list[Path]:
    node = str(node_id or "").strip()
    candidates: list[Path] = []
    if node:
        candidates.extend(
            [
                crop_path.with_name(f"crop_{node}_fullwidth.jpg"),
                crop_path.with_name(f"crop_{node}_4b_fullwidth.jpg"),
            ]
        )
    if crop_path.name.endswith(".jpg") and "fullwidth" not in crop_path.stem:
        candidates.extend(
            [
                crop_path.with_name(f"{crop_path.stem}_fullwidth.jpg"),
                crop_path.with_name(f"{crop_path.stem}_4b_fullwidth.jpg"),
            ]
        )
    return candidates


def locate_or_derive_full_width_crop(
    *,
    ocr: Mapping[str, Any],
    source_image: np.ndarray | None,
    output_dir: Path,
) -> tuple[Path | None, str]:
    """Find or create a full-width crop for a provenance record."""
    crop_path = Path(str(ocr.get("crop_path") or ""))
    if crop_path.exists() and "fullwidth" in crop_path.stem:
        return crop_path, "provenance_fullwidth"

    if crop_path.exists():
        for candidate in _fullwidth_sibling_candidates(crop_path, ocr.get("node_id")):
            if candidate.exists():
                return candidate, "sibling_fullwidth"

    if source_image is None:
        if crop_path.exists():
            return crop_path, "provenance_fallback_not_fullwidth"
        return None, "source image unavailable for full-width derivation"

    node_id = str(ocr.get("node_id") or "node")
    if node_id == "ROOT":
        full = source_image
    else:
        box = ocr.get("node_box")
        if not (
            isinstance(box, list)
            and len(box) == 4
            and all(isinstance(value, (int, float)) for value in box)
        ):
            return None, "missing node box for full-width derivation"
        _x, y, _w, h = [int(value) for value in box]
        h_img = source_image.shape[0]
        y1 = max(0, y - 10)
        masked_crop = cv2.imread(str(crop_path)) if crop_path.exists() else None
        if str(ocr.get("parent_id") or "") == "ROOT":
            y2 = h_img
        elif masked_crop is not None:
            y2 = min(h_img, y1 + masked_crop.shape[0])
        else:
            y2 = min(h_img, y + h)
        full = source_image[y1:y2, :].copy()
        if crop_path.exists():
            x1 = max(0, int(box[0]) - 10)
            if (
                masked_crop is not None
                and masked_crop.shape[0] == full.shape[0]
                and x1 < full.shape[1]
            ):
                paste_w = min(masked_crop.shape[1], full.shape[1] - x1)
                full[:, x1 : x1 + paste_w] = masked_crop[:, :paste_w]

    derived_dir = output_dir / "verifier_line_rereads_27b"
    derived_dir.mkdir(parents=True, exist_ok=True)
    full_path = derived_dir / f"node_{node_id}_fullwidth_source.jpg"
    cv2.imwrite(str(full_path), full)
    return full_path, "derived_from_source"


def detect_ocr_row_bounds(crop_path: Path, *, row_count: int, padding: int = 18) -> list[tuple[int, int]]:
    """Detect row y-bounds in the exact image sent to OCR."""
    img = cv2.imread(str(crop_path))
    if img is None or row_count < 1:
        return []

    bands = [band for band in detect_text_row_bands(img) if band_height(band) >= 18]
    if len(bands) != row_count:
        h_img = img.shape[0]
        step = h_img / float(row_count)
        bands = []
        for idx in range(row_count):
            center = (idx + 0.5) * step
            half_height = max(24.0, step * 0.42)
            bands.append((int(max(0, center - half_height)), int(min(h_img - 1, center + half_height))))

    h_img = img.shape[0]
    return [
        (max(0, int(y1) - padding), min(h_img, int(y2) + padding))
        for y1, y2 in bands
    ]


def detect_visible_gutter_line_bounds(
    crop_path: Path,
    *,
    target_line: int,
    first_line: int,
    last_line: int,
    node_box: list[Any] | None = None,
    padding: int = 18,
) -> tuple[int, int] | None:
    """Estimate a row y-window from visible left-gutter line-number bands."""
    if target_line < first_line or target_line > last_line:
        return None
    img = cv2.imread(str(crop_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    h_img, w_img = img.shape[:2]
    if node_box and len(node_box) >= 1 and isinstance(node_box[0], (int, float)):
        x_end = max(20, min(int(node_box[0]) - 45, w_img // 3))
    else:
        x_end = max(20, w_img // 6)
    x_start = max(0, x_end - 60)
    gutter = img[:, x_start:x_end]
    if gutter.size == 0:
        return None

    mask = gutter < 160
    projection = mask.sum(axis=1)
    raw_bands: list[tuple[int, int]] = []
    in_band = False
    start = 0
    for y, value in enumerate(projection):
        if value > 1 and not in_band:
            start = y
            in_band = True
        if (value <= 1 or y == len(projection) - 1) and in_band:
            end = y
            in_band = False
            if end - start >= 4:
                raw_bands.append((start, end))

    merged: list[tuple[int, int]] = []
    for band in raw_bands:
        if merged and band[0] - merged[-1][1] <= 8:
            merged[-1] = (merged[-1][0], band[1])
        else:
            merged.append(band)
    bands = [band for band in merged if band[1] - band[0] >= 8]

    expected_count = last_line - first_line + 1
    if len(bands) < expected_count:
        return None
    index = target_line - first_line
    if index >= len(bands):
        return None
    y1, y2 = bands[index]
    return max(0, int(y1) - padding), min(h_img, int(y2) + padding)


def extract_single_line_crop(
    crop_path: Path,
    *,
    local_index: int,
    expected_lines: int,
    output_path: Path,
    padding: int = 18,
    row_bounds: tuple[int, int] | None = None,
) -> Path:
    """Extract a generous full-width row crop from a node crop."""
    img = cv2.imread(str(crop_path))
    if img is None:
        raise FileNotFoundError(f"Could not read crop image: {crop_path}")
    if local_index < 1 or expected_lines < 1 or local_index > expected_lines:
        raise ValueError(f"Invalid row {local_index} of {expected_lines} for {crop_path}")

    if row_bounds is not None:
        y1, y2 = row_bounds
        y1 = max(0, int(y1))
        y2 = min(img.shape[0], int(y2))
    else:
        bands = [band for band in detect_text_row_bands(img) if band_height(band) >= 18]
        if len(bands) != expected_lines:
            h_img = img.shape[0]
            step = h_img / float(expected_lines)
            bands = []
            for idx in range(expected_lines):
                center = (idx + 0.5) * step
                half_height = max(24.0, step * 0.42)
                bands.append((int(max(0, center - half_height)), int(min(h_img - 1, center + half_height))))

        y1, y2 = bands[local_index - 1]
        y1 = max(0, int(y1) - padding)
        y2 = min(img.shape[0], int(y2) + padding)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), img[y1:y2, :])
    return output_path


def build_single_line_reread_prompt(suspicion: Mapping[str, Any], *, target_kind: str = "rejected") -> str:
    issues = "; ".join(str(item) for item in suspicion.get("issues", []))
    target_label = "rejected verifier line" if target_kind == "rejected" else "line cited by the rejected line"
    return (
        "You are rereading one cropped row from a handwritten Fitch proof because "
        f"the first OCR pass produced a verifier warning. Target: {target_label}.\n"
        f"Final proof line: {suspicion.get('line')}.\n"
        f"First OCR formula: {suspicion.get('wffstr', '')}\n"
        f"First OCR justification: {suspicion.get('jstr', '')}\n"
        f"Verifier warning: {issues}\n\n"
        "Read only the visible text in this single row image. Return exactly one JSON object "
        "with keys \"line\", \"wffstr\", and \"jstr\". Put only the formula in "
        "\"wffstr\" and only the printed rule/citations in \"jstr\". Use the visible "
        "left-gutter line number if present. Preserve visible citation numbers exactly "
        "as written. Do not prove, repair, or validate the line semantically. Do not "
        "infer a valid proof from logic alone. Return JSON only."
    )


def _json_from_text(text: str) -> Any:
    content = text.strip()
    if "```json" in content:
        content = content.split("```json", 1)[1].split("```", 1)[0]
    elif "```" in content:
        content = content.split("```", 1)[1].split("```", 1)[0]
    return json.loads(content)


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
            )
            prompt = build_single_line_reread_prompt(target, target_kind=target_kind)
            if _env_flag("TFL_FITCH_27B_NO_THINK"):
                prompt = f"/no_think\n{prompt}"
            if (
                reread_model
                and _env_flag("TFL_FITCH_27B_SINGLE_USER_MESSAGE")
                and hasattr(client, "build_single_user_image_message")
            ):
                messages = client.build_single_user_image_message(row_path, prompt)
            else:
                messages = client.build_messages_for_prompt(row_path, prompt)
            chat_kwargs = {
                "model": reread_model,
                "timeout": timeout,
                "max_tokens": max_tokens,
            }
            if extra_payload:
                chat_kwargs["extra_payload"] = extra_payload
            raw = client.chat(messages, **chat_kwargs)
            parsed = _json_from_text(raw)
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
