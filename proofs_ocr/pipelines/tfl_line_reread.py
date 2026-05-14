"""Verifier-guided single-line reread helpers for TFL/Fitch OCR.

The main OCR pipeline intentionally avoids semantic auto-repair: a verifier
error might be a real student mistake.  This module provides an opt-in probe
path that uses verifier issues only to choose which visual rows to reread with
the default small model.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping

import cv2

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
                    },
                }
            )
    return suspicions


def extract_single_line_crop(
    crop_path: Path,
    *,
    local_index: int,
    expected_lines: int,
    output_path: Path,
    padding: int = 18,
) -> Path:
    """Extract a generous full-width row crop from a node crop."""
    img = cv2.imread(str(crop_path))
    if img is None:
        raise FileNotFoundError(f"Could not read crop image: {crop_path}")
    if local_index < 1 or expected_lines < 1 or local_index > expected_lines:
        raise ValueError(f"Invalid row {local_index} of {expected_lines} for {crop_path}")

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


def build_single_line_reread_prompt(suspicion: Mapping[str, Any]) -> str:
    issues = "; ".join(str(item) for item in suspicion.get("issues", []))
    return (
        "You are rereading one cropped row from a handwritten Fitch proof because "
        "the first OCR pass produced a verifier warning.\n"
        f"Final proof line: {suspicion.get('line')}.\n"
        f"First OCR formula: {suspicion.get('wffstr', '')}\n"
        f"First OCR justification: {suspicion.get('jstr', '')}\n"
        f"Verifier warning: {issues}\n\n"
        "Read only the visible row in the image. Return exactly one JSON object "
        "with keys \"line\", \"wffstr\", and \"jstr\". Put only the formula in "
        "\"wffstr\" and only the printed rule/citations in \"jstr\". Preserve "
        "citation numbers exactly as written. Do not infer a valid proof and do "
        "not repair the student's work from logic alone. Return JSON only."
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
) -> list[dict[str, Any]]:
    """Reread verifier-suspicious leaf rows and return candidate records."""
    suspicions = find_verifier_suspicions(proof_structure, numprems=numprems)
    records: list[dict[str, Any]] = []
    reread_dir = output_dir / "verifier_line_rereads"

    for suspicion in suspicions:
        ocr = suspicion.get("ocr") if isinstance(suspicion.get("ocr"), dict) else {}
        crop_path = Path(str(ocr.get("crop_path") or ""))
        local_index = ocr.get("local_index")
        expected_lines = ocr.get("expected_lines")
        is_leaf = bool(ocr.get("is_leaf"))

        record: dict[str, Any] = {
            "line": suspicion.get("line"),
            "suspicion": suspicion,
            "status": "skipped",
        }
        if not is_leaf:
            record["reason"] = "line did not come from a leaf OCR crop"
            records.append(record)
            continue
        if not crop_path.exists() or not isinstance(local_index, int) or not isinstance(expected_lines, int):
            record["reason"] = "missing crop provenance"
            records.append(record)
            continue

        row_path = reread_dir / (
            f"line_{int(suspicion['line']):02d}_"
            f"node_{ocr.get('node_id')}_row_{local_index}.jpg"
        )
        try:
            extract_single_line_crop(
                crop_path,
                local_index=local_index,
                expected_lines=expected_lines,
                output_path=row_path,
            )
            prompt = build_single_line_reread_prompt(suspicion)
            messages = client.build_messages_for_prompt(row_path, prompt)
            raw = client.chat(messages, model=model, timeout=timeout, max_tokens=256)
            parsed = _json_from_text(raw)
            record.update(
                {
                    "status": "candidate",
                    "row_crop": str(row_path),
                    "prompt": prompt,
                    "raw": raw,
                    "parsed": parsed,
                    "metadata": getattr(client, "last_response_metadata", {}) or {},
                }
            )
        except Exception as exc:  # noqa: BLE001 - probe records failures.
            record.update({"status": "error", "error": str(exc)})
        records.append(record)

    return records
