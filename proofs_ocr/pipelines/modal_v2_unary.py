"""Passive unary-analysis helpers for the modal_v2 baseline pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2

from modal_ocr.verifier import (
    build_structured_unary_verifier_prompt,
    leading_unary_prefix,
    should_verify_unary_formula,
)
from proofs_ocr.common import artifacts
from proofs_ocr.domains import modal
from proofs_ocr.pipelines.modal_v2_core import ImagePromptClient
from proofs_ocr.regression import unary_blob_probe


def _int_field(value: Any, default: int = 0, minimum: int = 0, maximum: int = 3) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, parsed))


def _formula_components(formula_text: str) -> tuple[str, str]:
    normalized = modal.normalize_formula_text(formula_text)
    prefix = leading_unary_prefix(normalized)
    body = normalized[len(prefix) :].strip()
    return prefix, body


def derive_formula_from_structured_counts(
    *,
    formula_text: str,
    structured_parsed: Mapping[str, Any],
    scope_role: str = "derived_line",
) -> str | None:
    """Reconstruct a unary formula directly from the structured negation counts."""

    normalized_formula = modal.normalize_formula_text(formula_text if scope_role != "strict_open" else "□")
    _, body = _formula_components(normalized_formula)
    if not body:
        return None

    modal_value = str(structured_parsed.get("main_modal_operator") or "").strip().lower()
    if modal_value == "box":
        modal_symbol = "□"
    elif modal_value == "diamond":
        modal_symbol = "⋄"
    else:
        modal_symbol = ""

    before_modal = _int_field(structured_parsed.get("negations_before_modal"), 0, 0, 3)
    before_atom = _int_field(structured_parsed.get("negations_before_final_atom"), 0, 0, 3)

    if not modal_symbol:
        return f"{'¬' * before_atom}{body}"
    return f"{'¬' * before_modal}{modal_symbol}{'¬' * before_atom}{body}"


def _should_probe_entry(entry: Mapping[str, Any]) -> bool:
    return should_verify_unary_formula(
        dict(entry),
        normalize_formula_text=lambda value, scope_role: (
            "□" if str(scope_role or "").strip() == "strict_open" else modal.normalize_formula_text(value)
        ),
        normalize_justification_text=modal.normalize_justification_text,
    )


def _is_atomic_unary_body(body: str) -> bool:
    return bool(body) and (body.isalpha() or body == "⊥")


def _should_probe_structured_entry(entry: Mapping[str, Any]) -> bool:
    if not _should_probe_entry(entry):
        return False

    formula_text = modal.normalize_formula_text(entry.get("formula_text") or entry.get("wffstr") or "")
    if "¬" not in formula_text:
        return False
    if any(char in formula_text for char in "()∧∨→↔"):
        return False

    modal_count = formula_text.count("□") + formula_text.count("⋄")
    if modal_count > 1:
        return False

    _, body = _formula_components(formula_text)
    return _is_atomic_unary_body(body)


def _upscale_for_probe(crop_img, *, scale: int = 4):
    if crop_img.size == 0:
        return crop_img
    return cv2.resize(crop_img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)


def run_structured_unary_probe(
    *,
    entries_by_node: Mapping[str, Sequence[Mapping[str, Any]]],
    output_dir: Path,
    client: ImagePromptClient,
) -> list[dict[str, Any]]:
    """Run the structured unary verifier prompt on baseline line crops."""

    records: list[dict[str, Any]] = []

    for node_id in sorted(entries_by_node.keys(), key=str):
        for entry in entries_by_node.get(node_id, []):
            if not isinstance(entry, Mapping):
                continue
            if not _should_probe_structured_entry(entry):
                continue

            row_order = int(entry.get("_ocr_row_order", 0))
            crop_path = output_dir / artifacts.NODE_LINE_CROP_TEMPLATE.format(node_id=node_id, row_order=row_order)
            crop_img = cv2.imread(str(crop_path))
            if crop_img is None or crop_img.size == 0:
                continue

            probe_crop_path = output_dir / artifacts.UNARY_STRUCTURED_LINE_CROP_TEMPLATE.format(
                node_id=node_id,
                row_order=row_order,
            )
            cv2.imwrite(str(probe_crop_path), _upscale_for_probe(crop_img))

            prompt = build_structured_unary_verifier_prompt(
                dict(entry),
                normalize_formula_text=lambda value, scope_role: (
                    "□" if str(scope_role or "").strip() == "strict_open" else modal.normalize_formula_text(value)
                ),
                normalize_justification_text=modal.normalize_justification_text,
            )
            raw_response = client.chat_with_image_prompt(probe_crop_path, prompt)
            parsed = modal.parse_json_object_response(raw_response)
            baseline_formula = modal.normalize_formula_text(entry.get("formula_text") or entry.get("wffstr") or "")
            scope_role = str(entry.get("scope_role") or "derived_line").strip()
            structured_formula = derive_formula_from_structured_counts(
                formula_text=baseline_formula,
                structured_parsed=parsed,
                scope_role=scope_role,
            )
            records.append(
                {
                    "node_id": str(node_id),
                    "row_order": row_order,
                    "line_label": str(entry.get("line_label") or "").strip(),
                    "scope_role": scope_role,
                    "baseline_formula": baseline_formula,
                    "baseline_justification": modal.normalize_justification_text(
                        entry.get("justification_text") or entry.get("jstr") or ""
                    ),
                    "structured_counts": parsed,
                    "structured_formula": structured_formula or "",
                    "would_change_formula": bool(structured_formula and structured_formula != baseline_formula),
                    "line_crop_filename": crop_path.name,
                    "probe_crop_filename": probe_crop_path.name,
                    "raw_response": raw_response,
                }
            )

    (output_dir / artifacts.UNARY_STRUCTURED_PROBE_FILENAME).write_text(
        json.dumps(records, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return records


def run_blob_unary_probe(
    *,
    entries_by_node: Mapping[str, Sequence[Mapping[str, Any]]],
    output_dir: Path,
) -> list[dict[str, Any]]:
    """Run the passive unary blob-count probe on baseline line crops."""

    records: list[dict[str, Any]] = []

    for node_id in sorted(entries_by_node.keys(), key=str):
        for entry in entries_by_node.get(node_id, []):
            if not isinstance(entry, Mapping):
                continue
            if not _should_probe_entry(entry):
                continue

            row_order = int(entry.get("_ocr_row_order", 0))
            crop_path = output_dir / artifacts.NODE_LINE_CROP_TEMPLATE.format(node_id=node_id, row_order=row_order)
            if not crop_path.exists():
                continue

            line_output_dir = output_dir / f"unary_blob_probe_{node_id}_{row_order}"
            baseline_formula = modal.normalize_formula_text(entry.get("formula_text") or entry.get("wffstr") or "")
            payload = unary_blob_probe.run_blob_probe(
                image_path=crop_path,
                current_ocr_formula=baseline_formula,
                output_dir=line_output_dir,
            )

            formula_window_source = Path(str(payload.get("formula_window_path") or "")).resolve()
            formula_window_name = artifacts.UNARY_BLOB_LINE_WINDOW_TEMPLATE.format(node_id=node_id, row_order=row_order)
            formula_window_target = output_dir / formula_window_name
            if formula_window_source.exists() and formula_window_source.is_file():
                formula_window_target.write_bytes(formula_window_source.read_bytes())

            records.append(
                {
                    "node_id": str(node_id),
                    "row_order": row_order,
                    "line_label": str(entry.get("line_label") or "").strip(),
                    "scope_role": str(entry.get("scope_role") or "derived_line").strip(),
                    "baseline_formula": baseline_formula,
                    "ocr_symbol_count": int(payload.get("ocr_symbol_count") or 0),
                    "max_component_count": int(payload.get("max_component_count") or 0),
                    "any_blob_flagged": bool(payload.get("any_blob_flagged")),
                    "window": payload.get("window") or {},
                    "formula_window_filename": formula_window_name if formula_window_target.exists() else "",
                    "line_crop_filename": crop_path.name,
                    "variants": payload.get("variants") or [],
                }
            )

    (output_dir / artifacts.UNARY_BLOB_PROBE_FILENAME).write_text(
        json.dumps(records, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return records
