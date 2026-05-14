"""Candidate classification helpers for verifier-guided TFL row rereads."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping as MappingABC
from pathlib import Path
from typing import Any, Mapping

from proofs_ocr.domains import tfl as tfl_domain
from proofs_ocr.tricks.justification_snaps import (
    TFL_VALID_RULE_HEADS,
    clean_justification_for_tfl,
    fuzzy_snap_justification,
    snap_justification_rule_head,
)


SUMMARY_FILENAME = "verifier_line_reread_summary.json"


def normalize_candidate_text(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"\s+", "", text)
    text = text.replace("->", "→").replace("&", "∧")
    return text


def normalize_candidate_justification(value: Any) -> str:
    """Normalize a reread justification with the same TFL cleanup/snap stack."""
    cleaned = clean_justification_for_tfl(str(value or "").strip())
    snapped = snap_justification_rule_head(
        cleaned,
        valid_rules=TFL_VALID_RULE_HEADS,
    )
    return fuzzy_snap_justification(
        snapped,
        valid_rules=TFL_VALID_RULE_HEADS,
    )


def _canonical_formula(value: Any) -> str:
    return tfl_domain.canonicalize_formula_for_comparison(value)


def _canonical_justification(value: Any) -> str:
    return tfl_domain.canonicalize_justification_for_comparison(
        normalize_candidate_justification(value)
    )


def parsed_line_number(parsed: Any) -> int | None:
    if not isinstance(parsed, MappingABC):
        return None
    raw = parsed.get("line")
    if isinstance(raw, int):
        return raw
    text = str(raw or "").strip()
    if text.isdigit():
        return int(text)
    return None


def candidate_materially_changed(
    suspicion: Mapping[str, Any],
    parsed: Any,
    *,
    line_matches_target: bool,
) -> bool:
    if not isinstance(parsed, MappingABC):
        return False
    if not line_matches_target:
        return False
    old_wff = _canonical_formula(suspicion.get("wffstr"))
    old_j = _canonical_justification(suspicion.get("jstr"))
    new_wff = _canonical_formula(parsed.get("wffstr"))
    new_j = _canonical_justification(parsed.get("jstr"))
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


def _coerce_int(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    text = str(value or "").strip()
    return int(text) if text.isdigit() else None


def _gold_line_lookup(gold_metrics: Mapping[str, Any] | None) -> dict[int, Mapping[str, Any]]:
    if not isinstance(gold_metrics, MappingABC):
        return {}
    metrics = gold_metrics.get("metrics")
    if isinstance(metrics, MappingABC):
        gold_metrics = metrics
    records = gold_metrics.get("line_records")
    if not isinstance(records, list):
        return {}
    lookup: dict[int, Mapping[str, Any]] = {}
    for record in records:
        if not isinstance(record, MappingABC):
            continue
        line_no = _coerce_int(record.get("order_index"))
        if line_no is not None:
            lookup[line_no] = record
    return lookup


def _line_values(record: Mapping[str, Any]) -> tuple[int | None, Mapping[str, Any], Mapping[str, Any]]:
    suspicion = record.get("suspicion")
    if not isinstance(suspicion, MappingABC):
        suspicion = {}
    parsed = record.get("parsed")
    if not isinstance(parsed, MappingABC):
        parsed = {}
    line_no = _coerce_int(record.get("line"))
    if line_no is None:
        line_no = _coerce_int(suspicion.get("line"))
    return line_no, suspicion, parsed


def _comparison_block(line_no: int | None, suspicion: Mapping[str, Any], parsed: Mapping[str, Any]) -> dict[str, Any]:
    original_j = normalize_candidate_justification(suspicion.get("jstr"))
    candidate_j = normalize_candidate_justification(parsed.get("jstr"))
    original_wff = str(suspicion.get("wffstr") or "").strip()
    candidate_wff = str(parsed.get("wffstr") or "").strip()
    formula_changed = _canonical_formula(original_wff) != _canonical_formula(candidate_wff)
    justification_changed = _canonical_justification(original_j) != _canonical_justification(candidate_j)
    changed_fields = []
    if formula_changed:
        changed_fields.append("wffstr")
    if justification_changed:
        changed_fields.append("jstr")
    return {
        "line": line_no,
        "original": {
            "wffstr": original_wff,
            "jstr": str(suspicion.get("jstr") or "").strip(),
            "normalized_jstr": original_j,
        },
        "candidate": {
            "wffstr": candidate_wff,
            "jstr": str(parsed.get("jstr") or "").strip(),
            "normalized_jstr": candidate_j,
        },
        "changed_fields": changed_fields,
    }


def _gold_comparison(
    *,
    line_no: int | None,
    comparison: Mapping[str, Any],
    gold_lookup: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    if line_no is None or line_no not in gold_lookup:
        return {"available": False}
    record = gold_lookup[line_no]
    expected = record.get("expected")
    if not isinstance(expected, MappingABC):
        return {"available": False}
    original = comparison.get("original") if isinstance(comparison.get("original"), MappingABC) else {}
    candidate = comparison.get("candidate") if isinstance(comparison.get("candidate"), MappingABC) else {}
    original_formula = _canonical_formula(original.get("wffstr"))
    original_j = _canonical_justification(original.get("normalized_jstr"))
    candidate_formula = _canonical_formula(candidate.get("wffstr"))
    candidate_j = _canonical_justification(candidate.get("normalized_jstr"))
    expected_formula = _canonical_formula(expected.get("wffstr"))
    expected_j = _canonical_justification(expected.get("jstr"))
    original_matches = {
        "formula": original_formula == expected_formula,
        "justification": original_j == expected_j,
    }
    candidate_matches = {
        "formula": candidate_formula == expected_formula,
        "justification": candidate_j == expected_j,
    }
    original_matches["both"] = bool(original_matches["formula"] and original_matches["justification"])
    candidate_matches["both"] = bool(candidate_matches["formula"] and candidate_matches["justification"])
    return {
        "available": True,
        "expected": {
            "wffstr": str(expected.get("wffstr") or "").strip(),
            "jstr": str(expected.get("jstr") or "").strip(),
        },
        "original_matches_gold": original_matches,
        "candidate_matches_gold": candidate_matches,
    }


def _classify_candidate(
    *,
    record: Mapping[str, Any],
    line_no: int | None,
    comparison: Mapping[str, Any],
    gold: Mapping[str, Any],
) -> tuple[str, str]:
    status = str(record.get("status") or "").strip() or "unknown"
    if status == "skipped":
        return "skipped", str(record.get("reason") or "candidate was not reread").strip()
    if status == "error":
        return "error", str(record.get("error") or "reread failed").strip()
    if status != "candidate":
        return status, "record is not a candidate reread"

    parsed_line = _coerce_int(record.get("parsed_line"))
    if record.get("line_matches_target") is False:
        return "line_mismatch", f"Reread returned line {parsed_line}; target was line {line_no}."

    changed_fields = list(comparison.get("changed_fields") or [])
    if not changed_fields:
        return "unchanged", "Normalized reread matches the original OCR."

    if bool(gold.get("available")):
        original_matches = gold.get("original_matches_gold") if isinstance(gold.get("original_matches_gold"), MappingABC) else {}
        candidate_matches = gold.get("candidate_matches_gold") if isinstance(gold.get("candidate_matches_gold"), MappingABC) else {}
        if candidate_matches.get("both") and not original_matches.get("both"):
            return "would_match_gold", "Candidate would match the gold line; original OCR did not."
        if candidate_matches.get("justification") and not original_matches.get("justification"):
            return "would_fix_justification_gold", "Candidate justification would match gold."
        if candidate_matches.get("formula") and not original_matches.get("formula"):
            return "would_fix_formula_gold", "Candidate formula would match gold."
        if original_matches.get("both") and not candidate_matches.get("both"):
            return "would_regress_gold", "Original OCR matches gold; candidate would not."
        return "changed_no_gold_gain", "Candidate changes OCR but does not improve the gold match."

    if changed_fields == ["jstr"]:
        return "changed_justification", "Candidate changes the normalized justification."
    if changed_fields == ["wffstr"]:
        return "changed_formula", "Candidate changes the normalized formula."
    return "changed_formula_and_justification", "Candidate changes normalized formula and justification."


def score_verifier_reread_candidates(
    records: list[Mapping[str, Any]],
    *,
    gold_metrics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a report-only scoring summary for verifier reread candidates."""
    gold_lookup = _gold_line_lookup(gold_metrics)
    scored: list[dict[str, Any]] = []
    class_counts: dict[str, int] = {}
    for record in records:
        line_no, suspicion, parsed = _line_values(record)
        comparison = _comparison_block(line_no, suspicion, parsed)
        gold = _gold_comparison(line_no=line_no, comparison=comparison, gold_lookup=gold_lookup)
        candidate_class, reason = _classify_candidate(
            record=record,
            line_no=line_no,
            comparison=comparison,
            gold=gold,
        )
        class_counts[candidate_class] = class_counts.get(candidate_class, 0) + 1
        scored.append(
            {
                "line": line_no,
                "target_kind": record.get("target_kind"),
                "source_rejected_line": record.get("source_rejected_line"),
                "status": record.get("status"),
                "candidate_class": candidate_class,
                "reason": reason,
                **comparison,
                "parsed_line": record.get("parsed_line"),
                "line_matches_target": record.get("line_matches_target"),
                "material_change": record.get("material_change"),
                "gold": gold,
            }
        )
    return {
        "report_kind": "verifier_line_reread_summary",
        "gold_metrics_available": bool(gold_lookup),
        "record_count": len(scored),
        "candidate_class_counts": class_counts,
        "records": scored,
    }


def write_verifier_reread_summary(
    *,
    output_dir: Path,
    records: list[Mapping[str, Any]],
    gold_metrics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write ``verifier_line_reread_summary.json`` for candidate inspection."""
    payload = score_verifier_reread_candidates(records, gold_metrics=gold_metrics)
    (output_dir / SUMMARY_FILENAME).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return payload
