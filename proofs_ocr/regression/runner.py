"""Manifest loading and explicit gold-regression runner for proof OCR fixtures."""

from __future__ import annotations

import argparse
import difflib
import inspect
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence

import yaml

import proofs_ocr.regression.failed_line_reread as failed_line_reread
from proofs_ocr.domains import modal

_COMPARE_WS_RE = re.compile(r"\s+")
_COMPARE_DASH_BETWEEN_DIGITS_RE = re.compile(r"(?<=\d)\s*[-–—]\s*(?=\d)")
_COMPARE_COMMA_BETWEEN_DIGITS_RE = re.compile(r"(?<=\d)\s*,\s*(?=\d)")


@dataclass(frozen=True)
class GoldRegressionResult:
    case_id: str
    oracle_kind: str
    output_dir: Path
    image_path: Path
    expected_path: Path
    passed: bool
    run_summary: str
    granular_metrics: Dict[str, Any] | None = None
    diff_text: str = ""
    error: str = ""


@dataclass(frozen=True)
class FlattenedProofLine:
    order_index: int
    nesting_path: tuple[int, ...]
    wffstr: str
    jstr: str
    source_node_id: str = ""
    source_row_order: int = 0
    source_line_label: str = ""

    def to_json(self) -> Dict[str, Any]:
        return {
            "order_index": int(self.order_index),
            "nesting_path": list(self.nesting_path),
            "nesting_path_text": render_nesting_path(self.nesting_path),
            "wffstr": self.wffstr,
            "jstr": self.jstr,
            "source_node_id": self.source_node_id,
            "source_row_order": int(self.source_row_order),
            "source_line_label": self.source_line_label,
        }


def get_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _manifest_path() -> Path:
    return Path(__file__).resolve().with_name("manifest.yaml")


def load_manifest(path: Path | None = None) -> Dict[str, Any]:
    manifest_path = path or _manifest_path()
    parsed = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    if not isinstance(parsed, dict):
        raise ValueError("proof OCR regression manifest must parse to a mapping")
    return parsed


def manifest_cases(manifest: Dict[str, Any]) -> List[Dict[str, Any]]:
    cases = manifest.get("cases") or []
    if not isinstance(cases, list):
        raise ValueError("proof OCR regression manifest cases must be a list")
    return [case for case in cases if isinstance(case, dict)]


def _check_case_paths(case: Dict[str, Any], repo_root: Path) -> Dict[str, bool]:
    image_path = repo_root / str(case.get("image_path") or "")
    expected_path = repo_root / str(case.get("expected_structure_path") or "")
    return {
        "image_exists": image_path.exists(),
        "expected_exists": expected_path.exists(),
    }


def render_manifest_summary(manifest: Dict[str, Any], *, repo_root: Path | None = None) -> str:
    root = repo_root or get_repo_root()
    suite = manifest.get("suite") or {}
    cases = manifest_cases(manifest)

    lines: List[str] = []
    lines.append(f"Suite: {suite.get('name', 'unknown')}")
    lines.append(f"Cases: {len(cases)}")

    for case in cases:
        checks = _check_case_paths(case, root)
        status = []
        status.append("image:ok" if checks["image_exists"] else "image:missing")
        status.append("expected:ok" if checks["expected_exists"] else "expected:missing")
        case_id = str(case.get("id") or "unknown")
        domain = str(case.get("domain") or "?")
        kind = str(case.get("kind") or "?")
        oracle_kind = str(case.get("oracle_kind") or "unspecified")
        lines.append(
            f"- {case_id} [{domain}/{kind}] ({oracle_kind}) "
            + ", ".join(status)
        )

    return "\n".join(lines)


def select_cases(
    manifest: Dict[str, Any],
    *,
    oracle_prefixes: Sequence[str] = ("gold_",),
    case_ids: Sequence[str] | None = None,
    kinds: Sequence[str] | None = None,
    tags: Sequence[str] | None = None,
) -> List[Dict[str, Any]]:
    allowed_ids = {str(item).strip() for item in (case_ids or []) if str(item).strip()}
    allowed_kinds = {str(item).strip() for item in (kinds or []) if str(item).strip()}
    required_tags = {str(item).strip() for item in (tags or []) if str(item).strip()}

    selected: List[Dict[str, Any]] = []
    for case in manifest_cases(manifest):
        oracle_kind = str(case.get("oracle_kind") or "").strip()
        if oracle_prefixes and not any(oracle_kind.startswith(prefix) for prefix in oracle_prefixes):
            continue
        case_id = str(case.get("id") or "").strip()
        if allowed_ids and case_id not in allowed_ids:
            continue
        case_kind = str(case.get("kind") or "").strip()
        if allowed_kinds and case_kind not in allowed_kinds:
            continue
        case_tags = {str(item).strip() for item in (case.get("tags") or []) if str(item).strip()}
        if required_tags and not required_tags.issubset(case_tags):
            continue
        selected.append(case)
    return selected


def case_image_path(case: Mapping[str, Any], *, repo_root: Path) -> Path:
    return (repo_root / str(case.get("image_path") or "")).resolve()


def case_expected_path(case: Mapping[str, Any], *, repo_root: Path) -> Path:
    return (repo_root / str(case.get("expected_structure_path") or "")).resolve()


def case_semantic_notes_path(case: Mapping[str, Any], *, repo_root: Path) -> Path | None:
    path_text = str(case.get("semantic_notes_path") or "").strip()
    if not path_text:
        return None
    return (repo_root / path_text).resolve()


def load_expected_structure(case: Mapping[str, Any], *, repo_root: Path) -> Any:
    return json.loads(case_expected_path(case, repo_root=repo_root).read_text(encoding="utf-8"))


def load_semantic_notes(case: Mapping[str, Any], *, repo_root: Path) -> Dict[str, Any]:
    path = case_semantic_notes_path(case, repo_root=repo_root)
    if path is None or not path.exists():
        return {}
    parsed = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if isinstance(parsed, dict):
        return parsed
    return {}


def default_modal_config_for_case(case: Mapping[str, Any], *, repo_root: Path) -> Dict[str, Any]:
    config_path_text = str(case.get("config_path") or "").strip()
    if config_path_text:
        parsed = yaml.safe_load((repo_root / config_path_text).read_text(encoding="utf-8")) or {}
        if isinstance(parsed, dict):
            return {
                "system": str(parsed.get("system") or "K").strip() or "K",
                "include_derived_rules": bool(parsed.get("include_derived_rules", False)),
                "conclusion": str(parsed.get("conclusion") or "").strip(),
            }

    tags = {str(item).strip().lower() for item in (case.get("tags") or []) if str(item).strip()}
    system = "K"
    if "s4" in tags:
        system = "S4"
    elif "s5" in tags:
        system = "S5"
    elif "t" in tags:
        system = "T"

    return {
        "system": system,
        "include_derived_rules": False,
        "conclusion": "",
    }


def canonicalize_proof_structure(value: Any) -> Any:
    if isinstance(value, list):
        return [canonicalize_proof_structure(item) for item in value]
    if isinstance(value, dict):
        normalized: Dict[str, Any] = {}
        if "wffstr" in value:
            normalized["wffstr"] = _normalize_formula_for_compare(value.get("wffstr"))
        if "jstr" in value:
            normalized["jstr"] = _normalize_justification_for_compare(value.get("jstr"))
        return normalized
    return value


def _collapse_compare_whitespace(value: Any) -> str:
    return _COMPARE_WS_RE.sub(" ", str(value or "")).strip()


def _normalize_formula_for_compare(value: Any) -> str:
    return modal.normalize_formula_text(value)


def _normalize_justification_for_compare(value: Any) -> str:
    normalized = modal.normalize_formula_text(_collapse_compare_whitespace(value))
    normalized = normalized.replace("–", "-").replace("—", "-")
    normalized = _COMPARE_DASH_BETWEEN_DIGITS_RE.sub("-", normalized)
    normalized = _COMPARE_COMMA_BETWEEN_DIGITS_RE.sub(",", normalized)
    return normalized


def render_nesting_path(nesting_path: Sequence[int]) -> str:
    if not nesting_path:
        return "ROOT"
    return "ROOT/" + "/".join(str(int(part)) for part in nesting_path)


def flatten_proof_structure(value: Any) -> List[FlattenedProofLine]:
    flattened: List[FlattenedProofLine] = []

    def _walk(node: Any, nesting_path: tuple[int, ...]) -> None:
        if isinstance(node, list):
            for item_index, item in enumerate(node, start=1):
                if isinstance(item, list):
                    _walk(item, nesting_path + (item_index,))
                    continue
                if not isinstance(item, dict):
                    continue
                flattened.append(
                    FlattenedProofLine(
                        order_index=len(flattened) + 1,
                        nesting_path=nesting_path,
                        wffstr=str(item.get("wffstr") or "").strip(),
                        jstr=str(item.get("jstr") or "").strip(),
                        source_node_id=str(item.get("source_node_id") or "").strip(),
                        source_row_order=int(item.get("source_row_order") or 0),
                        source_line_label=str(item.get("source_line_label") or "").strip(),
                    )
                )

    _walk(value, ())
    return flattened


def compare_proof_structures_granular(actual: Any, expected: Any) -> Dict[str, Any]:
    actual_lines = flatten_proof_structure(actual)
    expected_lines = flatten_proof_structure(expected)
    matched_line_count = min(len(actual_lines), len(expected_lines))
    line_records: List[Dict[str, Any]] = []
    formula_correct_line_count = 0
    justification_correct_line_count = 0
    both_correct_line_count = 0
    structure_mismatch_count = 0

    for line_index in range(matched_line_count):
        expected_line = expected_lines[line_index]
        actual_line = actual_lines[line_index]
        formula_matches = _normalize_formula_for_compare(actual_line.wffstr) == _normalize_formula_for_compare(expected_line.wffstr)
        justification_matches = _normalize_justification_for_compare(actual_line.jstr) == _normalize_justification_for_compare(expected_line.jstr)
        structure_matches = actual_line.nesting_path == expected_line.nesting_path
        if formula_matches:
            formula_correct_line_count += 1
        if justification_matches:
            justification_correct_line_count += 1
        if formula_matches and justification_matches:
            both_correct_line_count += 1
        if not structure_matches:
            structure_mismatch_count += 1
        line_records.append(
            {
                "order_index": line_index + 1,
                "expected": expected_line.to_json(),
                "actual": actual_line.to_json(),
                "formula_matches": formula_matches,
                "justification_matches": justification_matches,
                "structure_matches": structure_matches,
            }
        )

    for line in expected_lines[matched_line_count:]:
        line_records.append(
            {
                "order_index": len(line_records) + 1,
                "expected": line.to_json(),
                "actual": None,
                "formula_matches": False,
                "justification_matches": False,
                "structure_matches": False,
                "status": "missing_actual_line",
            }
        )
    for line in actual_lines[matched_line_count:]:
        line_records.append(
            {
                "order_index": len(line_records) + 1,
                "expected": None,
                "actual": line.to_json(),
                "formula_matches": False,
                "justification_matches": False,
                "structure_matches": False,
                "status": "extra_actual_line",
            }
        )

    expected_line_count = len(expected_lines)
    actual_line_count = len(actual_lines)
    missing_line_count = max(0, expected_line_count - actual_line_count)
    extra_line_count = max(0, actual_line_count - expected_line_count)
    return {
        "expected_line_count": expected_line_count,
        "actual_line_count": actual_line_count,
        "matched_line_count": matched_line_count,
        "formula_correct_line_count": formula_correct_line_count,
        "justification_correct_line_count": justification_correct_line_count,
        "both_correct_line_count": both_correct_line_count,
        "missing_line_count": missing_line_count,
        "extra_line_count": extra_line_count,
        "structure_mismatch_count": structure_mismatch_count,
        "expected_lines": [line.to_json() for line in expected_lines],
        "actual_lines": [line.to_json() for line in actual_lines],
        "line_records": line_records,
    }


def compare_proof_structures(actual: Any, expected: Any) -> tuple[bool, str]:
    canonical_actual = canonicalize_proof_structure(actual)
    canonical_expected = canonicalize_proof_structure(expected)
    if canonical_actual == canonical_expected:
        return True, ""

    actual_text = json.dumps(canonical_actual, indent=2, ensure_ascii=False, sort_keys=True).splitlines()
    expected_text = json.dumps(canonical_expected, indent=2, ensure_ascii=False, sort_keys=True).splitlines()
    diff = "\n".join(
        difflib.unified_diff(
            expected_text,
            actual_text,
            fromfile="expected",
            tofile="actual",
            lineterm="",
        )
    )
    return False, diff


def _load_actual_final_structure(output_dir: Path) -> Any:
    final_path = output_dir / "final_structure_baseline.json"
    return json.loads(final_path.read_text(encoding="utf-8"))


def _write_regression_report(
    *,
    output_dir: Path,
    case: Mapping[str, Any],
    expected_path: Path,
    image_path: Path,
    passed: bool,
    run_summary: str,
    diff_text: str,
    error: str,
    granular_metrics: Mapping[str, Any] | None = None,
) -> None:
    payload = {
        "case_id": str(case.get("id") or ""),
        "oracle_kind": str(case.get("oracle_kind") or ""),
        "image_path": str(image_path),
        "expected_structure_path": str(expected_path),
        "passed": bool(passed),
        "run_summary": run_summary,
        "error": error,
    }
    (output_dir / "gold_regression_result.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    metrics_payload = {
        "case_id": str(case.get("id") or ""),
        "oracle_kind": str(case.get("oracle_kind") or ""),
        "error": error,
    }
    if granular_metrics is not None:
        metrics_payload["metrics"] = dict(granular_metrics)
    (output_dir / "gold_regression_metrics.json").write_text(
        json.dumps(metrics_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if diff_text:
        (output_dir / "gold_regression_diff.txt").write_text(diff_text, encoding="utf-8")


def _call_transcribe_fn(
    transcribe_fn: Callable[..., Mapping[str, Any]],
    *,
    kwargs: Mapping[str, Any],
) -> Mapping[str, Any]:
    signature = inspect.signature(transcribe_fn)
    supported = {
        key: value
        for key, value in kwargs.items()
        if key in signature.parameters
    }
    return transcribe_fn(**supported)


def run_gold_regressions(
    manifest: Dict[str, Any],
    *,
    repo_root: Path | None = None,
    output_root: Path | None = None,
    case_ids: Sequence[str] | None = None,
    kinds: Sequence[str] | None = None,
    tags: Sequence[str] | None = None,
    vision_base_url: str | None = None,
    vision_model: str | None = None,
    verifier_vision_model: str | None = None,
    enable_false_left_separator_filter: bool = False,
    enable_sparse_lower_compact_band_recovery: bool = False,
    enable_contradiction_reread: bool = False,
    enable_justification_head_reread: bool = False,
    line_reread_base_url: str | None = None,
    line_reread_model: str | None = None,
    line_reread_modes: Sequence[str] | None = None,
    line_reread_client: Any | None = None,
    transcribe_fn: Callable[..., Mapping[str, Any]] | None = None,
) -> List[GoldRegressionResult]:
    root = repo_root or get_repo_root()
    selected = select_cases(
        manifest,
        case_ids=case_ids,
        kinds=kinds,
        tags=tags,
    )
    destination_root = (output_root or (root / "working" / "modal_v2_gold_regressions")).resolve()
    destination_root.mkdir(parents=True, exist_ok=True)

    if transcribe_fn is None:
        from proofs_ocr.pipelines.modal_v2 import transcribe_image_to_modal_v2_debug_state

        transcribe_fn = transcribe_image_to_modal_v2_debug_state

    probe_client = line_reread_client
    if probe_client is None and str(line_reread_model or "").strip():
        from proofs_ocr.pipelines.modal_v2_runtime import build_vlm_client

        probe_client = build_vlm_client(
            base_url=line_reread_base_url or vision_base_url,
            model=line_reread_model,
        )

    results: List[GoldRegressionResult] = []
    for case in selected:
        case_id = str(case.get("id") or "").strip()
        image_path = case_image_path(case, repo_root=root)
        expected_path = case_expected_path(case, repo_root=root)
        output_dir = destination_root / case_id
        output_dir.mkdir(parents=True, exist_ok=True)
        modal_config = default_modal_config_for_case(case, repo_root=root)
        try:
            state = dict(
                _call_transcribe_fn(
                    transcribe_fn,
                    kwargs={
                        "image_path": image_path,
                        "output_dir": output_dir,
                        "modal_config": modal_config,
                        "vision_base_url": vision_base_url,
                        "vision_model": vision_model,
                        "verifier_vision_model": verifier_vision_model,
                        "enable_false_left_separator_filter": enable_false_left_separator_filter,
                        "enable_sparse_lower_compact_band_recovery": enable_sparse_lower_compact_band_recovery,
                        "enable_contradiction_reread": enable_contradiction_reread,
                        "enable_justification_head_reread": enable_justification_head_reread,
                    },
                )
            )
            actual = _load_actual_final_structure(output_dir)
            actual_with_source = failed_line_reread.load_source_aware_structure(output_dir)
            if not actual_with_source:
                actual_with_source = actual
            expected = load_expected_structure(case, repo_root=root)
            passed, diff_text = compare_proof_structures(actual, expected)
            granular_metrics = compare_proof_structures_granular(actual_with_source, expected)
            run_summary = str(state.get("run_summary") or "").strip()
            if probe_client is not None:
                line_records = granular_metrics.get("line_records") or []
                probe_records = failed_line_reread.run_failed_line_rereads(
                    output_dir=output_dir,
                    line_records=line_records if isinstance(line_records, list) else [],
                    client=probe_client,
                    modal_config=modal_config,
                    modes=line_reread_modes or ("whole_line", "boxed_context"),
                    model_name=str(line_reread_model or getattr(probe_client, "model", "") or "").strip(),
                )
                if probe_records:
                    run_summary = (
                        f"{run_summary} Alternate-model reread wrote {len(probe_records)} failed-line probe"
                        f"{'' if len(probe_records) == 1 else 's'}."
                    ).strip()
            _write_regression_report(
                output_dir=output_dir,
                case=case,
                expected_path=expected_path,
                image_path=image_path,
                passed=passed,
                run_summary=run_summary,
                diff_text=diff_text,
                error="",
                granular_metrics=granular_metrics,
            )
            results.append(
                GoldRegressionResult(
                    case_id=case_id,
                    oracle_kind=str(case.get("oracle_kind") or ""),
                    output_dir=output_dir,
                    image_path=image_path,
                    expected_path=expected_path,
                    passed=passed,
                    run_summary=run_summary,
                    granular_metrics=granular_metrics,
                    diff_text=diff_text,
                )
            )
        except Exception as exc:
            error = str(exc)
            _write_regression_report(
                output_dir=output_dir,
                case=case,
                expected_path=expected_path,
                image_path=image_path,
                passed=False,
                run_summary="",
                diff_text="",
                error=error,
                granular_metrics=None,
            )
            results.append(
                GoldRegressionResult(
                    case_id=case_id,
                    oracle_kind=str(case.get("oracle_kind") or ""),
                    output_dir=output_dir,
                    image_path=image_path,
                    expected_path=expected_path,
                    passed=False,
                    run_summary="",
                    granular_metrics=None,
                    error=error,
                )
            )
    return results


def render_gold_regression_summary(results: Iterable[GoldRegressionResult]) -> str:
    materialized = list(results)
    total = len(materialized)
    passed_count = sum(1 for result in materialized if result.passed)
    failed_count = total - passed_count
    totals = {
        "matched": 0,
        "formula": 0,
        "justification": 0,
        "both": 0,
        "expected": 0,
        "actual": 0,
        "missing": 0,
        "extra": 0,
        "structure": 0,
    }
    for result in materialized:
        metrics = result.granular_metrics or {}
        totals["matched"] += int(metrics.get("matched_line_count", 0))
        totals["formula"] += int(metrics.get("formula_correct_line_count", 0))
        totals["justification"] += int(metrics.get("justification_correct_line_count", 0))
        totals["both"] += int(metrics.get("both_correct_line_count", 0))
        totals["expected"] += int(metrics.get("expected_line_count", 0))
        totals["actual"] += int(metrics.get("actual_line_count", 0))
        totals["missing"] += int(metrics.get("missing_line_count", 0))
        totals["extra"] += int(metrics.get("extra_line_count", 0))
        totals["structure"] += int(metrics.get("structure_mismatch_count", 0))

    lines = [
        f"Gold cases: {total}",
        f"Passed: {passed_count}",
        f"Failed: {failed_count}",
        (
            "Aggregate lines: "
            f"formula {totals['formula']}/{totals['matched']}, "
            f"justification {totals['justification']}/{totals['matched']}, "
            f"both {totals['both']}/{totals['matched']}, "
            f"expected={totals['expected']}, actual={totals['actual']}, "
            f"missing={totals['missing']}, extra={totals['extra']}, "
            f"structural mismatches={totals['structure']}"
        ),
    ]
    for result in materialized:
        status = "PASS" if result.passed else "FAIL"
        line = f"- {status} {result.case_id} ({result.oracle_kind})"
        if result.error:
            line += f" error={result.error}"
        elif result.run_summary:
            line += f" {result.run_summary}"
        metrics = result.granular_metrics or {}
        if metrics:
            line += (
                " | "
                f"formula {int(metrics.get('formula_correct_line_count', 0))}/{int(metrics.get('matched_line_count', 0))}, "
                f"justification {int(metrics.get('justification_correct_line_count', 0))}/{int(metrics.get('matched_line_count', 0))}, "
                f"both {int(metrics.get('both_correct_line_count', 0))}/{int(metrics.get('matched_line_count', 0))}, "
                f"structure mismatches={int(metrics.get('structure_mismatch_count', 0))}, "
                f"missing={int(metrics.get('missing_line_count', 0))}, extra={int(metrics.get('extra_line_count', 0))}"
            )
        lines.append(line)
        if result.diff_text:
            diff_preview = "\n".join(result.diff_text.splitlines()[:20])
            if diff_preview.strip():
                lines.append(diff_preview)
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Proof OCR regression helpers.")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("summary", help="Print the manifest summary.")

    run_parser = subparsers.add_parser("run-gold", help="Run the slow modal_v2 gold regression suite.")
    run_parser.add_argument("--output-root", default="", help="Directory for per-case OCR outputs and diffs.")
    run_parser.add_argument("--case", action="append", default=[], help="Run only the named case id. May be repeated.")
    run_parser.add_argument("--kind", action="append", default=[], help="Filter selected gold cases by kind.")
    run_parser.add_argument("--tag", action="append", default=[], help="Require a tag on selected gold cases.")
    run_parser.add_argument("--vision-base-url", default="", help="LM Studio base URL override.")
    run_parser.add_argument("--vision-model", default="", help="LM Studio model override.")
    run_parser.add_argument("--verifier-model", default="", help="LM Studio verifier model override.")
    run_parser.add_argument(
        "--line-reread-base-url",
        default="",
        help="Base URL override for alternate failed-line rereads. Defaults to --vision-base-url.",
    )
    run_parser.add_argument(
        "--line-reread-model",
        default="",
        help="Alternate local vision model used to reread only mismatched lines after the baseline run.",
    )
    run_parser.add_argument(
        "--line-reread-mode",
        action="append",
        default=[],
        help="Failed-line reread crop mode: whole_line, boxed_context, or boxed_justification. May be repeated.",
    )
    run_parser.add_argument(
        "--enable-false-left-separator-filter",
        action="store_true",
        help="Enable the pre-tree handwritten false-left-separator verifier experiment.",
    )
    run_parser.add_argument(
        "--enable-sparse-lower-compact-band-recovery",
        action="store_true",
        help="Enable a narrow handwritten row-band rescue for tiny lower-half formula symbols.",
    )
    run_parser.add_argument(
        "--enable-contradiction-reread",
        action="store_true",
        help="Enable a narrow verifier reread for suspicious ¬E/~E contradiction lines.",
    )
    run_parser.add_argument(
        "--enable-justification-head-reread",
        action="store_true",
        help="Enable a boxed justification-head verifier reread for suspicious lines.",
    )

    return parser.parse_args()


def _safe_console_print(text: str) -> None:
    try:
        print(text)
    except UnicodeEncodeError:
        encoded = text.encode(getattr(sys.stdout, "encoding", None) or "utf-8", errors="replace")
        sys.stdout.buffer.write(encoded + b"\n")


def main() -> int:
    args = _parse_args()
    manifest = load_manifest()

    if args.command in {None, "summary"}:
        _safe_console_print(render_manifest_summary(manifest))
        return 0

    if args.command == "run-gold":
        results = run_gold_regressions(
            manifest,
            output_root=Path(args.output_root) if str(args.output_root or "").strip() else None,
            case_ids=args.case,
            kinds=args.kind,
            tags=args.tag,
            vision_base_url=args.vision_base_url or None,
            vision_model=args.vision_model or None,
            verifier_vision_model=args.verifier_model or None,
            line_reread_base_url=args.line_reread_base_url or None,
            line_reread_model=args.line_reread_model or None,
            line_reread_modes=args.line_reread_mode,
            enable_false_left_separator_filter=bool(args.enable_false_left_separator_filter),
            enable_sparse_lower_compact_band_recovery=bool(args.enable_sparse_lower_compact_band_recovery),
            enable_contradiction_reread=bool(args.enable_contradiction_reread),
            enable_justification_head_reread=bool(args.enable_justification_head_reread),
        )
        _safe_console_print(render_gold_regression_summary(results))
        return 0 if all(result.passed for result in results) else 1

    raise ValueError(f"Unknown regression command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
