"""TFL Fitch OCR gold-regression runner.

Mirrors the modal ``runner.run_gold_regressions`` pattern but uses TFL-specific
canonicalization and the Fitch chop-and-mask OCR pipeline.
"""

from __future__ import annotations

import difflib
import inspect
import json
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence

from proofs_ocr.domains import tfl as tfl_domain
from proofs_ocr.regression.runner import (
    GoldRegressionResult,
    case_expected_path,
    case_image_path,
    get_repo_root,
    load_expected_structure,
    load_manifest,
    select_cases,
)

# ── TFL-specific canonicalization ────────────────────────────────────


def _normalize_formula_for_compare(value: Any) -> str:
    return tfl_domain.canonicalize_formula_for_comparison(value)


def _normalize_justification_for_compare(value: Any) -> str:
    return tfl_domain.canonicalize_justification_for_comparison(value)


def _collapse_ws(value: Any) -> str:
    import re

    return re.compile(r"\s+").sub(" ", str(value or "")).strip()


# ── Canonical comparison ─────────────────────────────────────────────


def canonicalize_proof_structure(value: Any) -> Any:
    """Canonicalize a TFL proof structure for comparison."""
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


def compare_proof_structures(actual: Any, expected: Any) -> tuple[bool, str]:
    """Compare two TFL proof structures after canonicalization."""
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


# ── Granular line-by-line comparison ─────────────────────────────────


def _flatten_tfl_structure(value: Any) -> List[Dict[str, Any]]:
    """Flatten a nested TFL proof structure into a list of line dicts with nesting paths."""
    flattened: List[Dict[str, Any]] = []

    def _walk(node: Any, nesting_path: tuple[int, ...]) -> None:
        if isinstance(node, list):
            for item_index, item in enumerate(node, start=1):
                if isinstance(item, list):
                    _walk(item, nesting_path + (item_index,))
                    continue
                if not isinstance(item, dict):
                    continue
                flattened.append(
                    {
                        "order_index": len(flattened) + 1,
                        "nesting_path": nesting_path,
                        "wffstr": str(item.get("wffstr") or "").strip(),
                        "jstr": str(item.get("jstr") or "").strip(),
                    }
                )

    _walk(value, ())
    return flattened


def compare_proof_structures_granular(actual: Any, expected: Any) -> Dict[str, Any]:
    """Line-by-line granular comparison using TFL canonicalization."""
    actual_lines = _flatten_tfl_structure(actual)
    expected_lines = _flatten_tfl_structure(expected)
    matched_line_count = min(len(actual_lines), len(expected_lines))
    line_records: List[Dict[str, Any]] = []
    formula_correct_line_count = 0
    justification_correct_line_count = 0
    both_correct_line_count = 0
    structure_mismatch_count = 0

    for line_index in range(matched_line_count):
        expected_line = expected_lines[line_index]
        actual_line = actual_lines[line_index]
        formula_matches = _normalize_formula_for_compare(actual_line["wffstr"]) == _normalize_formula_for_compare(expected_line["wffstr"])
        justification_matches = _normalize_justification_for_compare(actual_line["jstr"]) == _normalize_justification_for_compare(expected_line["jstr"])
        structure_matches = actual_line["nesting_path"] == expected_line["nesting_path"]
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
                "expected": expected_line,
                "actual": actual_line,
                "formula_matches": formula_matches,
                "justification_matches": justification_matches,
                "structure_matches": structure_matches,
            }
        )

    for line in expected_lines[matched_line_count:]:
        line_records.append(
            {
                "order_index": len(line_records) + 1,
                "expected": line,
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
                "actual": line,
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
        "line_records": line_records,
    }


# ── Report writing ───────────────────────────────────────────────────


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
    metrics_payload: Dict[str, Any] = {
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


# ── Transcribe helper ────────────────────────────────────────────────


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


# ── Main regression entry point ──────────────────────────────────────


def _load_actual_final_structure(output_dir: Path) -> Any:
    final_path = output_dir / "final_structure.json"
    return json.loads(final_path.read_text(encoding="utf-8"))


def select_tfl_cases(
    manifest: Dict[str, Any],
    *,
    case_ids: Sequence[str] | None = None,
) -> List[Dict[str, Any]]:
    """Select TFL handwritten cases from the manifest."""
    return [
        case
        for case in select_cases(
            manifest,
            oracle_prefixes=("gold_",),
            kinds=("handwritten",),
            case_ids=case_ids,
        )
        if str(case.get("domain") or "").strip().lower() == "tfl"
    ]


def run_tfl_gold_regressions(
    manifest: Dict[str, Any],
    *,
    repo_root: Path | None = None,
    output_root: Path | None = None,
    case_ids: Sequence[str] | None = None,
    transcribe_fn: Callable[..., Mapping[str, Any]] | None = None,
) -> List[GoldRegressionResult]:
    """Run TFL gold regressions on all handwritten TFL fixtures.

    Parameters
    ----------
    manifest:
        Parsed manifest dict (from ``load_manifest()``).
    repo_root:
        Repository root path.  Auto-detected when ``None``.
    output_root:
        Where per-case output dirs are created.  Defaults to
        ``working/tfl_gold_regressions`` under the repo root.
    case_ids:
        Optional case-id filter.
    transcribe_fn:
        Callable that performs OCR.  Must accept keyword arguments
        ``image_path`` (Path) and ``output_dir`` (Path) at minimum.
        It must write ``final_structure.json`` into ``output_dir``.
        When ``None``, defaults to
        :func:`scripts.debug_fitch_ocr.run_fitch_ocr_pipeline`.
    """
    root = repo_root or get_repo_root()
    selected = select_tfl_cases(manifest, case_ids=case_ids)
    destination_root = (output_root or (root / "working" / "tfl_gold_regressions")).resolve()
    destination_root.mkdir(parents=True, exist_ok=True)

    if transcribe_fn is None:
        from proofs_ocr.pipelines.tfl_fitch import run_fitch_ocr_pipeline

        transcribe_fn = run_fitch_ocr_pipeline  # type: ignore[assignment]

    results: List[GoldRegressionResult] = []
    for case in selected:
        case_id = str(case.get("id") or "").strip()
        image_path = case_image_path(case, repo_root=root)
        expected_path = case_expected_path(case, repo_root=root)
        output_dir = destination_root / case_id
        output_dir.mkdir(parents=True, exist_ok=True)
        try:
            state = dict(
                _call_transcribe_fn(
                    transcribe_fn,
                    kwargs={
                        "image_path": image_path,
                        "output_dir": output_dir,
                    },
                )
            )
            actual = _load_actual_final_structure(output_dir)
            expected = load_expected_structure(case, repo_root=root)
            passed, diff_text = compare_proof_structures(actual, expected)
            granular_metrics = compare_proof_structures_granular(actual, expected)
            run_summary = str(state.get("run_summary") or "").strip()
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


# ── Summary rendering ────────────────────────────────────────────────


def render_tfl_gold_regression_summary(results: Iterable[GoldRegressionResult]) -> str:
    """Render a human-readable summary of TFL gold regression results."""
    materialized = list(results)

    def _totals(res_list: Sequence[GoldRegressionResult]) -> Dict[str, int]:
        total = len(res_list)
        passed = sum(1 for r in res_list if r.passed)
        t = {
            "total": total,
            "passed": passed,
            "failed": total - passed,
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
        for r in res_list:
            m = r.granular_metrics or {}
            t["matched"] += int(m.get("matched_line_count", 0))
            t["formula"] += int(m.get("formula_correct_line_count", 0))
            t["justification"] += int(m.get("justification_correct_line_count", 0))
            t["both"] += int(m.get("both_correct_line_count", 0))
            t["expected"] += int(m.get("expected_line_count", 0))
            t["actual"] += int(m.get("actual_line_count", 0))
            t["missing"] += int(m.get("missing_line_count", 0))
            t["extra"] += int(m.get("extra_line_count", 0))
            t["structure"] += int(m.get("structure_mismatch_count", 0))
        return t

    totals = _totals(materialized)
    lines = [
        f"TFL Gold cases: {totals['total']}",
        f"Passed: {totals['passed']}",
        f"Failed: {totals['failed']}",
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


# ── CLI ──────────────────────────────────────────────────────────────


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="TFL Fitch OCR gold regression runner.")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("summary", help="Print the manifest summary for TFL cases.")

    run_parser = subparsers.add_parser("run-gold", help="Run the TFL gold regression suite.")
    run_parser.add_argument("--output-root", default="", help="Directory for per-case OCR outputs and diffs.")
    run_parser.add_argument("--case", action="append", default=[], help="Run only the named case id. May be repeated.")

    args = parser.parse_args()
    manifest = load_manifest()

    if args.command in {None, "summary"}:
        selected = select_tfl_cases(manifest)
        print(f"TFL handwritten cases in manifest: {len(selected)}")
        for case in selected:
            print(f"  - {case.get('id')}")
        return 0

    if args.command == "run-gold":
        results = run_tfl_gold_regressions(
            manifest,
            output_root=Path(args.output_root) if str(args.output_root or "").strip() else None,
            case_ids=args.case,
        )
        print(render_tfl_gold_regression_summary(results))
        return 0 if all(r.passed for r in results) else 1

    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
