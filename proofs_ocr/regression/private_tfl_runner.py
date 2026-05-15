"""Private TFL/Fitch OCR regression runner for external FERPA data.

This module is intentionally separate from ``manifest.yaml`` and the tracked
fixture runner.  It accepts an operator-owned manifest outside the repo, runs
the normal TFL/Fitch OCR pipeline on private proof crops, and writes a
shareable report that omits paths, raw line text, diffs, timestamps, and image
references.
"""

from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence

import yaml

from proofs_ocr.regression.runner import GoldRegressionResult, get_repo_root
from proofs_ocr.regression.tfl_runner import (
    compare_proof_structures,
    compare_proof_structures_granular,
)


def _safe_console_print(text: str) -> None:
    try:
        print(text)
    except UnicodeEncodeError:
        encoded = text.encode(
            getattr(sys.stdout, "encoding", None) or "utf-8",
            errors="replace",
        )
        sys.stdout.buffer.write(encoded + b"\n")


def _call_transcribe_fn(
    transcribe_fn: Callable[..., Mapping[str, Any]],
    *,
    kwargs: Mapping[str, Any],
) -> Mapping[str, Any]:
    signature = inspect.signature(transcribe_fn)
    supported = {key: value for key, value in kwargs.items() if key in signature.parameters}
    return transcribe_fn(**supported)


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _resolve_manifest_relative(path_text: Any, *, manifest_dir: Path) -> Path:
    text = str(path_text or "").strip()
    if not text:
        raise ValueError("private manifest case path cannot be empty")
    path = Path(text).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (manifest_dir / path).resolve()


def _slug(value: Any, *, fallback: str = "case") -> str:
    text = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(value or "").strip())
    while "__" in text:
        text = text.replace("__", "_")
    return text.strip("_") or fallback


def _proof_lines_payload(value: Any) -> Any:
    """Unwrap debug-page saved structures into the proof line list."""
    if isinstance(value, dict):
        if isinstance(value.get("lines"), list):
            return value["lines"]
        if isinstance(value.get("structure"), list):
            return value["structure"]
    return value


def load_private_manifest(path: Path) -> Dict[str, Any]:
    """Load a private OCR manifest from an operator-owned YAML file."""
    manifest_path = Path(path).expanduser().resolve()
    parsed = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    if not isinstance(parsed, dict):
        raise ValueError("private TFL OCR manifest must parse to a mapping")
    cases = parsed.get("cases")
    if not isinstance(cases, list):
        raise ValueError("private TFL OCR manifest must contain a cases list")
    parsed["_manifest_path"] = str(manifest_path)
    return parsed


def private_manifest_cases(manifest: Mapping[str, Any]) -> List[Dict[str, Any]]:
    cases = manifest.get("cases") or []
    if not isinstance(cases, list):
        raise ValueError("private TFL OCR manifest cases must be a list")
    return [case for case in cases if isinstance(case, dict)]


def select_private_tfl_cases(
    manifest: Mapping[str, Any],
    *,
    case_ids: Sequence[str] | None = None,
    splits: Sequence[str] | None = None,
) -> List[Dict[str, Any]]:
    allowed_ids = {str(item).strip() for item in (case_ids or []) if str(item).strip()}
    allowed_splits = {str(item).strip() for item in (splits or []) if str(item).strip()}

    selected: List[Dict[str, Any]] = []
    for case in private_manifest_cases(manifest):
        case_id = str(case.get("id") or "").strip()
        if not case_id:
            continue
        if allowed_ids and case_id not in allowed_ids:
            continue
        split = str(case.get("split") or "unassigned").strip() or "unassigned"
        if allowed_splits and split not in allowed_splits:
            continue
        selected.append(case)
    return selected


def private_case_image_path(case: Mapping[str, Any], *, manifest_dir: Path) -> Path:
    return _resolve_manifest_relative(case.get("image_path"), manifest_dir=manifest_dir)


def private_case_expected_path(case: Mapping[str, Any], *, manifest_dir: Path) -> Path:
    return _resolve_manifest_relative(case.get("expected_structure_path"), manifest_dir=manifest_dir)


def _manifest_dir(manifest: Mapping[str, Any]) -> Path:
    path_text = str(manifest.get("_manifest_path") or "").strip()
    if not path_text:
        raise ValueError("private manifest was not loaded by load_private_manifest")
    return Path(path_text).resolve().parent


def _validate_private_paths(
    *,
    manifest: Mapping[str, Any],
    output_root: Path,
    selected: Sequence[Mapping[str, Any]],
    allow_repo_paths: bool,
) -> None:
    if allow_repo_paths:
        return
    repo_root = get_repo_root().resolve()
    manifest_path = Path(str(manifest.get("_manifest_path") or "")).resolve()
    guarded_paths = [manifest_path, output_root.resolve()]
    manifest_dir = _manifest_dir(manifest)
    for case in selected:
        guarded_paths.append(private_case_image_path(case, manifest_dir=manifest_dir))
        guarded_paths.append(private_case_expected_path(case, manifest_dir=manifest_dir))
    for path in guarded_paths:
        if _is_relative_to(path, repo_root):
            raise ValueError(
                "Private TFL OCR paths must stay outside the repository. "
                f"Move or override this path: {path}"
            )


def load_private_expected_structure(case: Mapping[str, Any], *, manifest_dir: Path) -> Any:
    path = private_case_expected_path(case, manifest_dir=manifest_dir)
    return _proof_lines_payload(json.loads(path.read_text(encoding="utf-8")))


def _load_actual_final_structure(output_dir: Path) -> Any:
    path = output_dir / "final_structure.json"
    return _proof_lines_payload(json.loads(path.read_text(encoding="utf-8")))


def _redacted_metrics(metrics: Mapping[str, Any] | None) -> Dict[str, int]:
    if not metrics:
        return {}
    keys = (
        "expected_line_count",
        "actual_line_count",
        "matched_line_count",
        "formula_correct_line_count",
        "justification_correct_line_count",
        "both_correct_line_count",
        "missing_line_count",
        "extra_line_count",
        "structure_mismatch_count",
    )
    return {key: int(metrics.get(key, 0)) for key in keys}


def _failure_categories(result: GoldRegressionResult) -> List[str]:
    if result.passed:
        return []
    if result.error:
        return ["runner_error"]
    metrics = result.granular_metrics or {}
    if not metrics:
        return ["missing_metrics"]

    categories: List[str] = []
    matched = int(metrics.get("matched_line_count", 0))
    if int(metrics.get("missing_line_count", 0)) or int(metrics.get("extra_line_count", 0)):
        categories.append("line_count")
    if int(metrics.get("structure_mismatch_count", 0)):
        categories.append("structure")
    if int(metrics.get("formula_correct_line_count", 0)) < matched:
        categories.append("formula")
    if int(metrics.get("justification_correct_line_count", 0)) < matched:
        categories.append("justification")
    return categories or ["non_exact_match"]


def build_shareable_summary(
    results: Iterable[GoldRegressionResult],
    *,
    cases_by_id: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    materialized = list(results)
    totals: Dict[str, int] = {
        "cases": len(materialized),
        "passed": sum(1 for result in materialized if result.passed),
        "failed": sum(1 for result in materialized if not result.passed),
        "expected_lines": 0,
        "actual_lines": 0,
        "matched_lines": 0,
        "formula_correct": 0,
        "justification_correct": 0,
        "both_correct": 0,
        "missing_lines": 0,
        "extra_lines": 0,
        "structure_mismatches": 0,
    }
    by_split: Dict[str, Dict[str, int]] = {}
    failure_categories: Dict[str, int] = {}
    case_summaries: List[Dict[str, Any]] = []

    for result in materialized:
        metrics = result.granular_metrics or {}
        totals["expected_lines"] += int(metrics.get("expected_line_count", 0))
        totals["actual_lines"] += int(metrics.get("actual_line_count", 0))
        totals["matched_lines"] += int(metrics.get("matched_line_count", 0))
        totals["formula_correct"] += int(metrics.get("formula_correct_line_count", 0))
        totals["justification_correct"] += int(metrics.get("justification_correct_line_count", 0))
        totals["both_correct"] += int(metrics.get("both_correct_line_count", 0))
        totals["missing_lines"] += int(metrics.get("missing_line_count", 0))
        totals["extra_lines"] += int(metrics.get("extra_line_count", 0))
        totals["structure_mismatches"] += int(metrics.get("structure_mismatch_count", 0))

        case = cases_by_id.get(result.case_id, {})
        split = str(case.get("split") or "unassigned").strip() or "unassigned"
        split_counts = by_split.setdefault(split, {"cases": 0, "passed": 0, "failed": 0})
        split_counts["cases"] += 1
        split_counts["passed" if result.passed else "failed"] += 1

        categories = _failure_categories(result)
        for category in categories:
            failure_categories[category] = failure_categories.get(category, 0) + 1

        case_summaries.append(
            {
                "case_id": result.case_id,
                "split": split,
                "passed": bool(result.passed),
                "categories": categories,
                "metrics": _redacted_metrics(result.granular_metrics),
                "error": "present" if result.error else "",
                "run_summary": result.run_summary,
            }
        )

    return {
        "kind": "private_tfl_fitch_ocr_shareable_summary",
        "privacy": {
            "contains_paths": False,
            "contains_images": False,
            "contains_line_text": False,
            "contains_diffs": False,
        },
        "totals": totals,
        "by_split": by_split,
        "failure_categories": failure_categories,
        "cases": case_summaries,
    }


def render_shareable_summary(summary: Mapping[str, Any]) -> str:
    totals = summary.get("totals") if isinstance(summary.get("totals"), dict) else {}
    lines = [
        "Private TFL/Fitch OCR shareable summary",
        f"Cases: {int(totals.get('cases', 0))}",
        f"Passed: {int(totals.get('passed', 0))}",
        f"Failed: {int(totals.get('failed', 0))}",
        (
            "Aggregate lines: "
            f"formula {int(totals.get('formula_correct', 0))}/{int(totals.get('matched_lines', 0))}, "
            f"justification {int(totals.get('justification_correct', 0))}/{int(totals.get('matched_lines', 0))}, "
            f"both {int(totals.get('both_correct', 0))}/{int(totals.get('matched_lines', 0))}, "
            f"expected={int(totals.get('expected_lines', 0))}, "
            f"actual={int(totals.get('actual_lines', 0))}, "
            f"missing={int(totals.get('missing_lines', 0))}, "
            f"extra={int(totals.get('extra_lines', 0))}, "
            f"structural mismatches={int(totals.get('structure_mismatches', 0))}"
        ),
    ]

    by_split = summary.get("by_split") if isinstance(summary.get("by_split"), dict) else {}
    if by_split:
        lines.append("Splits:")
        for split in sorted(by_split):
            counts = by_split.get(split) if isinstance(by_split.get(split), dict) else {}
            lines.append(
                f"- {split}: cases={int(counts.get('cases', 0))}, "
                f"passed={int(counts.get('passed', 0))}, failed={int(counts.get('failed', 0))}"
            )

    categories = summary.get("failure_categories") if isinstance(summary.get("failure_categories"), dict) else {}
    if categories:
        lines.append("Failure categories:")
        for category in sorted(categories):
            lines.append(f"- {category}: {int(categories.get(category, 0))}")

    cases = summary.get("cases") if isinstance(summary.get("cases"), list) else []
    if cases:
        lines.append("Cases:")
        for item in cases:
            if not isinstance(item, dict):
                continue
            status = "PASS" if item.get("passed") else "FAIL"
            categories_text = ",".join(str(cat) for cat in (item.get("categories") or [])) or "none"
            metrics = item.get("metrics") if isinstance(item.get("metrics"), dict) else {}
            lines.append(
                f"- {status} {item.get('case_id')} split={item.get('split')} "
                f"categories={categories_text} "
                f"formula={int(metrics.get('formula_correct_line_count', 0))}/{int(metrics.get('matched_line_count', 0))} "
                f"justification={int(metrics.get('justification_correct_line_count', 0))}/{int(metrics.get('matched_line_count', 0))} "
                f"structure_mismatches={int(metrics.get('structure_mismatch_count', 0))}"
            )

    return "\n".join(lines)


def _write_private_case_report(
    *,
    output_dir: Path,
    case: Mapping[str, Any],
    result: GoldRegressionResult,
) -> None:
    payload = {
        "case_id": result.case_id,
        "split": str(case.get("split") or "unassigned"),
        "oracle_kind": str(case.get("oracle_kind") or "gold_private_human_checked"),
        "passed": bool(result.passed),
        "run_summary": result.run_summary,
        "error": result.error,
    }
    (output_dir / "private_gold_result.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    metrics_payload = {
        "case_id": result.case_id,
        "split": str(case.get("split") or "unassigned"),
        "metrics": result.granular_metrics or {},
        "error": result.error,
    }
    (output_dir / "private_gold_metrics.local.json").write_text(
        json.dumps(metrics_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if result.diff_text:
        (output_dir / "private_gold_diff.local.txt").write_text(result.diff_text, encoding="utf-8")


def run_private_tfl_gold_regressions(
    manifest: Mapping[str, Any],
    *,
    output_root: Path | None = None,
    case_ids: Sequence[str] | None = None,
    splits: Sequence[str] | None = None,
    transcribe_fn: Callable[..., Mapping[str, Any]] | None = None,
    allow_repo_paths: bool = False,
) -> List[GoldRegressionResult]:
    """Run private TFL/Fitch gold regressions and write redacted summaries."""
    manifest_dir = _manifest_dir(manifest)
    selected = select_private_tfl_cases(manifest, case_ids=case_ids, splits=splits)
    destination_root = (output_root or (manifest_dir / "runs" / "tfl_private_gold_regressions")).resolve()

    _validate_private_paths(
        manifest=manifest,
        output_root=destination_root,
        selected=selected,
        allow_repo_paths=allow_repo_paths,
    )
    destination_root.mkdir(parents=True, exist_ok=True)

    if transcribe_fn is None:
        from proofs_ocr.pipelines.tfl_fitch import run_fitch_ocr_pipeline

        transcribe_fn = run_fitch_ocr_pipeline  # type: ignore[assignment]

    results: List[GoldRegressionResult] = []
    for case in selected:
        case_id = str(case.get("id") or "").strip()
        image_path = private_case_image_path(case, manifest_dir=manifest_dir)
        expected_path = private_case_expected_path(case, manifest_dir=manifest_dir)
        output_dir = destination_root / _slug(case_id, fallback="case")
        output_dir.mkdir(parents=True, exist_ok=True)
        try:
            state = dict(
                _call_transcribe_fn(
                    transcribe_fn,
                    kwargs={"image_path": image_path, "output_dir": output_dir},
                )
            )
            actual = _load_actual_final_structure(output_dir)
            expected = load_private_expected_structure(case, manifest_dir=manifest_dir)
            passed, diff_text = compare_proof_structures(actual, expected)
            granular_metrics = compare_proof_structures_granular(actual, expected)
            result = GoldRegressionResult(
                case_id=case_id,
                oracle_kind=str(case.get("oracle_kind") or "gold_private_human_checked"),
                output_dir=output_dir,
                image_path=image_path,
                expected_path=expected_path,
                passed=passed,
                run_summary=str(state.get("run_summary") or "").strip(),
                granular_metrics=granular_metrics,
                diff_text=diff_text,
            )
        except Exception as exc:  # noqa: BLE001 - regression should continue across cases.
            result = GoldRegressionResult(
                case_id=case_id,
                oracle_kind=str(case.get("oracle_kind") or "gold_private_human_checked"),
                output_dir=output_dir,
                image_path=image_path,
                expected_path=expected_path,
                passed=False,
                run_summary="",
                granular_metrics=None,
                error=str(exc),
            )
        _write_private_case_report(output_dir=output_dir, case=case, result=result)
        results.append(result)

    cases_by_id = {str(case.get("id") or "").strip(): case for case in selected}
    shareable = build_shareable_summary(results, cases_by_id=cases_by_id)
    (destination_root / "shareable_summary.json").write_text(
        json.dumps(shareable, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (destination_root / "shareable_summary.txt").write_text(
        render_shareable_summary(shareable),
        encoding="utf-8",
    )
    return results


def render_private_manifest_summary(manifest: Mapping[str, Any]) -> str:
    manifest_dir = _manifest_dir(manifest)
    cases = private_manifest_cases(manifest)
    split_counts: Dict[str, int] = {}
    lines = [f"Private TFL/Fitch cases: {len(cases)}"]
    for case in cases:
        split = str(case.get("split") or "unassigned").strip() or "unassigned"
        split_counts[split] = split_counts.get(split, 0) + 1
    if split_counts:
        lines.append("Splits:")
        for split in sorted(split_counts):
            lines.append(f"- {split}: {split_counts[split]}")
    lines.append("Cases:")
    for case in cases:
        case_id = str(case.get("id") or "").strip() or "(missing id)"
        split = str(case.get("split") or "unassigned").strip() or "unassigned"
        image_exists = private_case_image_path(case, manifest_dir=manifest_dir).exists()
        expected_exists = private_case_expected_path(case, manifest_dir=manifest_dir).exists()
        gold_status = str(case.get("gold_status") or "").strip() or "unspecified"
        lines.append(
            f"- {case_id} split={split} gold={gold_status} "
            f"image={'ok' if image_exists else 'missing'} expected={'ok' if expected_exists else 'missing'}"
        )
    return "\n".join(lines)


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Private TFL/Fitch OCR regression runner.")
    parser.add_argument("--manifest", required=True, help="Path to private manifest YAML outside the repo.")
    parser.add_argument(
        "--allow-repo-paths",
        action="store_true",
        help="Disable the outside-repo privacy guard. Intended only for fake tests.",
    )
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("summary", help="Print a redacted manifest summary.")

    run_parser = subparsers.add_parser("run-gold", help="Run private gold regressions.")
    run_parser.add_argument("--output-root", default="", help="External directory for OCR outputs and reports.")
    run_parser.add_argument("--case", action="append", default=[], help="Run only this anonymous case id. May be repeated.")
    run_parser.add_argument("--split", action="append", default=[], help="Run only this split. May be repeated.")

    args = parser.parse_args()
    manifest = load_private_manifest(Path(args.manifest))

    if args.command in {None, "summary"}:
        _safe_console_print(render_private_manifest_summary(manifest))
        return 0

    if args.command == "run-gold":
        results = run_private_tfl_gold_regressions(
            manifest,
            output_root=Path(args.output_root) if str(args.output_root or "").strip() else None,
            case_ids=args.case,
            splits=args.split,
            allow_repo_paths=bool(args.allow_repo_paths),
        )
        output_root = (
            Path(args.output_root).expanduser().resolve()
            if str(args.output_root or "").strip()
            else (_manifest_dir(manifest) / "runs" / "tfl_private_gold_regressions").resolve()
        )
        _safe_console_print((output_root / "shareable_summary.txt").read_text(encoding="utf-8"))
        return 0 if all(result.passed for result in results) else 1

    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
