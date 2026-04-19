"""Manifest loading and explicit gold-regression runner for proof OCR fixtures."""

from __future__ import annotations

import argparse
import difflib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence

import yaml


@dataclass(frozen=True)
class GoldRegressionResult:
    case_id: str
    oracle_kind: str
    output_dir: Path
    image_path: Path
    expected_path: Path
    passed: bool
    run_summary: str
    diff_text: str = ""
    error: str = ""


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
        for key in ("wffstr", "jstr"):
            if key in value:
                normalized[key] = str(value.get(key) or "").strip()
        return normalized
    return value


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
    if diff_text:
        (output_dir / "gold_regression_diff.txt").write_text(diff_text, encoding="utf-8")


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
                transcribe_fn(
                    image_path=image_path,
                    output_dir=output_dir,
                    modal_config=modal_config,
                    vision_base_url=vision_base_url,
                    vision_model=vision_model,
                )
            )
            actual = _load_actual_final_structure(output_dir)
            expected = load_expected_structure(case, repo_root=root)
            passed, diff_text = compare_proof_structures(actual, expected)
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
                    error=error,
                )
            )
    return results


def render_gold_regression_summary(results: Iterable[GoldRegressionResult]) -> str:
    materialized = list(results)
    total = len(materialized)
    passed_count = sum(1 for result in materialized if result.passed)
    failed_count = total - passed_count
    lines = [f"Gold cases: {total}", f"Passed: {passed_count}", f"Failed: {failed_count}"]
    for result in materialized:
        status = "PASS" if result.passed else "FAIL"
        line = f"- {status} {result.case_id} ({result.oracle_kind})"
        if result.error:
            line += f" error={result.error}"
        elif result.run_summary:
            line += f" {result.run_summary}"
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

    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    manifest = load_manifest()

    if args.command in {None, "summary"}:
        print(render_manifest_summary(manifest))
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
        )
        print(render_gold_regression_summary(results))
        return 0 if all(result.passed for result in results) else 1

    raise ValueError(f"Unknown regression command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
