"""Tiny manifest loader and summary runner for proof OCR regression suites."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import yaml


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
    cases = manifest.get("cases") or []
    if not isinstance(cases, list):
        raise ValueError("proof OCR regression manifest cases must be a list")

    lines: List[str] = []
    lines.append(f"Suite: {suite.get('name', 'unknown')}")
    lines.append(f"Cases: {len(cases)}")

    for case in cases:
        if not isinstance(case, dict):
            continue
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


def main() -> int:
    manifest = load_manifest()
    print(render_manifest_summary(manifest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

