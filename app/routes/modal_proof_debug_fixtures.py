from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List


def strip_yaml_scalar(raw: str) -> str:
    text = str(raw or "").strip()
    if not text:
        return ""
    if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
        return text[1:-1]
    return text


def load_modal_regression_fixtures(manifest_path: Path, repo_root: Path) -> List[Dict[str, str]]:
    if not manifest_path.exists():
        return []

    try:
        lines = manifest_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []

    fixtures: List[Dict[str, str]] = []
    current: Dict[str, str] | None = None
    in_cases = False

    for raw_line in lines:
        if not in_cases:
            if raw_line.strip() == "cases:":
                in_cases = True
            continue

        case_start = re.match(r"^  - id:\s*(.+)$", raw_line)
        if case_start:
            if current:
                fixtures.append(current)
            current = {"id": strip_yaml_scalar(case_start.group(1))}
            continue

        if current is None:
            continue

        field_match = re.match(r"^    ([a-zA-Z0-9_]+):\s*(.*)$", raw_line)
        if field_match:
            current[field_match.group(1)] = strip_yaml_scalar(field_match.group(2))

    if current:
        fixtures.append(current)

    resolved: List[Dict[str, str]] = []
    for fixture in fixtures:
        image_path = str(fixture.get("image_path") or "").strip()
        fixture_id = str(fixture.get("id") or "").strip()
        if not fixture_id or not image_path:
            continue
        absolute_image_path = (repo_root / image_path).resolve()
        resolved.append(
            {
                "id": fixture_id,
                "domain": str(fixture.get("domain") or "").strip(),
                "kind": str(fixture.get("kind") or "").strip(),
                "image_path": image_path,
                "absolute_image_path": str(absolute_image_path),
                "notes": str(fixture.get("notes") or "").strip(),
            }
        )
    return resolved


def fixture_option_label(fixture: Dict[str, str]) -> str:
    fixture_id = str(fixture.get("id") or "").strip()
    kind = str(fixture.get("kind") or "").strip()
    domain = str(fixture.get("domain") or "").strip()
    parts = [fixture_id]
    meta = " / ".join([item for item in (domain, kind) if item])
    if meta:
        parts.append(meta)
    return " - ".join(parts)


def fixture_lookup(fixtures: List[Dict[str, str]]) -> Dict[str, Dict[str, str]]:
    return {str(item.get("id") or "").strip(): item for item in fixtures if str(item.get("id") or "").strip()}
