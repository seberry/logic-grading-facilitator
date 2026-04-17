from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

SUPPORTED_SYSTEMS = {"K", "T", "S4", "S5"}
TOKEN_PATTERN = re.compile(r"<SUBPROOF_ID_([A-Za-z0-9_]+)>")
JUSTIFICATION_HEAD_OPTIONS = (
    "PR",
    "R",
    "∧I",
    "∧E",
    "∨I",
    "∨E",
    "→I",
    "→E",
    "↔I",
    "↔E",
    "¬I",
    "¬E",
    "IP",
    "DN",
    "MC",
    "□I",
    "□E",
    "R4",
    "RT",
    "Def⋄",
    "",
    "other",
)
JUSTIFICATION_HEAD_OPTION_SET = set(JUSTIFICATION_HEAD_OPTIONS)
LINE_SCOPE_ROLE_OPTIONS = (
    "premise",
    "ordinary_assumption",
    "derived_line",
    "strict_open",
    "subproof_token",
)
LINE_SCOPE_ROLE_OPTION_SET = set(LINE_SCOPE_ROLE_OPTIONS)


@dataclass
class SubproofNode:
    id: str
    box: Tuple[int, int, int, int]
    children: List["SubproofNode"]
    parent: Optional["SubproofNode"] = None

    def __init__(self, box_id: str, box: Tuple[int, int, int, int]) -> None:
        self.id = box_id
        self.box = box
        self.children = []
        self.parent = None

    @property
    def depth(self) -> int:
        depth = 0
        current = self.parent
        while current:
            depth += 1
            current = current.parent
        return depth


def load_modal_config(path: Path | None) -> Dict[str, Any]:
    if path is None or not path.exists():
        return {}

    raw_text = path.read_text(encoding="utf-8")
    parsed = yaml.safe_load(raw_text) or {}
    if not isinstance(parsed, dict):
        raise ValueError("modal autograde file must parse to an object/map")

    if "proof_json_text" in parsed or "system" in parsed:
        return {
            "system": str(parsed.get("system") or "").strip() or "K",
            "include_derived_rules": bool(parsed.get("include_derived_rules", False)),
            "conclusion": str(parsed.get("conclusion") or "").strip() or None,
        }

    nested_setup = parsed.get("problem_setup_text")
    if isinstance(nested_setup, str) and nested_setup.strip():
        nested = yaml.safe_load(nested_setup) or {}
        if not isinstance(nested, dict):
            raise ValueError("problem_setup_text must parse to an object/map")
        parsed = nested

    return {
        "system": str(parsed.get("system") or "").strip() or "K",
        "include_derived_rules": bool(parsed.get("include_derived_rules", False)),
        "conclusion": str(parsed.get("conclusion") or "").strip() or None,
    }


def canonical_modal_config(config: Dict[str, Any]) -> Dict[str, Any]:
    system = str(config.get("system") or "K").strip() or "K"
    if system not in SUPPORTED_SYSTEMS:
        system = "K"
    conclusion = config.get("conclusion")
    conclusion_text = str(conclusion).strip() if conclusion is not None else ""
    return {
        "system": system,
        "include_derived_rules": bool(config.get("include_derived_rules", False)),
        "conclusion": conclusion_text or None,
    }

