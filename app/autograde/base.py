from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class AutoGradeTraceBlock:
    title: str
    result: str
    body: str


@dataclass
class AutoGradeResult:
    problem_type: str
    status: str
    passed: Optional[bool]
    summary: str
    draft_comment: str
    score: Optional[float] = None
    trace_blocks: List[AutoGradeTraceBlock] = field(default_factory=list)
    extracted_answer: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)

