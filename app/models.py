from typing import List, Dict, Optional
from pydantic import BaseModel, Field


class RosterEntry(BaseModel):
    """Single student from the Canvas roster."""
    student_id: str
    name: str


class ProblemSpec(BaseModel):
    """Single problem on the quiz/exam."""
    id: str          # e.g. "Q1"
    label: str       # human-readable label, e.g. "Problem 1 (symbolization)"
    points: float    # max points for this problem


class GradingKey(BaseModel):
    """Grading key / rubric for one quiz."""
    quiz_id: str
    max_points: float
    problems: List[ProblemSpec]


class WorkUnit(BaseModel):
    """
    One student's submission.

    - unit_id: sequential integer ID for this session
    - page_filenames: list of JPG filenames in data/raw_scans/
    - student_id: Canvas student ID (from roster), once identified
    - per_problem: mapping from problem.id -> score (or None if ungraded)
    - comment: optional free-text comment
    """
    unit_id: int
    page_filenames: List[str]
    student_id: Optional[str]
    per_problem: Dict[str, Optional[float]]
    comment: Optional[str]
    per_problem_comments: Dict[str, Optional[str]] = Field(default_factory=dict)
    # PLANNED: free-text comments keyed by problem id


class Session(BaseModel):
    """
    Canonical session file, matching docs/SCHEMA.md:

    {
      "roster": [...],
      "grading_key": {...},
      "work_units": [...],
      "unclaimed_pages": [...]
    }
    """
    roster: List[RosterEntry]
    grading_key: GradingKey
    work_units: List[WorkUnit]
    unclaimed_pages: List[str]
    pinned_note: Optional[str] = None  # PLANNED: session-wide answer key / notes

    # NEW: optional per-problem notes (e.g. “Q1 = truth table for ¬(P→Q)”)
    problem_notes: Dict[str, str] = Field(default_factory=dict)

    # Assignment metadata (optional, used by the Start Assignment flow)
    assignment_name: Optional[str] = None
    raw_scans_subdir: Optional[str] = None
    roster_path: Optional[str] = None
