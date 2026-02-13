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


class ClassworkFileRef(BaseModel):
    """Single uploaded classwork file tied to one student."""
    stored_filename: str
    original_filename: str
    ext: str


class ClassworkEntry(BaseModel):
    """Roster-complete classwork row (submitted/missing + score/comment)."""
    student_id: str
    student_name: str
    has_submission: bool
    files: List[ClassworkFileRef] = Field(default_factory=list)
    score: float
    comment: Optional[str] = None


class ClassworkModeState(BaseModel):
    """State for effort-based classwork review mode."""
    assignment_id: str
    max_points: float
    gradebook_token: str
    id_column: str
    gradebook_column: Optional[str] = None
    entries: List[ClassworkEntry] = Field(default_factory=list)
    unmatched_files: List[str] = Field(default_factory=list)


class ClassworkBatchState(BaseModel):
    """Batch mode: one gradebook + multiple classwork assignments."""
    gradebook_token: str
    id_column: str
    assignments: List[ClassworkModeState] = Field(default_factory=list)


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
    # Optional per-problem type tags for autograde workflows (e.g., "fitch_proof")
    problem_types: Dict[str, Optional[str]] = Field(default_factory=dict)
    # Optional effort-grading mode state for classwork/attendance assignments.
    classwork_mode: Optional[ClassworkModeState] = None
    # Optional batch state for merging multiple classwork assignments at once.
    classwork_batch: Optional[ClassworkBatchState] = None
