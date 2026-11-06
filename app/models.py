from typing import Any, List, Dict, Optional
from pydantic import BaseModel, Field


class RosterEntry(BaseModel):
    """Single student from the Canvas roster."""
    student_id: str
    name: str


class ProblemVariantSpec(BaseModel):
    """Version-specific display and grading guidance for one problem."""

    label: Optional[str] = None
    prompt: Optional[str] = None
    # Relative path below DATA_DIR/rubric_assets. This is a crop of the blank
    # assessment, never a student scan or generated solution.
    question_image: Optional[str] = None
    answer_key: Optional[str] = None
    rubric_notes: Optional[str] = None
    segmentation_hint: Optional[str] = None


class ProblemSpec(BaseModel):
    """Single problem on the quiz/exam."""
    id: str          # e.g. "Q1"
    label: str       # human-readable label, e.g. "Problem 1 (symbolization)"
    points: float    # max points for this problem
    extra_credit: bool = False
    section: Optional[str] = None
    problem_number: Optional[str] = None
    prompt: Optional[str] = None
    question_image: Optional[str] = None
    answer_key: Optional[str] = None
    segmentation_hint: Optional[str] = None
    rubric_notes: Optional[str] = None
    problem_type: Optional[str] = None
    autograde: Dict[str, Any] = Field(default_factory=dict)
    variants: Dict[str, ProblemVariantSpec] = Field(default_factory=dict)


class GradingKey(BaseModel):
    """Grading key / rubric for one quiz."""
    quiz_id: str
    problems: List[ProblemSpec]
    # Explicit opt-in. Never infer an answer key from a printed prompt.
    fol_truth_assistance: Optional[Dict[str, Any]] = None


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
    # Printed quiz/exam version (for example "A", "B", or "C").
    quiz_version: Optional[str] = None
    per_problem: Dict[str, Optional[float]]
    comment: Optional[str]
    per_problem_comments: Dict[str, Optional[str]] = Field(default_factory=dict)
    # TF-confirmed transcription is separate from any future vision proposal.
    fol_truth_reviews: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    # PLANNED: free-text comments keyed by problem id


class ClassworkFileRef(BaseModel):
    """Single classwork file with separate storage and original display/export names."""
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


class ClassworkDownloadWarning(BaseModel):
    """Canvas download failure recorded after retries are exhausted."""
    student_id: str
    student_name: str
    filename: str
    error: str


class ClassworkModeState(BaseModel):
    """State for effort-based classwork review mode."""
    assignment_id: str
    max_points: float
    gradebook_token: str
    id_column: str
    gradebook_column: Optional[str] = None
    entries: List[ClassworkEntry] = Field(default_factory=list)
    unmatched_files: List[str] = Field(default_factory=list)
    download_warnings: List[ClassworkDownloadWarning] = Field(default_factory=list)
    # Canvas integration fields
    canvas_course_id: Optional[int] = None
    canvas_assignment_id: Optional[int] = None


class ClassworkBatchState(BaseModel):
    """Batch mode: one gradebook + multiple classwork assignments."""
    gradebook_token: str
    id_column: str
    assignments: List[ClassworkModeState] = Field(default_factory=list)
    # Canvas user ID mapping: sis_login_id (student_id) -> canvas_user_id
    # This is in-memory only, not persisted to session.json (FERPA-safe)
    canvas_user_ids: Dict[str, int] = Field(default_factory=dict)


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
    pinned_note: Optional[str] = None  # Legacy session-wide grading note

    # NEW: optional per-problem notes (e.g. “Q1 = truth table for ¬(P→Q)”)
    problem_notes: Dict[str, str] = Field(default_factory=dict)
    # Human-selected visual comparison answers, keyed by problem id and then
    # normalized quiz version (for example {"Q1": {"A": 7, "B": 12}}).
    # Values are WorkUnit ids; the scan/crop remains the source of truth.
    reference_pins: Dict[str, Dict[str, int]] = Field(default_factory=dict)
    # Optional per-problem type tags for autograde workflows (e.g., "fitch_proof")
    problem_types: Dict[str, Optional[str]] = Field(default_factory=dict)
    # Optional effort-grading mode state for classwork/attendance assignments.
    classwork_mode: Optional[ClassworkModeState] = None
    # Optional batch state for merging multiple classwork assignments at once.
    classwork_batch: Optional[ClassworkBatchState] = None
