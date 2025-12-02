from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import List

import yaml  # type: ignore

from .models import RosterEntry, GradingKey, ProblemSpec, Session

from pathlib import Path
import shutil
import json
from datetime import datetime
import re

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
RAW_SCANS_DIR = DATA_DIR / "raw_scans"
GRADES_DIR = DATA_DIR / "grades"
RUBRIC_DIR = DATA_DIR / "rubric"
NAME_REVIEW_CACHE_DIR = DATA_DIR / "name_review_cache"

DEFAULT_SESSION_PATH = GRADES_DIR / "session.json"
DEFAULT_ROSTER_PATH = DATA_DIR / "roster.csv"
DEFAULT_GRADING_KEY_PATH = RUBRIC_DIR / "quiz1.yaml"
DATA_ARCHIVE_DIR = ROOT_DIR / "data_archive"
DATA_ARCHIVE_DIR.mkdir(exist_ok=True)
NAME_REVIEW_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def load_roster(path: Path = DEFAULT_ROSTER_PATH) -> List[RosterEntry]:
    """
    Load a simplified roster CSV.

    Expected header: student_id,name

    You can create this by exporting Canvas gradebook and trimming it
    down to just these two columns.
    """
    if not path.exists():
        raise FileNotFoundError(f"Roster CSV not found at {path}")

    entries: List[RosterEntry] = []
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if "student_id" not in reader.fieldnames or "name" not in reader.fieldnames:
            raise ValueError(
                f"Roster CSV {path} must have headers: student_id,name "
                f"(found: {reader.fieldnames})"
            )
        for row in reader:
            sid = (row.get("student_id") or "").strip()
            name = (row.get("name") or "").strip()
            if not sid or not name:
                continue
            entries.append(RosterEntry(student_id=sid, name=name))
    return entries


def load_grading_key(path: Path = DEFAULT_GRADING_KEY_PATH) -> GradingKey:
    """
    Load grading key YAML.

    Example format:

    quiz_id: quiz1
    max_points: 10
    problems:
      - id: "Q1"
        label: "Problem 1"
        points: 5
      - id: "Q2"
        label: "Problem 2"
        points: 5
    """
    if not path.exists():
        raise FileNotFoundError(f"Grading key YAML not found at {path}")

    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)

    problems = [
        ProblemSpec(**p) for p in data.get("problems", [])
    ]
    return GradingKey(
        quiz_id=data["quiz_id"],
        max_points=float(data["max_points"]),
        problems=problems,
    )


def save_session(session: Session, path: Path = DEFAULT_SESSION_PATH) -> None:
    """Write canonical session.json with pretty indentation."""
    GRADES_DIR.mkdir(parents=True, exist_ok=True)
    obj = session.model_dump()
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def load_session(path: Path = DEFAULT_SESSION_PATH) -> Session:
    """Load an existing session.json."""
    if not path.exists():
        raise FileNotFoundError(f"Session file not found at {path}")
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    return Session(**data)


def init_session(
    roster_path: Path = DEFAULT_ROSTER_PATH,
    grading_key_path: Path = DEFAULT_GRADING_KEY_PATH,
    session_path: Path = DEFAULT_SESSION_PATH,
    raw_scans_dir: Path = RAW_SCANS_DIR,
) -> Session:
    """
    Initialize a brand-new session.json.

    - Reads roster CSV and grading key YAML.
    - Lists all scan files in data/raw_scans.
    - Creates an empty work_units list.
    - Puts all scans into unclaimed_pages (sorted by filename).
    - Overwrites any existing session.json at session_path.
    """
    roster = load_roster(roster_path)
    grading_key = load_grading_key(grading_key_path)

    if not raw_scans_dir.exists():
        raise FileNotFoundError(f"Raw scans directory not found: {raw_scans_dir}")

    exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    filenames = [
        p.name
        for p in raw_scans_dir.iterdir()
        if p.is_file() and p.suffix in exts
    ]
    filenames.sort()

    session = Session(
        roster=roster,
        grading_key=grading_key,
        work_units=[],
        unclaimed_pages=filenames,
    )
    save_session(session, session_path)
    return session

def _slugify_label(label: str) -> str:
    """
    Turn an arbitrary label into a filesystem-safe slug.
    Example: "Quiz 1 – Section A" -> "quiz_1_section_a"
    """
    s = label.strip().lower()
    s = re.sub(r"[\s\-–]+", "_", s)       # spaces/dashes -> _
    s = re.sub(r"[^a-z0-9_]", "", s)     # drop other punctuation
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "assignment"


def archive_current_data(label: str) -> str:
    """
    Archive the entire current `data/` directory into `data_archive/`,
    then clear raw_scans/ and session.json to prepare for a new assignment.

    Returns the snapshot_id (e.g. "2025-11-16T14-05-23_quiz1").
    """
    # Build snapshot id: timestamp + slugged label
    now = datetime.now()
    timestamp_str = now.strftime("%Y-%m-%dT%H-%M-%S")
    slug = _slugify_label(label)
    snapshot_id = f"{timestamp_str}_{slug}"

    snapshot_root = DATA_ARCHIVE_DIR / snapshot_id
    snapshot_root.parent.mkdir(exist_ok=True)

    # 1. Copy entire data/ tree into the snapshot.
    # This gives us a full image of roster.csv, rubric, raw_scans, grades, etc.
    shutil.copytree(DATA_DIR, snapshot_root)

    # 2. Write a small metadata.json into the snapshot.
    metadata = {
        "label": label,
        "timestamp": now.isoformat(),
    }
    (snapshot_root / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    # 3. Reset live state:
    #    - clear raw_scans/
    #    - delete grades/session.json (if present)
    raw_scans_dir = DATA_DIR / "raw_scans"
    if raw_scans_dir.exists():
        for child in raw_scans_dir.iterdir():
            if child.is_file():
                child.unlink()
            elif child.is_dir():
                shutil.rmtree(child)
    else:
        raw_scans_dir.mkdir(parents=True, exist_ok=True)

    # Ensure raw_scans directory exists (empty)
    raw_scans_dir.mkdir(parents=True, exist_ok=True)

    session_path = DATA_DIR / "grades" / "session.json"
    if session_path.exists():
        session_path.unlink()

    # Clear cached name review results so the setup wizard can start fresh
    if NAME_REVIEW_CACHE_DIR.exists():
        for child in NAME_REVIEW_CACHE_DIR.iterdir():
            if child.is_file():
                child.unlink()
            elif child.is_dir():
                shutil.rmtree(child)
    else:
        NAME_REVIEW_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Note: we intentionally *do not* touch roster.csv or rubric/ in DATA_DIR.
    return snapshot_id
