from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import List, Tuple

import yaml  # type: ignore

from .models import RosterEntry, GradingKey, ProblemSpec, Session

import shutil
from datetime import datetime
import re

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]


def _resolve_path_from_env(var_name: str, default: Path) -> Path:
    raw = os.getenv(var_name, "")
    if not raw:
        return default
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = (ROOT_DIR / path).resolve()
    return path


def _resolve_data_dir() -> Path:
    load_dotenv()
    default = (ROOT_DIR / "data" / "samples").resolve()
    data_dir = _resolve_path_from_env("DATA_DIR", default)

    env_mode = os.getenv("ENV_MODE", "").strip().upper()
    if env_mode == "PRODUCTION":
        repo_root = ROOT_DIR.resolve()
        if data_dir == repo_root or repo_root in data_dir.parents:
            raise RuntimeError(
                "ENV_MODE=PRODUCTION requires DATA_DIR outside the repo. "
                f"Repo root: {repo_root}, DATA_DIR: {data_dir}"
            )
    return data_dir


def _resolve_archive_dir(data_dir: Path) -> Path:
    default = data_dir / "data_archive"
    return _resolve_path_from_env("DATA_ARCHIVE_DIR", default)


DATA_DIR = _resolve_data_dir()
ENV_MODE = os.getenv("ENV_MODE", "").strip().upper()
IS_PRODUCTION = ENV_MODE == "PRODUCTION"
RAW_SCANS_DIR = DATA_DIR / "raw_scans"
GRADES_DIR = DATA_DIR / "grades"
RUBRIC_DIR = DATA_DIR / "rubric"
RUBRIC_ASSETS_DIR = DATA_DIR / "rubric_assets"
NAME_REVIEW_CACHE_DIR = DATA_DIR / "name_review_cache"
DEBUG_FITCH_DIR = DATA_DIR / "debug_fitch"
DEBUG_FOL_MODEL_DIR = DATA_DIR / "debug_fol_model"
DEBUG_MODAL_PROOF_DIR = DATA_DIR / "debug_modal_proof"
AUTO_GROUPED_DIR = DATA_DIR / "auto_grouped"
FLASHCARDS_DIR = DATA_DIR / "flashcards"

DEFAULT_SESSION_PATH = GRADES_DIR / "session.json"
DEFAULT_ROSTER_PATH = DATA_DIR / "roster.csv"
DEFAULT_GRADING_KEY_PATH = RUBRIC_DIR / "quiz1.yaml"
DATA_ARCHIVE_DIR = _resolve_archive_dir(DATA_DIR)
DATA_ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
NAME_REVIEW_CACHE_DIR.mkdir(parents=True, exist_ok=True)
DEBUG_FITCH_DIR.mkdir(parents=True, exist_ok=True)
DEBUG_FOL_MODEL_DIR.mkdir(parents=True, exist_ok=True)
DEBUG_MODAL_PROOF_DIR.mkdir(parents=True, exist_ok=True)
RAW_SCANS_DIR.mkdir(parents=True, exist_ok=True)
RUBRIC_ASSETS_DIR.mkdir(parents=True, exist_ok=True)
FLASHCARDS_DIR.mkdir(parents=True, exist_ok=True)


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
        problems=problems,
        fol_truth_assistance=data.get("fol_truth_assistance"),
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
    - Lists all scan files in DATA_DIR/raw_scans.
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


def _copy_tree(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    if src.is_dir():
        shutil.copytree(src, dst)
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def archive_current_data(label: str) -> str:
    """
    Archive the entire current DATA_DIR dataset into DATA_ARCHIVE_DIR,
    then clear raw_scans/ and session.json to prepare for a new assignment.

    Returns the snapshot_id (e.g. "2025-11-16T14-05-23_quiz1").
    """
    # Build snapshot id: timestamp + slugged label
    now = datetime.now()
    timestamp_str = now.strftime("%Y-%m-%dT%H-%M-%S")
    slug = _slugify_label(label)
    snapshot_id = f"{timestamp_str}_{slug}"

    snapshot_root = DATA_ARCHIVE_DIR / snapshot_id
    snapshot_root.mkdir(parents=True, exist_ok=True)

    # 1. Copy the live data paths into the snapshot.
    # Avoid copying data_archive into itself if it lives under DATA_DIR.
    copy_items: List[Tuple[Path, Path]] = [
        (RAW_SCANS_DIR, snapshot_root / "raw_scans"),
        (GRADES_DIR, snapshot_root / "grades"),
        (RUBRIC_DIR, snapshot_root / "rubric"),
        (RUBRIC_ASSETS_DIR, snapshot_root / "rubric_assets"),
        (DATA_DIR / "rosters", snapshot_root / "rosters"),
        (NAME_REVIEW_CACHE_DIR, snapshot_root / "name_review_cache"),
        (DATA_DIR / "debug_fitch", snapshot_root / "debug_fitch"),
        (DATA_DIR / "debug_fol_model", snapshot_root / "debug_fol_model"),
        (DATA_DIR / "debug_modal_proof", snapshot_root / "debug_modal_proof"),
        (DATA_DIR / "auto_grouped", snapshot_root / "auto_grouped"),
        (DATA_DIR / "classwork", snapshot_root / "classwork"),
        (DATA_DIR / "canvas_cache", snapshot_root / "canvas_cache"),
        (DATA_DIR / "gradebook_merge", snapshot_root / "gradebook_merge"),
        (DEFAULT_ROSTER_PATH, snapshot_root / "roster.csv"),
    ]
    for src, dst in copy_items:
        _copy_tree(src, dst)

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
    #    - clear grades/ artifacts (session + cached segment files)
    raw_scans_dir = RAW_SCANS_DIR
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

    grades_dir = GRADES_DIR
    if grades_dir.exists():
        for child in grades_dir.iterdir():
            if child.is_file():
                child.unlink()
            elif child.is_dir():
                shutil.rmtree(child)
    else:
        grades_dir.mkdir(parents=True, exist_ok=True)

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
