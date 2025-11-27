"""Populate `data/` with FERPA-safe fake inputs for bbox testing.

This copies the handcrafted assets in `docs/fake_data/` into the ignored
`data/` tree so you can open `/problem/Q1/strips` without any real scans.
"""

from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FAKE_SRC = ROOT / "docs" / "fake_data"
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw_scans"
GRADES_DIR = DATA_DIR / "grades"
RUBRIC_DIR = DATA_DIR / "rubric"


def copy(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dest)
    print(f"wrote {dest}")


def main() -> None:
    if not FAKE_SRC.exists():
        raise FileNotFoundError(f"Missing source assets: {FAKE_SRC}")

    expected_scans = [RAW_DIR / "fake_quiz_1.jpg", RAW_DIR / "fake_quiz_2.jpg"]
    missing = [p for p in expected_scans if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing raw scans. Place fake_quiz_1.jpg and fake_quiz_2.jpg in "
            f"{RAW_DIR} before running this script."
        )

    # Core artifacts (JPEGs are already in data/raw_scans/)
    copy(FAKE_SRC / "roster.csv", DATA_DIR / "roster.csv")
    copy(FAKE_SRC / "quiz1.yaml", RUBRIC_DIR / "quiz1.yaml")
    copy(FAKE_SRC / "session.json", GRADES_DIR / "session.json")
    copy(FAKE_SRC / "segments_Q1.json", GRADES_DIR / "segments_Q1.json")

    print()
    print("Fake bbox dataset ready.")
    print("Start the app and open /problem/Q1/strips to see the synthetic boxes.")
    print("Hint: /grade/1/page also loads with the same fake scan and roster.")


if __name__ == "__main__":
    main()
