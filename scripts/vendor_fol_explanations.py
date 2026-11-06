"""Vendor the approved, dependency-free upstream module at an explicit revision."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
FILES = (
    "index.mjs", "core.mjs", "commentary.mjs", "README.md", "INTEGRATION.md",
    "FEEDBACK-REVIEW.md", "examples/quiz9.mjs", "examples/generate-review.mjs",
)


def vendor(source: Path, revision: str) -> Path:
    # Command-local trust avoids changing the user's global Git configuration.
    git = ["git", "-c", f"safe.directory={source.resolve().as_posix()}", "-C", str(source)]
    commit = subprocess.check_output(git + ["rev-parse", "--verify", f"{revision}^{{commit}}"], text=True).strip()
    contents = {name: subprocess.check_output(git + ["show", f"{commit}:fol-explanations/{name}"])
                for name in FILES}
    target = ROOT / "app/assets/vendor/fol-explanations"
    for name, data in contents.items():
        path = target / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    provenance = {
        "repository": "https://github.com/seberry/FOL-Introduction-Worksheets",
        "directory": "fol-explanations", "commit": commit,
        "sha256": {name: hashlib.sha256(data).hexdigest() for name, data in contents.items()},
    }
    (target / "SOURCE.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()
    print(vendor(args.source, args.revision))
