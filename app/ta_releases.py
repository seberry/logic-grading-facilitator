"""Find locally available grader code that has passed a handoff smoke test."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import file_digest
import json
from pathlib import Path
import re
from zipfile import BadZipFile, ZipFile


ROOT = Path(__file__).resolve().parents[1]
TA_RELEASE_ROOT = ROOT / "working" / "ta_releases"
CAPABILITIES = "app/assets/fol_truth_capabilities.json"
EXPLAINER_SOURCE = "app/assets/vendor/fol-explanations/SOURCE.json"
_ARCHIVE_NAME = re.compile(r"logic-grading-facilitator-[0-9a-f]{7,40}\.zip")


@dataclass(frozen=True)
class GraderRelease:
    archive: Path
    sha256: str
    schemas: tuple[int, ...]
    explainer_revision: str | None

    def supports(self, schema: int | None, revision: str | None) -> bool:
        return schema is None or (schema in self.schemas and self.explainer_revision == revision)


def explainer_revision() -> str:
    return json.loads((ROOT / EXPLAINER_SOURCE).read_text(encoding="utf-8"))["commit"]


def inspect_code_zip(archive: Path) -> GraderRelease:
    """Read the same compatibility fields the TF launchers check."""
    if not _ARCHIVE_NAME.fullmatch(archive.name):
        raise ValueError("Expected a code ZIP built by scripts/build_ta_release.py.")
    prefix = archive.stem + "/"
    with archive.open("rb") as stream:
        digest = file_digest(stream, "sha256").hexdigest()
    with ZipFile(archive) as bundle:
        bundle.getinfo(prefix + "app/server.py")
        required = ("app/fol_truth_assistance.py", "app/assets/fol_truth_review.mjs",
                    "app/assets/fol_truth_scoring.mjs", "app/assets/vendor/fol-explanations/index.mjs",
                    EXPLAINER_SOURCE)
        names = set(bundle.namelist())
        if not all(prefix + name in names for name in required):
            return GraderRelease(archive, digest, (), None)
        revision = json.loads(bundle.read(prefix + EXPLAINER_SOURCE))["commit"]
        schemas = json.loads(bundle.read(prefix + CAPABILITIES))["schemas"] if prefix + CAPABILITIES in names else [1]
        if not isinstance(revision, str) or not isinstance(schemas, list) or any(type(s) is not int for s in schemas):
            raise ValueError("Invalid FOL compatibility metadata in the grader code ZIP.")
    return GraderRelease(archive, digest, tuple(schemas), revision)


def tested_releases():
    receipts = sorted(TA_RELEASE_ROOT.glob("*/logic-grading-facilitator-*.tested.json"),
                      key=lambda path: path.parent.name, reverse=True)
    for receipt in receipts:
        if not receipt.resolve().is_relative_to(TA_RELEASE_ROOT.resolve()):
            continue
        try:
            archive = receipt.with_name(receipt.name.removesuffix(".tested.json") + ".zip")
            if not archive.resolve().is_relative_to(TA_RELEASE_ROOT.resolve()):
                continue
            release = inspect_code_zip(archive)
            if json.loads(receipt.read_text(encoding="utf-8")) == {"sha256": release.sha256}:
                yield release
        except (OSError, ValueError, TypeError, KeyError, BadZipFile):
            continue
