"""Seed private TFL/Fitch OCR eval manifests from local debug artifacts."""

from __future__ import annotations

import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import yaml

from proofs_ocr.regression.runner import get_repo_root

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")


@dataclass(frozen=True)
class ExportedPrivateCase:
    case_id: str
    source_slug: str
    status: str
    image_path: Path
    expected_path: Path
    skipped: bool = False
    reason: str = ""


def _safe_console_print(text: str) -> None:
    try:
        print(text)
    except UnicodeEncodeError:
        encoded = text.encode(
            getattr(sys.stdout, "encoding", None) or "utf-8",
            errors="replace",
        )
        sys.stdout.buffer.write(encoded + b"\n")


def _slug(value: Any, *, fallback: str = "item") -> str:
    text = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(value or "").strip())
    while "__" in text:
        text = text.replace("__", "_")
    return text.strip("_") or fallback


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _require_external_output(path: Path, *, allow_repo_paths: bool) -> None:
    if allow_repo_paths:
        return
    repo_root = get_repo_root().resolve()
    resolved = path.resolve()
    if _is_relative_to(resolved, repo_root):
        raise ValueError(
            "Private TFL OCR eval output must stay outside the repository. "
            f"Choose an external --eval-root. Refused path: {resolved}"
        )


def _state_path_for_problem(grades_dir: Path, problem_id: str) -> Path:
    return Path(grades_dir) / "proof_ocr" / f"tfl_fitch_{_slug(problem_id, fallback='problem')}.json"


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_yaml_mapping(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    parsed = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(parsed, dict):
        raise ValueError(f"YAML file must parse to a mapping: {path}")
    return parsed


def _dump_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(dict(payload), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def _manifest_cases(manifest: Mapping[str, Any]) -> List[Dict[str, Any]]:
    raw = manifest.get("cases") or []
    if not isinstance(raw, list):
        raise ValueError("private eval manifest 'cases' must be a list")
    return [case for case in raw if isinstance(case, dict)]


def _find_primary_proof_image(debug_case_dir: Path, slug: str) -> Path | None:
    for ext in IMAGE_EXTENSIONS:
        candidate = debug_case_dir / f"{slug}{ext.lower()}"
        if candidate.exists():
            return candidate
        candidate = debug_case_dir / f"{slug}{ext.upper()}"
        if candidate.exists():
            return candidate

    image_candidates: List[Path] = []
    for child in sorted(debug_case_dir.iterdir() if debug_case_dir.exists() else []):
        if not child.is_file() or child.suffix not in IMAGE_EXTENSIONS:
            continue
        if child.name.startswith(("crop_", "debug_")):
            continue
        image_candidates.append(child)
    if image_candidates:
        return image_candidates[0]
    return None


def _relative_manifest_path(path: Path, *, manifest_path: Path) -> str:
    try:
        return path.resolve().relative_to(manifest_path.resolve().parent).as_posix()
    except ValueError:
        return str(path.resolve())


def _existing_case_ids(cases: Iterable[Mapping[str, Any]]) -> set[str]:
    return {str(case.get("id") or "").strip() for case in cases if str(case.get("id") or "").strip()}


def _next_case_id(*, existing_ids: set[str], prefix: str) -> str:
    number = 1
    while True:
        candidate = f"{prefix}{number:04d}"
        if candidate not in existing_ids:
            existing_ids.add(candidate)
            return candidate
        number += 1


def _selected_state_items(
    state: Mapping[str, Any],
    *,
    statuses: Sequence[str],
) -> List[Dict[str, Any]]:
    allowed = {str(status).strip() for status in statuses if str(status).strip()}
    raw_items = state.get("items") if isinstance(state.get("items"), dict) else {}
    selected: List[Dict[str, Any]] = []
    for key in sorted(raw_items, key=lambda value: int(value) if str(value).isdigit() else str(value)):
        item = raw_items.get(key)
        if not isinstance(item, dict):
            continue
        status = str(item.get("status") or "").strip()
        if allowed and status not in allowed:
            continue
        selected.append(item)
    return selected


def seed_private_manifest_from_strip_ocr(
    *,
    eval_root: Path,
    assignment_id: str,
    problem_id: str,
    grades_dir: Path,
    debug_fitch_dir: Path,
    manifest_path: Path | None = None,
    split: str = "discovery",
    statuses: Sequence[str] = ("accepted",),
    gold_status: str = "human_checked",
    copy_images: bool = True,
    allow_repo_paths: bool = False,
) -> List[ExportedPrivateCase]:
    """Create private eval cases from accepted/corrected strip OCR artifacts."""
    eval_root = Path(eval_root).expanduser().resolve()
    manifest_path = Path(manifest_path).expanduser().resolve() if manifest_path else eval_root / "manifest.yaml"
    _require_external_output(eval_root, allow_repo_paths=allow_repo_paths)
    _require_external_output(manifest_path, allow_repo_paths=allow_repo_paths)

    state_path = _state_path_for_problem(Path(grades_dir), problem_id)
    if not state_path.exists():
        raise FileNotFoundError(f"Fitch OCR review state not found: {state_path}")

    state = _load_json(state_path)
    if not isinstance(state, dict):
        raise ValueError(f"Fitch OCR review state must be an object: {state_path}")

    manifest = _load_yaml_mapping(manifest_path)
    manifest.setdefault(
        "suite",
        {
            "name": f"{_slug(assignment_id, fallback='assignment')}_private_tfl",
            "source": "seeded_from_strip_fitch_ocr_debug_artifacts",
        },
    )
    cases = _manifest_cases(manifest)
    manifest["cases"] = cases
    existing_ids = _existing_case_ids(cases)

    problem_slug = _slug(problem_id, fallback="problem")
    case_prefix = f"{_slug(assignment_id, fallback='assignment')}_{problem_slug}_case_"
    crop_dir = eval_root / "crops" / problem_slug
    gold_dir = eval_root / "gold" / problem_slug
    crop_dir.mkdir(parents=True, exist_ok=True)
    gold_dir.mkdir(parents=True, exist_ok=True)

    exported: List[ExportedPrivateCase] = []
    for item in _selected_state_items(state, statuses=statuses):
        slug = str(item.get("slug") or "").strip()
        status = str(item.get("status") or "").strip()
        if not slug:
            exported.append(
                ExportedPrivateCase(
                    case_id="",
                    source_slug="",
                    status=status,
                    image_path=Path(),
                    expected_path=Path(),
                    skipped=True,
                    reason="missing_slug",
                )
            )
            continue

        debug_case_dir = Path(debug_fitch_dir) / slug
        final_structure_path = debug_case_dir / "final_structure.json"
        source_image = _find_primary_proof_image(debug_case_dir, slug)
        if not final_structure_path.exists():
            exported.append(
                ExportedPrivateCase(
                    case_id="",
                    source_slug=slug,
                    status=status,
                    image_path=source_image or Path(),
                    expected_path=final_structure_path,
                    skipped=True,
                    reason="missing_final_structure",
                )
            )
            continue
        if source_image is None:
            exported.append(
                ExportedPrivateCase(
                    case_id="",
                    source_slug=slug,
                    status=status,
                    image_path=Path(),
                    expected_path=final_structure_path,
                    skipped=True,
                    reason="missing_proof_image",
                )
            )
            continue

        case_id = _next_case_id(existing_ids=existing_ids, prefix=case_prefix)
        target_image = crop_dir / f"{case_id}{source_image.suffix.lower()}"
        target_expected = gold_dir / f"{case_id}.json"

        if copy_images:
            shutil.copy2(source_image, target_image)
        else:
            target_image = source_image.resolve()

        final_structure = _load_json(final_structure_path)
        target_expected.write_text(
            json.dumps(final_structure, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        cases.append(
            {
                "id": case_id,
                "split": split,
                "image_path": _relative_manifest_path(target_image, manifest_path=manifest_path),
                "expected_structure_path": _relative_manifest_path(target_expected, manifest_path=manifest_path),
                "gold_status": gold_status,
                "oracle_kind": "gold_private_human_checked",
                "tags": ["student_scan", "tfl", "seeded_from_strip_fitch_ocr"],
            }
        )
        exported.append(
            ExportedPrivateCase(
                case_id=case_id,
                source_slug=slug,
                status=status,
                image_path=target_image,
                expected_path=target_expected,
            )
        )

    _dump_yaml(manifest_path, manifest)
    return exported


def render_export_summary(exported: Sequence[ExportedPrivateCase]) -> str:
    created = [item for item in exported if not item.skipped]
    skipped = [item for item in exported if item.skipped]
    lines = [
        f"Private TFL export created: {len(created)}",
        f"Private TFL export skipped: {len(skipped)}",
    ]
    for item in created:
        lines.append(f"- CREATED {item.case_id} from {item.source_slug} status={item.status}")
    if skipped:
        lines.append("Skipped:")
        for item in skipped:
            source = item.source_slug or "(unknown)"
            lines.append(f"- SKIPPED {source} status={item.status or '(unknown)'} reason={item.reason}")
    return "\n".join(lines)


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Seed private TFL/Fitch OCR eval data from local debug artifacts.")
    parser.add_argument(
        "--allow-repo-paths",
        action="store_true",
        help="Disable the outside-repo output guard. Intended only for fake tests.",
    )
    subparsers = parser.add_subparsers(dest="command")

    seed_parser = subparsers.add_parser("seed-from-strip", help="Seed cases from strip-page Fitch OCR review state.")
    seed_parser.add_argument("--eval-root", required=True, help="External private eval root.")
    seed_parser.add_argument("--assignment-id", required=True, help="Anonymous assignment id prefix, e.g. exam1_spring2026.")
    seed_parser.add_argument("--problem-id", required=True, help="Problem id whose TFL/Fitch OCR state should be exported.")
    seed_parser.add_argument("--grades-dir", required=True, help="Live DATA_DIR/grades folder.")
    seed_parser.add_argument("--debug-fitch-dir", required=True, help="Live DATA_DIR/debug_fitch folder.")
    seed_parser.add_argument("--manifest", default="", help="Manifest path. Defaults to <eval-root>/manifest.yaml.")
    seed_parser.add_argument("--split", default="discovery", help="Private split label for new cases.")
    seed_parser.add_argument(
        "--status",
        action="append",
        default=[],
        help="Review status to export. Defaults to accepted. May be repeated.",
    )
    seed_parser.add_argument("--gold-status", default="human_checked", help="Gold review status to write in manifest.")
    seed_parser.add_argument(
        "--reference-images",
        action="store_true",
        help="Reference existing debug crop images instead of copying them into eval-root.",
    )

    args = parser.parse_args()
    if args.command == "seed-from-strip":
        exported = seed_private_manifest_from_strip_ocr(
            eval_root=Path(args.eval_root),
            assignment_id=args.assignment_id,
            problem_id=args.problem_id,
            grades_dir=Path(args.grades_dir),
            debug_fitch_dir=Path(args.debug_fitch_dir),
            manifest_path=Path(args.manifest) if str(args.manifest or "").strip() else None,
            split=args.split,
            statuses=tuple(args.status or ["accepted"]),
            gold_status=args.gold_status,
            copy_images=not bool(args.reference_images),
            allow_repo_paths=bool(args.allow_repo_paths),
        )
        _safe_console_print(render_export_summary(exported))
        return 0

    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
