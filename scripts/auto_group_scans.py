# scripts/auto_group_scans.py
"""
Command-line tool to:
- run the VLM page classifier over a folder of raw scans,
- group pages into submissions by student (using filename order + high-confidence name hits),
- write CSV summaries,
- and emit a folder of renamed scan copies with canonical filenames.

This is intentionally **standalone** from the FastAPI grading app: it only depends on
`app.storage`, `app.vlm_client`, and `app.vlm_classifier`.
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

from app.storage import DEFAULT_ROSTER_PATH, load_roster  # type: ignore
from app.vlm_client import LMStudioQwenClient  # type: ignore
from app.vlm_classifier import classify_page  # type: ignore


# --- Helper types ------------------------------------------------------------

@dataclass
class SubmissionGroup:
    """A contiguous chunk of pages, inferred to be one student's submission."""

    group_index: int
    pages: List[Any]  # PagePrediction objects (duck-typed)
    inferred_student_id: Optional[str]
    inferred_canvas_name: Optional[str]
    flags: Set[str] = field(default_factory=set)
    notes: List[str] = field(default_factory=list)


# --- Filename / sorting utilities -------------------------------------------

EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


def natural_sort_key(path: Path) -> List[Any]:
    """
    Return a key for natural sorting of filenames, so that
    ..._1.jpg, ..._2.jpg, ..., ..._10.jpg is ordered numerically.
    """
    parts = re.split(r"(\d+)", path.name)
    key: List[Any] = []
    for part in parts:
        if part.isdigit():
            key.append(int(part))
        else:
            key.append(part.lower())
    return key


def slugify_name(name: Optional[str]) -> str:
    """
    Turn a Canvas-style 'Last, First' name into 'last_first'.
    If name is None, return empty string.
    """
    if not name:
        return ""
    # Replace commas with space, collapse whitespace, lower-case, replace spaces with underscores.
    s = name.replace(",", " ")
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s.replace(" ", "_")


def canonical_filename(
    quiz_id: str,
    student_id: Optional[str],
    canvas_name: Optional[str],
    page_index: int,
    unknown_index: Optional[int],
    orig_suffix: str,
) -> str:
    """
    Build a filename like:

      quiz1_berrysh_berry_sharon_p01.jpg
      quiz1_UNKNOWN01_p01.jpg
    """
    ext = orig_suffix.lower() if orig_suffix else ".jpg"

    if student_id is None:
        sid_part = f"UNKNOWN{unknown_index:02d}" if unknown_index is not None else "UNKNOWN"
        slug_part = ""
    else:
        sid_part = student_id
        slug = slugify_name(canvas_name)
        slug_part = f"_{slug}" if slug else ""

    return f"{quiz_id}_{sid_part}{slug_part}_p{page_index:02d}{ext}"


# --- Grouping logic ----------------------------------------------------------

def build_roster_index(roster: Iterable[Any]) -> Dict[str, str]:
    """
    Build a mapping student_id -> name from load_roster() output.
    """
    index: Dict[str, str] = {}
    for entry in roster:
        # RosterEntry has fields: student_id, name
        sid = getattr(entry, "student_id", None)
        name = getattr(entry, "name", None)
        if sid:
            index[sid] = name or ""
    return index


def group_pages(
    preds: List[Any],
    roster_by_id: Dict[str, str],
    min_high_conf: float = 0.85,
) -> List[SubmissionGroup]:
    """
    Group pages into submissions based on scan order and high-confidence name hits.

    Rules (v1):
    - We walk pages in filename order maintaining a single 'current' group.
    - When we see a high-confidence name for a NEW student_id, we start a new group.
    - When we see a high-confidence name for the SAME student_id, we keep appending.
    - Pages with no reliable name just attach to the current group.
    - After the pass, we flag students who show up in multiple groups as 'split_submission'.
    """
    groups: List[SubmissionGroup] = []
    current_group: Optional[SubmissionGroup] = None
    current_student_id: Optional[str] = None

    for p in preds:
        student_id = getattr(p, "student_id", None)
        confidence = float(getattr(p, "confidence", 0.0))
        name_visible = bool(getattr(p, "name_visible", False))

        is_high = student_id is not None and confidence >= min_high_conf

        if current_group is None:
            # Start the very first group
            inferred_sid = student_id if is_high else None
            inferred_name = roster_by_id.get(inferred_sid) if inferred_sid else None
            current_group = SubmissionGroup(
                group_index=len(groups) + 1,
                pages=[p],
                inferred_student_id=inferred_sid,
                inferred_canvas_name=inferred_name,
            )
            current_student_id = inferred_sid

            if not name_visible:
                current_group.flags.add("has_no_name_page")
            if student_id is None:
                current_group.flags.add("has_unmatched_page")
            elif not is_high:
                current_group.flags.add("has_low_conf_page")

            continue

        if is_high:
            if current_student_id is None:
                # We were in an anonymous group; now we've got a name. Attach & set.
                current_group.pages.append(p)
                current_group.inferred_student_id = student_id
                current_group.inferred_canvas_name = roster_by_id.get(student_id)
                current_student_id = student_id
            elif student_id == current_student_id:
                # Same student, high-confidence page
                current_group.pages.append(p)
            else:
                # High-confidence different student → boundary
                current_group.flags.add("name_switch_boundary")
                groups.append(current_group)

                current_group = SubmissionGroup(
                    group_index=len(groups) + 1,
                    pages=[p],
                    inferred_student_id=student_id,
                    inferred_canvas_name=roster_by_id.get(student_id),
                )
                current_student_id = student_id
        else:
            # No reliable name; just attach to current group
            current_group.pages.append(p)
            if not name_visible:
                current_group.flags.add("has_no_name_page")
            if student_id is None:
                current_group.flags.add("has_unmatched_page")
            else:
                current_group.flags.add("has_low_conf_page")

    # End for
    if current_group is not None:
        groups.append(current_group)

    # Second pass: mark split submissions
    student_to_groups: Dict[str, List[int]] = {}
    for g in groups:
        if g.inferred_student_id:
            student_to_groups.setdefault(g.inferred_student_id, []).append(g.group_index)

    for sid, idxs in student_to_groups.items():
        if len(idxs) > 1:
            for i in idxs:
                groups[i - 1].flags.add("split_submission")

    # Mark unknown groups (no name inferred at all)
    for g in groups:
        if g.inferred_student_id is None:
            g.flags.add("unknown_student")

    return groups


# --- CSV writing -------------------------------------------------------------

def write_page_predictions_csv(
    csv_path: Path,
    preds: List[Any],
    group_index_by_filename: Dict[str, int],
    groups_by_index: Dict[int, SubmissionGroup],
    min_high_conf: float,
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "page_index",
        "filename",
        "group_index",
        "name_visible",
        "raw_name",
        "quiz_version",
        "predicted_student_id",
        "predicted_canvas_name",
        "confidence",
        "reasoning",
        "alert_low_confidence",
        "alert_no_name",
        "alert_unmatched",
        "group_inferred_student_id",
        "group_inferred_canvas_name",
        "group_flags",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i, p in enumerate(preds, start=1):
            filename = getattr(p, "filename", None) or getattr(p, "path", None)
            # classify_page likely sets .filename to the basename
            if isinstance(filename, Path):
                filename = filename.name
            if filename is None:
                # Fallback: try 'name' attr
                filename = getattr(p, "name", f"page_{i:02d}")

            student_id = getattr(p, "student_id", None)
            confidence = float(getattr(p, "confidence", 0.0))
            name_visible = bool(getattr(p, "name_visible", False))
            raw_name = getattr(p, "raw_name", None)
            quiz_version = getattr(p, "quiz_version", None)
            reasoning = getattr(p, "reasoning", "")

            group_index = group_index_by_filename.get(filename)
            group = groups_by_index.get(group_index) if group_index is not None else None

            alert_low_conf = (
                name_visible and student_id is not None and confidence < min_high_conf
            )
            alert_no_name = not name_visible
            alert_unmatched = name_visible and student_id is None

            row = {
                "page_index": i,
                "filename": filename,
                "group_index": group_index,
                "name_visible": name_visible,
                "raw_name": raw_name,
                "quiz_version": quiz_version,
                "predicted_student_id": student_id,
                "predicted_canvas_name": getattr(group, "inferred_canvas_name", None),
                "confidence": confidence,
                "reasoning": reasoning,
                "alert_low_confidence": alert_low_conf,
                "alert_no_name": alert_no_name,
                "alert_unmatched": alert_unmatched,
                "group_inferred_student_id": getattr(group, "inferred_student_id", None),
                "group_inferred_canvas_name": getattr(group, "inferred_canvas_name", None),
                "group_flags": ";".join(sorted(getattr(group, "flags", []))) if group else "",
            }
            writer.writerow(row)


def write_groups_csv(
    csv_path: Path,
    groups: List[SubmissionGroup],
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "group_index",
        "page_filenames",
        "page_count",
        "inferred_student_id",
        "inferred_canvas_name",
        "min_confidence",
        "max_confidence",
        "flags",
        "notes",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for g in groups:
            confidences = [float(getattr(p, "confidence", 0.0)) for p in g.pages]
            filenames = []
            for p in g.pages:
                fn = getattr(p, "filename", None) or getattr(p, "path", None)
                if isinstance(fn, Path):
                    fn = fn.name
                if fn is None:
                    fn = getattr(p, "name", "")
                filenames.append(fn)

            row = {
                "group_index": g.group_index,
                "page_filenames": ";".join(filenames),
                "page_count": len(g.pages),
                "inferred_student_id": g.inferred_student_id,
                "inferred_canvas_name": g.inferred_canvas_name,
                "min_confidence": min(confidences) if confidences else None,
                "max_confidence": max(confidences) if confidences else None,
                "flags": ";".join(sorted(g.flags)),
                "notes": " ".join(g.notes),
            }
            writer.writerow(row)


# --- Main CLI ---------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Group and rename scanned quiz pages by student using a local VLM."
    )
    parser.add_argument("--quiz-id", required=True, help="Short identifier for this quiz, e.g. 'quiz1'.")
    parser.add_argument(
        "--raw-scans-dir",
        type=Path,
        required=True,
        help="Directory containing raw scan image files for ONE quiz.",
    )
    parser.add_argument(
        "--roster",
        type=Path,
        default=DEFAULT_ROSTER_PATH,
        help="Path to roster CSV.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Output directory for renamed scans and CSVs.",
    )
    parser.add_argument(
        "--predictions-csv",
        type=Path,
        default=None,
        help="Optional explicit path for per-page predictions CSV.",
    )
    parser.add_argument(
        "--groups-csv",
        type=Path,
        default=None,
        help="Optional explicit path for grouped submissions CSV.",
    )
    parser.add_argument(
        "--lmstudio-base-url",
        default="http://127.0.0.1:1234/v1",
        help="Base URL for LM Studio HTTP server.",
    )
    parser.add_argument(
        "--lmstudio-model",
        default="qwen/qwen3-vl-4b",
        help="Model identifier to send to LM Studio.",
    )
    parser.add_argument(
        "--min-high-confidence",
        type=float,
        default=0.85,
        help="Confidence threshold for treating a name match as 'high confidence'.",
    )

    args = parser.parse_args()

    quiz_id: str = args.quiz_id
    raw_scans_dir: Path = args.raw_scans_dir
    out_dir: Path = args.out_dir
    out_dir_scans: Path = out_dir / "scans"
    out_dir_scans.mkdir(parents=True, exist_ok=True)

    predictions_csv_path: Path = (
        args.predictions_csv
        if args.predictions_csv is not None
        else out_dir / f"{quiz_id}_page_predictions.csv"
    )
    groups_csv_path: Path = (
        args.groups_csv if args.groups_csv is not None else out_dir / f"{quiz_id}_groups.csv"
    )

    # 1. Load roster
    roster = load_roster(args.roster)
    roster_by_id = build_roster_index(roster)

    # 2. Build LM Studio client
    client = LMStudioQwenClient(
        base_url=args.lmstudio_base_url,
        model=args.lmstudio_model,
    )

    # 3. Collect image paths in natural-sorted order
    all_paths = [
        p
        for p in raw_scans_dir.iterdir()
        if p.is_file() and p.suffix in EXTS
    ]
    if not all_paths:
        raise SystemExit(f"No image files found in {raw_scans_dir}")

    all_paths.sort(key=natural_sort_key)

    print(f"Found {len(all_paths)} pages in {raw_scans_dir}")
    print(f"Running VLM classifier (min_high_conf={args.min_high_confidence})...")
    print()

    # 4. Run classifier per page
    preds: List[Any] = []
    for i, path in enumerate(all_paths, start=1):
        pred = classify_page(path, roster, client)
        # Make sure we have a filename attribute for downstream code / CSV
        if getattr(pred, "filename", None) is None:
            setattr(pred, "filename", path.name)
        preds.append(pred)

        print(f"{i:02d}. {path.name}")
        print(f"    name_visible   = {pred.name_visible}")
        print(f"    raw_name       = {getattr(pred, 'raw_name', None)!r}")
        print(f"    student_id     = {getattr(pred, 'student_id', None)!r}")
        print(f"    confidence     = {float(getattr(pred, 'confidence', 0.0)):.3f}")
        print(f"    reasoning      = {getattr(pred, 'reasoning', '')}")
        print()

    # 5. Group pages into submissions
    groups = group_pages(preds, roster_by_id, min_high_conf=args.min_high_confidence)

    # Build a quick index: filename -> group_index
    group_index_by_filename: Dict[str, int] = {}
    for g in groups:
        for p in g.pages:
            fn = getattr(p, "filename", None) or getattr(p, "path", None)
            if isinstance(fn, Path):
                fn = fn.name
            if fn is None:
                continue
            group_index_by_filename[fn] = g.group_index

    groups_by_index: Dict[int, SubmissionGroup] = {g.group_index: g for g in groups}

    # 6. Print group summary / alerts
    print("=== Group Summary ===")
    for g in groups:
        confidences = [float(getattr(p, "confidence", 0.0)) for p in g.pages]
        min_conf = min(confidences) if confidences else None
        max_conf = max(confidences) if confidences else None
        label = (
            f"{g.inferred_student_id} ({g.inferred_canvas_name})"
            if g.inferred_student_id
            else "UNKNOWN"
        )
        if min_conf is not None:
            msg = (
                f"Group {g.group_index:02d}: {label}, "
                f"pages={len(g.pages)}, "
                f"min_conf={min_conf:.3f} max_conf={max_conf:.3f}"
            )
        else:
            msg = f"Group {g.group_index:02d}: {label}, pages={len(g.pages)}"
        print(msg)
        if g.flags:
            print(f"    flags: {', '.join(sorted(g.flags))}")
    print()

    print("=== Alerts ===")
    any_alerts = False
    for g in groups:
        if not g.flags:
            continue
        any_alerts = True
        label = (
            f"{g.inferred_student_id} ({g.inferred_canvas_name})"
            if g.inferred_student_id
            else "UNKNOWN"
        )
        print(f"- Group {g.group_index:02d}: {label}")
        print(f"    flags: {', '.join(sorted(g.flags))}")
    if not any_alerts:
        print("No group-level alerts.")
    print()

    # 7. Write CSVs
    write_page_predictions_csv(
        predictions_csv_path,
        preds,
        group_index_by_filename,
        groups_by_index,
        min_high_conf=args.min_high_confidence,
    )
    write_groups_csv(groups_csv_path, groups)
    print(f"Wrote per-page predictions to {predictions_csv_path}")
    print(f"Wrote group summary to {groups_csv_path}")

    # 8. Copy / rename scans into out_dir/scans
    # Assign UNKNOWN indices for groups with no inferred_student_id
    unknown_index_by_group: Dict[int, int] = {}
    unknown_counter = 0
    for g in groups:
        if g.inferred_student_id is None:
            unknown_counter += 1
            unknown_index_by_group[g.group_index] = unknown_counter

    print(f"Writing renamed scan copies to {out_dir_scans} ...")

    for g in groups:
        unknown_index = unknown_index_by_group.get(g.group_index)
        # Page index is per-group, starting at 1
        for page_idx, p in enumerate(g.pages, start=1):
            fn = getattr(p, "filename", None) or getattr(p, "path", None)
            if isinstance(fn, Path):
                fn = fn.name
            if fn is None:
                continue

            orig_suffix = Path(fn).suffix
            src_path = raw_scans_dir / fn

            new_name = canonical_filename(
                quiz_id=quiz_id,
                student_id=g.inferred_student_id,
                canvas_name=g.inferred_canvas_name,
                page_index=page_idx,
                unknown_index=unknown_index,
                orig_suffix=orig_suffix,
            )
            dst_path = out_dir_scans / new_name

            # Use copy rather than rename, so we don't mutate the raw scans folder.
            data = src_path.read_bytes()
            dst_path.write_bytes(data)

    print("Done.")


if __name__ == "__main__":
    main()
