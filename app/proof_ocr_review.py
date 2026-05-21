from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional
from urllib.parse import quote

from PIL import Image

from app.verifier import ProofLine, check_proof, flatten_proof

FITCH_OCR_STATE_DIRNAME = "proof_ocr"
FITCH_OCR_KIND = "tfl_fitch"
TERMINAL_READY_STATUSES = {"needs_review", "verifier_issue", "accepted"}
RUNNING_STATUSES = {"queued", "processing"}

_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tfl_fitch_ocr")
_state_lock = threading.Lock()
_active_jobs: set[tuple[str, int]] = set()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clean_slug(raw: str, fallback: str = "proof_ocr") -> str:
    text = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(raw or "").strip())
    while "__" in text:
        text = text.replace("__", "_")
    return text.strip("_") or fallback


def fitch_ocr_state_path(grades_dir: Path, problem_id: str) -> Path:
    safe_problem = _clean_slug(problem_id, "problem")
    return Path(grades_dir) / FITCH_OCR_STATE_DIRNAME / f"{FITCH_OCR_KIND}_{safe_problem}.json"


def fitch_ocr_slug(problem_id: str, unit_id: int, page_filename: str) -> str:
    return _clean_slug(f"{problem_id}_{unit_id}_{Path(page_filename).stem}", "fitch_debug")


def load_fitch_ocr_state(grades_dir: Path, problem_id: str) -> Dict[str, Any]:
    path = fitch_ocr_state_path(grades_dir, problem_id)
    if not path.exists():
        return {
            "problem_id": problem_id,
            "kind": FITCH_OCR_KIND,
            "items": {},
            "updated_at": None,
        }
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        parsed = {}
    if not isinstance(parsed, dict):
        parsed = {}
    items = parsed.get("items")
    if not isinstance(items, dict):
        items = {}
    parsed["items"] = items
    parsed.setdefault("problem_id", problem_id)
    parsed.setdefault("kind", FITCH_OCR_KIND)
    parsed.setdefault("updated_at", None)
    return parsed


def save_fitch_ocr_state(grades_dir: Path, problem_id: str, state: Mapping[str, Any]) -> None:
    path = fitch_ocr_state_path(grades_dir, problem_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(state)
    payload["problem_id"] = problem_id
    payload["kind"] = FITCH_OCR_KIND
    payload["updated_at"] = _now_iso()
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def public_fitch_ocr_status(grades_dir: Path, problem_id: str) -> Dict[str, Any]:
    state = load_fitch_ocr_state(grades_dir, problem_id)
    raw_items = state.get("items") if isinstance(state.get("items"), dict) else {}
    items = []
    counts: Dict[str, int] = {}
    for key in sorted(raw_items, key=lambda value: int(value) if str(value).isdigit() else str(value)):
        item = raw_items.get(key)
        if not isinstance(item, dict):
            continue
        status = str(item.get("status") or "queued")
        counts[status] = counts.get(status, 0) + 1
        items.append(_public_item(item))
    return {
        "problem_id": problem_id,
        "kind": FITCH_OCR_KIND,
        "updated_at": state.get("updated_at"),
        "counts": counts,
        "items": items,
    }


def _public_item(item: Mapping[str, Any]) -> Dict[str, Any]:
    allowed = {
        "unit_id",
        "page_filename",
        "slug",
        "status",
        "message",
        "debug_url",
        "proof_image_url",
        "run_summary",
        "line_count",
        "updated_at",
        "started_at",
        "completed_at",
        "accepted_at",
        "preview_lines",
        "checker_issues",
        "reread_summary",
        "reread_records",
        "ocr_diagnostics",
    }
    return {key: item.get(key) for key in allowed if key in item}


def accept_fitch_ocr_review(grades_dir: Path, problem_id: str, unit_id: int) -> Dict[str, Any]:
    with _state_lock:
        state = load_fitch_ocr_state(grades_dir, problem_id)
        items = state.setdefault("items", {})
        item = items.get(str(unit_id))
        if not isinstance(item, dict):
            raise KeyError(f"No OCR review state for unit {unit_id}.")
        item["status"] = "accepted"
        item["message"] = "Accepted by grader."
        item["accepted_at"] = _now_iso()
        item["updated_at"] = item["accepted_at"]
        save_fitch_ocr_state(grades_dir, problem_id, state)
        return _public_item(item)


def enqueue_fitch_ocr_jobs(
    *,
    problem_id: str,
    targets: Iterable[Mapping[str, Any]],
    raw_scans_dir: Path,
    debug_fitch_dir: Path,
    grades_dir: Path,
    force: bool = False,
) -> Dict[str, Any]:
    queued = 0
    skipped = 0
    failed = 0
    submitted: List[Dict[str, Any]] = []

    with _state_lock:
        state = load_fitch_ocr_state(grades_dir, problem_id)
        items = state.setdefault("items", {})

        for target in targets:
            try:
                unit_id = int(target.get("unit_id"))
            except (TypeError, ValueError):
                failed += 1
                continue

            page_filename = str(target.get("page_filename") or "").strip()
            if not page_filename:
                skipped += 1
                continue

            key = str(unit_id)
            existing = items.get(key)
            existing_status = str(existing.get("status") or "") if isinstance(existing, dict) else ""
            if not force and existing_status in (RUNNING_STATUSES | TERMINAL_READY_STATUSES):
                skipped += 1
                continue

            image_path = Path(raw_scans_dir) / page_filename
            slug = fitch_ocr_slug(problem_id, unit_id, page_filename)
            target_dir = Path(debug_fitch_dir) / slug
            target_image = target_dir / f"{slug}{image_path.suffix.lower() or '.jpg'}"

            try:
                _write_cropped_image(
                    image_path=image_path,
                    target_image=target_image,
                    bbox=target.get("bbox"),
                )
            except Exception as exc:
                failed += 1
                items[key] = {
                    "unit_id": unit_id,
                    "page_filename": page_filename,
                    "slug": slug,
                    "status": "ocr_failed",
                    "message": f"Could not prepare proof crop: {exc}",
                    "updated_at": _now_iso(),
                    "debug_url": _debug_url(slug),
                }
                continue

            item = {
                "unit_id": unit_id,
                "page_filename": page_filename,
                "slug": slug,
                "status": "queued",
                "message": "Queued for TFL/Fitch OCR.",
                "updated_at": _now_iso(),
                "debug_url": _debug_url(slug),
                "proof_image_url": _debug_asset_url(slug, target_image.name),
            }
            items[key] = item
            submitted.append(
                {
                    "problem_id": problem_id,
                    "unit_id": unit_id,
                    "image_path": target_image,
                    "output_dir": target_dir,
                    "grades_dir": Path(grades_dir),
                    "debug_fitch_dir": Path(debug_fitch_dir),
                }
            )
            queued += 1

        save_fitch_ocr_state(grades_dir, problem_id, state)

    for job in submitted:
        submit_fitch_ocr_job(**job)

    return {
        "status": "ok",
        "queued": queued,
        "skipped": skipped,
        "failed": failed,
        "items": public_fitch_ocr_status(grades_dir, problem_id)["items"],
    }


def submit_fitch_ocr_job(
    *,
    problem_id: str,
    unit_id: int,
    image_path: Path,
    output_dir: Path,
    grades_dir: Path,
    debug_fitch_dir: Path,
) -> None:
    key = (problem_id, unit_id)
    with _state_lock:
        if key in _active_jobs:
            return
        _active_jobs.add(key)
    _executor.submit(
        _run_fitch_ocr_job,
        problem_id=problem_id,
        unit_id=unit_id,
        image_path=Path(image_path),
        output_dir=Path(output_dir),
        grades_dir=Path(grades_dir),
        debug_fitch_dir=Path(debug_fitch_dir),
    )


def _run_fitch_ocr_job(
    *,
    problem_id: str,
    unit_id: int,
    image_path: Path,
    output_dir: Path,
    grades_dir: Path,
    debug_fitch_dir: Path,
) -> None:
    try:
        _update_item(
            grades_dir,
            problem_id,
            unit_id,
            {
                "status": "processing",
                "message": "OCR is running.",
                "started_at": _now_iso(),
            },
        )

        from proofs_ocr.pipelines.tfl_fitch import run_fitch_ocr_pipeline

        result = run_fitch_ocr_pipeline(image_path=image_path, output_dir=output_dir)
        final_structure = _load_json_file(output_dir / "final_structure.json", default=[])
        checker_issues = _checker_issues(final_structure)
        reread_summary, reread_records = _load_reread_reports(output_dir, debug_fitch_dir)
        ocr_diagnostics = _summarize_vlm_diagnostics(output_dir)
        has_reread_flags = bool((reread_summary or {}).get("record_count"))
        next_status = "verifier_issue" if checker_issues or has_reread_flags else "needs_review"
        message = "OCR ready for review."
        if next_status == "verifier_issue":
            message = "OCR ready with verifier/reread flags."

        _update_item(
            grades_dir,
            problem_id,
            unit_id,
            {
                "status": next_status,
                "message": message,
                "completed_at": _now_iso(),
                "run_summary": str(result.get("run_summary") or ""),
                "line_count": result.get("line_count"),
                "preview_lines": _preview_lines(final_structure),
                "checker_issues": checker_issues,
                "reread_summary": reread_summary,
                "reread_records": reread_records,
                "ocr_diagnostics": ocr_diagnostics,
            },
        )
    except Exception as exc:  # noqa: BLE001 - background job state must record failures.
        _update_item(
            grades_dir,
            problem_id,
            unit_id,
            {
                "status": "ocr_failed",
                "message": str(exc),
                "completed_at": _now_iso(),
            },
        )
    finally:
        with _state_lock:
            _active_jobs.discard((problem_id, unit_id))


def _update_item(grades_dir: Path, problem_id: str, unit_id: int, updates: Mapping[str, Any]) -> None:
    with _state_lock:
        state = load_fitch_ocr_state(grades_dir, problem_id)
        items = state.setdefault("items", {})
        item = items.setdefault(str(unit_id), {"unit_id": unit_id})
        item.update(updates)
        item["updated_at"] = _now_iso()
        save_fitch_ocr_state(grades_dir, problem_id, state)


def _write_cropped_image(*, image_path: Path, target_image: Path, bbox: Any) -> None:
    if not image_path.exists():
        raise FileNotFoundError(f"Scan image not found: {image_path.name}")

    target_image.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(image_path) as im:
        working = im.convert("RGB")
        normalized = _coerce_bbox(bbox)
        if normalized:
            width, height = working.size
            x1 = max(0, min(width - 1, int(round(normalized[0] * width))))
            y1 = max(0, min(height - 1, int(round(normalized[1] * height))))
            x2 = max(x1 + 1, min(width, int(round(normalized[2] * width))))
            y2 = max(y1 + 1, min(height, int(round(normalized[3] * height))))
            working = working.crop((x1, y1, x2, y2))
        working.save(target_image)


def _coerce_bbox(value: Any) -> Optional[List[float]]:
    if not isinstance(value, list) or len(value) != 4:
        return None
    try:
        x1, y1, x2, y2 = [float(v) for v in value]
    except (TypeError, ValueError):
        return None
    if 0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1:
        return [x1, y1, x2, y2]
    return None


def _summarize_vlm_diagnostics(output_dir: Path) -> Dict[str, Any]:
    metadata_files = sorted(Path(output_dir).glob("response_metadata_*.json"))
    if not metadata_files:
        return {}

    call_count = 0
    total_elapsed = 0.0
    max_elapsed = 0.0
    slowest_node = ""
    thinking_chars = 0
    content_chars = 0
    finish_reasons: Dict[str, int] = {}

    for path in metadata_files:
        data = _load_json_file(path, default={})
        if not isinstance(data, dict):
            continue
        call_count += 1
        try:
            elapsed = float(data.get("elapsed_s") or 0.0)
        except (TypeError, ValueError):
            elapsed = 0.0
        total_elapsed += elapsed
        if elapsed > max_elapsed:
            max_elapsed = elapsed
            slowest_node = path.stem.removeprefix("response_metadata_")

        try:
            thinking_chars += int(data.get("thinking_chars") or 0)
        except (TypeError, ValueError):
            pass
        try:
            content_chars += int(data.get("content_chars") or 0)
        except (TypeError, ValueError):
            pass

        finish = str(data.get("finish_reason") or "unknown").strip() or "unknown"
        finish_reasons[finish] = finish_reasons.get(finish, 0) + 1

    error_count = len(list(Path(output_dir).glob("error_*.txt")))
    thinking_file_count = len(list(Path(output_dir).glob("thinking_*.txt")))
    note_parts = [f"VLM calls {call_count}", f"elapsed {total_elapsed:.1f}s"]
    if slowest_node:
        note_parts.append(f"slowest node {slowest_node} {max_elapsed:.1f}s")
    if thinking_chars:
        note_parts.append(f"thinking {thinking_chars} chars")
    if error_count:
        note_parts.append(f"errors {error_count}")

    return {
        "call_count": call_count,
        "total_elapsed_s": round(total_elapsed, 3),
        "max_elapsed_s": round(max_elapsed, 3),
        "slowest_node": slowest_node,
        "thinking_chars": thinking_chars,
        "thinking_file_count": thinking_file_count,
        "content_chars": content_chars,
        "finish_reasons": finish_reasons,
        "error_count": error_count,
        "note": "; ".join(note_parts),
    }


def _debug_url(slug: str) -> str:
    return f"/debug/fitch?scan={quote(slug)}"


def _debug_asset_url(slug: str, filename: str) -> str:
    return f"/debug_fitch_assets/{quote(slug)}/{quote(filename)}"


def _asset_url_for_path(path_text: Any, debug_fitch_dir: Path) -> Optional[str]:
    text = str(path_text or "").strip()
    if not text:
        return None
    path = Path(text)
    try:
        rel = path.resolve().relative_to(Path(debug_fitch_dir).resolve())
    except Exception:
        return None
    return "/debug_fitch_assets/" + "/".join(quote(part) for part in rel.parts)


def _load_json_file(path: Path, *, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _preview_lines(structure: Any) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    def walk(node: Any, depth: int) -> None:
        if isinstance(node, list):
            for child in node:
                walk(child, depth + 1 if isinstance(child, list) else depth)
        elif isinstance(node, dict):
            rows.append(
                {
                    "line": node.get("line") if isinstance(node.get("line"), int) else len(rows) + 1,
                    "depth": max(0, depth),
                    "wffstr": str(node.get("wffstr") or "").strip(),
                    "jstr": str(node.get("jstr") or "").strip(),
                }
            )

    walk(structure, 0)
    return rows


def _checker_issues(structure: Any) -> List[str]:
    if not isinstance(structure, list):
        return []
    try:
        proof_lines = _structure_to_proof_lines(structure)
        flattened = flatten_proof(proof_lines, [])
        conclusion = flattened[-1].wffstr if flattened else ""
        result = check_proof(proof_lines, numprems=_count_premises(structure), conc=conclusion)
        return [str(issue) for issue in result.issues]
    except Exception as exc:  # noqa: BLE001 - verifier failures should not hide OCR.
        return [f"Verifier unavailable: {exc}"]


def _structure_to_proof_lines(structure: List[Any]) -> List[Any]:
    proof_lines: List[Any] = []
    for item in structure:
        if isinstance(item, list):
            proof_lines.append(_structure_to_proof_lines(item))
        elif isinstance(item, dict):
            proof_lines.append(
                ProofLine(
                    wffstr=str(item.get("wffstr", "")).strip(),
                    jstr=str(item.get("jstr", "")).strip(),
                    location=[],
                )
            )
    return proof_lines


def _count_premises(structure: List[Any]) -> int:
    count = 0
    for item in structure:
        if isinstance(item, list):
            break
        if not isinstance(item, dict):
            continue
        jstr = str(item.get("jstr") or "").strip().lower()
        if jstr in {"pr", "premise"}:
            count += 1
            continue
        break
    return count


def _load_reread_reports(output_dir: Path, debug_fitch_dir: Path) -> tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    summary = _load_json_file(output_dir / "verifier_line_reread_summary.json", default=None)
    if not isinstance(summary, dict):
        return None, []

    raw_records = summary.get("records")
    if not isinstance(raw_records, list):
        return summary, []

    candidates = _load_json_file(output_dir / "verifier_line_reread_candidates.json", default=[])
    crop_urls_by_line: Dict[tuple[Any, Any], str] = {}
    if isinstance(candidates, list):
        for record in candidates:
            if not isinstance(record, dict):
                continue
            url = _asset_url_for_path(record.get("row_crop"), debug_fitch_dir)
            if not url:
                continue
            crop_urls_by_line[(record.get("line"), record.get("target_kind"))] = url

    public_records: List[Dict[str, Any]] = []
    for record in raw_records:
        if not isinstance(record, dict):
            continue
        original = record.get("original") if isinstance(record.get("original"), dict) else {}
        candidate = record.get("candidate") if isinstance(record.get("candidate"), dict) else {}
        public_records.append(
            {
                "line": record.get("line"),
                "target_kind": record.get("target_kind"),
                "candidate_class": record.get("candidate_class"),
                "reason": record.get("reason"),
                "original": {
                    "wffstr": str(original.get("wffstr") or "").strip(),
                    "jstr": str(original.get("jstr") or "").strip(),
                },
                "candidate": {
                    "wffstr": str(candidate.get("wffstr") or "").strip(),
                    "jstr": str(candidate.get("jstr") or "").strip(),
                },
                "row_crop_url": crop_urls_by_line.get((record.get("line"), record.get("target_kind"))),
            }
        )

    public_summary = {
        "record_count": summary.get("record_count", len(public_records)),
        "candidate_class_counts": summary.get("candidate_class_counts", {}),
        "gold_metrics_available": bool(summary.get("gold_metrics_available")),
    }
    return public_summary, public_records
