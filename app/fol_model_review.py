from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional
from urllib.parse import quote

from PIL import Image

from app.autograde import AutoGradeResult, autograde_submission

FOL_MODEL_STATE_DIRNAME = "fol_model_ocr"
FOL_MODEL_KIND = "fol_model"
RUNNING_STATUSES = {"queued", "processing"}
READY_STATUSES = {"needs_review", "checker_passed", "checker_failed", "invalid_input"}

_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="fol_model_ocr")
_state_lock = threading.Lock()
_active_jobs: set[tuple[str, int]] = set()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clean_slug(raw: str, fallback: str = "fol_model") -> str:
    text = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(raw or "").strip())
    while "__" in text:
        text = text.replace("__", "_")
    return text.strip("_") or fallback


def fol_model_state_path(grades_dir: Path, problem_id: str) -> Path:
    safe_problem = _clean_slug(problem_id, "problem")
    return Path(grades_dir) / FOL_MODEL_STATE_DIRNAME / f"{FOL_MODEL_KIND}_{safe_problem}.json"


def fol_model_slug(problem_id: str, unit_id: int, page_filename: str) -> str:
    return _clean_slug(f"{problem_id}_{unit_id}_{Path(page_filename).stem}", "fol_model")


def load_fol_model_state(grades_dir: Path, problem_id: str) -> Dict[str, Any]:
    path = fol_model_state_path(grades_dir, problem_id)
    if not path.exists():
        return {
            "problem_id": problem_id,
            "kind": FOL_MODEL_KIND,
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
    parsed.setdefault("kind", FOL_MODEL_KIND)
    parsed.setdefault("updated_at", None)
    return parsed


def save_fol_model_state(grades_dir: Path, problem_id: str, state: Mapping[str, Any]) -> None:
    path = fol_model_state_path(grades_dir, problem_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(state)
    payload["problem_id"] = problem_id
    payload["kind"] = FOL_MODEL_KIND
    payload["updated_at"] = _now_iso()
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def public_fol_model_status(grades_dir: Path, problem_id: str) -> Dict[str, Any]:
    state = load_fol_model_state(grades_dir, problem_id)
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
        "kind": FOL_MODEL_KIND,
        "updated_at": state.get("updated_at"),
        "counts": counts,
        "items": items,
    }


def public_fol_item(grades_dir: Path, problem_id: str, unit_id: int) -> Dict[str, Any]:
    state = load_fol_model_state(grades_dir, problem_id)
    item = state.get("items", {}).get(str(unit_id))
    if not isinstance(item, dict):
        raise KeyError(f"No FOL model review state for unit {unit_id}.")
    return _public_item(item)


def _public_item(item: Mapping[str, Any]) -> Dict[str, Any]:
    allowed = {
        "unit_id",
        "page_filename",
        "slug",
        "status",
        "message",
        "debug_url",
        "image_url",
        "model_text",
        "passed",
        "summary",
        "suggested_score",
        "suggested_comment",
        "trace_blocks",
        "requirement_text",
        "checker_error",
        "updated_at",
        "started_at",
        "completed_at",
    }
    return {key: item.get(key) for key in allowed if key in item}


def enqueue_fol_model_ocr_jobs(
    *,
    problem_id: str,
    targets: Iterable[Mapping[str, Any]],
    raw_scans_dir: Path,
    debug_fol_model_dir: Path,
    grades_dir: Path,
    autograde_config: Mapping[str, Any],
    force: bool = False,
) -> Dict[str, Any]:
    queued = 0
    skipped = 0
    failed = 0
    submitted: List[Dict[str, Any]] = []
    requirement_text = requirement_text_from_config(autograde_config)

    with _state_lock:
        state = load_fol_model_state(grades_dir, problem_id)
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
            if not force and existing_status in (RUNNING_STATUSES | READY_STATUSES):
                skipped += 1
                continue

            image_path = Path(raw_scans_dir) / page_filename
            slug = fol_model_slug(problem_id, unit_id, page_filename)
            target_dir = Path(debug_fol_model_dir) / slug
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
                    "message": f"Could not prepare model crop: {exc}",
                    "requirement_text": requirement_text,
                    "updated_at": _now_iso(),
                    "debug_url": _debug_url(slug),
                }
                continue

            item = {
                "unit_id": unit_id,
                "page_filename": page_filename,
                "slug": slug,
                "status": "queued",
                "message": "Queued for FOL model OCR.",
                "requirement_text": requirement_text,
                "updated_at": _now_iso(),
                "debug_url": _debug_url(slug),
                "image_url": _debug_asset_url(slug, target_image.name),
                "model_text": "",
            }
            items[key] = item
            _write_debug_state(target_dir, item, autograde_config)
            submitted.append(
                {
                    "problem_id": problem_id,
                    "unit_id": unit_id,
                    "image_path": target_image,
                    "output_dir": target_dir,
                    "grades_dir": Path(grades_dir),
                    "autograde_config": dict(autograde_config),
                }
            )
            queued += 1

        save_fol_model_state(grades_dir, problem_id, state)

    for job in submitted:
        submit_fol_model_ocr_job(**job)

    return {
        "status": "ok",
        "queued": queued,
        "skipped": skipped,
        "failed": failed,
        "items": public_fol_model_status(grades_dir, problem_id)["items"],
    }


def submit_fol_model_ocr_job(
    *,
    problem_id: str,
    unit_id: int,
    image_path: Path,
    output_dir: Path,
    grades_dir: Path,
    autograde_config: Mapping[str, Any],
) -> None:
    key = (problem_id, unit_id)
    with _state_lock:
        if key in _active_jobs:
            return
        _active_jobs.add(key)
    _executor.submit(
        _run_fol_model_ocr_job,
        problem_id=problem_id,
        unit_id=unit_id,
        image_path=Path(image_path),
        output_dir=Path(output_dir),
        grades_dir=Path(grades_dir),
        autograde_config=dict(autograde_config),
    )


def _run_fol_model_ocr_job(
    *,
    problem_id: str,
    unit_id: int,
    image_path: Path,
    output_dir: Path,
    grades_dir: Path,
    autograde_config: Mapping[str, Any],
) -> None:
    try:
        _update_item(
            grades_dir,
            problem_id,
            unit_id,
            {"status": "processing", "message": "OCR is running.", "started_at": _now_iso()},
        )

        from scripts.debug_fol_model_ocr import transcribe_image_to_canonical_text

        _, model_text = transcribe_image_to_canonical_text(
            image_path=image_path,
            output_dir=output_dir,
            autograde_config=dict(autograde_config),
        )
        item_updates = _updates_from_checker_result(
            model_text=model_text,
            result=autograde_submission(
                problem_type="fol_model",
                answer_text=model_text,
                autograde_config=dict(autograde_config),
                runtime_options={"trace_mode": "terse"},
            ),
            autograde_config=autograde_config,
        )
        item_updates.update({"completed_at": _now_iso()})
        _update_item(grades_dir, problem_id, unit_id, item_updates)
        _sync_debug_state(output_dir, model_text, autograde_config)
    except Exception as exc:  # noqa: BLE001 - background job state must record failures.
        _update_item(
            grades_dir,
            problem_id,
            unit_id,
            {
                "status": "ocr_failed",
                "message": str(exc),
                "checker_error": str(exc),
                "completed_at": _now_iso(),
            },
        )
    finally:
        with _state_lock:
            _active_jobs.discard((problem_id, unit_id))


def check_fol_model_text(
    *,
    grades_dir: Path,
    problem_id: str,
    unit_id: int,
    model_text: str,
    autograde_config: Mapping[str, Any],
    trace_mode: str = "terse",
) -> Dict[str, Any]:
    result = autograde_submission(
        problem_type="fol_model",
        answer_text=model_text,
        autograde_config=dict(autograde_config),
        runtime_options={"trace_mode": trace_mode},
    )
    updates = _updates_from_checker_result(
        model_text=model_text,
        result=result,
        autograde_config=autograde_config,
    )
    _update_item(grades_dir, problem_id, unit_id, updates)
    return public_fol_item(grades_dir, problem_id, unit_id)


def seed_fol_model_item(
    *,
    grades_dir: Path,
    problem_id: str,
    unit_id: int,
    page_filename: str,
    autograde_config: Mapping[str, Any],
    model_text: str = "",
) -> Dict[str, Any]:
    with _state_lock:
        state = load_fol_model_state(grades_dir, problem_id)
        items = state.setdefault("items", {})
        item = items.setdefault(str(unit_id), {"unit_id": unit_id})
        item.update(
            {
                "unit_id": unit_id,
                "page_filename": page_filename,
                "status": "needs_review" if model_text.strip() else "not_started",
                "message": "Enter or OCR a proposed model.",
                "model_text": model_text,
                "requirement_text": requirement_text_from_config(autograde_config),
                "updated_at": _now_iso(),
            }
        )
        save_fol_model_state(grades_dir, problem_id, state)
        return _public_item(item)


def _updates_from_checker_result(
    *,
    model_text: str,
    result: AutoGradeResult,
    autograde_config: Mapping[str, Any],
) -> Dict[str, Any]:
    trace_blocks = [
        {"title": block.title, "result": block.result, "body": block.body}
        for block in result.trace_blocks
    ]
    if result.status == "scored":
        status = "checker_passed" if result.passed else "checker_failed"
        message = "Checker passed." if result.passed else "Checker found this model does not satisfy the target."
    else:
        status = "invalid_input"
        message = result.summary or "Checker could not parse this model."

    return {
        "status": status,
        "message": message,
        "model_text": model_text,
        "passed": result.passed,
        "summary": result.summary,
        "suggested_score": result.score,
        "suggested_comment": result.draft_comment,
        "trace_blocks": trace_blocks,
        "checker_error": "; ".join(result.errors),
        "requirement_text": requirement_text_from_config(autograde_config),
        "updated_at": _now_iso(),
    }


def requirement_text_from_config(autograde_config: Mapping[str, Any]) -> str:
    task_kind = str(autograde_config.get("task_kind") or "truth_value").strip() or "truth_value"
    prompt_data = autograde_config.get("prompt_data") if isinstance(autograde_config.get("prompt_data"), dict) else {}
    if task_kind == "non_validity":
        premises = prompt_data.get("premises") if isinstance(prompt_data, dict) else []
        conclusion = prompt_data.get("conclusion") if isinstance(prompt_data, dict) else ""
        premise_count = len(premises) if isinstance(premises, list) else 0
        return f"Need: all {premise_count} premise(s) true and the conclusion false ({conclusion})."
    if task_kind == "non_equivalence":
        return "Need: the two target sentences have different truth values."
    if task_kind == "joint_consistency":
        return "Need: every target sentence true in one model."
    if task_kind in {"make_false", "non_logical_falsehood"}:
        return "Need: every target sentence false in this model."
    return "Need: every target sentence true in this model."


def task_details_from_config(autograde_config: Mapping[str, Any]) -> List[str]:
    task_kind = str(autograde_config.get("task_kind") or "truth_value").strip() or "truth_value"
    prompt_data = autograde_config.get("prompt_data") if isinstance(autograde_config.get("prompt_data"), dict) else {}
    details = [requirement_text_from_config(autograde_config)]
    if not isinstance(prompt_data, dict):
        return details

    if task_kind == "non_validity":
        premises = prompt_data.get("premises")
        if isinstance(premises, list) and premises:
            details.append("Premises: " + "; ".join(str(item) for item in premises))
        conclusion = prompt_data.get("conclusion")
        if conclusion:
            details.append("Conclusion: " + str(conclusion))
        return details

    formulas = prompt_data.get("formulas")
    if isinstance(formulas, list) and formulas:
        details.append("Targets: " + "; ".join(str(item) for item in formulas))
    return details


def _update_item(grades_dir: Path, problem_id: str, unit_id: int, updates: Mapping[str, Any]) -> None:
    with _state_lock:
        state = load_fol_model_state(grades_dir, problem_id)
        items = state.setdefault("items", {})
        item = items.setdefault(str(unit_id), {"unit_id": unit_id})
        item.update(updates)
        item["updated_at"] = _now_iso()
        save_fol_model_state(grades_dir, problem_id, state)


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


def _debug_url(slug: str) -> str:
    return f"/debug/fol-model?scan={quote(slug)}"


def _debug_asset_url(slug: str, filename: str) -> str:
    return f"/debug_fol_model_assets/{quote(slug)}/{quote(filename)}"


def _write_debug_state(target_dir: Path, item: Mapping[str, Any], autograde_config: Mapping[str, Any]) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    state = {
        "problem_type": str(autograde_config.get("task_kind") or "truth_value"),
        "problem_id": "",
        "problem_label": "FOL model strip",
        "problem_setup_text": _yaml_dump(autograde_config),
        "model_text": str(item.get("model_text") or ""),
        "sentences_text": "",
        "trace_blocks": [],
        "draft_comment": "",
        "draft_grade": "",
        "checker_enabled": True,
        "trace_mode": "terse",
        "ocr_log": "",
        "ocr_response_text": "",
    }
    (target_dir / "state.json").write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")


def _sync_debug_state(output_dir: Path, model_text: str, autograde_config: Mapping[str, Any]) -> None:
    state_path = output_dir / "state.json"
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except Exception:
        state = {}
    if not isinstance(state, dict):
        state = {}
    state["model_text"] = model_text
    state["problem_setup_text"] = _yaml_dump(autograde_config)
    state_path.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")


def _yaml_dump(value: Mapping[str, Any]) -> str:
    try:
        import yaml

        return yaml.safe_dump(dict(value), sort_keys=False, allow_unicode=True).strip() + "\n"
    except Exception:
        return json.dumps(dict(value), indent=2, ensure_ascii=False)
