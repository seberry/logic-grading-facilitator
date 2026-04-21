from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import UploadFile


def clean_example_slug(raw: Optional[str], starter_example: str) -> str:
    text = str(raw or "").strip().lower()
    text = re.sub(r"[\s\-–]+", "_", text)
    text = re.sub(r"[^a-z0-9_]", "", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or starter_example


def list_modal_proof_examples(debug_dir: Path) -> list[str]:
    if not debug_dir.exists():
        return []
    return sorted([path.name for path in debug_dir.iterdir() if path.is_dir()], key=str.lower)


def state_path(debug_dir: Path, example_slug: str, state_filename: str) -> Path:
    return debug_dir / example_slug / state_filename


def default_page_state(default_proof_json_text: str) -> Dict[str, Any]:
    return {
        "system": "K",
        "include_derived_rules": False,
        "conclusion": "A",
        "vision_base_url": "",
        "vision_model": "",
        "verifier_vision_model": "",
        "ocr_pipeline": "legacy_modal_debug",
        "enable_formula_justification_separation": False,
        "fixture_id": "",
        "source_image_name": "",
        "source_image_path": "",
        "proof_json_text": default_proof_json_text,
        "original_proof_json_text": default_proof_json_text,
        "verifier_rewrite_summary": [],
        "review_decisions": {},
        "draft_comment": "",
        "draft_grade": "",
        "checker_request": None,
        "checker_response": None,
        "run_summary": "Edit the starter proof JSON and run the modal checker.",
        "last_run_at": "",
        "status": "",
        "message": "",
        "checker_error": "",
    }


def parse_review_decisions(raw: Any) -> Dict[str, Dict[str, str]]:
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return {}
        try:
            raw = json.loads(text)
        except json.JSONDecodeError:
            return {}

    if not isinstance(raw, dict):
        return {}

    parsed: Dict[str, Dict[str, str]] = {}
    for key, value in raw.items():
        line_key = str(key or "").strip()
        if not line_key or not isinstance(value, dict):
            continue
        mode = str(value.get("mode") or "").strip()
        if mode not in {"keep_current", "candidate", "manual"}:
            continue
        parsed[line_key] = {
            "mode": mode,
            "formula": str(value.get("formula") or "").strip(),
            "source_key": str(value.get("source_key") or "").strip(),
        }
    return parsed


def load_example_state(
    debug_dir: Path,
    example_slug: Optional[str],
    state_filename: str,
    default_proof_json_text: str,
) -> Dict[str, Any]:
    state = default_page_state(default_proof_json_text)
    if not example_slug:
        return state

    path = state_path(debug_dir, example_slug, state_filename)
    if not path.exists():
        return state

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return state

    for key in (
        "system",
        "include_derived_rules",
        "conclusion",
        "proof_json_text",
        "original_proof_json_text",
        "vision_base_url",
        "vision_model",
        "verifier_vision_model",
        "ocr_pipeline",
        "enable_formula_justification_separation",
        "fixture_id",
        "source_image_name",
        "source_image_path",
        "verifier_rewrite_summary",
        "review_decisions",
        "draft_comment",
        "draft_grade",
        "checker_request",
        "checker_response",
        "run_summary",
        "last_run_at",
    ):
        if key in raw:
            state[key] = raw[key]
    return state


def save_example_state(
    debug_dir: Path,
    example_slug: str,
    state: Dict[str, Any],
    state_filename: str,
    default_proof_json_text: str,
) -> None:
    path = state_path(debug_dir, example_slug, state_filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "system": state.get("system", "K"),
                "include_derived_rules": bool(state.get("include_derived_rules", False)),
                "conclusion": state.get("conclusion", ""),
                "vision_base_url": state.get("vision_base_url", ""),
                "vision_model": state.get("vision_model", ""),
                "verifier_vision_model": state.get("verifier_vision_model", ""),
                "ocr_pipeline": state.get("ocr_pipeline", "legacy_modal_debug"),
                "enable_formula_justification_separation": bool(
                    state.get("enable_formula_justification_separation", False)
                ),
                "fixture_id": state.get("fixture_id", ""),
                "source_image_name": state.get("source_image_name", ""),
                "source_image_path": state.get("source_image_path", ""),
                "proof_json_text": state.get("proof_json_text", default_proof_json_text),
                "original_proof_json_text": state.get("original_proof_json_text", default_proof_json_text),
                "verifier_rewrite_summary": state.get("verifier_rewrite_summary") or [],
                "review_decisions": state.get("review_decisions") or {},
                "draft_comment": state.get("draft_comment", ""),
                "draft_grade": state.get("draft_grade", ""),
                "checker_request": state.get("checker_request"),
                "checker_response": state.get("checker_response"),
                "run_summary": state.get("run_summary", ""),
                "last_run_at": state.get("last_run_at", ""),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def state_from_form(
    *,
    example: Optional[str],
    system: str,
    include_derived_rules: bool,
    conclusion: str,
    vision_base_url: str,
    vision_model: str,
    verifier_vision_model: str,
    ocr_pipeline: str,
    enable_formula_justification_separation: bool,
    fixture_id: str,
    proof_json_text: str,
    review_decisions_json: str,
    draft_comment: str,
    draft_grade: str,
    clean_slug,
    load_state,
    parse_decisions,
) -> tuple[str, Dict[str, Any]]:
    example_slug = clean_slug(example)
    state = load_state(example_slug)
    state.update(
        {
            "system": str(system or "K").strip(),
            "include_derived_rules": bool(include_derived_rules),
            "conclusion": str(conclusion or "").strip(),
            "vision_base_url": str(vision_base_url or "").strip(),
            "vision_model": str(vision_model or "").strip(),
            "verifier_vision_model": str(verifier_vision_model or "").strip(),
            "ocr_pipeline": str(ocr_pipeline or "legacy_modal_debug").strip() or "legacy_modal_debug",
            "enable_formula_justification_separation": bool(enable_formula_justification_separation),
            "fixture_id": str(fixture_id or "").strip(),
            "proof_json_text": proof_json_text,
            "review_decisions": parse_decisions(review_decisions_json),
            "draft_comment": draft_comment,
            "draft_grade": draft_grade,
            "status": "",
            "message": "",
            "checker_error": "",
        }
    )
    return example_slug, state


def build_autograde_config(state: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "system": state.get("system", "K"),
        "include_derived_rules": bool(state.get("include_derived_rules", False)),
        "conclusion": state.get("conclusion") or None,
    }


def find_saved_source_image(debug_dir: Path, example_slug: str, state: Dict[str, Any]) -> Optional[Path]:
    source_image_name = str(state.get("source_image_name") or "").strip()
    example_dir = debug_dir / example_slug
    if source_image_name:
        candidate = example_dir / source_image_name
        if candidate.exists() and candidate.is_file():
            return candidate
    candidates = sorted(example_dir.iterdir(), key=lambda path: path.name.lower()) if example_dir.exists() else []
    for candidate in candidates:
        if candidate.is_file() and candidate.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
            return candidate
    return None


def resolve_source_image_path(
    debug_dir: Path,
    raw_scans_dir: Path,
    example_slug: str,
    raw_path: str,
) -> Optional[Path]:
    text = str(raw_path or "").strip()
    if not text:
        return None
    candidate = Path(text)
    if candidate.is_absolute():
        return candidate

    example_candidate = (debug_dir / example_slug / candidate).resolve()
    if example_candidate.exists():
        return example_candidate

    raw_scans_candidate = (raw_scans_dir / candidate).resolve()
    if raw_scans_candidate.exists():
        return raw_scans_candidate

    return raw_scans_candidate


async def persist_uploaded_source_image(
    debug_dir: Path,
    example_slug: str,
    proof_image: UploadFile | None,
) -> Optional[Path]:
    if proof_image is None or not getattr(proof_image, "filename", ""):
        return None

    suffix = Path(proof_image.filename or "").suffix.lower() or ".png"
    if suffix not in {".png", ".jpg", ".jpeg", ".webp"}:
        raise ValueError("OCR image must be PNG, JPG, JPEG, or WEBP")

    example_dir = debug_dir / example_slug
    example_dir.mkdir(parents=True, exist_ok=True)
    target_path = example_dir / f"source_image{suffix}"
    with target_path.open("wb") as handle:
        shutil.copyfileobj(proof_image.file, handle)
    return target_path
