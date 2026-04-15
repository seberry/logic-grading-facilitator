from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import quote

from fastapi import APIRouter, File, Form, Request, Response, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from app.autograde import AutoGradeResult, autograde_submission
from app.storage import RAW_SCANS_DIR
from app.storage import DEBUG_MODAL_PROOF_DIR
from scripts.debug_modal_proof_ocr import build_vlm_client, transcribe_image_to_modal_debug_state

router = APIRouter()

TEMPLATES_DIR = Path(__file__).resolve().parents[1] / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

STATE_FILENAME = "state.json"
OPTIONAL_TEXT_ARTIFACTS = ("notes.md", "source.txt")
OCR_TEXT_ARTIFACTS = (
    "ocr_prompt.txt",
    "intermediate_ROOT.json",
    "intermediate_1.json",
    "intermediate_2.json",
    "response_ROOT.json",
    "response_1.json",
    "response_2.json",
    "final_structure.json",
)
IMAGE_ARTIFACTS = ("source_image", "debug_bars.jpg", "crop_ROOT.jpg", "crop_1.jpg", "crop_2.jpg")
SYSTEM_OPTIONS = ["K", "T", "S4", "S5"]
STARTER_EXAMPLE = "starter"

DEFAULT_PROOF_JSON_TEXT = json.dumps(
    {
        "lines": [
            {"wffstr": "A", "jstr": "PR"},
        ],
        "meta": {},
    },
    indent=2,
    ensure_ascii=False,
)


def _clean_example_slug(raw: Optional[str]) -> str:
    text = str(raw or "").strip().lower()
    text = re.sub(r"[\s\-–]+", "_", text)
    text = re.sub(r"[^a-z0-9_]", "", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or STARTER_EXAMPLE


def list_modal_proof_examples() -> List[str]:
    if not DEBUG_MODAL_PROOF_DIR.exists():
        return []
    return sorted([p.name for p in DEBUG_MODAL_PROOF_DIR.iterdir() if p.is_dir()], key=str.lower)


def _state_path(example_slug: str) -> Path:
    return DEBUG_MODAL_PROOF_DIR / example_slug / STATE_FILENAME


def _artifact_text(example_slug: str, filename: str) -> str:
    path = DEBUG_MODAL_PROOF_DIR / example_slug / filename
    if not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _default_page_state() -> Dict[str, Any]:
    return {
        "system": "K",
        "include_derived_rules": False,
        "conclusion": "A",
        "vision_base_url": "",
        "vision_model": "",
        "source_image_name": "",
        "proof_json_text": DEFAULT_PROOF_JSON_TEXT,
        "draft_comment": "",
        "draft_grade": "",
        "checker_request": None,
        "checker_response": None,
        "run_summary": "Edit the starter proof JSON and run the modal checker.",
        "status": "",
        "message": "",
        "checker_error": "",
    }


def _load_example_state(example_slug: Optional[str]) -> Dict[str, Any]:
    state = _default_page_state()
    if not example_slug:
        return state

    path = _state_path(example_slug)
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
        "vision_base_url",
        "vision_model",
        "source_image_name",
        "draft_comment",
        "draft_grade",
        "checker_request",
        "checker_response",
        "run_summary",
    ):
        if key in raw:
            state[key] = raw[key]
    return state


def _save_example_state(example_slug: str, state: Dict[str, Any]) -> None:
    path = _state_path(example_slug)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "system": state.get("system", "K"),
                "include_derived_rules": bool(state.get("include_derived_rules", False)),
                "conclusion": state.get("conclusion", ""),
                "vision_base_url": state.get("vision_base_url", ""),
                "vision_model": state.get("vision_model", ""),
                "source_image_name": state.get("source_image_name", ""),
                "proof_json_text": state.get("proof_json_text", DEFAULT_PROOF_JSON_TEXT),
                "draft_comment": state.get("draft_comment", ""),
                "draft_grade": state.get("draft_grade", ""),
                "checker_request": state.get("checker_request"),
                "checker_response": state.get("checker_response"),
                "run_summary": state.get("run_summary", ""),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def _checker_issues(checker_response: Any, key: str) -> List[Dict[str, Any]]:
    if not isinstance(checker_response, dict):
        return []
    issues = checker_response.get(key)
    return issues if isinstance(issues, list) else []


def _checker_lines(checker_response: Any) -> List[Dict[str, Any]]:
    if not isinstance(checker_response, dict):
        return []
    lines = checker_response.get("normalized_lines")
    return lines if isinstance(lines, list) else []


def _checker_request_lines(checker_request: Any) -> List[Dict[str, Any]]:
    if not isinstance(checker_request, dict):
        return []
    lines = checker_request.get("lines")
    return lines if isinstance(lines, list) else []


def _render_checker_request_preview(checker_request: Any) -> str:
    lines = _checker_request_lines(checker_request)
    if not lines:
        return ""

    rendered: List[str] = []
    for idx, item in enumerate(lines, start=1):
        if not isinstance(item, dict):
            continue
        try:
            depth = max(0, int(item.get("depth", 0)))
        except (TypeError, ValueError):
            depth = 0
        sentence = str(item.get("sentence", "")).strip()
        citation = str(item.get("citation", "")).strip()
        indent = "  " * depth
        line = f"{idx:>2}. {indent}{sentence}"
        if citation:
            line += f"    {citation}"
        rendered.append(line)
    return "\n".join(rendered)


def _render_page(
    request: Request,
    *,
    example: Optional[str],
    state: Dict[str, Any],
) -> Response:
    examples = list_modal_proof_examples()
    if example and (example == STARTER_EXAMPLE or example in examples):
        selected = example
    elif examples:
        selected = examples[0]
    else:
        selected = STARTER_EXAMPLE

    artifact_texts = {}
    image_artifacts: List[Dict[str, str]] = []
    if selected and selected != STARTER_EXAMPLE:
        for filename in (*OPTIONAL_TEXT_ARTIFACTS, *OCR_TEXT_ARTIFACTS):
            text = _artifact_text(selected, filename)
            if text.strip():
                artifact_texts[filename] = text
        example_dir = DEBUG_MODAL_PROOF_DIR / selected
        source_image_name = str(state.get("source_image_name") or "").strip()
        image_candidates = list(IMAGE_ARTIFACTS)
        if source_image_name:
            image_candidates.insert(0, source_image_name)
        seen = set()
        for filename in image_candidates:
            if not filename or filename in seen:
                continue
            seen.add(filename)
            path = example_dir / filename
            if path.exists() and path.is_file():
                image_artifacts.append(
                    {
                        "filename": filename,
                        "url": f"/debug_modal_proof_assets/{selected}/{quote(filename)}",
                    }
                )

    return templates.TemplateResponse(
        request,
        "debug_modal_proof.html",
        {
            "available_examples": examples,
            "selected_example": selected,
            "system_options": SYSTEM_OPTIONS,
            "parse_errors": _checker_issues(state.get("checker_response"), "parse_errors"),
            "check_errors": _checker_issues(state.get("checker_response"), "check_errors"),
            "normalized_lines": _checker_lines(state.get("checker_response")),
            "checker_request_preview": _render_checker_request_preview(state.get("checker_request")),
            "optional_artifacts": artifact_texts,
            "image_artifacts": image_artifacts,
            **state,
        },
    )


def _state_from_form(
    *,
    example: Optional[str],
    system: str,
    include_derived_rules: bool,
    conclusion: str,
    vision_base_url: str,
    vision_model: str,
    proof_json_text: str,
    draft_comment: str,
    draft_grade: str,
) -> tuple[str, Dict[str, Any]]:
    example_slug = _clean_example_slug(example)
    state = _load_example_state(example_slug)
    state.update(
        {
            "system": str(system or "K").strip(),
            "include_derived_rules": bool(include_derived_rules),
            "conclusion": str(conclusion or "").strip(),
            "vision_base_url": str(vision_base_url or "").strip(),
            "vision_model": str(vision_model or "").strip(),
            "proof_json_text": proof_json_text,
            "draft_comment": draft_comment,
            "draft_grade": draft_grade,
            "status": "",
            "message": "",
            "checker_error": "",
        }
    )
    return example_slug, state


def _build_autograde_config(state: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "system": state.get("system", "K"),
        "include_derived_rules": bool(state.get("include_derived_rules", False)),
        "conclusion": state.get("conclusion") or None,
    }


def _find_saved_source_image(example_slug: str, state: Dict[str, Any]) -> Optional[Path]:
    source_image_name = str(state.get("source_image_name") or "").strip()
    example_dir = DEBUG_MODAL_PROOF_DIR / example_slug
    if source_image_name:
        candidate = example_dir / source_image_name
        if candidate.exists() and candidate.is_file():
            return candidate
    for candidate in sorted(example_dir.iterdir(), key=lambda p: p.name.lower()) if example_dir.exists() else []:
        if candidate.is_file() and candidate.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
            return candidate
    return None


def _resolve_source_image_path(example_slug: str, raw_path: str) -> Optional[Path]:
    text = str(raw_path or "").strip()
    if not text:
        return None
    candidate = Path(text)
    if candidate.is_absolute():
        return candidate

    example_candidate = (DEBUG_MODAL_PROOF_DIR / example_slug / candidate).resolve()
    if example_candidate.exists():
        return example_candidate

    raw_scans_candidate = (RAW_SCANS_DIR / candidate).resolve()
    if raw_scans_candidate.exists():
        return raw_scans_candidate

    return raw_scans_candidate


async def _persist_uploaded_source_image(
    example_slug: str,
    proof_image: UploadFile | None,
) -> Optional[Path]:
    if proof_image is None or not getattr(proof_image, "filename", ""):
        return None

    suffix = Path(proof_image.filename or "").suffix.lower() or ".png"
    if suffix not in {".png", ".jpg", ".jpeg", ".webp"}:
        raise ValueError("OCR image must be PNG, JPG, JPEG, or WEBP")

    example_dir = DEBUG_MODAL_PROOF_DIR / example_slug
    example_dir.mkdir(parents=True, exist_ok=True)
    target_path = example_dir / f"source_image{suffix}"
    with target_path.open("wb") as handle:
        shutil.copyfileobj(proof_image.file, handle)
    return target_path


@router.get("/debug/modal-proof", response_class=HTMLResponse)
async def modal_proof_debug_page(
    request: Request,
    example: Optional[str] = None,
) -> Response:
    example_slug = _clean_example_slug(example)
    state = _load_example_state(example_slug if example else None)
    status = request.query_params.get("status", "")
    message = request.query_params.get("message", "")
    if status or message:
        state["status"] = status
        state["message"] = message
    return _render_page(request, example=example_slug if example else None, state=state)


@router.post("/debug/modal-proof/save", response_class=HTMLResponse)
async def save_modal_proof_state(
    request: Request,
    example: str = Form(STARTER_EXAMPLE),
    system: str = Form("K"),
    include_derived_rules: bool = Form(False),
    conclusion: str = Form(""),
    vision_base_url: str = Form(""),
    vision_model: str = Form(""),
    proof_json_text: str = Form(...),
    draft_comment: str = Form(""),
    draft_grade: str = Form(""),
) -> Response:
    example_slug, state = _state_from_form(
        example=example,
        system=system,
        include_derived_rules=include_derived_rules,
        conclusion=conclusion,
        vision_base_url=vision_base_url,
        vision_model=vision_model,
        proof_json_text=proof_json_text,
        draft_comment=draft_comment,
        draft_grade=draft_grade,
    )
    state["status"] = "success"
    state["message"] = "Saved current modal proof debug state."
    _save_example_state(example_slug, state)

    return RedirectResponse(
        url=f"/debug/modal-proof?example={quote(example_slug)}&status=success&message={quote(state['message'])}",
        status_code=303,
    )


@router.post("/debug/modal-proof/run_checker", response_class=HTMLResponse)
async def run_modal_proof_debug_checker(
    request: Request,
    example: str = Form(STARTER_EXAMPLE),
    system: str = Form("K"),
    include_derived_rules: bool = Form(False),
    conclusion: str = Form(""),
    vision_base_url: str = Form(""),
    vision_model: str = Form(""),
    proof_json_text: str = Form(...),
    draft_comment: str = Form(""),
    draft_grade: str = Form(""),
) -> Response:
    example_slug, state = _state_from_form(
        example=example,
        system=system,
        include_derived_rules=include_derived_rules,
        conclusion=conclusion,
        vision_base_url=vision_base_url,
        vision_model=vision_model,
        proof_json_text=proof_json_text,
        draft_comment=draft_comment,
        draft_grade=draft_grade,
    )

    result: AutoGradeResult = autograde_submission(
        problem_type="modal_proof",
        answer_text=proof_json_text,
        autograde_config=_build_autograde_config(state),
    )

    state["run_summary"] = result.summary
    state["draft_comment"] = result.draft_comment or draft_comment
    if result.score is not None:
        state["draft_grade"] = str(result.score)
    state["checker_request"] = result.metadata.get("checker_request")
    state["checker_response"] = result.metadata.get("checker_response")

    if result.status == "scored":
        state["status"] = "success"
        state["message"] = "Checker ran successfully through the shared autograde layer."
    else:
        state["status"] = "error"
        state["message"] = "Checker could not run."
        state["checker_error"] = "; ".join(result.errors) or result.summary

    _save_example_state(example_slug, state)
    return _render_page(request, example=example_slug, state=state)


@router.post("/debug/modal-proof/run_ocr", response_class=HTMLResponse)
async def run_modal_proof_debug_ocr(
    request: Request,
    example: str = Form(STARTER_EXAMPLE),
    system: str = Form("K"),
    include_derived_rules: bool = Form(False),
    conclusion: str = Form(""),
    vision_base_url: str = Form(""),
    vision_model: str = Form(""),
    proof_json_text: str = Form(DEFAULT_PROOF_JSON_TEXT),
    draft_comment: str = Form(""),
    draft_grade: str = Form(""),
    proof_image_path: str = Form(""),
    proof_image: UploadFile | None = File(None),
) -> Response:
    example_slug, state = _state_from_form(
        example=example,
        system=system,
        include_derived_rules=include_derived_rules,
        conclusion=conclusion,
        vision_base_url=vision_base_url,
        vision_model=vision_model,
        proof_json_text=proof_json_text,
        draft_comment=draft_comment,
        draft_grade=draft_grade,
    )

    example_dir = DEBUG_MODAL_PROOF_DIR / example_slug
    example_dir.mkdir(parents=True, exist_ok=True)

    try:
        uploaded_image_path = await _persist_uploaded_source_image(example_slug, proof_image)
        source_image_path = uploaded_image_path or _resolve_source_image_path(example_slug, proof_image_path) or _find_saved_source_image(example_slug, state)
        if source_image_path is None:
            raise ValueError("Provide a proof image upload or an existing image path before running OCR.")
        if not source_image_path.exists() or not source_image_path.is_file():
            raise ValueError(f"OCR image not found: {source_image_path}")

        client = build_vlm_client(
            base_url=state.get("vision_base_url") or None,
            model=state.get("vision_model") or None,
        )
        ocr_state = transcribe_image_to_modal_debug_state(
            image_path=source_image_path,
            output_dir=example_dir,
            modal_config=_build_autograde_config(state),
            client=client,
            vision_base_url=state.get("vision_base_url") or None,
            vision_model=state.get("vision_model") or None,
        )
        ocr_state["draft_comment"] = draft_comment
        ocr_state["draft_grade"] = draft_grade
        ocr_state["status"] = "success"
        ocr_state["message"] = "OCR ran and updated the modal proof state."
        state = ocr_state
    except Exception as exc:
        state["status"] = "error"
        state["message"] = f"Modal OCR failed: {exc}"

    _save_example_state(example_slug, state)
    return _render_page(request, example=example_slug, state=state)
