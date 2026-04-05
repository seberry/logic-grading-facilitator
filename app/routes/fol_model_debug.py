from __future__ import annotations

import asyncio
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import quote

import yaml
from fastapi import APIRouter, Form, HTTPException, Request, Response
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from app.fol_model_checker import FolModelParseError, check_fol_task
from app.session_state import get_session
from app.storage import DEBUG_FOL_MODEL_DIR, RAW_SCANS_DIR

router = APIRouter()

TEMPLATES_DIR = Path(__file__).resolve().parents[1] / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")
STATE_FILENAME = "state.json"
OCR_REQUEST_FILENAME = "ocr_request.yaml"
OCR_PROMPT_FILENAME = "ocr_prompt.txt"
OCR_RESPONSE_FILENAME = "ocr_raw_response.txt"
OCR_TRANSCRIPTION_FILENAME = "ocr_model_text.txt"
SUPPORTED_PROBLEM_TYPES = [
    {"id": "truth_value", "label": "Determine truth value"},
    {"id": "non_equivalence", "label": "Show non-equivalence"},
    {"id": "joint_consistency", "label": "Show joint consistency"},
    {"id": "non_validity", "label": "Show non-validity"},
    {"id": "non_logical_truth", "label": "Show non-logical truth"},
    {"id": "non_logical_falsehood", "label": "Show non-logical falsehood"},
]
TRACE_MODE_OPTIONS = [
    {"id": "terse", "label": "Terse"},
    {"id": "full", "label": "Full"},
]

DEFAULT_MODEL_TEXT = """DOMAIN: {1,2,3}

CONSTANTS:
a: 1
b: 2

PREDICATES:
P: {1,2}
Q: {2}
R: {(1,2),(2,3)}
"""

DEFAULT_SENTENCES_TEXT = """1. Pa
2. ∃x Rxa
3. ∀x (Px → Qx)
"""

DEFAULT_DRAFT_COMMENT = """Question: FOL model
Checker: FOL model checker v0

Model summary:
DOMAIN: {1,2,3}
CONSTANTS:
  a: 1
  b: 2
PREDICATES:
  P: {1,2}
  Q: {2}
  R: {(1,2),(2,3)}

Sentence results:
- 1. Pa: true
- 2. ∃x Rxa: false
- 3. ∀x (Px → Qx): false

Instructor note:
No default scoring rule is configured yet. Edit this comment and the grade manually.
"""

DEFAULT_TRACE_BLOCKS = [
    {
        "title": "Sentence 1: Pa",
        "result": "True",
        "body": """Check Pa
  a names 1
  P = {1,2}
  So this is P(1)
  (1) is in P
Result: True""",
    },
    {
        "title": "Sentence 3: ∀x (Px → Qx)",
        "result": "False",
        "body": """Check ∀x(Px → Qx)
  Try object 1
    Check P1 → Q1
      Check P1
        P = {1,2}
        (1) is in P
      Result: True
      Check Q1
        Q = {2}
        (1) is not in Q
      Result: False
    Result: False
Result: False because object 1 is a counterexample""",
    },
]

PROBLEM_TYPE_PRESETS: Dict[str, Dict[str, Any]] = {
    "truth_value": {
        "problem_setup_text": """task_kind: truth_value
default_score_correct: 2
default_score_incorrect: 0
prompt_data:
  formulas:
    - "Pa"
    - "∀x (Px → Qx)"
""",
        "model_text": DEFAULT_MODEL_TEXT,
        "sentences_text": """1. Pa
2. ∀x (Px → Qx)
""",
        "trace_blocks": DEFAULT_TRACE_BLOCKS,
        "draft_comment": DEFAULT_DRAFT_COMMENT,
        "draft_grade": "",
        "checker_enabled": True,
    },
    "non_equivalence": {
        "problem_setup_text": """task_kind: non_equivalence
default_score_correct: 3
default_score_incorrect: 0
prompt_data:
  formulas:
    - "∀x Px"
    - "∃x Px"
""",
        "model_text": """DOMAIN: {1,2}

CONSTANTS:
a: 1

PREDICATES:
P: {1}
""",
        "sentences_text": """1. ∀x Px
2. ∃x Px
""",
        "trace_blocks": DEFAULT_TRACE_BLOCKS,
        "draft_comment": """Question: non-equivalence witness
Checker: FOL model checker v0

This sample checks whether the two formulas come apart in one model.
""",
        "draft_grade": "",
        "checker_enabled": True,
    },
    "joint_consistency": {
        "problem_setup_text": """task_kind: joint_consistency
default_score_correct: 3
default_score_incorrect: 0
prompt_data:
  formulas:
    - "∃x Px"
    - "∃x Qx"
    - "∀x (Px → ¬Qx)"
""",
        "model_text": """DOMAIN: {1,2}

CONSTANTS:

PREDICATES:
P: {1}
Q: {2}
""",
        "sentences_text": """1. ∃x Px
2. ∃x Qx
3. ∀x (Px → ¬Qx)
""",
        "trace_blocks": DEFAULT_TRACE_BLOCKS,
        "draft_comment": """Question: joint consistency
Checker: FOL model checker v0

This sample checks whether all target formulas are true together in one model.
""",
        "draft_grade": "",
        "checker_enabled": True,
    },
    "non_validity": {
        "problem_setup_text": """task_kind: non_validity
default_score_correct: 3
default_score_incorrect: 0
prompt_data:
  premises:
    - "∀x (Px → Qx)"
    - "∃x Px"
  conclusion: "∀x Qx"
""",
        "model_text": """DOMAIN: {1,2}

CONSTANTS:

PREDICATES:
P: {1}
Q: {1}
""",
        "sentences_text": """1. ∀x (Px → Qx)
2. ∃x Px
3. ∀x Qx
""",
        "trace_blocks": DEFAULT_TRACE_BLOCKS,
        "draft_comment": """Question: non-validity countermodel
Checker: FOL model checker v0

This sample checks whether the premises are all true and the conclusion false.
""",
        "draft_grade": "",
        "checker_enabled": True,
    },
    "non_logical_truth": {
        "problem_setup_text": """task_kind: non_logical_truth
default_score_correct: 2
default_score_incorrect: 0
prompt_data:
  formulas:
    - "∃x Px"
""",
        "model_text": """DOMAIN: {1}

CONSTANTS:

PREDICATES:
P: {1}
""",
        "sentences_text": """1. ∃x Px
""",
        "trace_blocks": DEFAULT_TRACE_BLOCKS,
        "draft_comment": """Question: non-logical truth
Checker: FOL model checker v0

This sample checks whether the target formula is true in at least one model.
""",
        "draft_grade": "",
        "checker_enabled": True,
    },
    "non_logical_falsehood": {
        "problem_setup_text": """task_kind: non_logical_falsehood
default_score_correct: 2
default_score_incorrect: 0
prompt_data:
  formulas:
    - "∀x Px"
""",
        "model_text": """DOMAIN: {1,2}

CONSTANTS:

PREDICATES:
P: {1}
""",
        "sentences_text": """1. ∀x Px
""",
        "trace_blocks": DEFAULT_TRACE_BLOCKS,
        "draft_comment": """Question: non-logical falsehood
Checker: FOL model checker v0

This sample checks whether the target formula is false in at least one model.
""",
        "draft_grade": "",
        "checker_enabled": True,
    },
}

LIVE_TASK_KINDS = {
    "truth_value",
    "truth_in_given_model",
    "true_false_in_given_model",
    "make_true",
    "make_false",
    "non_equivalence",
    "joint_consistency",
    "non_validity",
    "non_logical_truth",
    "non_logical_falsehood",
}


def list_fol_model_debug_scans() -> List[str]:
    if not DEBUG_FOL_MODEL_DIR.exists():
        return []
    return sorted([p.name for p in DEBUG_FOL_MODEL_DIR.iterdir() if p.is_dir()], key=str.lower)


def _find_scan_image_path(scan_slug: str) -> Optional[Path]:
    for ext in IMAGE_EXTENSIONS:
        candidate = RAW_SCANS_DIR / f"{scan_slug}{ext}"
        if candidate.exists():
            return candidate

    debug_dir = DEBUG_FOL_MODEL_DIR / scan_slug
    if not debug_dir.exists():
        return None

    for ext in IMAGE_EXTENSIONS:
        candidate = debug_dir / f"{scan_slug}{ext}"
        if candidate.exists():
            return candidate

    for child in sorted(debug_dir.iterdir()):
        if child.is_file() and child.suffix in IMAGE_EXTENSIONS:
            return child
    return None


def _scan_image_url(scan_slug: str) -> Optional[str]:
    path = _find_scan_image_path(scan_slug)
    if path is None:
        return None

    if RAW_SCANS_DIR in path.parents:
        return f"/scans/{path.name}"

    if DEBUG_FOL_MODEL_DIR in path.parents:
        rel = path.relative_to(DEBUG_FOL_MODEL_DIR)
        return f"/debug_fol_model_assets/{rel.as_posix()}"

    return None


def _state_path(scan_slug: str) -> Path:
    return DEBUG_FOL_MODEL_DIR / scan_slug / STATE_FILENAME


def _artifact_path(scan_slug: str, filename: str) -> Path:
    return DEBUG_FOL_MODEL_DIR / scan_slug / filename


def _default_page_state() -> Dict[str, Any]:
    return {
        "problem_id": "",
        "problem_label": "",
        "problem_type": "truth_value",
        "problem_setup_text": PROBLEM_TYPE_PRESETS["truth_value"]["problem_setup_text"],
        "model_text": DEFAULT_MODEL_TEXT,
        "sentences_text": DEFAULT_SENTENCES_TEXT,
        "trace_blocks": DEFAULT_TRACE_BLOCKS,
        "draft_comment": DEFAULT_DRAFT_COMMENT,
        "draft_grade": "",
        "status": "",
        "message": "",
        "checker_error": "",
        "checker_ran": False,
        "checker_enabled": True,
        "trace_mode": "terse",
        "ocr_log": "",
        "ocr_response_text": "",
        "available_session_problems": [],
    }


def _load_scan_state(scan_slug: Optional[str]) -> Dict[str, Any]:
    state = _default_page_state()
    if not scan_slug:
        return state

    path = _state_path(scan_slug)
    if not path.exists():
        return state

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return state

    for key in (
        "problem_type",
        "problem_id",
        "problem_label",
        "problem_setup_text",
        "model_text",
        "sentences_text",
        "trace_blocks",
        "draft_comment",
        "draft_grade",
        "checker_enabled",
        "trace_mode",
        "ocr_log",
        "ocr_response_text",
    ):
        if key in raw:
            state[key] = raw[key]
    return state


def _save_scan_state(scan_slug: str, state: Dict[str, Any]) -> None:
    path = _state_path(scan_slug)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "model_text": state.get("model_text", ""),
                "problem_type": state.get("problem_type", "truth_value"),
                "problem_id": state.get("problem_id", ""),
                "problem_label": state.get("problem_label", ""),
                "problem_setup_text": state.get("problem_setup_text", ""),
                "sentences_text": state.get("sentences_text", ""),
                "trace_blocks": state.get("trace_blocks", []),
                "draft_comment": state.get("draft_comment", ""),
                "draft_grade": state.get("draft_grade", ""),
                "checker_enabled": state.get("checker_enabled", True),
                "trace_mode": state.get("trace_mode", "terse"),
                "ocr_log": state.get("ocr_log", ""),
                "ocr_response_text": state.get("ocr_response_text", ""),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def _render_page(
    request: Request,
    *,
    scan: Optional[str],
    state: Dict[str, Any],
) -> Response:
    scans = list_fol_model_debug_scans()
    selected = scan if scan in scans else (scan if scan else (scans[0] if scans else None))
    image_url = _scan_image_url(selected) if selected else None

    return templates.TemplateResponse(
        "debug_fol_model.html",
        {
            "request": request,
            "available_scans": scans,
            "selected_scan": selected,
            "image_url": image_url,
            "problem_type_options": SUPPORTED_PROBLEM_TYPES,
            "trace_mode_options": TRACE_MODE_OPTIONS,
            **state,
        },
    )


def _merge_preset(problem_type: str, state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    merged = _default_page_state()
    if state:
        merged.update(state)
    preset = PROBLEM_TYPE_PRESETS.get(problem_type, PROBLEM_TYPE_PRESETS["truth_value"])
    merged.update(
        {
            "problem_type": problem_type if problem_type in PROBLEM_TYPE_PRESETS else "truth_value",
            "problem_setup_text": preset["problem_setup_text"],
            "model_text": preset["model_text"],
            "sentences_text": preset["sentences_text"],
            "trace_blocks": preset["trace_blocks"],
            "draft_comment": preset["draft_comment"],
            "draft_grade": preset["draft_grade"],
            "checker_enabled": preset["checker_enabled"],
            "checker_ran": False,
            "checker_error": "",
            "status": "",
            "message": "",
        }
    )
    return merged


def _available_session_fol_problems() -> List[Dict[str, str]]:
    try:
        session = get_session()
    except Exception:
        return []

    items: List[Dict[str, str]] = []
    for spec in session.grading_key.problems:
        if spec.problem_type == "fol_model":
            items.append({"id": spec.id, "label": spec.label})
    return items


def _setup_text_from_autograde(autograde: Dict[str, Any]) -> str:
    return yaml.safe_dump(autograde, sort_keys=False, allow_unicode=True).strip() + "\n"


def _load_state_from_problem(problem_id: str, state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    session = get_session()
    spec = next((p for p in session.grading_key.problems if p.id == problem_id), None)
    if spec is None:
        raise HTTPException(status_code=404, detail=f"Unknown problem id {problem_id!r}")

    merged = _default_page_state()
    if state:
        merged.update(state)

    autograde = spec.autograde if isinstance(spec.autograde, dict) else {}
    task_kind = str(autograde.get("task_kind") or "truth_value")
    merged.update(
        {
            "problem_id": spec.id,
            "problem_label": spec.label,
            "problem_type": task_kind,
            "problem_setup_text": _setup_text_from_autograde(autograde),
            "checker_enabled": _checker_enabled_for_task(task_kind),
            "checker_ran": False,
            "checker_error": "",
            "status": "",
            "message": f"Loaded rubric config for {spec.id}.",
        }
    )
    return merged


def _parse_problem_setup_text(problem_setup_text: str) -> Dict[str, Any]:
    try:
        parsed = yaml.safe_load(problem_setup_text) or {}
    except yaml.YAMLError as exc:
        raise FolModelParseError(f"Problem setup YAML could not be parsed: {exc}")
    if not isinstance(parsed, dict):
        raise FolModelParseError("Problem setup must parse to a YAML object/map.")
    return parsed


def _checker_enabled_for_task(task_kind: str) -> bool:
    return task_kind in LIVE_TASK_KINDS


def _load_artifact_text(scan_slug: Optional[str], filename: str) -> str:
    if not scan_slug:
        return ""
    path = _artifact_path(scan_slug, filename)
    if not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _ensure_raw_scan_available(scan_slug: str) -> Path:
    existing = _find_scan_image_path(scan_slug)
    if existing and RAW_SCANS_DIR in existing.parents:
        return existing

    if existing and DEBUG_FOL_MODEL_DIR in existing.parents:
        RAW_SCANS_DIR.mkdir(parents=True, exist_ok=True)
        destination = RAW_SCANS_DIR / existing.name
        if not destination.exists():
            shutil.copy(existing, destination)
        return destination

    raise HTTPException(
        status_code=404,
        detail=f"Could not locate a scan image for '{scan_slug}' in {RAW_SCANS_DIR} or {DEBUG_FOL_MODEL_DIR}.",
    )


async def _run_subprocess_with_logs(command: List[str]) -> tuple[int, str]:
    proc = await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd=Path(__file__).resolve().parents[2],
    )
    stdout, _ = await proc.communicate()
    output = stdout.decode("utf-8", errors="replace")
    return proc.returncode, output


@router.get("/debug/fol-model", response_class=HTMLResponse)
async def fol_model_debug_page(
    request: Request,
    scan: Optional[str] = None,
    problem_type: Optional[str] = None,
    problem_id: Optional[str] = None,
) -> Response:
    state = _load_scan_state(scan)
    state["available_session_problems"] = _available_session_fol_problems()
    if problem_id:
        state = _load_state_from_problem(problem_id, state if scan and _state_path(scan).exists() else None)
        state["available_session_problems"] = _available_session_fol_problems()
        return _render_page(request, scan=scan, state=state)
    selected_type = problem_type or state.get("problem_type", "truth_value")
    if problem_type:
        state = _merge_preset(selected_type, state if scan and _state_path(scan).exists() else None)
        state["available_session_problems"] = _available_session_fol_problems()
    status = request.query_params.get("status", "")
    message = request.query_params.get("message", "")
    if status or message:
        state["status"] = status
        state["message"] = message
    return _render_page(request, scan=scan, state=state)


@router.post("/debug/fol-model/save", response_class=HTMLResponse)
async def save_fol_model_transcription(
    request: Request,
    scan: Optional[str] = Form(None),
    problem_id: str = Form(""),
    problem_type: str = Form("truth_value"),
    trace_mode: str = Form("terse"),
    problem_setup_text: str = Form(""),
    model_text: str = Form(...),
    sentences_text: str = Form(...),
    draft_comment: str = Form(""),
    draft_grade: str = Form(""),
) -> Response:
    if not scan or not scan.strip():
        parsed_setup: Dict[str, Any] = {}
        try:
            if problem_setup_text.strip():
                parsed_setup = _parse_problem_setup_text(problem_setup_text)
        except FolModelParseError:
            parsed_setup = {}
        effective_task_kind = str(parsed_setup.get("task_kind") or problem_type or "truth_value")
        state = _default_page_state()
        state.update(
            {
                "problem_type": effective_task_kind,
                "problem_id": problem_id,
                "problem_setup_text": problem_setup_text,
                "model_text": model_text,
                "sentences_text": sentences_text,
                "draft_comment": draft_comment,
                "draft_grade": draft_grade,
                "checker_enabled": _checker_enabled_for_task(effective_task_kind),
                "trace_mode": trace_mode,
                "status": "error",
                "message": "Pick a scan slug before saving state to disk.",
            }
        )
        return _render_page(request, scan=None, state=state)

    scan_slug = scan.strip()
    parsed_setup: Dict[str, Any] = {}
    try:
        if problem_setup_text.strip():
            parsed_setup = _parse_problem_setup_text(problem_setup_text)
    except FolModelParseError:
        parsed_setup = {}
    effective_task_kind = str(parsed_setup.get("task_kind") or problem_type or "truth_value")
    state = _load_scan_state(scan_slug)
    state.update(
        {
            "problem_type": effective_task_kind,
            "problem_id": problem_id,
            "problem_setup_text": problem_setup_text,
            "model_text": model_text,
            "sentences_text": sentences_text,
            "draft_comment": draft_comment,
            "draft_grade": draft_grade,
            "checker_enabled": _checker_enabled_for_task(effective_task_kind),
            "trace_mode": trace_mode,
            "status": "success",
            "message": "Saved current transcription state.",
        }
    )
    _save_scan_state(scan_slug, state)

    return RedirectResponse(
        url=f"/debug/fol-model?scan={quote(scan_slug)}&status=success&message={quote('Saved current transcription state.')}",
        status_code=303,
    )


@router.post("/debug/fol-model/rerun", response_class=HTMLResponse)
async def rerun_fol_model_ocr(
    request: Request,
    scan: Optional[str] = Form(None),
    problem_id: str = Form(""),
    problem_type: str = Form("truth_value"),
    trace_mode: str = Form("terse"),
    problem_setup_text: str = Form(""),
    model_text: str = Form(...),
    sentences_text: str = Form(...),
    draft_comment: str = Form(""),
    draft_grade: str = Form(""),
) -> Response:
    scan_slug = scan.strip() if scan else None
    state = _load_scan_state(scan_slug)
    state.update(
        {
            "problem_type": problem_type,
            "problem_id": problem_id,
            "problem_setup_text": problem_setup_text,
            "model_text": model_text,
            "sentences_text": sentences_text,
            "draft_comment": draft_comment,
            "draft_grade": draft_grade,
            "checker_enabled": _checker_enabled_for_task(problem_type),
            "trace_mode": trace_mode,
            "status": "",
            "message": "",
            "checker_error": "",
            "checker_ran": False,
            "ocr_log": "",
            "ocr_response_text": "",
        }
    )
    state["available_session_problems"] = _available_session_fol_problems()

    if not scan_slug:
        state["status"] = "error"
        state["message"] = "Pick a scan slug before running OCR."
        return _render_page(request, scan=None, state=state)

    try:
        autograde = _parse_problem_setup_text(problem_setup_text)
        task_kind = str(autograde.get("task_kind") or problem_type or "truth_value")
        state["problem_type"] = task_kind
        state["checker_enabled"] = _checker_enabled_for_task(task_kind)

        image_path = _ensure_raw_scan_available(scan_slug)
        debug_dir = DEBUG_FOL_MODEL_DIR / scan_slug
        debug_dir.mkdir(parents=True, exist_ok=True)

        request_path = _artifact_path(scan_slug, OCR_REQUEST_FILENAME)
        request_path.write_text(
            yaml.safe_dump(autograde, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )

        command = [
            sys.executable,
            "scripts/debug_fol_model_ocr.py",
            "--image",
            str(image_path),
            "--output-dir",
            str(debug_dir),
            "--autograde-file",
            str(request_path),
        ]
        code, output = await _run_subprocess_with_logs(command)
        state["ocr_log"] = output
        state["ocr_response_text"] = _load_artifact_text(scan_slug, OCR_RESPONSE_FILENAME)
        proposed_text = _load_artifact_text(scan_slug, OCR_TRANSCRIPTION_FILENAME)
        if proposed_text.strip():
            state["model_text"] = proposed_text

        if code == 0:
            state["status"] = "success"
            state["message"] = "OCR reran successfully. Review the proposed model text."
        else:
            state["status"] = "error"
            state["message"] = "OCR rerun failed. See log output below."
    except (FolModelParseError, HTTPException) as exc:
        state["checker_error"] = str(exc)
        state["status"] = "error"
        state["message"] = "OCR rerun failed."

    if scan_slug:
        _save_scan_state(scan_slug, state)
    return _render_page(request, scan=scan_slug, state=state)


@router.post("/debug/fol-model/run_checker", response_class=HTMLResponse)
async def run_fol_model_checker(
    request: Request,
    scan: Optional[str] = Form(None),
    problem_id: str = Form(""),
    problem_type: str = Form("truth_value"),
    trace_mode: str = Form("terse"),
    problem_setup_text: str = Form(""),
    model_text: str = Form(...),
    sentences_text: str = Form(...),
    draft_comment: str = Form(""),
    draft_grade: str = Form(""),
) -> Response:
    scan_slug = scan.strip() if scan else None
    state = _load_scan_state(scan_slug)
    state.update(
        {
            "problem_type": problem_type,
            "problem_id": problem_id,
            "problem_setup_text": problem_setup_text,
            "model_text": model_text,
            "sentences_text": sentences_text,
            "draft_comment": draft_comment,
            "draft_grade": draft_grade,
            "checker_enabled": _checker_enabled_for_task(problem_type),
            "trace_mode": trace_mode,
            "status": "",
            "message": "",
            "checker_error": "",
            "checker_ran": False,
        }
    )
    state["available_session_problems"] = _available_session_fol_problems()

    try:
        autograde = _parse_problem_setup_text(problem_setup_text)
        task_kind = str(autograde.get("task_kind") or problem_type or "truth_value")
        state["problem_type"] = task_kind
        if task_kind in LIVE_TASK_KINDS:
            task_result = check_fol_task(model_text, autograde, trace_mode=trace_mode)
            state["trace_blocks"] = task_result.trace_blocks
            state["draft_comment"] = task_result.draft_comment
            state["checker_enabled"] = _checker_enabled_for_task(task_kind)
            state["checker_ran"] = True
            state["status"] = "success"
            state["message"] = "Checker ran successfully."
        elif task_kind in PROBLEM_TYPE_PRESETS:
            preset = PROBLEM_TYPE_PRESETS[task_kind]
            state["trace_blocks"] = preset["trace_blocks"]
            state["draft_comment"] = draft_comment or preset["draft_comment"]
            state["checker_enabled"] = False
            state["status"] = "success"
            state["message"] = "This task kind is currently in design-preview mode."
        else:
            raise FolModelParseError(f"Unsupported task_kind: {task_kind}")
    except FolModelParseError as exc:
        state["checker_error"] = str(exc)
        state["status"] = "error"
        state["message"] = "Checker could not run."

    if scan_slug:
        _save_scan_state(scan_slug, state)

    return _render_page(request, scan=scan_slug, state=state)
