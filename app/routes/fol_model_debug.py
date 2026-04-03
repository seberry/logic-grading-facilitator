from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import quote

from fastapi import APIRouter, Form, HTTPException, Request, Response
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from app.fol_model_checker import (
    FolModelParseError,
    build_draft_comment,
    build_trace_blocks,
    check_sentences,
)
from app.storage import DEBUG_FOL_MODEL_DIR, RAW_SCANS_DIR

router = APIRouter()

TEMPLATES_DIR = Path(__file__).resolve().parents[1] / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")
STATE_FILENAME = "state.json"
SUPPORTED_PROBLEM_TYPES = [
    {"id": "truth_value", "label": "Determine truth value"},
    {"id": "non_equivalence", "label": "Show non-equivalence"},
    {"id": "joint_consistency", "label": "Show joint consistency"},
    {"id": "non_validity", "label": "Show non-validity"},
    {"id": "non_logical_truth", "label": "Show non-logical truth"},
    {"id": "non_logical_falsehood", "label": "Show non-logical falsehood"},
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
        "problem_setup_text": """TYPE: truth_value
QUESTION: Determine whether each sentence is true or false in the given model.
DEFAULT_CORRECT_SCORE: 2
DEFAULT_INCORRECT_SCORE: 0
TARGETS:
- Pa
- ∀x (Px → Qx)
""",
        "model_text": DEFAULT_MODEL_TEXT,
        "sentences_text": DEFAULT_SENTENCES_TEXT,
        "trace_blocks": DEFAULT_TRACE_BLOCKS,
        "draft_comment": DEFAULT_DRAFT_COMMENT,
        "draft_grade": "",
        "checker_enabled": True,
    },
    "non_equivalence": {
        "problem_setup_text": """TYPE: non_equivalence
QUESTION: Give a model showing the two sentences are not logically equivalent.
DEFAULT_CORRECT_SCORE: 3
DEFAULT_INCORRECT_SCORE: 0
LEFT: ∀x Px
RIGHT: ∃x Px
SUCCESS_CONDITION: exactly one of LEFT or RIGHT must be true in the model
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
        "trace_blocks": [
            {
                "title": "Dummy non-equivalence trace",
                "result": "Design sketch",
                "body": """Goal: show exactly one sentence is true.

Check LEFT = ∀x Px
  Object 2 is a counterexample
Result: LEFT is false

Check RIGHT = ∃x Px
  Witness x = 1 works
Result: RIGHT is true

Conclusion: the two sentences differ in truth value in this model, so they are not logically equivalent.""",
            }
        ],
        "draft_comment": """Question: non-equivalence witness
Checker: planned FOL model checker

Design note:
For this problem type the future checker should evaluate both sentences in the same model and confirm that they come apart.
""",
        "draft_grade": "",
        "checker_enabled": False,
    },
    "joint_consistency": {
        "problem_setup_text": """TYPE: joint_consistency
QUESTION: Give a model in which all listed sentences are true together.
DEFAULT_CORRECT_SCORE: 3
DEFAULT_INCORRECT_SCORE: 0
TARGETS:
- ∃x Px
- ∃x Qx
- ∀x (Px → ¬Qx)
SUCCESS_CONDITION: every target sentence must be true
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
        "trace_blocks": [
            {
                "title": "Dummy joint-consistency trace",
                "result": "Design sketch",
                "body": """Goal: show all target sentences come out true in one model.

Sentence 1 succeeds with witness x = 1.
Sentence 2 succeeds with witness x = 2.
Sentence 3 succeeds because every P-object fails to be Q.

Conclusion: the set is jointly consistent because there is at least one model making them all true.""",
            }
        ],
        "draft_comment": """Question: joint consistency
Checker: planned FOL model checker

Design note:
For this type the future checker should mark the answer correct only if every target sentence is true in the supplied model.
""",
        "draft_grade": "",
        "checker_enabled": False,
    },
    "non_validity": {
        "problem_setup_text": """TYPE: non_validity
QUESTION: Give a model in which all premises are true and the conclusion is false.
DEFAULT_CORRECT_SCORE: 3
DEFAULT_INCORRECT_SCORE: 0
PREMISES:
- ∀x (Px → Qx)
- ∃x Px
CONCLUSION: ∀x Qx
SUCCESS_CONDITION: every premise true, conclusion false
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
        "trace_blocks": [
            {
                "title": "Dummy non-validity trace",
                "result": "Design sketch",
                "body": """Goal: premises true, conclusion false.

Premise 1: true
Premise 2: true
Conclusion: false because object 2 is not Q

Conclusion: this model is a countermodel to the argument, so the argument is not valid.""",
            }
        ],
        "draft_comment": """Question: non-validity countermodel
Checker: planned FOL model checker

Design note:
This type is just a structured special case: evaluate premises as all true, then conclusion as false.
""",
        "draft_grade": "",
        "checker_enabled": False,
    },
    "non_logical_truth": {
        "problem_setup_text": """TYPE: non_logical_truth
QUESTION: Give a model in which the target sentence is true.
DEFAULT_CORRECT_SCORE: 2
DEFAULT_INCORRECT_SCORE: 0
TARGET: ∃x Px
SUCCESS_CONDITION: target sentence true
""",
        "model_text": """DOMAIN: {1}

CONSTANTS:

PREDICATES:
P: {1}
""",
        "sentences_text": """1. ∃x Px
""",
        "trace_blocks": [
            {
                "title": "Dummy non-logical-truth trace",
                "result": "Design sketch",
                "body": """Goal: show that the sentence is satisfiable.

The checker should verify only that the target sentence is true in the supplied model.
That demonstrates non-logical truth by example.""",
            }
        ],
        "draft_comment": """Question: non-logical truth
Checker: planned FOL model checker

Design note:
This is the one-sentence satisfiability case.
""",
        "draft_grade": "",
        "checker_enabled": False,
    },
    "non_logical_falsehood": {
        "problem_setup_text": """TYPE: non_logical_falsehood
QUESTION: Give a model in which the target sentence is false.
DEFAULT_CORRECT_SCORE: 2
DEFAULT_INCORRECT_SCORE: 0
TARGET: ∀x Px
SUCCESS_CONDITION: target sentence false
""",
        "model_text": """DOMAIN: {1,2}

CONSTANTS:

PREDICATES:
P: {1}
""",
        "sentences_text": """1. ∀x Px
""",
        "trace_blocks": [
            {
                "title": "Dummy non-logical-falsehood trace",
                "result": "Design sketch",
                "body": """Goal: show that the sentence is falsifiable.

The checker should verify only that the target sentence is false in the supplied model.
That demonstrates non-logical falsehood by counterexample.""",
            }
        ],
        "draft_comment": """Question: non-logical falsehood
Checker: planned FOL model checker

Design note:
This is the one-sentence falsifiability case.
""",
        "draft_grade": "",
        "checker_enabled": False,
    },
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


def _default_page_state() -> Dict[str, Any]:
    return {
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
        "problem_setup_text",
        "model_text",
        "sentences_text",
        "trace_blocks",
        "draft_comment",
        "draft_grade",
        "checker_enabled",
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
                "problem_setup_text": state.get("problem_setup_text", ""),
                "sentences_text": state.get("sentences_text", ""),
                "trace_blocks": state.get("trace_blocks", []),
                "draft_comment": state.get("draft_comment", ""),
                "draft_grade": state.get("draft_grade", ""),
                "checker_enabled": state.get("checker_enabled", True),
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


@router.get("/debug/fol-model", response_class=HTMLResponse)
async def fol_model_debug_page(
    request: Request,
    scan: Optional[str] = None,
    problem_type: Optional[str] = None,
) -> Response:
    state = _load_scan_state(scan)
    selected_type = problem_type or state.get("problem_type", "truth_value")
    if problem_type:
        state = _merge_preset(selected_type, state if scan and _state_path(scan).exists() else None)
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
    problem_type: str = Form("truth_value"),
    problem_setup_text: str = Form(""),
    model_text: str = Form(...),
    sentences_text: str = Form(...),
    draft_comment: str = Form(""),
    draft_grade: str = Form(""),
) -> Response:
    if not scan or not scan.strip():
        state = _default_page_state()
        state.update(
            {
                "problem_type": problem_type,
                "problem_setup_text": problem_setup_text,
                "model_text": model_text,
                "sentences_text": sentences_text,
                "draft_comment": draft_comment,
                "draft_grade": draft_grade,
                "checker_enabled": problem_type == "truth_value",
                "status": "error",
                "message": "Pick a scan slug before saving state to disk.",
            }
        )
        return _render_page(request, scan=None, state=state)

    scan_slug = scan.strip()
    state = _load_scan_state(scan_slug)
    state.update(
        {
            "problem_type": problem_type,
            "problem_setup_text": problem_setup_text,
            "model_text": model_text,
            "sentences_text": sentences_text,
            "draft_comment": draft_comment,
            "draft_grade": draft_grade,
            "checker_enabled": problem_type == "truth_value",
            "status": "success",
            "message": "Saved current transcription state.",
        }
    )
    _save_scan_state(scan_slug, state)

    return RedirectResponse(
        url=f"/debug/fol-model?scan={quote(scan_slug)}&status=success&message={quote('Saved current transcription state.')}",
        status_code=303,
    )


@router.post("/debug/fol-model/run_checker", response_class=HTMLResponse)
async def run_fol_model_checker(
    request: Request,
    scan: Optional[str] = Form(None),
    problem_type: str = Form("truth_value"),
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
            "problem_setup_text": problem_setup_text,
            "model_text": model_text,
            "sentences_text": sentences_text,
            "draft_comment": draft_comment,
            "draft_grade": draft_grade,
            "checker_enabled": problem_type == "truth_value",
            "status": "",
            "message": "",
            "checker_error": "",
            "checker_ran": problem_type == "truth_value",
        }
    )

    if problem_type != "truth_value":
        preset = PROBLEM_TYPE_PRESETS.get(problem_type, PROBLEM_TYPE_PRESETS["truth_value"])
        state["trace_blocks"] = preset["trace_blocks"]
        state["draft_comment"] = draft_comment or preset["draft_comment"]
        state["status"] = "success"
        state["message"] = "This problem type is currently in design-preview mode. The live checker still only runs for truth-in-model."
    else:
        try:
            result = check_sentences(model_text, sentences_text)
            state["trace_blocks"] = build_trace_blocks(result)
            state["draft_comment"] = build_draft_comment(result)
            state["status"] = "success"
            state["message"] = "Checker ran successfully."
        except FolModelParseError as exc:
            state["checker_error"] = str(exc)
            state["status"] = "error"
            state["message"] = "Checker could not run."

    if scan_slug:
        _save_scan_state(scan_slug, state)

    return _render_page(request, scan=scan_slug, state=state)
