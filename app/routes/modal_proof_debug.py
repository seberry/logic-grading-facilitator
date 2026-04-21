from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import quote

from fastapi import APIRouter, File, Form, Request, Response, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from app.autograde import AutoGradeResult, autograde_submission
from app.storage import DEBUG_MODAL_PROOF_DIR, RAW_SCANS_DIR
from proofs_ocr.pipelines.modal_v2 import (
    build_vlm_client as build_modal_v2_vlm_client,
    transcribe_image_to_modal_v2_debug_state,
)
from scripts.debug_modal_proof_ocr import (
    build_vlm_client as build_legacy_vlm_client,
    default_unary_verifier_model,
    transcribe_image_to_modal_debug_state,
)

from . import modal_proof_debug_artifacts as artifact_helpers
from . import modal_proof_debug_fixtures as fixture_helpers
from . import modal_proof_debug_page as page_helpers
from . import modal_proof_debug_state as state_helpers

router = APIRouter()

TEMPLATES_DIR = Path(__file__).resolve().parents[1] / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
REPO_ROOT = Path(__file__).resolve().parents[2]
MODAL_REGRESSION_MANIFEST_PATH = REPO_ROOT / "proofs_ocr" / "regression" / "manifest.yaml"

STATE_FILENAME = "state.json"
OPTIONAL_TEXT_ARTIFACTS = ("notes.md", "source.txt")
OCR_TEXT_ARTIFACTS = (
    "ocr_prompt.txt",
    "intermediate_ROOT.json",
    "intermediate_1.json",
    "intermediate_2.json",
    "node_rows_ROOT.json",
    "node_rows_1.json",
    "node_rows_2.json",
    "node_entries_ROOT.json",
    "node_entries_1.json",
    "node_entries_2.json",
    "strict_opener_1.json",
    "strict_opener_2.json",
    "response_ROOT.json",
    "response_1.json",
    "response_2.json",
    "bar_filter_debug.json",
    "unary_structured_probe.json",
    "unary_structured_formula_probe.json",
    "unary_blob_probe.json",
    "final_structure_baseline.json",
    "final_structure.json",
    "unary_verifier.json",
    "verifier_rewrites.json",
)
IMAGE_ARTIFACTS = (
    "source_image",
    "debug_bars.jpg",
    "crop_ROOT.jpg",
    "crop_1.jpg",
    "crop_2.jpg",
    "node_rows_overlay_ROOT.jpg",
    "node_rows_overlay_1.jpg",
    "node_rows_overlay_2.jpg",
    "strict_opener_line_1.jpg",
    "strict_opener_line_2.jpg",
    "margin_anchor_margin_base.jpg",
    "margin_anchor_margin_upscaled_4x.jpg",
    "margin_anchor_binary.png",
    "margin_anchor_candidate_boxes.jpg",
    "margin_anchor_strip_overlay.jpg",
)
STRICT_OPENER_STAGE_SPECS = (
    ("1", "Subproof opener 1", "strict_opener_line_1.jpg", "strict_opener_1.json"),
    ("2", "Subproof opener 2", "strict_opener_line_2.jpg", "strict_opener_2.json"),
)
LINE_NUMBER_DEBUG_IMAGE_NAMES = {
    "margin_anchor_margin_base.jpg",
    "margin_anchor_margin_upscaled_4x.jpg",
    "margin_anchor_binary.png",
    "margin_anchor_candidate_boxes.jpg",
    "margin_anchor_strip_overlay.jpg",
}
SYSTEM_OPTIONS = ["K", "T", "S4", "S5"]
OCR_PIPELINE_OPTIONS = (
    ("legacy_modal_debug", "Legacy modal OCR"),
    ("modal_v2_baseline", "modal_v2 baseline"),
)
OCR_PIPELINE_LABELS = {pipeline_id: pipeline_label for pipeline_id, pipeline_label in OCR_PIPELINE_OPTIONS}
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
    return state_helpers.clean_example_slug(raw, STARTER_EXAMPLE)


def list_modal_proof_examples() -> List[str]:
    return state_helpers.list_modal_proof_examples(DEBUG_MODAL_PROOF_DIR)


def _state_path(example_slug: str) -> Path:
    return state_helpers.state_path(DEBUG_MODAL_PROOF_DIR, example_slug, STATE_FILENAME)


def _artifact_text(example_slug: str, filename: str) -> str:
    return artifact_helpers.artifact_text(DEBUG_MODAL_PROOF_DIR, example_slug, filename)


def _artifact_url(example_slug: str, filename: str) -> str:
    return artifact_helpers.artifact_url(DEBUG_MODAL_PROOF_DIR, example_slug, filename)


def _default_page_state() -> Dict[str, Any]:
    return state_helpers.default_page_state(DEFAULT_PROOF_JSON_TEXT)


def _load_modal_regression_fixtures() -> List[Dict[str, str]]:
    return fixture_helpers.load_modal_regression_fixtures(MODAL_REGRESSION_MANIFEST_PATH, REPO_ROOT)


def _fixture_option_label(fixture: Dict[str, str]) -> str:
    return fixture_helpers.fixture_option_label(fixture)


def _fixture_lookup(fixtures: List[Dict[str, str]]) -> Dict[str, Dict[str, str]]:
    return fixture_helpers.fixture_lookup(fixtures)


def _load_example_state(example_slug: Optional[str]) -> Dict[str, Any]:
    return state_helpers.load_example_state(
        DEBUG_MODAL_PROOF_DIR,
        example_slug,
        STATE_FILENAME,
        DEFAULT_PROOF_JSON_TEXT,
    )


def _save_example_state(example_slug: str, state: Dict[str, Any]) -> None:
    state_helpers.save_example_state(
        DEBUG_MODAL_PROOF_DIR,
        example_slug,
        state,
        STATE_FILENAME,
        DEFAULT_PROOF_JSON_TEXT,
    )


def _checker_issues(checker_response: Any, key: str) -> List[Dict[str, Any]]:
    return page_helpers.checker_issues(checker_response, key)


def _checker_lines(checker_response: Any) -> List[Dict[str, Any]]:
    return page_helpers.checker_lines(checker_response)


def _checker_request_lines(checker_request: Any) -> List[Dict[str, Any]]:
    return page_helpers.checker_request_lines(checker_request)


def _build_ascii_proof_records(lines: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    return page_helpers.build_ascii_proof_records(lines)


def _parse_review_decisions(raw: Any) -> Dict[str, Dict[str, str]]:
    return state_helpers.parse_review_decisions(raw)


def _proof_lines_from_text(text: str) -> Dict[int, Dict[str, Any]]:
    return page_helpers.proof_lines_from_text(text)


def _checker_issues_by_line(checker_response: Any) -> Dict[int, List[Dict[str, str]]]:
    return page_helpers.checker_issues_by_line(checker_response)


def _read_json_artifact(path: Path) -> Any:
    return artifact_helpers.read_json_artifact(path)


def _build_review_panel(
    *,
    state: Dict[str, Any],
    unary_verifier_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    return page_helpers.build_review_panel(
        state=state,
        unary_verifier_results=unary_verifier_results,
        parse_review_decisions=_parse_review_decisions,
    )


def _load_unary_verifier_results(example_slug: str) -> List[Dict[str, Any]]:
    return artifact_helpers.load_unary_verifier_results(DEBUG_MODAL_PROOF_DIR, example_slug, STARTER_EXAMPLE)


def _render_checker_request_preview(checker_request: Any) -> str:
    return page_helpers.render_checker_request_preview(checker_request)


def _render_page(
    request: Request,
    *,
    example: Optional[str],
    state: Dict[str, Any],
) -> Response:
    return page_helpers.render_page(
        request,
        example=example,
        state=state,
        deps=page_helpers.RenderDeps(
            templates=templates,
            debug_dir=DEBUG_MODAL_PROOF_DIR,
            starter_example=STARTER_EXAMPLE,
            optional_text_artifacts=OPTIONAL_TEXT_ARTIFACTS,
            ocr_text_artifacts=OCR_TEXT_ARTIFACTS,
            image_artifacts=IMAGE_ARTIFACTS,
            strict_opener_stage_specs=STRICT_OPENER_STAGE_SPECS,
            line_number_debug_image_names=LINE_NUMBER_DEBUG_IMAGE_NAMES,
            system_options=SYSTEM_OPTIONS,
            ocr_pipeline_options=OCR_PIPELINE_OPTIONS,
            ocr_pipeline_labels=OCR_PIPELINE_LABELS,
            list_modal_proof_examples=list_modal_proof_examples,
            load_modal_regression_fixtures=_load_modal_regression_fixtures,
            fixture_option_label=_fixture_option_label,
            artifact_text=_artifact_text,
            artifact_url=_artifact_url,
            load_unary_verifier_results=_load_unary_verifier_results,
            checker_issues=_checker_issues,
            checker_lines=_checker_lines,
            build_review_panel=_build_review_panel,
            render_checker_request_preview=_render_checker_request_preview,
            build_ascii_proof_records=_build_ascii_proof_records,
            proof_lines_from_text=_proof_lines_from_text,
            read_json_artifact=_read_json_artifact,
            load_node_response=artifact_helpers.load_node_response,
            node_response_summary=artifact_helpers.node_response_summary,
            build_stitched_line_lookup=artifact_helpers.build_stitched_line_lookup,
            find_line_artifact=artifact_helpers.find_line_artifact,
            row_order_to_visible_line_order=artifact_helpers.row_order_to_visible_line_order,
        ),
    )


def _state_from_form(
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
) -> tuple[str, Dict[str, Any]]:
    return state_helpers.state_from_form(
        example=example,
        system=system,
        include_derived_rules=include_derived_rules,
        conclusion=conclusion,
        vision_base_url=vision_base_url,
        vision_model=vision_model,
        verifier_vision_model=verifier_vision_model,
        ocr_pipeline=ocr_pipeline,
        enable_formula_justification_separation=enable_formula_justification_separation,
        fixture_id=fixture_id,
        proof_json_text=proof_json_text,
        review_decisions_json=review_decisions_json,
        draft_comment=draft_comment,
        draft_grade=draft_grade,
        clean_slug=_clean_example_slug,
        load_state=_load_example_state,
        parse_decisions=_parse_review_decisions,
    )


def _build_autograde_config(state: Dict[str, Any]) -> Dict[str, Any]:
    return state_helpers.build_autograde_config(state)


def _find_saved_source_image(example_slug: str, state: Dict[str, Any]) -> Optional[Path]:
    return state_helpers.find_saved_source_image(DEBUG_MODAL_PROOF_DIR, example_slug, state)


def _resolve_source_image_path(example_slug: str, raw_path: str) -> Optional[Path]:
    return state_helpers.resolve_source_image_path(DEBUG_MODAL_PROOF_DIR, RAW_SCANS_DIR, example_slug, raw_path)


async def _persist_uploaded_source_image(
    example_slug: str,
    proof_image: UploadFile | None,
) -> Optional[Path]:
    return await state_helpers.persist_uploaded_source_image(DEBUG_MODAL_PROOF_DIR, example_slug, proof_image)


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


@router.get("/debug/modal-proof/save", response_class=HTMLResponse)
@router.get("/debug/modal-proof/run_checker", response_class=HTMLResponse)
@router.get("/debug/modal-proof/run_ocr", response_class=HTMLResponse)
async def modal_proof_debug_action_redirect(
    example: Optional[str] = None,
) -> Response:
    example_slug = _clean_example_slug(example)
    target = "/debug/modal-proof"
    if example_slug:
        target += f"?example={quote(example_slug)}"
    return RedirectResponse(url=target, status_code=303)


@router.post("/debug/modal-proof/save", response_class=HTMLResponse)
async def save_modal_proof_state(
    request: Request,
    example: str = Form(STARTER_EXAMPLE),
    system: str = Form("K"),
    include_derived_rules: bool = Form(False),
    conclusion: str = Form(""),
    vision_base_url: str = Form(""),
    vision_model: str = Form(""),
    verifier_vision_model: str = Form(""),
    ocr_pipeline: str = Form("legacy_modal_debug"),
    enable_formula_justification_separation: bool = Form(False),
    fixture_id: str = Form(""),
    proof_json_text: str = Form(...),
    review_decisions_json: str = Form(""),
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
        verifier_vision_model=verifier_vision_model,
        ocr_pipeline=ocr_pipeline,
        enable_formula_justification_separation=enable_formula_justification_separation,
        fixture_id=fixture_id,
        proof_json_text=proof_json_text,
        review_decisions_json=review_decisions_json,
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
    example: str = Form(STARTER_EXAMPLE),
    system: str = Form("K"),
    include_derived_rules: bool = Form(False),
    conclusion: str = Form(""),
    vision_base_url: str = Form(""),
    vision_model: str = Form(""),
    verifier_vision_model: str = Form(""),
    ocr_pipeline: str = Form("legacy_modal_debug"),
    enable_formula_justification_separation: bool = Form(False),
    fixture_id: str = Form(""),
    proof_json_text: str = Form(...),
    review_decisions_json: str = Form(""),
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
        verifier_vision_model=verifier_vision_model,
        ocr_pipeline=ocr_pipeline,
        enable_formula_justification_separation=enable_formula_justification_separation,
        fixture_id=fixture_id,
        proof_json_text=proof_json_text,
        review_decisions_json=review_decisions_json,
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
    return RedirectResponse(
        url=f"/debug/modal-proof?example={quote(example_slug)}&status={quote(str(state.get('status') or ''))}&message={quote(str(state.get('message') or ''))}",
        status_code=303,
    )


@router.post("/debug/modal-proof/run_ocr", response_class=HTMLResponse)
async def run_modal_proof_debug_ocr(
    example: str = Form(STARTER_EXAMPLE),
    system: str = Form("K"),
    include_derived_rules: bool = Form(False),
    conclusion: str = Form(""),
    vision_base_url: str = Form(""),
    vision_model: str = Form(""),
    verifier_vision_model: str = Form(""),
    ocr_pipeline: str = Form("legacy_modal_debug"),
    enable_formula_justification_separation: bool = Form(False),
    fixture_id: str = Form(""),
    proof_json_text: str = Form(DEFAULT_PROOF_JSON_TEXT),
    review_decisions_json: str = Form(""),
    draft_comment: str = Form(""),
    draft_grade: str = Form(""),
    proof_image_path: str = Form(""),
    proof_image: UploadFile | None = File(None),
    use_fixture_source: bool = Form(False),
) -> Response:
    regression_fixtures = _load_modal_regression_fixtures()
    fixture_map = _fixture_lookup(regression_fixtures)
    fixture = fixture_map.get(str(fixture_id or "").strip())
    example_value = fixture_id if use_fixture_source and fixture else example
    example_slug, state = _state_from_form(
        example=example_value,
        system=system,
        include_derived_rules=include_derived_rules,
        conclusion=conclusion,
        vision_base_url=vision_base_url,
        vision_model=vision_model,
        verifier_vision_model=verifier_vision_model,
        ocr_pipeline=ocr_pipeline,
        enable_formula_justification_separation=enable_formula_justification_separation,
        fixture_id=fixture_id,
        proof_json_text=proof_json_text,
        review_decisions_json=review_decisions_json,
        draft_comment=draft_comment,
        draft_grade=draft_grade,
    )

    example_dir = DEBUG_MODAL_PROOF_DIR / example_slug
    example_dir.mkdir(parents=True, exist_ok=True)

    try:
        uploaded_image_path = await _persist_uploaded_source_image(example_slug, proof_image)
        fixture_source_path: Optional[Path] = None
        if use_fixture_source:
            if fixture is None:
                raise ValueError("Choose a regression fixture before running fixture OCR.")
            fixture_source_path = Path(str(fixture.get("absolute_image_path") or "")).resolve()
        source_image_path = (
            uploaded_image_path
            or fixture_source_path
            or _resolve_source_image_path(example_slug, proof_image_path)
            or _find_saved_source_image(example_slug, state)
        )
        if source_image_path is None:
            raise ValueError("Provide a proof image upload or an existing image path before running OCR.")
        if not source_image_path.exists() or not source_image_path.is_file():
            raise ValueError(f"OCR image not found: {source_image_path}")

        pipeline = str(state.get("ocr_pipeline") or "legacy_modal_debug").strip() or "legacy_modal_debug"
        if pipeline == "modal_v2_baseline":
            client = build_modal_v2_vlm_client(
                base_url=state.get("vision_base_url") or None,
                model=state.get("vision_model") or None,
            )
            ocr_state = transcribe_image_to_modal_v2_debug_state(
                image_path=source_image_path,
                output_dir=example_dir,
                modal_config=_build_autograde_config(state),
                client=client,
                vision_base_url=state.get("vision_base_url") or None,
                vision_model=state.get("vision_model") or None,
                verifier_vision_model=state.get("verifier_vision_model") or None,
                enable_formula_justification_separation=bool(
                    state.get("enable_formula_justification_separation", False)
                ),
            )
        else:
            client = build_legacy_vlm_client(
                base_url=state.get("vision_base_url") or None,
                model=state.get("vision_model") or None,
            )
            verifier_client = build_legacy_vlm_client(
                base_url=state.get("vision_base_url") or None,
                model=state.get("verifier_vision_model") or default_unary_verifier_model(),
            )
            ocr_state = transcribe_image_to_modal_debug_state(
                image_path=source_image_path,
                output_dir=example_dir,
                modal_config=_build_autograde_config(state),
                client=client,
                verifier_client=verifier_client,
                vision_base_url=state.get("vision_base_url") or None,
                vision_model=state.get("vision_model") or None,
                verifier_vision_model=state.get("verifier_vision_model") or None,
            )
        ocr_state["ocr_pipeline"] = pipeline
        ocr_state["enable_formula_justification_separation"] = bool(
            state.get("enable_formula_justification_separation", False)
        )
        ocr_state["draft_comment"] = draft_comment
        ocr_state["draft_grade"] = draft_grade
        ocr_state["review_decisions"] = {}
        ocr_state["fixture_id"] = str(fixture.get("id") or "").strip() if fixture else ""
        ocr_state["source_image_path"] = str(source_image_path.resolve())
        ocr_state["last_run_at"] = datetime.now().isoformat(timespec="seconds")
        ocr_state["status"] = "success"
        ocr_state["message"] = "OCR ran and updated the modal proof state."
        state = ocr_state
    except Exception as exc:
        state["status"] = "error"
        state["message"] = f"Modal OCR failed: {exc}"

    _save_example_state(example_slug, state)
    return RedirectResponse(
        url=f"/debug/modal-proof?example={quote(example_slug)}&status={quote(str(state.get('status') or ''))}&message={quote(str(state.get('message') or ''))}",
        status_code=303,
    )
