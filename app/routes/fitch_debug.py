from __future__ import annotations

import asyncio
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Form, HTTPException, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from app.storage import DEBUG_FITCH_DIR, RAW_SCANS_DIR
from app.verifier import ProofLine, check_proof, flatten_proof

router = APIRouter()

TEMPLATES_DIR = Path(__file__).resolve().parents[1] / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")


def scan_pipeline_artifacts(scan_slug: str) -> Dict[str, Any]:
    """Scan debug dir for VLM pipeline artifacts (crops, responses, bar detection)."""
    debug_dir = DEBUG_FITCH_DIR / scan_slug
    if not debug_dir.exists():
        return {"nodes": [], "bar_image": None}

    nodes: List[Dict[str, Any]] = []

    # Find crop/response pairs
    for child in sorted(debug_dir.iterdir()):
        name = child.name
        if name.startswith("crop_") and name.rsplit(".", 1)[-1].lower() in ("jpg", "jpeg", "png"):
            node_id = name[5:].rsplit(".", 1)[0]  # e.g. "ROOT", "1", "2"
            response_path = debug_dir / f"response_{node_id}.json"
            response_text = ""
            if response_path.exists():
                response_text = response_path.read_text(encoding="utf-8")
            img_url = f"/debug_fitch_assets/{scan_slug}/{name}"
            nodes.append({
                "id": node_id,
                "crop_url": img_url,
                "response_text": response_text,
            })

    bar_image = None
    bars_path = debug_dir / "debug_bars.jpg"
    if bars_path.exists():
        bar_image = f"/debug_fitch_assets/{scan_slug}/debug_bars.jpg"

    return {"nodes": nodes, "bar_image": bar_image}


def list_fitch_debug_scans() -> List[str]:
    if not DEBUG_FITCH_DIR.exists():
        return []
    return sorted([p.name for p in DEBUG_FITCH_DIR.iterdir() if p.is_dir()], key=str.lower)


def final_structure_path(scan_slug: str) -> Path:
    return DEBUG_FITCH_DIR / scan_slug / "final_structure.json"


def load_structure_text(scan_slug: str) -> str:
    path = final_structure_path(scan_slug)
    if path.exists():
        return path.read_text(encoding="utf-8")
    return ""


def find_scan_image_path(scan_slug: str) -> Optional[Path]:
    for ext in IMAGE_EXTENSIONS:
        candidate = RAW_SCANS_DIR / f"{scan_slug}{ext}"
        if candidate.exists():
            return candidate

    debug_dir = DEBUG_FITCH_DIR / scan_slug
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


def scan_image_url(scan_slug: str) -> Optional[str]:
    path = find_scan_image_path(scan_slug)
    if path is None:
        return None

    if RAW_SCANS_DIR in path.parents:
        return f"/scans/{path.name}"

    if DEBUG_FITCH_DIR in path.parents:
        rel = path.relative_to(DEBUG_FITCH_DIR)
        return f"/debug_fitch_assets/{rel.as_posix()}"

    return None


def _clean_meta(meta: Any) -> Dict[str, Any]:
    if not isinstance(meta, dict):
        return {"num_premises": None, "conclusion": ""}

    raw_num = meta.get("num_premises")
    num_premises: Optional[int] = None
    if raw_num is not None and raw_num != "":
        try:
            num_premises = max(0, int(raw_num))
        except (TypeError, ValueError):
            num_premises = None

    conclusion = meta.get("conclusion")
    conclusion_str = str(conclusion) if conclusion is not None else ""

    return {
        "num_premises": num_premises,
        "conclusion": conclusion_str,
    }


def normalize_structure_payload(payload: Any) -> Dict[str, Any]:
    lines: Any = []
    meta: Dict[str, Any] = {}

    if isinstance(payload, list):
        lines = payload
        meta = {}
    elif isinstance(payload, dict):
        candidate_lines = payload.get("lines")
        lines = candidate_lines if isinstance(candidate_lines, list) else payload.get("structure", [])
        meta = payload.get("meta", {})
    else:
        raise HTTPException(status_code=400, detail="final_structure.json must be a list or object containing lines/meta.")

    if not isinstance(lines, list):
        raise HTTPException(status_code=400, detail="'lines' must be a list (allowing nested subproofs).")

    return {
        "lines": lines,
        "meta": _clean_meta(meta),
    }


def load_structure_json(scan_slug: str) -> Dict[str, Any]:
    text = load_structure_text(scan_slug)
    if not text:
        return {"lines": [], "meta": {"num_premises": None, "conclusion": ""}}

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid JSON in final_structure.json: {exc}")

    return normalize_structure_payload(parsed)


def ensure_raw_scan_available(scan_slug: str) -> Path:
    existing = find_scan_image_path(scan_slug)
    if existing and RAW_SCANS_DIR in existing.parents:
        return existing

    if existing and DEBUG_FITCH_DIR in existing.parents:
        RAW_SCANS_DIR.mkdir(parents=True, exist_ok=True)
        destination = RAW_SCANS_DIR / existing.name
        if not destination.exists():
            shutil.copy(existing, destination)
        return destination

    raise HTTPException(
        status_code=404,
        detail=f"Could not locate a scan image for '{scan_slug}' in {RAW_SCANS_DIR} or {DEBUG_FITCH_DIR}.",
    )


async def run_subprocess_with_logs(command: List[str]) -> tuple[int, str]:
    proc = await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd=Path(__file__).resolve().parents[2],
    )
    stdout, _ = await proc.communicate()
    output = stdout.decode("utf-8", errors="replace")
    return proc.returncode, output


def structure_to_proof_lines(structure: List[Any]) -> List[Any]:
    lines: List[Any] = []
    for item in structure:
        if isinstance(item, list):
            lines.append(structure_to_proof_lines(item))
        elif isinstance(item, dict):
            lines.append(
                ProofLine(
                    wffstr=str(item.get("wffstr", "")).strip(),
                    jstr=str(item.get("jstr", "")).strip(),
                    location=[],
                )
            )
    return lines


def label_and_count_premises(structure: List[Any]) -> int:
    count = 0
    for item in structure:
        if isinstance(item, list):
            break
        if isinstance(item, dict):
            jstr = str(item.get("jstr", "")).strip()
            if not jstr:
                item["jstr"] = "Pr"
                count += 1
            elif jstr.lower() != "pr":
                break
    return count


@router.get("/debug/fitch", response_class=HTMLResponse)
async def fitch_debug_page(request: Request, scan: Optional[str] = None) -> Response:
    scans = list_fitch_debug_scans()
    selected = scan if scan in scans else (scans[0] if scans else None)

    structure_text = load_structure_text(selected) if selected else ""
    final_path = final_structure_path(selected) if selected else None

    pipeline_artifacts = scan_pipeline_artifacts(selected) if selected else {"nodes": [], "bar_image": None}

    return templates.TemplateResponse(
        "debug_fitch.html",
        {
            "request": request,
            "available_scans": scans,
            "selected_scan": selected,
            "structure_text": structure_text,
            "structure_json": load_structure_json(selected) if selected else {"lines": [], "meta": {"num_premises": None, "conclusion": ""}},
            "image_url": scan_image_url(selected) if selected else None,
            "final_structure_exists": bool(final_path and final_path.exists()),
            "pipeline_artifacts": pipeline_artifacts,
        },
    )


@router.post("/debug/fitch/save")
async def save_fitch_structure(scan: str = Form(...), structure: str = Form(...)) -> Response:
    target_dir = DEBUG_FITCH_DIR / scan
    if not target_dir.exists():
        raise HTTPException(status_code=404, detail=f"Debug folder not found for '{scan}'.")

    try:
        parsed_raw = json.loads(structure)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {exc}")

    normalized = normalize_structure_payload(parsed_raw)

    path = final_structure_path(scan)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(normalized, indent=2, ensure_ascii=False), encoding="utf-8")

    return JSONResponse({"status": "ok", "message": f"Saved {path.name}."})


@router.post("/debug/fitch/rerun")
async def rerun_fitch_pipeline(scan: str = Form(...)) -> Response:
    if not (DEBUG_FITCH_DIR / scan).exists():
        raise HTTPException(status_code=404, detail=f"Debug folder not found for '{scan}'.")

    ensure_raw_scan_available(scan)
    command = [sys.executable, "scripts/debug_fitch_ocr.py", f"{scan}.jpg"]

    image_path = find_scan_image_path(scan)
    if image_path:
        command[-1] = image_path.name

    code, output = await run_subprocess_with_logs(command)
    structure_text = load_structure_text(scan)
    normalized = load_structure_json(scan)

    status = "ok" if code == 0 else "error"
    return JSONResponse(
        {
            "status": status,
            "log": output,
            "structure": structure_text,
            "normalized": normalized,
        }
    )


@router.post("/debug/fitch/run_grader")
async def run_fitch_grader(scan: str = Form(...)) -> Response:
    normalized = load_structure_json(scan)

    structure_copy = json.loads(json.dumps(normalized["lines"]))
    meta = normalized.get("meta", {})

    num_premises = meta.get("num_premises")
    if num_premises is None:
        num_premises = label_and_count_premises(structure_copy)

    proof_lines = structure_to_proof_lines(structure_copy)
    flattened = flatten_proof(proof_lines, [])
    conclusion_guess = meta.get("conclusion") or (flattened[-1].wffstr if flattened else "")

    result = check_proof(proof_lines, numprems=num_premises, conc=conclusion_guess)

    log_lines = [
        f"Scan: {scan}",
        f"Lines: {len(flattened)}",
        f"Premises (used): {num_premises}",
        f"Target conclusion: {conclusion_guess or '(empty)'}",
        f"Conclusion reached: {'yes' if result.conc_reached else 'no'}",
    ]

    if result.issues:
        log_lines.append("Issues:")
        log_lines.extend(result.issues)
    else:
        log_lines.append("No rule issues detected.")

    return JSONResponse(
        {
            "status": "ok",
            "log": "\n".join(log_lines),
        }
    )
