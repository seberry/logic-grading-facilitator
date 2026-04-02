from __future__ import annotations

import io
import json
import re
import shutil
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import quote
from uuid import uuid4

import fitz  # type: ignore
from docx import Document
from fastapi import APIRouter, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from PIL import Image

from app.helpers import _unique_path
from app.models import RosterEntry
from app.storage import DATA_DIR, FLASHCARDS_DIR, load_roster

router = APIRouter()

TEMPLATES_DIR = Path(__file__).resolve().parents[1] / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

FLASHCARDS_STATE_FILENAME = "state.json"
FLASHCARDS_IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def _flashcards_session_dir(session_id: str) -> Path:
    return FLASHCARDS_DIR / session_id


def _flashcards_state_path(session_id: str) -> Path:
    return _flashcards_session_dir(session_id) / FLASHCARDS_STATE_FILENAME


def _load_flashcards_state(session_id: str) -> Dict[str, Any]:
    path = _flashcards_state_path(session_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Flashcards session not found.")
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _save_flashcards_state(session_id: str, state: Dict[str, Any]) -> None:
    path = _flashcards_state_path(session_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def _normalize_for_match(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"[^a-z0-9]", "", value)
    return value


def _guess_name_from_filename(stem: str, roster_index: Dict[str, str]) -> str:
    cleaned = re.sub(r"\d+", "", stem).replace("_", "")
    key = _normalize_for_match(cleaned)
    return roster_index.get(key, "")


def _safe_extract_zip(zf: zipfile.ZipFile, dest_dir: Path) -> List[Path]:
    extracted: List[Path] = []
    for entry in zf.infolist():
        if entry.is_dir():
            continue
        name = entry.filename.replace("\\", "/")
        if name.startswith("__MACOSX/"):
            continue
        target = Path(name)
        if target.is_absolute() or ".." in target.parts:
            continue
        out_path = dest_dir / target
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with zf.open(entry) as src, out_path.open("wb") as f:
            shutil.copyfileobj(src, f)
        extracted.append(out_path)
    return extracted


def _render_pdf_to_images(pdf_path: Path, output_dir: Path) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    images: List[Path] = []
    doc = fitz.open(str(pdf_path))
    for idx, page in enumerate(doc):
        matrix = fitz.Matrix(2, 2)
        pix = page.get_pixmap(matrix=matrix)
        filename = f"{pdf_path.stem}_p{idx + 1}.jpg"
        target = _unique_path(output_dir / filename)
        pix.save(str(target))
        images.append(target)
    doc.close()
    return images


def _extract_docx_assets(docx_path: Path, output_dir: Path) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    doc = Document(str(docx_path))
    text = "\n".join(p.text for p in doc.paragraphs if p.text).strip()
    images: List[Path] = []
    for rel in doc.part.rels.values():
        rel_type = getattr(rel, "reltype", "")
        if "image" not in rel_type:
            continue
        blob = rel.target_part.blob
        suffix = Path(rel.target_ref).suffix or ".jpg"
        filename = f"{docx_path.stem}_img_{len(images) + 1}{suffix}"
        target = _unique_path(output_dir / filename)
        target.write_bytes(blob)
        images.append(target)
    return {"text": text, "images": images}


def _slugify_name(name: str) -> str:
    slug = name.strip().lower()
    slug = re.sub(r"[^a-z0-9]+", "_", slug)
    slug = slug.strip("_")
    return slug or "student"


def _parse_bbox_value(value: str) -> Optional[List[float]]:
    if not value:
        return None
    parts = [p.strip() for p in value.split(",")]
    if len(parts) != 4:
        return None
    try:
        coords = [float(p) for p in parts]
    except ValueError:
        return None
    clamped = [max(0.0, min(1.0, c)) for c in coords]
    return clamped


@router.get("/flashcards", response_class=HTMLResponse)
async def flashcards_setup(request: Request, status: str = "", message: str = "") -> Response:
    roster_dir = DATA_DIR / "rosters"
    roster_dir.mkdir(parents=True, exist_ok=True)
    available_rosters = sorted(
        [p.name for p in roster_dir.iterdir() if p.is_file() and p.suffix.lower() == ".csv"],
        key=str.lower,
    )
    return templates.TemplateResponse(
        "flashcards_setup.html",
        {
            "request": request,
            "available_rosters": available_rosters,
            "rosters_dir": str(roster_dir),
            "status": status,
            "message": message,
        },
    )


@router.post("/flashcards/upload")
async def flashcards_upload(
    roster_choice: str = Form(""),
    roster_upload: Optional[UploadFile] = File(None),
    dataset_zip: UploadFile = File(...),
) -> Response:
    if not dataset_zip or not dataset_zip.filename:
        raise HTTPException(status_code=400, detail="Please upload a ZIP file.")

    roster_dir = DATA_DIR / "rosters"
    roster_dir.mkdir(parents=True, exist_ok=True)

    roster_path: Optional[Path] = None
    if roster_upload and roster_upload.filename:
        filename = Path(roster_upload.filename).name
        if Path(filename).suffix.lower() != ".csv":
            raise HTTPException(status_code=400, detail="Roster upload must be a CSV.")
        dest = _unique_path(roster_dir / filename)
        with dest.open("wb") as f:
            shutil.copyfileobj(roster_upload.file, f)
        roster_path = dest
    elif roster_choice:
        candidate = roster_dir / roster_choice
        if not candidate.exists():
            raise HTTPException(status_code=400, detail="Selected roster not found.")
        roster_path = candidate

    roster_entries: List[RosterEntry] = []
    if roster_path:
        roster_entries = load_roster(roster_path)

    roster_index = {
        _normalize_for_match(entry.name): entry.name for entry in roster_entries
    }

    session_id = f"{datetime.now().strftime('%Y-%m-%d_%H%M%S')}_{uuid4().hex[:6]}"
    session_dir = _flashcards_session_dir(session_id)
    sources_dir = session_dir / "sources"
    raw_dir = sources_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    zip_bytes = await dataset_zip.read()
    if not zip_bytes:
        raise HTTPException(status_code=400, detail="Uploaded ZIP was empty.")

    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            _safe_extract_zip(zf, raw_dir)
    except zipfile.BadZipFile:
        raise HTTPException(status_code=400, detail="Uploaded ZIP could not be read.")

    items: List[Dict[str, Any]] = []
    for file_path in sorted(raw_dir.rglob("*")):
        if not file_path.is_file():
            continue
        suffix = file_path.suffix.lower()
        if suffix == ".pdf":
            name_guess = _guess_name_from_filename(file_path.stem, roster_index)
            try:
                images = _render_pdf_to_images(file_path, sources_dir)
            except Exception:
                continue
            for img_path in images:
                items.append(
                    {
                        "id": len(items) + 1,
                        "source_image": img_path.relative_to(session_dir).as_posix(),
                        "source_text": "",
                        "name": name_guess,
                        "facts": "",
                        "bbox": None,
                        "skip": False,
                    }
                )
            continue
        if suffix == ".docx":
            name_guess = _guess_name_from_filename(file_path.stem, roster_index)
            try:
                assets = _extract_docx_assets(file_path, sources_dir)
            except Exception:
                continue
            text = assets.get("text", "")
            images = assets.get("images", [])
            for img_path in images:
                items.append(
                    {
                        "id": len(items) + 1,
                        "source_image": img_path.relative_to(session_dir).as_posix(),
                        "source_text": text,
                        "name": name_guess,
                        "facts": "",
                        "bbox": None,
                        "skip": False,
                    }
                )
            continue
        if suffix in FLASHCARDS_IMAGE_EXTS:
            name_guess = _guess_name_from_filename(file_path.stem, roster_index)
            target = _unique_path(sources_dir / file_path.name)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file_path, target)
            items.append(
                {
                    "id": len(items) + 1,
                    "source_image": target.relative_to(session_dir).as_posix(),
                    "source_text": "",
                    "name": name_guess,
                    "facts": "",
                    "bbox": None,
                    "skip": False,
                }
            )

    if not items:
        return RedirectResponse(
            url="/flashcards?status=error&message="
            + quote("No usable files found in the ZIP."),
            status_code=303,
        )

    state = {
        "session_id": session_id,
        "created_at": datetime.now().isoformat(),
        "roster_path": str(roster_path) if roster_path else None,
        "items": items,
    }
    _save_flashcards_state(session_id, state)

    return RedirectResponse(
        url=f"/flashcards/review?session={quote(session_id)}&index=0",
        status_code=303,
    )


@router.get("/flashcards/review", response_class=HTMLResponse)
async def flashcards_review(
    request: Request,
    session: str,
    index: int = 0,
) -> Response:
    state = _load_flashcards_state(session)
    items = state.get("items", [])
    session_dir = _flashcards_session_dir(session)
    dataset_path = session_dir / "dataset.json"
    dataset_url = None
    if dataset_path.exists():
        dataset_url = f"/flashcards_assets/{session}/dataset.json"

    if not items or index >= len(items):
        return templates.TemplateResponse(
            "flashcards_done.html",
            {
                "request": request,
                "session_id": session,
                "dataset_url": dataset_url,
                "dataset_count": 0,
            },
        )

    index = max(0, index)
    item = items[index]

    roster_names: List[str] = []
    roster_path = state.get("roster_path")
    if roster_path:
        roster_names = [entry.name for entry in load_roster(Path(roster_path))]

    source_image = item.get("source_image", "")
    source_url = (
        f"/flashcards_assets/{session}/{source_image}"
        if source_image
        else ""
    )
    bbox = item.get("bbox")
    bbox_value = ",".join(str(v) for v in bbox) if bbox else ""

    return templates.TemplateResponse(
        "flashcards_review.html",
        {
            "request": request,
            "session_id": session,
            "index": index,
            "total": len(items),
            "item": item,
            "source_url": source_url,
            "bbox_value": bbox_value,
            "roster_names": roster_names,
            "has_prev": index > 0,
            "has_next": index + 1 < len(items),
        },
    )


@router.post("/flashcards/review")
async def flashcards_review_submit(
    request: Request,
    session_id: str = Form(...),
    index: int = Form(...),
    name: str = Form(""),
    facts: str = Form(""),
    bbox: str = Form(""),
    action: str = Form("next"),
) -> Response:
    state = _load_flashcards_state(session_id)
    items = state.get("items", [])
    if index < 0 or index >= len(items):
        raise HTTPException(status_code=400, detail="Invalid flashcards index.")

    item = items[index]
    item["name"] = name.strip()
    item["facts"] = facts.strip()
    item["bbox"] = _parse_bbox_value(bbox)
    item["skip"] = action == "skip"

    _save_flashcards_state(session_id, state)

    if action == "export":
        return await flashcards_export(request, session_id)

    if action == "prev":
        target_index = max(0, index - 1)
    else:
        target_index = index + 1

    return RedirectResponse(
        url=f"/flashcards/review?session={quote(session_id)}&index={target_index}",
        status_code=303,
    )


@router.post("/flashcards/export", response_class=HTMLResponse)
async def flashcards_export(request: Request, session_id: str = Form(...)) -> Response:
    state = _load_flashcards_state(session_id)
    items = state.get("items", [])
    session_dir = _flashcards_session_dir(session_id)
    images_dir = session_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    name_counts: Dict[str, int] = {}
    dataset: List[Dict[str, str]] = []

    for item in items:
        if item.get("skip"):
            continue
        name = (item.get("name") or "").strip()
        bbox = item.get("bbox")
        if not name or not bbox:
            continue
        source_rel = item.get("source_image") or ""
        if not source_rel:
            continue
        source_path = session_dir / source_rel
        if not source_path.exists():
            continue

        with Image.open(source_path) as img:
            width, height = img.size
            x1 = int(round(bbox[0] * width))
            y1 = int(round(bbox[1] * height))
            x2 = int(round(bbox[2] * width))
            y2 = int(round(bbox[3] * height))
            x1 = max(0, min(width, x1))
            y1 = max(0, min(height, y1))
            x2 = max(0, min(width, x2))
            y2 = max(0, min(height, y2))
            if x2 <= x1 or y2 <= y1:
                continue
            crop = img.crop((x1, y1, x2, y2))

            base = _slugify_name(name)
            count = name_counts.get(base, 0) + 1
            name_counts[base] = count

            suffix = source_path.suffix.lower()
            if suffix == ".jpeg":
                suffix = ".jpg"
            if suffix not in {".jpg", ".png"}:
                suffix = ".jpg"

            filename = f"{base}{suffix}" if count == 1 else f"{base}_{count:02d}{suffix}"
            target = images_dir / filename
            crop.save(target)

        dataset.append(
            {
                "name": name,
                "photo_path": f"images/{filename}",
                "facts": (item.get("facts") or "").strip(),
            }
        )

    dataset_path = session_dir / "dataset.json"
    dataset_path.write_text(json.dumps(dataset, indent=2), encoding="utf-8")
    dataset_url = f"/flashcards_assets/{session_id}/dataset.json"

    return templates.TemplateResponse(
        "flashcards_done.html",
        {
            "request": request,
            "session_id": session_id,
            "dataset_url": dataset_url,
            "dataset_count": len(dataset),
        },
    )
