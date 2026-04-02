from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import quote


def _unique_path(dest: Path) -> Path:
    if not dest.exists():
        return dest
    stem = dest.stem
    suffix = dest.suffix
    parent = dest.parent
    counter = 2
    while True:
        candidate = parent / f"{stem}_{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]


def _is_image_filename(filename: str) -> bool:
    return Path(filename).suffix in IMAGE_EXTENSIONS


def _select_display_filename(
    page_filenames: List[str],
    seg_filename: Optional[str],
) -> Optional[str]:
    if seg_filename and _is_image_filename(seg_filename):
        return seg_filename
    for fname in page_filenames:
        if _is_image_filename(fname):
            return fname
    return None


def _build_page_options(page_filenames: List[str]) -> List[Dict[str, str]]:
    options: List[Dict[str, str]] = []
    for fname in page_filenames:
        if not _is_image_filename(fname):
            continue
        options.append(
            {
                "filename": fname,
                "url": f"/scans/{quote(fname)}",
            }
        )
    return options
