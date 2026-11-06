"""Register Quiz 9 blank pages to scans and prepare private question locations.

Only geometry and aggregate diagnostics are printed. Student images remain in
DATA_DIR; this script never sends them to a model or stores derived crops.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import cv2
import fitz
import numpy as np

from app.routes.problem_strips import _segments_cache_matches_session
from app.storage import DEFAULT_SESSION_PATH, GRADES_DIR, RAW_SCANS_DIR, load_session


def question_rects(page: fitz.Page) -> dict[str, fitz.Rect]:
    starts: dict[int, float] = {}
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            content = "".join(span["text"] for span in line["spans"]).strip()
            match = re.match(r"([1-7])\.\s", content)
            if match:
                starts[int(match.group(1))] = line["bbox"][1]
    if set(starts) != set(range(1, 8)):
        raise ValueError("Blank page lacks the seven numbered questions")
    return {
        f"Q{n}": fitz.Rect(50, starts[n] - 11,
                              565, starts.get(n + 1, starts[n] + 55) - 8)
        for n in range(1, 8)
    }


def prepare(pdf_path: Path, write: bool = False) -> dict[str, object]:
    session = load_session(DEFAULT_SESSION_PATH)
    if (session.grading_key.quiz_id != "quiz9_monadic_fol_truth"
            or {p.id for p in session.grading_key.problems} != {f"Q{n}" for n in range(1, 8)}):
        raise ValueError("Active session is not the expected Quiz 9")
    doc = fitz.open(pdf_path)
    if len(doc) != 3:
        raise ValueError("Expected three blank PDF pages in A/B/C order")
    detector = cv2.SIFT_create(nfeatures=5000)
    matcher = cv2.BFMatcher()
    templates = {}
    for version, page in zip("ABC", doc):
        if f"Version {version}" not in page.get_text():
            raise ValueError(f"Blank page order differs at Version {version}")
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
        image = cv2.cvtColor(
            np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n),
            cv2.COLOR_RGB2GRAY)
        keypoints, descriptors = detector.detectAndCompute(image, None)
        if descriptors is None:
            raise ValueError(f"No blank-page features for Version {version}")
        templates[version] = (pix, keypoints, descriptors, question_rects(page))

    rows = {f"Q{n}": [] for n in range(1, 8)}
    inlier_counts: list[int] = []
    inlier_ratios: list[float] = []
    page_boxes: list[list[float]] = []
    for unit in session.work_units:
        version = unit.quiz_version
        if version not in templates or len(unit.page_filenames) != 1:
            raise ValueError("Each submission must have one scan and a confirmed A/B/C version")
        filename = unit.page_filenames[0]
        scan = cv2.imread(str(RAW_SCANS_DIR / filename), cv2.IMREAD_GRAYSCALE)
        if scan is None:
            raise ValueError("A submission scan could not be read")
        height, width = scan.shape
        pix, blank_points, blank_desc, rects = templates[version]
        scan_points, scan_desc = detector.detectAndCompute(scan, None)
        if scan_desc is None:
            raise ValueError("A scan has no registration features")
        pairs = matcher.knnMatch(blank_desc, scan_desc, k=2)
        matches = [a for a, b in pairs if a.distance < 0.75 * b.distance]
        if len(matches) < 100:
            raise ValueError("A scan has too few blank-page matches")
        source = np.float32([blank_points[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        target = np.float32([scan_points[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        transform, mask = cv2.findHomography(source, target, cv2.RANSAC, 5.0)
        if transform is None or mask is None:
            raise ValueError("A scan could not be aligned to its blank page")
        inliers = int(mask.sum())
        ratio = float(mask.mean())
        if inliers < 200 or ratio < 0.45:
            raise ValueError("A scan's blank-page alignment is uncertain")
        page_corners = np.float32(
            [[0, 0], [pix.width, 0], [pix.width, pix.height], [0, pix.height]]
        ).reshape(-1, 1, 2)
        mapped_page = cv2.perspectiveTransform(page_corners, transform).reshape(-1, 2)
        page_box = [float(mapped_page[:, 0].min()), float(mapped_page[:, 1].min()),
                    float(mapped_page[:, 0].max()), float(mapped_page[:, 1].max())]
        # A wrong version or spurious homography can pass a raw inlier count.
        if not (0 <= page_box[0] <= 60 and 75 <= page_box[1] <= 140
                and width - 35 <= page_box[2] <= width + 40
                and height - 85 <= page_box[3] <= height - 15):
            raise ValueError("A scan's mapped page geometry is implausible")
        inlier_counts.append(inliers)
        inlier_ratios.append(ratio)
        page_boxes.append(page_box)
        for problem_id, rect in rects.items():
            corners = np.float32([
                [rect.x0 * 2, rect.y0 * 2], [rect.x1 * 2, rect.y0 * 2],
                [rect.x1 * 2, rect.y1 * 2], [rect.x0 * 2, rect.y1 * 2],
            ]).reshape(-1, 1, 2)
            mapped = cv2.perspectiveTransform(corners, transform).reshape(-1, 2)
            x0 = max(0, float(mapped[:, 0].min()) - 8)
            y0 = max(0, float(mapped[:, 1].min()) - 5)
            x1 = min(width, float(mapped[:, 0].max()) + 8)
            y1 = min(height, float(mapped[:, 1].max()) + 5)
            bbox = [round(x0 / width, 6), round(y0 / height, 6),
                    round(x1 / width, 6), round(y1 / height, 6)]
            rows[problem_id].append({
                "unit_id": unit.unit_id, "page_filename": filename, "page_index": 0,
                "bbox": bbox, "present": True, "comment": "Registered from Quiz 9 blank PDF",
                "confidence": round(ratio, 3), "recognition_model": "blank_pdf_registration",
                "recognition_provider": "local_geometry",
            })

    if not session.work_units:
        raise ValueError("Session has no submissions")
    for problem_id, problem_rows in rows.items():
        if not _segments_cache_matches_session(session, problem_rows):
            raise ValueError(f"Invalid prepared cache for {problem_id}")
    report: dict[str, object] = {
        "submissions": len(session.work_units), "questions": len(rows),
        "min_inliers": min(inlier_counts), "min_inlier_ratio": round(min(inlier_ratios), 3),
        "page_box_range": [round(min(box[i] for box in page_boxes)) if i < 2 else
                           round(max(box[i] for box in page_boxes)) for i in range(4)],
        "written": write,
    }
    if write:
        for problem_id, problem_rows in rows.items():
            target = GRADES_DIR / f"segments_{problem_id}.json"
            if target.exists():
                raise FileExistsError(f"Refusing to overwrite existing {problem_id} cache")
        for problem_id, problem_rows in rows.items():
            target = GRADES_DIR / f"segments_{problem_id}.json"
            temporary = target.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(problem_rows, indent=2), encoding="utf-8")
            os.replace(temporary, target)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", required=True, type=Path)
    parser.add_argument("--write", action="store_true", help="Write validated private location caches")
    args = parser.parse_args()
    print(json.dumps(prepare(args.pdf, args.write), indent=2))
