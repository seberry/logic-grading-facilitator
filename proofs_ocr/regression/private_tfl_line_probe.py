"""Private single-row probes for handwritten TFL/Fitch OCR artifacts.

This module is for local FERPA/private debugging only.  It writes raw model
responses under the private run directory, but prints and writes only redacted
shape diagnostics in its shareable summary.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.vlm_client import LMStudioQwenClient
from proofs_ocr.pipelines.tfl_line_reread_geometry import extract_single_line_crop
from proofs_ocr.pipelines.tfl_line_reread_vlm import json_from_text
from proofs_ocr.tricks.justification_snaps import clean_justification_for_tfl


RULE_HEAD_RE = re.compile(
    r"(?:∧|∨|→|↔|¬|⊥|&|v|~|-|8|A|B|ε|∈)\s*[EI]\b|(?:Pr|Hyp)\b",
    re.IGNORECASE,
)


def _contains_rule_head(value: object) -> bool:
    cleaned = clean_justification_for_tfl(str(value or ""))
    return bool(RULE_HEAD_RE.search(cleaned))


def _response_shape(payload: Any) -> tuple[str, dict[str, Any] | None]:
    if isinstance(payload, dict):
        return "object", payload
    if isinstance(payload, list):
        dict_items = [item for item in payload if isinstance(item, dict)]
        if len(dict_items) == 1:
            return "single_object_array", dict_items[0]
        return f"array_{len(payload)}", dict_items[0] if dict_items else None
    return type(payload).__name__, None


def _build_prompt(row_index: int, expected_lines: int) -> str:
    return (
        "Read only this single cropped row from a handwritten Fitch proof. "
        "Return exactly one JSON object with keys \"line\", \"wffstr\", and \"jstr\". "
        "Put only the formula on the left into \"wffstr\". Put only the right-side "
        "rule head and citations into \"jstr\". If no right-side justification is "
        "printed, use an empty string. Do not include explanation or markdown. "
        f"This is row {row_index} of {expected_lines} in its local subproof crop."
    )


def _build_side_prompt(*, side: str, row_index: int, expected_lines: int) -> str:
    if side == "left":
        return (
            "Read only the formula area in this cropped handwritten Fitch proof row. "
            "Ignore any right-side rule names or citation numbers if visible. "
            "Return exactly one JSON object with keys \"wffstr\" and \"jstr\". "
            "Put the formula in \"wffstr\" and always set \"jstr\" to an empty string. "
            "Return JSON only. "
            f"This is row {row_index} of {expected_lines} in its local subproof crop."
        )
    return (
        "Read only the right-side justification area in this cropped handwritten Fitch proof row. "
        "Ignore the formula if visible. Return exactly one JSON object with keys \"wffstr\" and \"jstr\". "
        "Always set \"wffstr\" to an empty string. Put only the printed rule head and citations in \"jstr\". "
        "If no right-side justification is printed, set \"jstr\" to an empty string. Return JSON only. "
        f"This is row {row_index} of {expected_lines} in its local subproof crop."
    )


def _build_justification_column_prompt(expected_justifications: int) -> str:
    return (
        "Read this cropped right-side justification column from a handwritten Fitch subproof. "
        "Ignore any formula text if visible. Return exactly one JSON object with key "
        "\"justifications\". The value must be a top-to-bottom array of strings. "
        f"Return exactly {expected_justifications} justification string(s), one for each "
        "derived row after the subproof assumption. If a row has no visible justification, "
        "use an empty string for that slot. Put only rule heads and citation numbers in the "
        "array. Do not explain or use markdown."
    )


def _split_row_sides(
    row_path: Path,
    *,
    left_path: Path,
    right_path: Path,
    split_ratio: float,
    overlap_px: int,
) -> tuple[Path, Path]:
    img = cv2.imread(str(row_path))
    if img is None:
        raise FileNotFoundError(f"Could not read row image: {row_path}")
    _h, w = img.shape[:2]
    split_x = max(1, min(w - 1, int(w * split_ratio)))
    left_x2 = max(1, min(w, split_x + overlap_px))
    right_x1 = max(0, min(w - 1, split_x - overlap_px))
    left_path.parent.mkdir(parents=True, exist_ok=True)
    right_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(left_path), img[:, :left_x2])
    cv2.imwrite(str(right_path), img[:, right_x1:])
    return left_path, right_path


def estimate_row_split_x(
    img: np.ndarray,
    *,
    fallback_ratio: float = 0.34,
) -> tuple[int, str]:
    """Estimate formula/justification boundary from foreground column gaps."""
    if img.size == 0:
        return 1, "empty"
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    h, w = gray.shape[:2]
    if w <= 2:
        return 1, "too_narrow"

    # Remove long horizontal strokes/ruling, then look for vertical whitespace
    # separating left formula ink from right justification ink.
    dark = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        51,
        11,
    )
    horizontal = cv2.morphologyEx(
        dark,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (max(30, int(w * 0.10)), 1)),
    )
    foreground = cv2.subtract(dark, horizontal)
    column_counts = (foreground > 0).sum(axis=0)

    # Ignore crop edges and very sparse speckles.
    min_ink = max(1, int(h * 0.025))
    active = np.where(column_counts >= min_ink)[0]
    if active.size < 2:
        return max(1, min(w - 1, int(w * fallback_ratio))), "fallback_sparse"

    segments: list[tuple[int, int]] = []
    start = int(active[0])
    prev = int(active[0])
    max_inner_gap = max(3, int(w * 0.006))
    for raw_col in active[1:]:
        col = int(raw_col)
        if col <= prev + max_inner_gap:
            prev = col
            continue
        segments.append((start, prev))
        start = col
        prev = col
    segments.append((start, prev))

    segments = [
        (x1, x2)
        for x1, x2 in segments
        if (x2 - x1 + 1) >= max(3, int(w * 0.006))
    ]
    if len(segments) < 2:
        return max(1, min(w - 1, int(w * fallback_ratio))), "fallback_one_cluster"

    max_boundary = int(w * min(max(fallback_ratio, 0.05), 0.78))
    candidates: list[tuple[int, int, int, int]] = []
    for left, right in zip(segments, segments[1:]):
        left_end = left[1]
        right_start = right[0]
        gap = right_start - left_end - 1
        if gap <= 0:
            continue
        boundary = (left_end + right_start) // 2
        if int(w * 0.18) <= boundary <= max_boundary:
            candidates.append((gap, boundary, left_end, right_start))
    if not candidates:
        return max(1, min(w - 1, int(w * fallback_ratio))), "fallback_no_gap"

    substantial_gap = max(6, int(w * 0.012))
    substantial = [candidate for candidate in candidates if candidate[0] >= substantial_gap]
    if substantial:
        # For tight formula/justification bleed, a later gap inside citations
        # can be wider than the actual column boundary. Prefer the first
        # substantial whitespace break.
        gap, boundary, _left_end, _right_start = min(substantial, key=lambda item: item[1])
    else:
        gap, boundary, _left_end, _right_start = max(candidates, key=lambda item: item[0])
    return max(1, min(w - 1, boundary)), f"gap_{gap}"


def _split_row_sides_auto(
    row_path: Path,
    *,
    left_path: Path,
    right_path: Path,
    fallback_ratio: float,
    overlap_px: int,
) -> tuple[Path, Path, int, str]:
    img = cv2.imread(str(row_path))
    if img is None:
        raise FileNotFoundError(f"Could not read row image: {row_path}")
    _h, w = img.shape[:2]
    split_x, split_source = estimate_row_split_x(img, fallback_ratio=fallback_ratio)
    left_x2 = max(1, min(w, split_x + overlap_px))
    right_x1 = max(0, min(w - 1, split_x - overlap_px))
    left_path.parent.mkdir(parents=True, exist_ok=True)
    right_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(left_path), img[:, :left_x2])
    cv2.imwrite(str(right_path), img[:, right_x1:])
    return left_path, right_path, split_x, split_source


def _call_json(
    *,
    client: Any,
    image_path: Path,
    prompt: str,
    model: str | None,
    timeout: int,
    max_tokens: int,
    disable_thinking: bool = False,
) -> tuple[str, Any, dict[str, Any], str]:
    try:
        if disable_thinking:
            prompt = f"/no_think\n{prompt}"
        messages = client.build_messages_for_prompt(image_path, prompt)
        extra_payload = {"chat_template_kwargs": {"enable_thinking": False}} if disable_thinking else None
        raw = client.chat(
            messages,
            model=model,
            timeout=timeout,
            max_tokens=max_tokens,
            extra_payload=extra_payload,
        )
        return "ok", json_from_text(raw), getattr(client, "last_response_metadata", {}) or {}, raw
    except Exception as exc:
        return f"error:{type(exc).__name__}", None, getattr(client, "last_response_metadata", {}) or {}, str(exc)


def run_probe(
    *,
    crop_path: Path,
    output_dir: Path,
    expected_lines: int,
    model: str | None,
    timeout: int,
    max_tokens: int,
    disable_thinking: bool = False,
) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_dir = output_dir / "rows"
    responses_dir = output_dir / "raw_responses"
    rows_dir.mkdir(parents=True, exist_ok=True)
    responses_dir.mkdir(parents=True, exist_ok=True)

    client = LMStudioQwenClient()
    records: list[dict[str, Any]] = []

    for row_index in range(1, expected_lines + 1):
        row_path = rows_dir / f"row_{row_index:02d}.jpg"
        extract_single_line_crop(
            crop_path,
            local_index=row_index,
            expected_lines=expected_lines,
            output_path=row_path,
            padding=18,
        )
        messages = client.build_messages_for_prompt(
            row_path,
            _build_prompt(row_index, expected_lines),
        )
        raw_path = responses_dir / f"row_{row_index:02d}.json"
        metadata_path = responses_dir / f"row_{row_index:02d}_metadata.json"

        status = "ok"
        payload: Any = None
        error = ""
        try:
            raw = client.chat(
                messages,
                model=model,
                timeout=timeout,
                max_tokens=max_tokens,
                extra_payload={"chat_template_kwargs": {"enable_thinking": False}} if disable_thinking else None,
            )
            raw_path.write_text(raw, encoding="utf-8")
            payload = json_from_text(raw)
        except Exception as exc:
            status = "error"
            error = type(exc).__name__
            raw_path.write_text(str(exc), encoding="utf-8")

        metadata = getattr(client, "last_response_metadata", {}) or {}
        metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

        shape, line_obj = _response_shape(payload)
        wff_has_rule = bool(line_obj and _contains_rule_head(line_obj.get("wffstr")))
        jstr_has_rule = bool(line_obj and _contains_rule_head(line_obj.get("jstr")))
        records.append(
            {
                "row_index": row_index,
                "status": status,
                "error_type": error,
                "response_shape": shape,
                "has_single_line_object": line_obj is not None and shape in {"object", "single_object_array"},
                "wff_has_rule_head": wff_has_rule,
                "jstr_has_rule_head": jstr_has_rule,
                "rule_head_location": (
                    "both"
                    if wff_has_rule and jstr_has_rule
                    else "wffstr"
                    if wff_has_rule
                    else "jstr"
                    if jstr_has_rule
                    else "neither"
                ),
                "thinking_chars": metadata.get("thinking_chars"),
                "content_chars": metadata.get("content_chars"),
                "finish_reason": metadata.get("finish_reason"),
            }
        )

    summary = {
        "kind": "private_tfl_single_line_probe_summary",
        "privacy": {
            "contains_paths": False,
            "contains_images": False,
            "contains_line_text": False,
            "contains_diffs": False,
        },
        "expected_lines": expected_lines,
        "rows": records,
    }
    (output_dir / "shareable_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    lines = ["Private TFL single-line probe", f"Rows: {expected_lines}"]
    for record in records:
        lines.append(
            "- row {row_index}: status={status} shape={response_shape} "
            "rule_location={rule_head_location} finish={finish_reason}".format(**record)
        )
    (output_dir / "shareable_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return records


def run_split_probe(
    *,
    crop_path: Path,
    output_dir: Path,
    expected_lines: int,
    model: str | None,
    timeout: int,
    max_tokens: int,
    split_ratio: float = 0.58,
    overlap_px: int = 18,
    split_strategy: str = "fixed",
    disable_thinking: bool = False,
) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_dir = output_dir / "rows"
    sides_dir = output_dir / "sides"
    responses_dir = output_dir / "raw_responses"
    rows_dir.mkdir(parents=True, exist_ok=True)
    sides_dir.mkdir(parents=True, exist_ok=True)
    responses_dir.mkdir(parents=True, exist_ok=True)

    client = LMStudioQwenClient()
    records: list[dict[str, Any]] = []

    for row_index in range(1, expected_lines + 1):
        row_path = rows_dir / f"row_{row_index:02d}.jpg"
        extract_single_line_crop(
            crop_path,
            local_index=row_index,
            expected_lines=expected_lines,
            output_path=row_path,
            padding=18,
        )
        if split_strategy == "auto-gap":
            left_path, right_path, split_x, split_source = _split_row_sides_auto(
                row_path,
                left_path=sides_dir / f"row_{row_index:02d}_left.jpg",
                right_path=sides_dir / f"row_{row_index:02d}_right.jpg",
                fallback_ratio=split_ratio,
                overlap_px=overlap_px,
            )
        else:
            left_path, right_path = _split_row_sides(
                row_path,
                left_path=sides_dir / f"row_{row_index:02d}_left.jpg",
                right_path=sides_dir / f"row_{row_index:02d}_right.jpg",
                split_ratio=split_ratio,
                overlap_px=overlap_px,
            )
            row_img = cv2.imread(str(row_path))
            split_x = int((row_img.shape[1] if row_img is not None else 1) * split_ratio)
            split_source = "fixed_ratio"

        left_status, left_payload, left_meta, left_raw = _call_json(
            client=client,
            image_path=left_path,
            prompt=_build_side_prompt(side="left", row_index=row_index, expected_lines=expected_lines),
            model=model,
            timeout=timeout,
            max_tokens=max_tokens,
            disable_thinking=disable_thinking,
        )
        right_status, right_payload, right_meta, right_raw = _call_json(
            client=client,
            image_path=right_path,
            prompt=_build_side_prompt(side="right", row_index=row_index, expected_lines=expected_lines),
            model=model,
            timeout=timeout,
            max_tokens=max_tokens,
            disable_thinking=disable_thinking,
        )
        (responses_dir / f"row_{row_index:02d}_left.json").write_text(left_raw, encoding="utf-8")
        (responses_dir / f"row_{row_index:02d}_right.json").write_text(right_raw, encoding="utf-8")
        (responses_dir / f"row_{row_index:02d}_left_metadata.json").write_text(
            json.dumps(left_meta, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (responses_dir / f"row_{row_index:02d}_right_metadata.json").write_text(
            json.dumps(right_meta, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        _left_shape, left_obj = _response_shape(left_payload)
        _right_shape, right_obj = _response_shape(right_payload)
        left_wff_has_rule = bool(left_obj and _contains_rule_head(left_obj.get("wffstr")))
        left_jstr_has_rule = bool(left_obj and _contains_rule_head(left_obj.get("jstr")))
        right_wff_has_rule = bool(right_obj and _contains_rule_head(right_obj.get("wffstr")))
        right_jstr_has_rule = bool(right_obj and _contains_rule_head(right_obj.get("jstr")))
        records.append(
            {
                "row_index": row_index,
                "split_x": int(split_x),
                "split_source": split_source,
                "left_status": left_status,
                "right_status": right_status,
                "left_has_single_object": left_obj is not None,
                "right_has_single_object": right_obj is not None,
                "left_wff_has_rule_head": left_wff_has_rule,
                "left_jstr_has_rule_head": left_jstr_has_rule,
                "right_wff_has_rule_head": right_wff_has_rule,
                "right_jstr_has_rule_head": right_jstr_has_rule,
                "left_rule_leak": left_wff_has_rule or left_jstr_has_rule,
                "right_rule_in_jstr": right_jstr_has_rule,
                "right_rule_in_wff": right_wff_has_rule,
                "left_finish_reason": left_meta.get("finish_reason"),
                "right_finish_reason": right_meta.get("finish_reason"),
            }
        )

    summary = {
        "kind": "private_tfl_split_line_probe_summary",
        "privacy": {
            "contains_paths": False,
            "contains_images": False,
            "contains_line_text": False,
            "contains_diffs": False,
        },
        "expected_lines": expected_lines,
        "split_ratio": split_ratio,
        "overlap_px": overlap_px,
        "split_strategy": split_strategy,
        "rows": records,
    }
    (output_dir / "shareable_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    lines = [
        "Private TFL split-line probe",
        f"Rows: {expected_lines}",
        f"Split ratio: {split_ratio}",
        f"Split strategy: {split_strategy}",
    ]
    for record in records:
        lines.append(
            "- row {row_index}: split={split_x} source={split_source} "
            "left_status={left_status} left_rule_leak={left_rule_leak} "
            "right_status={right_status} right_rule_in_jstr={right_rule_in_jstr} "
            "right_rule_in_wff={right_rule_in_wff}".format(**record)
        )
    (output_dir / "shareable_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return records


def run_justification_column_probe(
    *,
    crop_path: Path,
    output_dir: Path,
    expected_lines: int,
    model: str | None,
    timeout: int,
    max_tokens: int,
    split_ratio: float,
    overlap_px: int,
    disable_thinking: bool = False,
) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    responses_dir = output_dir / "raw_responses"
    responses_dir.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(str(crop_path))
    if img is None:
        raise FileNotFoundError(f"Could not read crop image: {crop_path}")
    _h, w = img.shape[:2]
    split_x = max(1, min(w - 1, int(w * split_ratio)))
    right_x1 = max(0, min(w - 1, split_x - overlap_px))
    column_path = output_dir / "justification_column.jpg"
    cv2.imwrite(str(column_path), img[:, right_x1:])

    client = LMStudioQwenClient()
    expected_justifications = max(0, expected_lines - 1)
    status, payload, metadata, raw = _call_json(
        client=client,
        image_path=column_path,
        prompt=_build_justification_column_prompt(expected_justifications),
        model=model,
        timeout=timeout,
        max_tokens=max_tokens,
        disable_thinking=disable_thinking,
    )
    (responses_dir / "justification_column.json").write_text(raw, encoding="utf-8")
    (responses_dir / "justification_column_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    justifications = payload.get("justifications") if isinstance(payload, dict) else None
    if not isinstance(justifications, list):
        justifications = []
    records: list[dict[str, Any]] = []
    for index in range(expected_justifications):
        value = justifications[index] if index < len(justifications) else ""
        records.append(
            {
                "slot_index": index + 1,
                "present": index < len(justifications),
                "has_rule_head": _contains_rule_head(value),
            }
        )

    summary = {
        "kind": "private_tfl_justification_column_probe_summary",
        "privacy": {
            "contains_paths": False,
            "contains_images": False,
            "contains_line_text": False,
            "contains_diffs": False,
        },
        "status": status,
        "expected_justifications": expected_justifications,
        "returned_count": len(justifications),
        "split_ratio": split_ratio,
        "overlap_px": overlap_px,
        "slots": records,
        "thinking_chars": metadata.get("thinking_chars"),
        "content_chars": metadata.get("content_chars"),
        "finish_reason": metadata.get("finish_reason"),
    }
    (output_dir / "shareable_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    lines = [
        "Private TFL justification-column probe",
        f"Expected justifications: {expected_justifications}",
        f"Returned count: {len(justifications)}",
        f"Status: {status}",
    ]
    for record in records:
        lines.append(
            "- slot {slot_index}: present={present} has_rule_head={has_rule_head}".format(**record)
        )
    (output_dir / "shareable_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a private redacted single-line OCR probe.")
    parser.add_argument("--crop", required=True, type=Path, help="Private full-width crop path.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Private output directory.")
    parser.add_argument("--expected-lines", required=True, type=int)
    parser.add_argument("--model", default="", help="Optional LM Studio model override.")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--max-tokens", type=int, default=160)
    parser.add_argument("--disable-thinking", action="store_true")
    parser.add_argument(
        "--mode",
        choices=["whole-row", "split-row", "justification-column"],
        default="whole-row",
        help="Probe whole rows, split left/right row zones, or the right-side column.",
    )
    parser.add_argument("--split-ratio", type=float, default=0.58)
    parser.add_argument("--overlap-px", type=int, default=18)
    parser.add_argument(
        "--split-strategy",
        choices=["fixed", "auto-gap"],
        default="fixed",
        help="Use a fixed ratio or estimate the row split from foreground gaps.",
    )
    args = parser.parse_args(argv)

    if args.mode == "split-row":
        records = run_split_probe(
            crop_path=args.crop,
            output_dir=args.output_dir,
            expected_lines=args.expected_lines,
            model=args.model.strip() or None,
            timeout=args.timeout,
            max_tokens=args.max_tokens,
            split_ratio=args.split_ratio,
            overlap_px=args.overlap_px,
            split_strategy=args.split_strategy,
            disable_thinking=args.disable_thinking,
        )
    elif args.mode == "justification-column":
        records = run_justification_column_probe(
            crop_path=args.crop,
            output_dir=args.output_dir,
            expected_lines=args.expected_lines,
            model=args.model.strip() or None,
            timeout=args.timeout,
            max_tokens=args.max_tokens,
            split_ratio=args.split_ratio,
            overlap_px=args.overlap_px,
            disable_thinking=args.disable_thinking,
        )
    else:
        records = run_probe(
            crop_path=args.crop,
            output_dir=args.output_dir,
            expected_lines=args.expected_lines,
            model=args.model.strip() or None,
            timeout=args.timeout,
            max_tokens=args.max_tokens,
            disable_thinking=args.disable_thinking,
        )
    print((args.output_dir / "shareable_summary.txt").read_text(encoding="utf-8"))
    if args.mode == "split-row":
        return 1 if any(
            record["left_status"] != "ok" or record["right_status"] != "ok"
            for record in records
        ) else 0
    if args.mode == "justification-column":
        return 0
    return 1 if any(record["status"] != "ok" for record in records) else 0


if __name__ == "__main__":
    raise SystemExit(main())
