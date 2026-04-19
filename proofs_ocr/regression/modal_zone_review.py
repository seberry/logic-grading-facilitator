"""HTML review helpers for modal formula/justification zone OCR artifacts."""

from __future__ import annotations

import argparse
import html
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2


@dataclass(frozen=True)
class ModalZoneReviewRecord:
    case_id: str
    node_id: str
    row_order: int
    line_crop_filename: str
    formula_crop_filename: str
    justification_crop_filename: str
    formula_text: str
    justification_text: str
    scope_role: str
    formula_confidence: str
    justification_confidence: str
    skip_reason: str
    formula_parse_error: str
    justification_parse_error: str
    formula_raw_response: str
    justification_raw_response: str
    suspicious_reasons: tuple[str, ...]


def _json_files(output_dir: Path) -> Iterable[Path]:
    return sorted(output_dir.glob("node_line_*.json"))


def _load_json(path: Path) -> dict[str, Any]:
    parsed = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(parsed, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return parsed


def _crop_has_visible_ink(path: Path) -> bool:
    if not path.exists():
        return False
    image = cv2.imread(str(path))
    if image is None or image.size == 0:
        return False
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return int((gray < 210).sum()) >= 20


def _collapse(value: Any) -> str:
    return " ".join(str(value or "").split())


def _suspicious_reasons(payload: dict[str, Any], *, output_dir: Path) -> tuple[str, ...]:
    reasons: list[str] = []
    skip_reason = _collapse(payload.get("skip_reason"))
    if skip_reason:
        reasons.append(f"skip={skip_reason}")
    if _collapse(payload.get("formula_parse_error")):
        reasons.append("formula_parse_error")
    if _collapse(payload.get("justification_parse_error")):
        reasons.append("justification_parse_error")

    formula_crop_name = _collapse(payload.get("formula_crop_filename"))
    justification_crop_name = _collapse(payload.get("justification_crop_filename"))
    formula_resolved = payload.get("formula_resolved") or {}
    justification_resolved = payload.get("justification_resolved") or {}
    resolved = payload.get("resolved") or {}

    formula_text = _collapse(formula_resolved.get("formula_text") or resolved.get("formula_text"))
    justification_text = _collapse(justification_resolved.get("justification_text") or resolved.get("justification_text"))
    raw_formula_label = _collapse(formula_resolved.get("line_label"))

    if formula_crop_name and not formula_text and _crop_has_visible_ink(output_dir / formula_crop_name):
        reasons.append("blank_formula_with_ink")
    if justification_crop_name and not justification_text and _crop_has_visible_ink(output_dir / justification_crop_name):
        reasons.append("blank_justification_with_ink")
    if raw_formula_label and not _collapse(resolved.get("line_label")) and justification_text == raw_formula_label:
        reasons.append("justification_recovered_from_formula_label")
    return tuple(reasons)


def collect_modal_zone_review_records(*, output_dir: Path, case_id: str = "") -> list[ModalZoneReviewRecord]:
    records: list[ModalZoneReviewRecord] = []
    for path in _json_files(output_dir):
        payload = _load_json(path)
        if _collapse(payload.get("mode")) != "formula_justification_separation":
            continue

        formula_resolved = payload.get("formula_resolved") or {}
        justification_resolved = payload.get("justification_resolved") or {}
        resolved = payload.get("resolved") or {}
        records.append(
            ModalZoneReviewRecord(
                case_id=case_id or output_dir.name,
                node_id=_collapse(payload.get("node_id")),
                row_order=int(payload.get("row_order") or 0),
                line_crop_filename=path.with_suffix(".jpg").name,
                formula_crop_filename=_collapse(payload.get("formula_crop_filename")),
                justification_crop_filename=_collapse(payload.get("justification_crop_filename")),
                formula_text=_collapse(resolved.get("formula_text")),
                justification_text=_collapse(resolved.get("justification_text")),
                scope_role=_collapse(resolved.get("scope_role")),
                formula_confidence=_collapse(formula_resolved.get("confidence")),
                justification_confidence=_collapse(justification_resolved.get("confidence")),
                skip_reason=_collapse(payload.get("skip_reason")),
                formula_parse_error=_collapse(payload.get("formula_parse_error")),
                justification_parse_error=_collapse(payload.get("justification_parse_error")),
                formula_raw_response=str(payload.get("formula_raw_response") or ""),
                justification_raw_response=str(payload.get("justification_raw_response") or ""),
                suspicious_reasons=_suspicious_reasons(payload, output_dir=output_dir),
            )
        )
    return sorted(records, key=lambda item: (item.case_id, item.node_id, item.row_order))


def filter_modal_zone_review_records(
    records: Iterable[ModalZoneReviewRecord],
    *,
    suspicious_only: bool = False,
    selected_row_ids: set[tuple[str, str, int]] | None = None,
) -> list[ModalZoneReviewRecord]:
    materialized = list(records)
    if selected_row_ids:
        materialized = [
            record
            for record in materialized
            if (record.case_id, record.node_id, int(record.row_order)) in selected_row_ids
        ]
    if not suspicious_only:
        return materialized
    return [record for record in materialized if record.suspicious_reasons]


def build_modal_zone_review_html(
    *,
    title: str,
    report_dir: Path,
    sections: list[tuple[str, Path, list[ModalZoneReviewRecord]]],
) -> str:
    style = """
body { font-family: Georgia, 'Times New Roman', serif; background: #f5f1e8; color: #1f1c18; margin: 0; }
.page { max-width: 1500px; margin: 0 auto; padding: 24px; }
h1, h2 { margin: 0 0 12px; }
.intro { margin-bottom: 24px; color: #4a423a; }
.case { margin: 28px 0 36px; }
.card { background: #fffdf8; border: 1px solid #d7cdbf; border-radius: 14px; padding: 16px; margin: 14px 0; box-shadow: 0 2px 8px rgba(55, 42, 25, 0.06); }
.meta { margin-bottom: 10px; font-size: 14px; color: #5d544a; }
.flags { margin: 8px 0 0; }
.flag { display: inline-block; margin-right: 8px; margin-bottom: 6px; padding: 4px 8px; border-radius: 999px; background: #f3dfc4; color: #6b3f16; font-size: 12px; }
.grid { display: grid; grid-template-columns: 1.1fr 1fr 1fr; gap: 14px; align-items: start; }
.panel { border: 1px solid #e3d9ca; border-radius: 10px; background: #fff; padding: 10px; }
.panel h3 { margin: 0 0 8px; font-size: 15px; }
.imgwrap { background: #faf8f3; border: 1px solid #ece3d7; border-radius: 8px; padding: 8px; min-height: 70px; display: flex; align-items: center; justify-content: center; }
.imgwrap img { max-width: 100%; height: auto; display: block; }
.missing { color: #8a7d70; font-style: italic; }
pre { margin: 10px 0 0; padding: 10px; border-radius: 8px; background: #f8f4ed; overflow-x: auto; white-space: pre-wrap; word-break: break-word; font-size: 12px; }
.ocr { font-size: 14px; line-height: 1.45; }
.ocr code { background: #f5efe5; padding: 1px 4px; border-radius: 4px; }
"""

    parts = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'>",
        f"<title>{html.escape(title)}</title>",
        f"<style>{style}</style>",
        "</head><body><div class='page'>",
        f"<h1>{html.escape(title)}</h1>",
        "<p class='intro'>Each row shows the crop sent to the model beside what the model returned. "
        "Highlighted flags call out suspicious blank reads, parse issues, or formula/justification spill.</p>",
    ]

    for heading, output_dir, records in sections:
        parts.append("<section class='case'>")
        parts.append(f"<h2>{html.escape(heading)}</h2>")
        parts.append(f"<p class='intro'>Artifacts from <code>{html.escape(str(output_dir))}</code></p>")
        for record in records:
            parts.append("<article class='card'>")
            parts.append(
                "<div class='meta'>"
                f"node=<code>{html.escape(record.node_id)}</code> "
                f"row=<code>{record.row_order}</code> "
                f"scope_role=<code>{html.escape(record.scope_role or '?')}</code>"
                "</div>"
            )
            if record.suspicious_reasons:
                parts.append("<div class='flags'>" + "".join(
                    f"<span class='flag'>{html.escape(reason)}</span>" for reason in record.suspicious_reasons
                ) + "</div>")
            parts.append("<div class='grid'>")
            parts.append(_render_panel("Full line crop", output_dir, report_dir, record.line_crop_filename, [
                f"resolved formula: <code>{html.escape(record.formula_text or '')}</code>",
                f"resolved justification: <code>{html.escape(record.justification_text or '')}</code>",
            ]))
            parts.append(_render_panel("Formula crop", output_dir, report_dir, record.formula_crop_filename, [
                f"confidence: <code>{html.escape(record.formula_confidence or '')}</code>",
                "raw response:",
                f"<pre>{html.escape(record.formula_raw_response or '')}</pre>",
            ]))
            parts.append(_render_panel("Justification crop", output_dir, report_dir, record.justification_crop_filename, [
                f"confidence: <code>{html.escape(record.justification_confidence or '')}</code>",
                "raw response:",
                f"<pre>{html.escape(record.justification_raw_response or '')}</pre>",
            ]))
            parts.append("</div></article>")
        parts.append("</section>")
    parts.append("</div></body></html>")
    return "".join(parts)


def _render_panel(title: str, output_dir: Path, report_dir: Path, filename: str, lines: list[str]) -> str:
    image_block = "<div class='missing'>missing crop</div>"
    if filename:
        image_path = output_dir / filename
        if image_path.exists():
            relative_path = image_path.resolve().relative_to(report_dir.resolve()) if image_path.resolve().is_relative_to(report_dir.resolve()) else None
            src = (
                relative_path.as_posix()
                if relative_path is not None
                else image_path.resolve().as_uri()
            )
            image_block = f"<img src='{html.escape(src)}' alt='{html.escape(title)}'>"
    body = "".join(f"<div class='ocr'>{line}</div>" for line in lines)
    return (
        "<section class='panel'>"
        f"<h3>{html.escape(title)}</h3>"
        f"<div class='imgwrap'>{image_block}</div>"
        f"{body}</section>"
    )


def write_modal_zone_review_report(
    *,
    cases: list[tuple[str, Path]],
    report_path: Path,
    title: str = "Modal Zone OCR Review",
    suspicious_only: bool = False,
    selected_row_ids: set[tuple[str, str, int]] | None = None,
) -> Path:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    sections: list[tuple[str, Path, list[ModalZoneReviewRecord]]] = []
    for heading, output_dir in cases:
        records = collect_modal_zone_review_records(output_dir=output_dir, case_id=heading)
        sections.append(
            (
                heading,
                output_dir,
                filter_modal_zone_review_records(
                    records,
                    suspicious_only=suspicious_only,
                    selected_row_ids=selected_row_ids,
                ),
            )
        )
    html_text = build_modal_zone_review_html(
        title=title,
        report_dir=report_path.parent,
        sections=sections,
    )
    report_path.write_text(html_text, encoding="utf-8")
    return report_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an HTML review page for modal zone OCR artifacts.")
    parser.add_argument("--case-dir", action="append", default=[], help="One modal_v2 output directory. May be repeated.")
    parser.add_argument("--report-path", required=True, help="HTML file to write.")
    parser.add_argument("--title", default="Modal Zone OCR Review", help="HTML title.")
    parser.add_argument(
        "--suspicious-only",
        action="store_true",
        help="Include only rows with suspicious flags such as blank OCR despite visible ink.",
    )
    parser.add_argument(
        "--select-row",
        action="append",
        default=[],
        help="Include only one specific row, formatted as case_id:node_id:row_order. May be repeated.",
    )
    return parser.parse_args()


def _parse_selected_row_ids(raw_values: list[str]) -> set[tuple[str, str, int]]:
    selected: set[tuple[str, str, int]] = set()
    for raw in raw_values:
        text = str(raw or "").strip()
        if not text:
            continue
        case_id, node_id, row_order_text = text.split(":", 2)
        selected.add((case_id.strip(), node_id.strip(), int(row_order_text.strip())))
    return selected


def main() -> int:
    args = _parse_args()
    cases = [(Path(case_dir).name, Path(case_dir).resolve()) for case_dir in args.case_dir]
    if not cases:
        raise ValueError("Provide at least one --case-dir")
    path = write_modal_zone_review_report(
        cases=cases,
        report_path=Path(args.report_path).resolve(),
        title=str(args.title or "Modal Zone OCR Review"),
        suspicious_only=bool(args.suspicious_only),
        selected_row_ids=_parse_selected_row_ids(list(args.select_row)),
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
