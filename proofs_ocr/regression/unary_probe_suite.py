"""Batch unary-verifier probe suite for isolated formula-line images."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import unary_probe

DEFAULT_FONT_CANDIDATES = (
    "DejaVuSans.ttf",
    "C:/Windows/Fonts/DejaVuSans.ttf",
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/seguisym.ttf",
    "C:/Windows/Fonts/cambria.ttc",
)


@dataclass(frozen=True)
class UnaryProbeSuiteCase:
    case_id: str
    image_formula: str
    current_ocr_formula: str
    expected_formula: str
    justification_text: str = ""
    notes: str = ""


@dataclass(frozen=True)
class UnaryProbeSuiteResult:
    case_id: str
    expected_formula: str
    current_ocr_formula: str
    structured_formula: str
    forced_choice_formula: str
    structured_passed: bool
    forced_choice_passed: bool
    structured_changed: bool
    forced_choice_changed: bool
    image_blob_count: int
    current_ocr_symbol_count: int
    blob_count_exceeds_ocr: bool
    confidence: str
    result_path: str
    notes: str


def default_probe_suite_cases() -> list[UnaryProbeSuiteCase]:
    """Return the initial synthetic unary-repair cases to probe."""

    return [
        UnaryProbeSuiteCase(
            case_id="atom_double_negation",
            image_formula="¬¬A",
            current_ocr_formula="¬A",
            expected_formula="¬¬A",
            notes="Simple repeated negation on an atom.",
        ),
        UnaryProbeSuiteCase(
            case_id="atom_triple_negation",
            image_formula="¬¬¬A",
            current_ocr_formula="¬¬A",
            expected_formula="¬¬¬A",
            notes="Rarer stress test for one-more-negation repair.",
        ),
        UnaryProbeSuiteCase(
            case_id="box_inner_double_negation",
            image_formula="□¬¬A",
            current_ocr_formula="□¬A",
            expected_formula="□¬¬A",
            justification_text="□I 5-7",
            notes="Matches the handwritten inner-box failure shape.",
        ),
        UnaryProbeSuiteCase(
            case_id="negated_box_inner_double_negation",
            image_formula="¬□¬¬A",
            current_ocr_formula="¬□A",
            expected_formula="¬□¬¬A",
            justification_text="MC 3",
            notes="Matches the handwritten outer negated-box failure shape.",
        ),
        UnaryProbeSuiteCase(
            case_id="diamond_inner_double_negation",
            image_formula="⋄¬¬A",
            current_ocr_formula="⋄¬A",
            expected_formula="⋄¬¬A",
            justification_text="Def⋄ 3",
            notes="Checks whether the same count logic works for diamonds.",
        ),
        UnaryProbeSuiteCase(
            case_id="control_single_negation",
            image_formula="¬A",
            current_ocr_formula="¬A",
            expected_formula="¬A",
            notes="Control: should not introduce an extra negation.",
        ),
        UnaryProbeSuiteCase(
            case_id="control_box_single_inner_negation",
            image_formula="□¬A",
            current_ocr_formula="□¬A",
            expected_formula="□¬A",
            justification_text="□I 4-5",
            notes="Control: should keep a one-negation box formula stable.",
        ),
    ]


def load_font(*, font_size: int = 48, font_candidates: Sequence[str] | None = None) -> ImageFont.FreeTypeFont:
    """Load a font that can render modal/unary symbols for the probe suite."""

    for candidate in (font_candidates or DEFAULT_FONT_CANDIDATES):
        try:
            return ImageFont.truetype(candidate, font_size)
        except OSError:
            continue
    raise RuntimeError("Could not load a usable TrueType font for unary probe rendering")


def render_formula_image(
    *,
    formula_text: str,
    output_path: Path,
    font: ImageFont.FreeTypeFont | None = None,
    padding_x: int = 24,
    padding_y: int = 20,
) -> Path:
    """Render one isolated formula image for a unary-probe case."""

    resolved_font = font or load_font()
    dummy = Image.new("RGB", (1, 1), "white")
    draw = ImageDraw.Draw(dummy)
    left, top, right, bottom = draw.textbbox((0, 0), formula_text, font=resolved_font)
    width = max(120, (right - left) + (2 * padding_x))
    height = max(70, (bottom - top) + (2 * padding_y))

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((padding_x - left, padding_y - top), formula_text, fill="black", font=resolved_font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return output_path


def count_formula_blob_components(
    image_path: Path,
    *,
    dark_threshold: int = 200,
    min_area: int = 20,
) -> int:
    """Count glyph-like connected components in a clean formula-only image."""

    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read rendered formula image at {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dark = (gray < dark_threshold).astype(np.uint8)
    closed = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8))
    component_count, _, stats, _ = cv2.connectedComponentsWithStats(closed, 8)
    visible_components = 0
    for label_index in range(1, component_count):
        area = int(stats[label_index, cv2.CC_STAT_AREA])
        if area >= min_area:
            visible_components += 1
    return visible_components


def count_unary_formula_symbol_slots(formula_text: str) -> int:
    """Count the visible unary/modal symbols plus one body slot in a simple formula."""

    normalized = unary_probe.normalize_probe_formula_text(formula_text)
    prefix = unary_probe.leading_unary_prefix(normalized)
    body = normalized[len(prefix) :].strip()
    return len(prefix) + (1 if body else 0)


def run_probe_suite(
    *,
    cases: Sequence[UnaryProbeSuiteCase],
    output_root: Path,
    verifier_base_url: str | None = None,
    verifier_model: str | None = None,
    font_candidates: Sequence[str] | None = None,
) -> list[UnaryProbeSuiteResult]:
    """Render each synthetic case and run the unary probe on it."""

    output_root.mkdir(parents=True, exist_ok=True)
    font = load_font(font_candidates=font_candidates)
    results: list[UnaryProbeSuiteResult] = []

    for case in cases:
        case_dir = output_root / case.case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        image_path = render_formula_image(
            formula_text=case.image_formula,
            output_path=case_dir / "source_formula.png",
            font=font,
        )
        payload = unary_probe.run_unary_probe(
            image_path=image_path,
            formula_text=case.current_ocr_formula,
            justification_text=case.justification_text,
            output_dir=case_dir,
            verifier_base_url=verifier_base_url,
            verifier_model=verifier_model,
        )
        current_ocr_formula = unary_probe.normalize_probe_formula_text(case.current_ocr_formula)
        forced_choice_formula = unary_probe.normalize_probe_formula_text(
            payload.get("forced_choice_parsed", {}).get("selected_formula") or "",
        )
        structured_formula = unary_probe.normalize_probe_formula_text(
            payload.get("structured_formula") or "",
        )
        expected_formula = unary_probe.normalize_probe_formula_text(case.expected_formula)
        image_blob_count = count_formula_blob_components(image_path)
        current_ocr_symbol_count = count_unary_formula_symbol_slots(current_ocr_formula)
        confidence = str(payload.get("forced_choice_parsed", {}).get("confidence") or "")
        result = UnaryProbeSuiteResult(
            case_id=case.case_id,
            expected_formula=expected_formula,
            current_ocr_formula=current_ocr_formula,
            structured_formula=structured_formula,
            forced_choice_formula=forced_choice_formula,
            structured_passed=(structured_formula == expected_formula),
            forced_choice_passed=(forced_choice_formula == expected_formula),
            structured_changed=(bool(structured_formula) and structured_formula != current_ocr_formula),
            forced_choice_changed=(bool(forced_choice_formula) and forced_choice_formula != current_ocr_formula),
            image_blob_count=image_blob_count,
            current_ocr_symbol_count=current_ocr_symbol_count,
            blob_count_exceeds_ocr=(image_blob_count > current_ocr_symbol_count),
            confidence=confidence,
            result_path=str(case_dir / "unary_probe_result.json"),
            notes=case.notes,
        )
        (case_dir / "suite_case.json").write_text(
            json.dumps(asdict(case), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (case_dir / "suite_result.json").write_text(
            json.dumps(asdict(result), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        results.append(result)

    return results


def render_probe_suite_summary(results: Sequence[UnaryProbeSuiteResult]) -> str:
    """Render a compact human-readable summary of the probe-suite outcomes."""

    total = len(results)
    structured_passed = sum(1 for result in results if result.structured_passed)
    forced_passed = sum(1 for result in results if result.forced_choice_passed)
    structured_changes = sum(1 for result in results if result.structured_changed)
    forced_changes = sum(1 for result in results if result.forced_choice_changed)
    blob_excess_flags = sum(1 for result in results if result.blob_count_exceeds_ocr)
    lines = [
        f"Unary probe cases: {total}",
        f"Structured-count reconstruction passed: {structured_passed}",
        f"Structured-count reconstruction failed: {total - structured_passed}",
        f"Structured-count reconstruction changed OCR on: {structured_changes}",
        f"Forced choice passed: {forced_passed}",
        f"Forced choice failed: {total - forced_passed}",
        f"Forced choice changed OCR on: {forced_changes}",
        f"CV blob-count flags (image > OCR symbols): {blob_excess_flags}",
    ]
    for result in results:
        structured_status = "PASS" if result.structured_passed else "FAIL"
        forced_status = "PASS" if result.forced_choice_passed else "FAIL"
        lines.append(
            f"- {result.case_id}: expected {result.expected_formula}, "
            f"structured={result.structured_formula or '<empty>'} ({structured_status}), "
            f"forced={result.forced_choice_formula or '<empty>'} ({forced_status}), "
            f"blobs={result.image_blob_count}, ocr_symbols={result.current_ocr_symbol_count}, "
            f"blob_flag={'yes' if result.blob_count_exceeds_ocr else 'no'}"
        )
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a batch suite of synthetic unary-verifier probe cases.")
    parser.add_argument("--output-root", required=True, help="Directory where per-case probe artifacts should be written.")
    parser.add_argument("--verifier-base-url", default=None, help="LM Studio base URL override.")
    parser.add_argument("--verifier-model", default=None, help="LM Studio model override.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    results = run_probe_suite(
        cases=default_probe_suite_cases(),
        output_root=Path(args.output_root),
        verifier_base_url=args.verifier_base_url,
        verifier_model=args.verifier_model,
    )
    summary = render_probe_suite_summary(results)
    print(summary)
    (Path(args.output_root) / "suite_summary.txt").write_text(summary, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
