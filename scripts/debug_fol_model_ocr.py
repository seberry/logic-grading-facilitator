from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Set

import yaml

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from app.vlm_client import LMStudioQwenClient


def _load_autograde_config(path: Path | None) -> Dict[str, Any]:
    """
    Load the OCR/checker setup for one FOL-model task.

    This helper intentionally accepts two shapes:

    1. A plain autograde YAML/JSON file, e.g.
         task_kind: non_equivalence
         prompt_data: ...

    2. A saved debug-page state.json file, where the real autograde YAML is
       nested inside the "problem_setup_text" field.

    Accepting both shapes makes this script usable both from the browser route
    and from batch harnesses over debug sample folders.
    """
    if path is None or not path.exists():
        return {}

    raw_text = path.read_text(encoding="utf-8")
    parsed = yaml.safe_load(raw_text) or {}
    if not isinstance(parsed, dict):
        raise ValueError("autograde file must parse to a YAML object/map")

    nested_setup = parsed.get("problem_setup_text")
    if isinstance(nested_setup, str) and nested_setup.strip():
        nested_parsed = yaml.safe_load(nested_setup) or {}
        if not isinstance(nested_parsed, dict):
            raise ValueError("problem_setup_text must parse to a YAML object/map")
        return nested_parsed

    return parsed


def _formulas_for_hint(prompt_data: Dict[str, Any]) -> List[str]:
    for key in ("formulas", "premises"):
        value = prompt_data.get(key)
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
    conclusion = prompt_data.get("conclusion")
    if isinstance(conclusion, str) and conclusion.strip():
        return [conclusion.strip()]
    return []


def _symbols_from_formulas(formulas: List[str]) -> tuple[Set[str], Set[str]]:
    constants: Set[str] = set()
    predicates: Set[str] = set()
    for formula in formulas:
        predicates.update(re.findall(r"\b([A-Z])(?=[a-z])", formula))
        bound_variables = set(re.findall(r"[∀∃]\s*([a-z])", formula))
        for match in re.finditer(r"\b[A-Z]([a-z]+)", formula):
            constants.update(ch for ch in match.group(1) if ch not in bound_variables)
        for match in re.finditer(r"\b([a-z])\b", formula):
            token = match.group(1)
            if token not in bound_variables:
                constants.add(token)
    return constants, predicates


def _expected_symbols(autograde: Dict[str, Any], formulas: List[str]) -> tuple[List[str], List[str]]:
    constants, predicates = _symbols_from_formulas(formulas)
    ocr = autograde.get("ocr") or {}
    if isinstance(ocr, dict):
        for item in ocr.get("constants") or []:
            text = str(item).strip()
            if re.fullmatch(r"[a-z]", text):
                constants.add(text)
        for item in ocr.get("predicates") or []:
            text = str(item).strip()
            if re.fullmatch(r"[A-Z]", text):
                predicates.add(text)
    return sorted(constants), sorted(predicates)


def _expected_answer_shape(task_kind: str, autograde: Dict[str, Any]) -> str:
    ocr = autograde.get("ocr") or {}
    if isinstance(ocr, dict):
        shape = ocr.get("expected_answer_shape")
        if isinstance(shape, str) and shape.strip():
            return shape.strip()
    if task_kind == "true_false_in_given_model":
        return "truth_value_only"
    return "model_only"


def build_prompt(autograde: Dict[str, Any]) -> str:
    task_kind = str(autograde.get("task_kind") or "truth_value").strip() or "truth_value"
    prompt_data = autograde.get("prompt_data") or {}
    if not isinstance(prompt_data, dict):
        prompt_data = {}
    formulas = _formulas_for_hint(prompt_data)
    constants, predicates = _expected_symbols(autograde, formulas)
    answer_shape = _expected_answer_shape(task_kind, autograde)

    lines = [
        "You are transcribing a student's FOL model answer from an image.",
        "Return only a transcription in the requested canonical format.",
        "",
        f"Task kind: {task_kind}",
        f"Expected answer shape: {answer_shape}",
    ]
    if formulas:
        lines.append("Relevant formulas from the rubric:")
        lines.extend(f"- {formula}" for formula in formulas)
    if constants or predicates:
        lines.append("Expected nonlogical symbols from the rubric:")
        lines.append("- Constants: " + (", ".join(constants) if constants else "(none)"))
        lines.append("- Predicates: " + (", ".join(predicates) if predicates else "(none)"))
    if answer_shape == "truth_value_only":
        lines.extend(
            [
                "",
                "Return ONLY one of these strings:",
                "TRUE",
                "FALSE",
                "UNCLEAR",
            ]
        )
        return "\n".join(lines)

    lines.extend(
        [
            "",
            "Return ONLY canonical model text in this shape:",
            "DOMAIN: {1,2,3}",
            "",
            "CONSTANTS:",
            "(none)",
            "",
            "PREDICATES:",
            "P: {1,2}",
            "",
            "Rules:",
            "- Use integers as domain objects.",
            "- Only include constants that are visible in the student's answer or listed above as expected constants.",
            "- If no constants are visible or expected, leave the CONSTANTS section empty.",
            "- Only include predicate letters that are visible in the student's answer or listed above as expected predicates.",
            "- Unary predicates use sets like {1,2}.",
            "- Binary predicates use sets like {(1,2),(2,3)}.",
            "- If a section is empty, still include its header.",
            "- Do not invent default constants.",
            "- Do not explain your answer.",
            "- Do not include markdown fences.",
        ]
    )
    return "\n".join(lines)


def _strip_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        parts = stripped.split("```")
        if len(parts) >= 3:
            middle = parts[1]
            if "\n" in middle:
                middle = middle.split("\n", 1)[1]
            return middle.strip()
    return stripped


def transcribe_image_to_canonical_text(
    *,
    image_path: Path,
    output_dir: Path,
    autograde_config: Dict[str, Any],
    client: LMStudioQwenClient | None = None,
) -> tuple[str, str]:
    """
    Stage 1 of the workflow:

    image -> local vision model -> proposed canonical text

    This function does NOT run the deterministic checker. Its job is only to
    produce the OCR proposal that a human could later review in the debug UI.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt = build_prompt(autograde_config)
    (output_dir / "ocr_prompt.txt").write_text(prompt, encoding="utf-8")

    if client is None:
        client = LMStudioQwenClient()

    raw_response = client.chat_with_image_prompt(image_path=image_path, prompt=prompt)
    cleaned = _strip_fences(raw_response).strip() + "\n"

    (output_dir / "ocr_raw_response.txt").write_text(raw_response, encoding="utf-8")
    (output_dir / "ocr_model_text.txt").write_text(cleaned, encoding="utf-8")
    return raw_response, cleaned


def main() -> int:
    parser = argparse.ArgumentParser(description="Run local-model OCR for a FOL model crop.")
    parser.add_argument("--image", required=True, help="Path to the image to transcribe.")
    parser.add_argument("--output-dir", required=True, help="Directory where OCR artifacts should be written.")
    parser.add_argument("--autograde-file", help="YAML file containing fol_model autograde config.")
    args = parser.parse_args()

    image_path = Path(args.image)
    output_dir = Path(args.output_dir)
    autograde_file = Path(args.autograde_file) if args.autograde_file else None

    if not image_path.exists():
        print(f"Error: image not found: {image_path}")
        return 1

    autograde = _load_autograde_config(autograde_file)

    print(f"[OCR] Image: {image_path}")
    print(f"[OCR] Output dir: {output_dir}")
    print("[OCR] Sending prompt to local vision model...")

    try:
        transcribe_image_to_canonical_text(
            image_path=image_path,
            output_dir=output_dir,
            autograde_config=autograde,
            client=LMStudioQwenClient(),
        )
    except Exception as exc:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "ocr_raw_response.txt").write_text(f"ERROR: {exc}", encoding="utf-8")
        print(f"[OCR] Model call failed: {exc}")
        return 1

    print("[OCR] Wrote OCR artifacts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
