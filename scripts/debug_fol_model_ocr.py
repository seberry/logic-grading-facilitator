from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import yaml

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from app.vlm_client import LMStudioQwenClient


def _load_autograde(path: Path | None) -> Dict[str, Any]:
    if path is None or not path.exists():
        return {}
    parsed = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(parsed, dict):
        raise ValueError("autograde file must parse to a YAML object/map")
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
            "a: 1",
            "b: 2",
            "",
            "PREDICATES:",
            "P: {1,2}",
            "Q: {2}",
            "R: {(1,2),(2,3)}",
            "",
            "Rules:",
            "- Use integers as domain objects.",
            "- Use lowercase single letters for constants.",
            "- Use uppercase single letters for predicates.",
            "- Unary predicates use sets like {1,2}.",
            "- Binary predicates use sets like {(1,2),(2,3)}.",
            "- If a section is empty, still include its header.",
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

    output_dir.mkdir(parents=True, exist_ok=True)
    autograde = _load_autograde(autograde_file)
    prompt = build_prompt(autograde)

    (output_dir / "ocr_prompt.txt").write_text(prompt, encoding="utf-8")

    print(f"[OCR] Image: {image_path}")
    print(f"[OCR] Output dir: {output_dir}")
    print("[OCR] Sending prompt to local vision model...")

    client = LMStudioQwenClient()
    try:
        raw_response = client.chat_with_image_prompt(image_path=image_path, prompt=prompt)
    except Exception as exc:
        (output_dir / "ocr_raw_response.txt").write_text(f"ERROR: {exc}", encoding="utf-8")
        print(f"[OCR] Model call failed: {exc}")
        return 1

    cleaned = _strip_fences(raw_response)
    (output_dir / "ocr_raw_response.txt").write_text(raw_response, encoding="utf-8")
    (output_dir / "ocr_model_text.txt").write_text(cleaned.strip() + "\n", encoding="utf-8")
    print("[OCR] Wrote OCR artifacts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
