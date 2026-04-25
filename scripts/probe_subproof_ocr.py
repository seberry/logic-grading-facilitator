#!/usr/bin/env python
"""Probe subproof OCR with different VLM models and prompt variants.

Designed for experimenting with larger models (27b) on small subproof crops
that stymie the 4b/8b models used in the main pipeline.  Captures thinking
text and compares transcriptions against expected proof structures.

Usage examples
--------------
    # All Q4 subproof crops, all prompt variants, 27b model:
    python scripts/probe_subproof_ocr.py

    # Specific case only:
    python scripts/probe_subproof_ocr.py --case tfl_handwritten_q4_N_or_K_and_W

    # One crop within a case (substring match on filename):
    python scripts/probe_subproof_ocr.py --case tfl_handwritten_q4_N_or_K_and_W --only-crops crop_3

    # Specific prompt variant:
    python scripts/probe_subproof_ocr.py --prompt transcriber-v2

    # Quick smoke-test with the fast 4b model:
    python scripts/probe_subproof_ocr.py --model qwen/qwen3-vl-4b --prompt pipeline

    # Custom crop files (no case auto-discovery):
    python scripts/probe_subproof_ocr.py --crops path/to/crop.jpg
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

# ---------------------------------------------------------------------------
# Ensure repo root is importable
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from app.vlm_client import default_lmstudio_base_url

# ============================================================================
# Prompt variants
# ============================================================================

TRANSCRIBER_V1 = """\
Role: You are a logic quiz transcriber performing OCR on subproof chunks \
cropped from the middle of handwritten Fitch-style natural deduction proofs.

Context: You will be given cropped images of specific subproofs. Each subproof \
begins with an assumption (no justification, above a horizontal line) and \
derives consequences from that assumption and earlier lines. The top line in \
the crop might not be line 1 in the overall proof.

Task: Extract the logical formulas and their justifications.

Handwriting Translation Guide (CRITICAL):
Students have sloppy handwriting. Use the following known distortions to \
correct your visual reading:
- A sloppy K frequently looks like |< or a vertical line next to a <.
- A sloppy ampersand & (conjunction) frequently looks like an 8.
- A sloppy S frequently looks like a 5.

Output format: Return a JSON list of objects with keys "line", "wffstr" \
(the formula text), and "jstr" (the justification, or "" if none visible).
Return JSON only, no markdown fences or explanation.""".strip()


TRANSCRIBER_V2 = """\
Role: You are a logic quiz transcriber performing OCR on subproof chunks \
cropped from the middle of handwritten Fitch-style natural deduction proofs.

Context: You will be given cropped images of specific subproofs. Each subproof \
begins with an assumption (no justification, above a horizontal line) and \
derives consequences from that assumption and earlier lines. The top line in \
the crop might not be line 1 in the overall proof.

Task: Extract the logical formulas and their justifications.

CRITICAL — You are a TRANSCRIBER, not a proof checker:
- Write EXACTLY what you see in the image, character by character.
- Do NOT assess whether the proof steps are logically correct.
- Do NOT "fix" formulas that seem wrong or inconsistent.
- Do NOT infer missing steps or add lines you think should be there.
- Trust that the student wrote valid syntax and cited genuine rules.
- The ONLY corrections to apply are those in the Handwriting Translation Guide \
below.

Handwriting Translation Guide:
Students have sloppy handwriting. Use the following known distortions to \
correct your visual reading:
- A sloppy K frequently looks like |< or a vertical line next to a <.
- A sloppy ampersand & (conjunction) frequently looks like an 8.
- A sloppy S frequently looks like a 5.

Output format: Return a JSON list of objects with keys "line", "wffstr" \
(the formula text), and "jstr" (the justification, or "" if none visible).
Return JSON only, no markdown fences or explanation.""".strip()


TRANSCRIBER_V3 = """\
Role: You are a logic quiz transcriber performing OCR on subproof chunks \
cropped from the middle of handwritten Fitch-style natural deduction proofs.

Context: You will be given cropped images of specific subproofs. Each subproof \
begins with an assumption (no justification, above a horizontal line) and \
derives consequences from that assumption and earlier lines. The top line in \
the crop might not be line 1 in the overall proof.

Task: Extract the logical formulas and their justifications.

CRITICAL — You are a TRANSCRIBER, not a proof checker:
- Write EXACTLY what you see in the image, character by character.
- Do NOT assess whether the proof steps are logically correct.
- Do NOT "fix" formulas that seem wrong or inconsistent.
- Do NOT infer missing steps or add lines you think should be there.
- Trust that the student wrote valid syntax and cited genuine rules.
- The ONLY corrections to apply are those in the Handwriting Translation Guide \
below.

Handwriting Translation Guide:
Students have sloppy handwriting. Use the following known distortions to \
correct your visual reading:
- A sloppy K frequently looks like |< or a vertical line next to a <.
- A sloppy ampersand & (conjunction) frequently looks like an 8.
- A sloppy S frequently looks like a 5.

Layout Guide (avoid column mixing):
- Each proof line has TWO distinct regions: a FORMULA on the left, and a \
JUSTIFICATION on the right.
- The justification is a rule name followed by line numbers, e.g. "&E 5", \
"->I 3-7", "vE 1,3-4,5-6".
- The first line of a subproof typically has NO justification (it is an \
assumption).
- Do NOT place justification text in the wffstr field or vice versa.
- Count the visible text rows carefully — each horizontal row is ONE proof line.

Output format: Return a JSON list of objects with keys "line", "wffstr" \
(the formula text), and "jstr" (the justification, or "" if none visible).
Return JSON only, no markdown fences or explanation.""".strip()


def _get_pipeline_prompt() -> str:
    """Return the current pipeline leaf-node prompt."""
    from proofs_ocr.pipelines.tfl_fitch import build_ocr_prompt
    return build_ocr_prompt(has_children=False)


PROMPT_REGISTRY = {
    "pipeline": None,  # resolved lazily
    "transcriber-v1": TRANSCRIBER_V1,
    "transcriber-v2": TRANSCRIBER_V2,
    "transcriber-v3": TRANSCRIBER_V3,
}


def get_prompt(name: str) -> str:
    if name == "pipeline":
        return _get_pipeline_prompt()
    return PROMPT_REGISTRY[name]


ALL_PROMPT_NAMES = list(PROMPT_REGISTRY.keys())


# ============================================================================
# Crop / expected-structure discovery
# ============================================================================

WORKING_DIR = REPO_ROOT / "working" / "tfl_gold_regressions"
FIXTURE_DIR = REPO_ROOT / "proofs_ocr" / "regression" / "fixtures" / "tfl" / "handwritten"


def discover_crops(case_id: str, only_crops: list[str] | None = None) -> list[Path]:
    """Find subproof crop files for a case, sorted by node number.

    Returns crop_N.jpg files (N != "ROOT"), sorted numerically.
    """
    case_dir = WORKING_DIR / case_id
    if not case_dir.is_dir():
        print(f"[ERROR] Case directory not found: {case_dir}", file=sys.stderr)
        print(f"  Run a gold regression first: python -m proofs_ocr.regression.tfl_runner run-gold", file=sys.stderr)
        sys.exit(1)

    crops = []
    for f in sorted(case_dir.iterdir()):
        if not f.name.startswith("crop_") or not f.name.endswith(".jpg"):
            continue
        node_id = f.stem.replace("crop_", "")
        if node_id == "ROOT":
            continue
        # If --only-crops filter given, check for substring match
        if only_crops and not any(oc in f.name for oc in only_crops):
            continue
        crops.append(f)
    return crops


def extract_expected_subproofs(case_id: str) -> list[list[dict]]:
    """Load expected_structure.json and extract subproof blocks (list entries)."""
    fixture = FIXTURE_DIR / case_id / "expected_structure.json"
    if not fixture.exists():
        print(f"[WARN] No expected structure at {fixture}", file=sys.stderr)
        return []

    structure = json.loads(fixture.read_text(encoding="utf-8"))
    subproofs = []
    for item in structure:
        if isinstance(item, list):
            subproofs.append(item)
    return subproofs


# ============================================================================
# VLM API call with thinking capture
# ============================================================================

def encode_image_data_uri(image_path: Path) -> str:
    """Encode image as base64 data URI."""
    suffix = image_path.suffix.lower()
    mime = "image/jpeg" if suffix in (".jpg", ".jpeg") else "image/png"
    raw = image_path.read_bytes()
    b64 = base64.b64encode(raw).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def call_vlm_with_thinking(
    image_path: Path,
    prompt: str,
    *,
    model: str,
    base_url: str,
    timeout: int = 360,
) -> Tuple[str, str]:
    """Send image + prompt to VLM, return (content, thinking_text).

    Uses the LM Studio OpenAI-compatible API directly so we can capture
    ``reasoning_content`` (thinking text) from the response.
    """
    data_uri = encode_image_data_uri(image_path)

    messages = [
        {
            "role": "system",
            "content": prompt,
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Transcribe the proof lines visible in this image."},
                {"type": "image_url", "image_url": {"url": data_uri}},
            ],
        },
    ]

    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": 2048,
    }

    resp = requests.post(
        f"{base_url}/chat/completions",
        json=payload,
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()

    message = data.get("choices", [{}])[0].get("message", {})
    content = str(message.get("content", ""))
    thinking = ""

    # Qwen3 models return thinking in reasoning_content
    if message.get("reasoning_content"):
        thinking = str(message["reasoning_content"])

    # Fallback: check for <think/> tags in content
    if not thinking and "<think" in content:
        think_match = re.search(r"<think[^>]*>(.*?)</think?>", content, re.DOTALL)
        if think_match:
            thinking = think_match.group(1).strip()
            content = content[:think_match.start()] + content[think_match.end():]
            content = content.strip()

    return content, thinking


# ============================================================================
# Response parsing and normalization
# ============================================================================

def parse_vlm_json(content: str) -> list[dict]:
    """Extract JSON list from VLM response text."""
    # Strip markdown fences
    text = content.strip()
    if "```json" in text:
        text = text.split("```json", 1)[1].split("```", 1)[0]
    elif "```" in text:
        text = text.split("```", 1)[1].split("```", 1)[0]
    text = text.strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        # Try to find a JSON array in the text
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if match:
            parsed = json.loads(match.group())
        else:
            return [{"line": "?", "wffstr": f"PARSE_ERROR: {text[:200]}", "jstr": ""}]

    if isinstance(parsed, dict):
        parsed = [parsed]
    if not isinstance(parsed, list):
        return [{"line": "?", "wffstr": f"UNEXPECTED_TYPE: {type(parsed).__name__}", "jstr": ""}]

    # Normalize keys
    result = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        result.append({
            "line": item.get("line", "?"),
            "wffstr": str(item.get("wffstr", "")).strip(),
            "jstr": str(item.get("jstr", "")).strip(),
        })
    return result


# -- Lightweight normalization for comparison --

_SYMBOL_MAP = {
    "~": "¬", "-": "¬", "&": "∧", ".": "∧",
    "v": "∨", "V": "∨", "->": "→", "=>": "→",
    "<->": "↔", "=": "↔", "_|_": "⊥",
}

def _normalize_formula(wff: str) -> str:
    """Apply the same symbol normalization as the pipeline."""
    if not wff:
        return ""
    # Blanket OCR artifact repair
    wff = wff.replace("∈", "E").replace("ε", "E")
    wff = wff.replace("8", "∧")
    wff = wff.replace("5", "S")
    # Multi-char operators first
    wff = wff.replace("->", "→").replace("<->", "↔").replace("<→", "↔")
    # Single-char replacement + uppercase
    chars = []
    for c in wff:
        if c in _SYMBOL_MAP:
            chars.append(_SYMBOL_MAP[c])
        elif c.isalpha():
            chars.append(c.upper())
        elif c.isspace():
            continue  # strip whitespace for comparison
        else:
            chars.append(c)
    return "".join(chars)


def _normalize_justification(jstr: str) -> str:
    """Apply the same justification normalization as the pipeline."""
    from proofs_ocr.tricks.justification_snaps import (
        clean_justification_for_tfl,
        fuzzy_snap_justification,
        snap_justification_rule_head,
        TFL_VALID_RULE_HEADS,
    )
    j = clean_justification_for_tfl(jstr)
    j = snap_justification_rule_head(j, valid_rules=TFL_VALID_RULE_HEADS)
    j = fuzzy_snap_justification(j, valid_rules=TFL_VALID_RULE_HEADS)
    return j


def normalize_line(line: dict) -> dict:
    """Return a normalized copy of a line dict."""
    return {
        "line": line.get("line", "?"),
        "wffstr": _normalize_formula(line.get("wffstr", "")),
        "jstr": _normalize_justification(line.get("jstr", "")),
    }


# ============================================================================
# Comparison
# ============================================================================

def compare_lines(
    actual: list[dict],
    expected: list[dict],
) -> list[dict]:
    """Compare normalized actual lines against expected.

    Returns a list of per-line result dicts with keys:
        position, formula_match, jstr_match, actual, expected
    """
    results = []
    max_len = max(len(actual), len(expected))
    for i in range(max_len):
        a = normalize_line(actual[i]) if i < len(actual) else None
        e = expected[i] if i < len(expected) else None

        entry = {"position": i + 1}

        if a is None:
            entry.update({
                "formula_match": False,
                "jstr_match": False,
                "actual_formula": "(missing)",
                "actual_jstr": "(missing)",
                "expected_formula": e.get("wffstr", ""),
                "expected_jstr": e.get("jstr", ""),
                "status": "MISSING_LINE",
            })
        elif e is None:
            entry.update({
                "formula_match": False,
                "jstr_match": False,
                "actual_formula": a["wffstr"],
                "actual_jstr": a["jstr"],
                "expected_formula": "(extra)",
                "expected_jstr": "(extra)",
                "status": "EXTRA_LINE",
            })
        else:
            # Normalize expected formula the same way
            exp_wff = _normalize_formula(e.get("wffstr", ""))
            exp_jstr = e.get("jstr", "")

            # For first line of subproof, "Hyp" and "" are both acceptable
            jstr_ok = (a["jstr"] == exp_jstr) or (
                i == 0 and exp_jstr == "Hyp" and a["jstr"] in ("", "Hyp", "Pr", "premise")
            )

            entry.update({
                "formula_match": a["wffstr"] == exp_wff,
                "jstr_match": jstr_ok,
                "actual_formula": a["wffstr"],
                "actual_jstr": a["jstr"],
                "expected_formula": exp_wff,
                "expected_jstr": exp_jstr,
                "status": "OK" if (a["wffstr"] == exp_wff and jstr_ok) else "MISMATCH",
            })
        results.append(entry)
    return results


# ============================================================================
# Display helpers
# ============================================================================

def _bar(sep: str = "=", width: int = 80) -> str:
    return sep * width


def _truncate(text: str, max_lines: int = 30, max_line_len: int = 120) -> str:
    """Truncate text for display."""
    lines = text.split("\n")
    truncated = False
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        truncated = True
    result = []
    for line in lines:
        if len(line) > max_line_len:
            line = line[:max_line_len - 3] + "..."
        result.append(line)
    if truncated:
        result.append(f"  ... ({len(lines)} total lines, showing first {max_lines})")
    return "\n".join(result)


def print_result(
    *,
    crop_name: str,
    prompt_name: str,
    case_id: str,
    thinking: str,
    raw_content: str,
    parsed_lines: list[dict],
    comparison: list[dict],
    elapsed: float,
    show_thinking: bool = True,
    model: str,
):
    """Print a nicely formatted result block."""
    formulas_ok = sum(1 for r in comparison if r["formula_match"])
    jstrs_ok = sum(1 for r in comparison if r["jstr_match"])
    total = len(comparison)
    all_ok = all(r["status"] == "OK" for r in comparison)

    status_icon = "✅" if all_ok else "❌"

    print()
    print(_bar())
    print(f"  {status_icon}  CASE: {case_id}  |  CROP: {crop_name}  |  "
          f"PROMPT: {prompt_name}  |  MODEL: {model}")
    print(_bar())

    # Timing
    print(f"  TIMING: {elapsed:.1f}s")

    # Thinking text
    if show_thinking and thinking:
        think_lines = thinking.split("\n")
        print()
        print(f"  THINKING ({len(thinking)} chars, ~{len(think_lines)} lines):")
        print(_truncate(thinking, max_lines=25))
        if len(think_lines) > 25:
            print(f"  ... ({len(think_lines)} total thinking lines)")

    # Raw response
    print()
    print("  RAW RESPONSE:")
    print(f"    {_truncate(raw_content.strip(), max_lines=8, max_line_len=100)}")

    # Parsed (normalized)
    print()
    print("  NORMALIZED:")
    for line in parsed_lines:
        n = normalize_line(line)
        print(f"    Line {n['line']}: {n['wffstr']:<30s}  jstr: {n['jstr']}")

    # Expected
    print()
    print("  EXPECTED:")
    for r in comparison:
        print(f"    Pos {r['position']}: {r['expected_formula']:<30s}  jstr: {r['expected_jstr']}")

    # Comparison detail
    print()
    print("  COMPARISON:")
    for r in comparison:
        fm = "✓" if r["formula_match"] else "✗"
        jm = "✓" if r["jstr_match"] else "✗"
        status = "" if r["status"] == "OK" else f"  ← {r['status']}"
        print(f"    Pos {r['position']}: formula={fm}  jstr={jm}{status}")
        if not r["formula_match"]:
            print(f"      formula: got '{r['actual_formula']}' expected '{r['expected_formula']}'")
        if not r["jstr_match"]:
            print(f"      jstr:    got '{r['actual_jstr']}' expected '{r['expected_jstr']}'")

    # Summary
    print()
    print(f"  RESULT: {formulas_ok}/{total} formulas, {jstrs_ok}/{total} justifications  "
          f"{'PASS' if all_ok else 'FAIL'}")
    print(_bar("-"))


# ============================================================================
# Main
# ============================================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Probe subproof OCR with different VLM models and prompts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--case", "-c",
        default="tfl_handwritten_q4_N_or_K_and_W",
        help="Regression case ID (default: Q4). "
             "Available cases are directory names under working/tfl_gold_regressions/.",
    )
    p.add_argument(
        "--only-crops",
        nargs="*",
        default=None,
        help="Only test crops whose filename contains one of these substrings "
             "(e.g. 'crop_3' to test only crop_3.jpg).",
    )
    p.add_argument(
        "--crops",
        nargs="*",
        default=None,
        help="Explicit crop file paths (bypasses case auto-discovery).",
    )
    p.add_argument(
        "--prompt", "-p",
        nargs="*",
        default=ALL_PROMPT_NAMES,
        choices=ALL_PROMPT_NAMES,
        help=f"Prompt variant(s) to test (default: all). Choices: {ALL_PROMPT_NAMES}",
    )
    p.add_argument(
        "--model", "-m",
        default="qwen/qwen3.6-27b",
        help="LM Studio model ID (default: qwen/qwen3.6-27b).",
    )
    p.add_argument(
        "--timeout",
        type=int,
        default=360,
        help="API timeout in seconds (default: 360 — the 27b model is slow).",
    )
    p.add_argument(
        "--base-url",
        default=None,
        help="LM Studio API base URL (default: from LMSTUDIO_BASE_URL env or http://127.0.0.1:1234/v1).",
    )
    p.add_argument(
        "--no-thinking",
        action="store_true",
        help="Don't display thinking text.",
    )
    p.add_argument(
        "--output", "-o",
        default=None,
        help="Write results to this file (in addition to stdout). "
             "Default: working/subproof_probe_results/<timestamp>.txt",
    )
    p.add_argument(
        "--all-cases",
        action="store_true",
        help="Run on all TFL handwritten cases that have subproofs (Q3, Q4, Q6).",
    )
    return p


def run_probe(
    crop_path: Path,
    expected_lines: list[dict],
    *,
    model: str,
    base_url: str,
    prompt_name: str,
    prompt_text: str,
    timeout: int,
    show_thinking: bool,
    case_id: str,
) -> dict:
    """Run a single crop × prompt probe and return results dict."""
    t0 = time.time()
    try:
        content, thinking = call_vlm_with_thinking(
            crop_path, prompt_text,
            model=model, base_url=base_url, timeout=timeout,
        )
    except requests.Timeout:
        elapsed = time.time() - t0
        print(f"  TIMEOUT after {elapsed:.0f}s for {crop_path.name} / {prompt_name}")
        return {"status": "timeout", "elapsed": elapsed}
    except Exception as e:
        elapsed = time.time() - t0
        print(f"  ERROR: {e}")
        return {"status": "error", "error": str(e), "elapsed": elapsed}

    elapsed = time.time() - t0

    parsed = parse_vlm_json(content)
    comparison = compare_lines(parsed, expected_lines)

    print_result(
        crop_name=crop_path.name,
        prompt_name=prompt_name,
        case_id=case_id,
        thinking=thinking,
        raw_content=content,
        parsed_lines=parsed,
        comparison=comparison,
        elapsed=elapsed,
        show_thinking=show_thinking,
        model=model,
    )

    return {
        "status": "ok",
        "crop": crop_path.name,
        "prompt": prompt_name,
        "model": model,
        "elapsed": round(elapsed, 1),
        "thinking": thinking,
        "raw_content": content,
        "parsed_lines": parsed,
        "comparison": comparison,
        "formula_score": f"{sum(1 for r in comparison if r['formula_match'])}/{len(comparison)}",
        "jstr_score": f"{sum(1 for r in comparison if r['jstr_match'])}/{len(comparison)}",
    }


def main():
    parser = build_parser()
    args = parser.parse_args()

    base_url = (args.base_url or default_lmstudio_base_url()).rstrip("/")
    show_thinking = not args.no_thinking
    prompt_names = args.prompt if args.prompt else ALL_PROMPT_NAMES

    # Determine which cases to run
    if args.all_cases:
        case_ids = [
            "tfl_handwritten_q3_P_and_Q_imp_R",
            "tfl_handwritten_q4_N_or_K_and_W",
            "tfl_handwritten_q6_S_iff_W",
        ]
    else:
        case_ids = [args.case]

    # Determine crops
    if args.crops:
        # Explicit crop paths
        crop_map: dict[str, list[Path]] = {}
        for crop_str in args.crops:
            p = Path(crop_str)
            if not p.exists():
                print(f"[ERROR] Crop file not found: {p}", file=sys.stderr)
                sys.exit(1)
            case_key = args.case
            crop_map.setdefault(case_key, []).append(p)
    else:
        # Auto-discover from case
        crop_map = {}
        for cid in case_ids:
            crops = discover_crops(cid, args.only_crops)
            if not crops:
                print(f"[WARN] No subproof crops found for {cid}", file=sys.stderr)
                continue
            crop_map[cid] = crops

    # Resolve prompts
    prompts = [(name, get_prompt(name)) for name in prompt_names]

    # Run probes
    all_results = []
    total_probes = sum(len(crops) * len(prompts) for crops in crop_map.values())
    probe_num = 0

    print(f"SUBPROOF OCR PROBE: {total_probes} probe(s) planned")
    print(f"  Model: {args.model}")
    print(f"  Prompts: {prompt_names}")
    print(f"  Cases: {list(crop_map.keys())}")

    for cid, crops in crop_map.items():
        expected_subproofs = extract_expected_subproofs(cid)

        for crop_path in crops:
            # Extract crop number to pair with expected subproof
            crop_num_str = crop_path.stem.replace("crop_", "")
            try:
                crop_idx = int(crop_num_str) - 2  # crop_2 → index 0 (first subproof)
            except ValueError:
                crop_idx = 0

            # Get expected lines for this subproof
            if 0 <= crop_idx < len(expected_subproofs):
                expected_lines = expected_subproofs[crop_idx]
            else:
                print(f"\n[WARN] No expected subproof at index {crop_idx} for {crop_path.name}")
                print(f"  Available subproofs: {len(expected_subproofs)}")
                print(f"  Will compare against empty expected.")
                expected_lines = []

            for prompt_name, prompt_text in prompts:
                probe_num += 1
                print(f"\n{'#' * 80}")
                print(f"  PROBE {probe_num}/{total_probes}: "
                      f"{crop_path.name} × {prompt_name}")
                print(f"{'#' * 80}")

                result = run_probe(
                    crop_path,
                    expected_lines,
                    model=args.model,
                    base_url=base_url,
                    prompt_name=prompt_name,
                    prompt_text=prompt_text,
                    timeout=args.timeout,
                    show_thinking=show_thinking,
                    case_id=cid,
                )
                all_results.append(result)

    # Summary table
    print("\n\n" + _bar("="))
    print("SUMMARY")
    print(_bar("="))

    ok_results = [r for r in all_results if r.get("status") == "ok"]
    fail_results = [r for r in all_results if r.get("status") != "ok"]

    if fail_results:
        print(f"\n  ERRORS/TIMEOUTS: {len(fail_results)}")
        for r in fail_results:
            print(f"    {r.get('crop', '?')} × {r.get('prompt', '?')}: {r.get('status')} ({r.get('error', '')})")

    if ok_results:
        print(f"\n  {'Crop':<16s} {'Prompt':<18s} {'Formula':>10s} {'Jstr':>10s} {'Time':>8s} {'Status':>6s}")
        print(f"  {'-'*16} {'-'*18} {'-'*10} {'-'*10} {'-'*8} {'-'*6}")
        for r in ok_results:
            f_ok = sum(1 for c in r["comparison"] if c["formula_match"])
            j_ok = sum(1 for c in r["comparison"] if c["jstr_match"])
            total = len(r["comparison"])
            all_ok = f_ok == total and j_ok == total
            icon = "✅" if all_ok else "❌"
            print(f"  {r['crop']:<16s} {r['prompt']:<18s} "
                  f"{f_ok}/{total:<2d}      "
                  f"{j_ok}/{total:<2d}      "
                  f"{r['elapsed']:>6.1f}s "
                  f"{icon:>4s}")

    # Save results
    output_path = args.output
    if output_path is None:
        ts = time.strftime("%Y%m%d_%H%M%S")
        output_dir = REPO_ROOT / "working" / "subproof_probe_results"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"probe_{ts}.json"

    # Clean up results for JSON serialization
    serializable = []
    for r in all_results:
        s = {k: v for k, v in r.items() if k != "thinking"}  # skip huge thinking text
        serializable.append(s)

    Path(output_path).write_text(
        json.dumps(serializable, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print(f"\n  Results saved to: {output_path}")


if __name__ == "__main__":
    main()
