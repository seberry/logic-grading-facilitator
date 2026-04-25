#!/usr/bin/env python
"""Quick probe: send a single subproof crop directly to the 27b model.

Bypasses the full pipeline (bar detection, tree building, masking, stitching)
and tests the VLM OCR prompt + 27b model on one crop image.

Usage:
    python scripts/probe_27b_crop.py
    python scripts/probe_27b_crop.py path/to/crop.jpg
    python scripts/probe_27b_crop.py path/to/crop.jpg --expected-lines 3
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

# Force UTF-8 output on Windows console to handle logic symbols
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Ensure project root is on sys.path so we can import app/ and proofs_ocr/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.vlm_client import LMStudioQwenClient
from proofs_ocr.pipelines.tfl_fitch import build_ocr_prompt


def probe_crop(
    image_path: Path,
    *,
    expected_lines: int = 2,
    model: str | None = None,
    timeout: int = 600,
) -> None:
    """Send *image_path* to the 27b model and print everything."""

    model = model or os.getenv("LMSTUDIO_MODEL_27B", "qwen/qwen3.6-27b")

    # --- Build the exact prompt the pipeline would use ---
    prompt = build_ocr_prompt(has_children=False, expected_lines=expected_lines)

    print("=" * 72)
    print("PROBE: 27b model on single crop")
    print("=" * 72)
    print(f"Image:    {image_path}")
    print(f"Model:    {model}")
    print(f"Timeout:  {timeout}s")
    print(f"Expected lines: {expected_lines}")
    print()
    print("--- PROMPT ---")
    print(prompt)
    print()

    # --- Create client and build messages ---
    client = LMStudioQwenClient(model=model, timeout=timeout)
    messages = client.build_messages_for_prompt(image_path, prompt)

    # --- Call the model ---
    print(f"[probe] Sending to {model} ... (this may take several minutes)")
    t0 = time.time()
    try:
        raw_response = client.chat(messages, model=model, timeout=timeout)
    except Exception as exc:
        print(f"\n[ERROR] VLM call failed: {exc}")
        sys.exit(1)
    elapsed = time.time() - t0
    print(f"[probe] Response received in {elapsed:.1f}s")
    print()

    # --- Print thinking text ---
    thinking = getattr(client, "last_thinking", "") or ""
    if thinking:
        print("=" * 72)
        print("THINKING TEXT (reasoning_content)")
        print("=" * 72)
        print(thinking)
        print()
        print(f"[thinking: {len(thinking)} chars, ~{len(thinking.splitlines())} lines]")
    else:
        print("[no thinking text captured]")
    print()

    # --- Print raw response ---
    print("=" * 72)
    print("RAW VLM RESPONSE")
    print("=" * 72)
    print(raw_response)
    print()

    # --- Try to parse JSON ---
    content = raw_response
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0]
    elif "```" in content:
        content = content.split("```")[1].split("```")[0]

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        print(f"[ERROR] Could not parse JSON: {exc}")
        parsed = None

    if parsed is not None:
        print("=" * 72)
        print("PARSED JSON")
        print("=" * 72)
        print(json.dumps(parsed, indent=2, ensure_ascii=False))
        print()

    # --- Show expected answer ---
    print("=" * 72)
    print("EXPECTED ANSWER (for crop_3 of Q4: K∧W subproof)")
    print("=" * 72)
    expected = [
        {"line": 1, "wffstr": "K∧W", "jstr": "Hyp"},
        {"line": 2, "wffstr": "W", "jstr": "∧E 5"},
    ]
    print(json.dumps(expected, indent=2, ensure_ascii=False))
    print()

    # --- Quick comparison ---
    if parsed and isinstance(parsed, list):
        print("=" * 72)
        print("COMPARISON")
        print("=" * 72)
        for i, (exp, act) in enumerate(
            zip(expected, parsed)
        ):
            if not isinstance(act, dict):
                print(f"  Line {i+1}: actual is not a dict: {act!r}")
                continue
            wff_ok = act.get("wffstr", "").strip() == exp["wffstr"]
            jstr_ok = act.get("jstr", "").strip() == exp["jstr"]
            status = "✅" if (wff_ok and jstr_ok) else "❌"
            print(
                f"  Line {i+1} {status}: "
                f"wff={'✅' if wff_ok else '❌'} "
                f"(exp={exp['wffstr']!r}, got={act.get('wffstr','')!r})  "
                f"jstr={'✅' if jstr_ok else '❌'} "
                f"(exp={exp['jstr']!r}, got={act.get('jstr','')!r})"
            )
        if len(parsed) > len(expected):
            print(f"  ⚠️  Extra lines: {len(parsed) - len(expected)}")
        elif len(parsed) < len(expected):
            print(f"  ⚠️  Missing lines: {len(expected) - len(parsed)}")


def main():
    default_crop = (
        PROJECT_ROOT
        / "working"
        / "tfl_gold_regressions"
        / "tfl_handwritten_q4_N_or_K_and_W"
        / "crop_3.jpg"
    )

    import argparse
    parser = argparse.ArgumentParser(description="Probe the 27b model on a single crop")
    parser.add_argument(
        "image",
        nargs="?",
        default=str(default_crop),
        help=f"Path to crop image (default: {default_crop})",
    )
    parser.add_argument(
        "--expected-lines",
        type=int,
        default=2,
        help="Number of expected lines to include in prompt (default: 2)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name (default: LMSTUDIO_MODEL_27B env or qwen/qwen3.6-27b)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Timeout in seconds (default: 600)",
    )
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"[ERROR] Image not found: {image_path}")
        sys.exit(1)

    probe_crop(
        image_path,
        expected_lines=args.expected_lines,
        model=args.model,
        timeout=args.timeout,
    )


if __name__ == "__main__":
    main()