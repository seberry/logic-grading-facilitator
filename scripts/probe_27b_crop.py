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

MANUAL_PROMPT = (
    "Role: You are a logic quiz transcriber performing OCR on subproof chunks "
    "cropped out from the middle of handwritten Fitch-style natural deduction proofs.\n"
    "Context: You will be given cropped images of specific subproofs (beginning with an "
    "assumption with no justification above a horizontal line, and deriving various "
    "consequences from that line and earlier lines). So don't worry that those are missing, "
    "and remember that the top line in the crop you are seeing might not be line 1 in the proof.\n"
    "Task: Extract the logical formulas and their justifications.\n"
    "Handwriting Translation Guide (CRITICAL):\n"
    "Students have sloppy handwriting. Use the following known distortions to correct your visual reading:\n"
    "A sloppy K frequently looks like 1 < or a vertical line next to a <.\n"
    "A sloppy ampersand & (conjunction) frequently looks like an 8.\n"
    "Output format: JSON array of objects with keys 'line', 'wffstr', 'jstr'."
)


def probe_crop(
    image_path: Path,
    *,
    expected_lines: int = 2,
    model: str | None = None,
    timeout: int = 600,
    manual_prompt: bool = False,
    no_think: bool = False,
    no_max_tokens: bool = False,
    max_tokens: int = 16384,
    single_user_message: bool = False,
    enable_thinking_false: bool = False,
) -> None:
    """Send *image_path* to the 27b model and print everything."""

    model = model or os.getenv("LMSTUDIO_MODEL_27B", "qwen/qwen3.6-27b")

    # --- Build the prompt ---
    if manual_prompt:
        prompt = MANUAL_PROMPT
        prompt_label = "MANUAL PROMPT (exact text from LM Studio GUI)"
    else:
        prompt = build_ocr_prompt(has_children=False, expected_lines=expected_lines)
        prompt_label = "PIPELINE PROMPT (build_ocr_prompt)"

    if no_think:
        prompt += " /no_think"
        prompt_label += " + /no_think"

    print("=" * 72)
    print("PROBE: 27b model on single crop")
    print("=" * 72)
    print(f"Image:    {image_path}")
    print(f"Model:    {model}")
    print(f"Timeout:  {timeout}s")
    print(f"Expected lines: {expected_lines}")
    print(f"Prompt mode: {prompt_label}")
    print(f"Message mode: {'single user' if (manual_prompt or single_user_message) else 'system + user'}")
    print(f"Max tokens: {'server default' if no_max_tokens else max_tokens}")
    print(f"enable_thinking=false payload: {enable_thinking_false}")
    print()
    print("--- PROMPT ---")
    print(prompt)
    print()

    # --- Create client and build messages ---
    client = LMStudioQwenClient(model=model, timeout=timeout)

    if manual_prompt or single_user_message:
        # Match LM Studio GUI: put everything in a single user message with the image,
        # no system message.
        messages = client.build_single_user_image_message(image_path, prompt)
    else:
        messages = client.build_messages_for_prompt(image_path, prompt)

    # --- Call the model ---
    print(f"[probe] Sending to {model} ... (this may take several minutes)")
    t0 = time.time()
    try:
        max_tok = None if no_max_tokens else max_tokens
        extra_payload = {"chat_template_kwargs": {"enable_thinking": False}} if enable_thinking_false else None
        raw_response = client.chat(
            messages,
            model=model,
            timeout=timeout,
            max_tokens=max_tok,
            extra_payload=extra_payload,
        )
    except Exception as exc:
        print(f"\n[ERROR] VLM call failed: {exc}")
        sys.exit(1)
    elapsed = time.time() - t0
    print(f"[probe] Response received in {elapsed:.1f}s")
    print()
    metadata = getattr(client, "last_response_metadata", {}) or {}
    if metadata:
        print("=" * 72)
        print("CALL METADATA")
        print("=" * 72)
        print(json.dumps(metadata, indent=2, ensure_ascii=False))
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

    out_dir = PROJECT_ROOT / "working" / "thinking_captures"
    out_dir.mkdir(parents=True, exist_ok=True)
    from datetime import datetime

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model = model.replace("/", "_")
    prompt_slug = "manual" if manual_prompt else "pipeline"
    message_slug = "single" if (manual_prompt or single_user_message) else "system"
    result_path = out_dir / f"probe_27b_crop_{safe_model}_{prompt_slug}_{message_slug}_{ts}.json"
    result_payload = {
        "image": str(image_path),
        "model": model,
        "timeout": timeout,
        "expected_lines": expected_lines,
        "manual_prompt": manual_prompt,
        "no_think": no_think,
        "single_user_message": single_user_message,
        "no_max_tokens": no_max_tokens,
        "max_tokens": None if no_max_tokens else max_tokens,
        "enable_thinking_false": enable_thinking_false,
        "elapsed_s": round(elapsed, 1),
        "metadata": metadata,
        "thinking_chars": len(thinking),
        "response": raw_response,
        "parsed": parsed,
    }
    result_path.write_text(json.dumps(result_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[saved] {result_path}")


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
    parser.add_argument(
        "--manual-prompt",
        action="store_true",
        help="Use the exact manual prompt that worked in LM Studio GUI "
             "(single user message, no system prompt, Role/Context/Task framing)",
    )
    parser.add_argument(
        "--no-think",
        action="store_true",
        help="Append /no_think to disable Qwen3 extended thinking",
    )
    parser.add_argument(
        "--no-max-tokens",
        action="store_true",
        help="Don't send max_tokens parameter (let LM Studio use its server default)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=16384,
        help="max_tokens/max_completion_tokens to send unless --no-max-tokens is set (default: 16384)",
    )
    parser.add_argument(
        "--single-user-message",
        action="store_true",
        help="Use a GUI-like single user message with prompt text and image, even for the pipeline prompt",
    )
    parser.add_argument(
        "--enable-thinking-false",
        action="store_true",
        help="Send chat_template_kwargs.enable_thinking=false in the OpenAI-compatible payload",
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
        manual_prompt=args.manual_prompt,
        no_think=args.no_think,
        no_max_tokens=args.no_max_tokens,
        max_tokens=args.max_tokens,
        single_user_message=args.single_user_message,
        enable_thinking_false=args.enable_thinking_false,
    )


if __name__ == "__main__":
    main()
