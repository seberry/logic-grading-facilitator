#!/usr/bin/env python
"""Minimal probe: send a text-only query to the 27b model, capture thinking text.

No images, no OCR — just a simple question to verify thinking text capture works.

Usage:
    python -u scripts/probe_27b_thinking.py
    python -u scripts/probe_27b_thinking.py --model "qwen/qwen3.6-27b"
    python -u scripts/probe_27b_thinking.py --prompt "Explain why 7 is prime."
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Force UTF-8 on Windows
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.vlm_client import LMStudioQwenClient

DEFAULT_PROMPT = "What is 2+2? Think step by step, then give the answer."
OUTPUT_DIR = PROJECT_ROOT / "working" / "thinking_captures"


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Probe 27b thinking text capture")
    parser.add_argument("--model", default=None,
                        help="Model name (default: LMSTUDIO_MODEL_27B env or qwen/qwen3.6-27b)")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT,
                        help="Prompt to send (default: simple math)")
    parser.add_argument("--timeout", type=int, default=300,
                        help="Timeout in seconds (default: 300)")
    args = parser.parse_args()

    model = args.model or os.getenv("LMSTUDIO_MODEL_27B", "qwen/qwen3.6-27b")

    print("=" * 72)
    print("THINKING TEXT CAPTURE PROBE")
    print("=" * 72)
    print(f"Model:   {model}")
    print(f"Timeout: {args.timeout}s")
    print(f"Prompt:  {args.prompt}")
    print()

    # Create client
    client = LMStudioQwenClient(model=model, timeout=args.timeout)

    # Simple text-only messages (no image)
    messages = [{"role": "user", "content": args.prompt}]

    print(f"[probe] Sending to {model} ...")
    t0 = time.time()
    try:
        response = client.chat(messages, model=model, timeout=args.timeout)
    except Exception as exc:
        print(f"\n[ERROR] Call failed after {time.time()-t0:.1f}s: {exc}")
        print("\nTip: If LM Studio is busy processing other requests, wait for them")
        print("     to finish or restart LM Studio.")
        sys.exit(1)
    elapsed = time.time() - t0

    print(f"[probe] Response received in {elapsed:.1f}s")
    print()

    # --- Capture thinking text ---
    thinking = getattr(client, "last_thinking", "") or ""
    if thinking:
        print("=" * 72)
        print("THINKING TEXT (reasoning_content)")
        print("=" * 72)
        print(thinking)
        print()
        print(f"[thinking: {len(thinking)} chars, {len(thinking.splitlines())} lines]")
    else:
        print("[no thinking text captured — model may not support thinking or it was empty]")
    print()

    # --- Print response ---
    print("=" * 72)
    print("RESPONSE")
    print("=" * 72)
    print(response)
    print()

    # --- Save to file ---
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model = model.replace("/", "_")
    outfile = OUTPUT_DIR / f"thinking_{safe_model}_{ts}.txt"

    with open(outfile, "w", encoding="utf-8") as f:
        f.write(f"Model: {model}\n")
        f.write(f"Time:  {datetime.now().isoformat()}\n")
        f.write(f"Elapsed: {elapsed:.1f}s\n")
        f.write(f"Prompt: {args.prompt}\n")
        f.write("=" * 72 + "\n")
        if thinking:
            f.write("THINKING TEXT\n")
            f.write("=" * 72 + "\n")
            f.write(thinking + "\n")
            f.write("=" * 72 + "\n")
        f.write("RESPONSE\n")
        f.write("=" * 72 + "\n")
        f.write(response + "\n")

    print(f"[saved] {outfile}")

    # Also save as JSON for programmatic access
    json_out = outfile.with_suffix(".json")
    data = {
        "model": model,
        "prompt": args.prompt,
        "elapsed_s": round(elapsed, 1),
        "thinking": thinking,
        "response": response,
        "thinking_chars": len(thinking),
        "thinking_lines": len(thinking.splitlines()) if thinking else 0,
    }
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"[saved] {json_out}")


if __name__ == "__main__":
    main()