#!/usr/bin/env python
"""Diagnostic probe: test whether the 27b model can handle ANY image via API.

Creates a tiny 100x100 white JPEG and sends it with a simple prompt.
If this hangs, the problem is fundamental (model can't process images via API),
not related to image size or complexity.

Also supports /no_think tag to test if extended thinking is the hang cause.

Usage:
    python -u scripts/probe_27b_diagnostic.py --model "qwen/qwen3.6-27b"
    python -u scripts/probe_27b_diagnostic.py --model "qwen/qwen3.6-27b" --no-think
    python -u scripts/probe_27b_diagnostic.py --model "google/gemma-4-26b-a4b"
"""

from __future__ import annotations

import io
import json
import os
import sys
import time
from pathlib import Path

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image


def create_tiny_test_image() -> Path:
    """Create a 100x100 white JPEG with a simple black 'A' on it."""
    img = Image.new("RGB", (100, 100), "white")
    # Draw a simple mark so it's not just blank white
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    draw.text((30, 30), "A", fill="black")
    
    out_path = PROJECT_ROOT / "working" / "_tiny_test_100x100.jpg"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, format="JPEG", quality=90)
    return out_path


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Diagnostic: test if model handles any image via API")
    parser.add_argument("--model", default=None,
                        help="Model name (default: LMSTUDIO_MODEL_27B env or qwen/qwen3.6-27b)")
    parser.add_argument("--timeout", type=int, default=120,
                        help="Timeout in seconds (default: 120)")
    parser.add_argument("--no-think", action="store_true",
                        help="Append /no_think to disable extended thinking")
    parser.add_argument("--use-crop", action="store_true",
                        help="Use the real crop_3.jpg instead of tiny test image")
    parser.add_argument("--text-only", action="store_true",
                        help="Skip image entirely, text-only prompt")
    args = parser.parse_args()

    model = args.model or os.getenv("LMSTUDIO_MODEL_27B", "qwen/qwen3.6-27b")

    print("=" * 72)
    print("DIAGNOSTIC PROBE: Image handling test")
    print("=" * 72)
    print(f"Model:      {model}")
    print(f"Timeout:    {args.timeout}s")
    print(f"No-think:   {args.no_think}")
    print(f"Text-only:  {args.text_only}")
    print()

    # Build prompt
    prompt = "Describe what you see in this image in one sentence."
    if args.no_think:
        prompt += " /no_think"
    
    print(f"Prompt:     {prompt}")

    # Build messages
    if args.text_only:
        messages = [{"role": "user", "content": prompt}]
        print("Mode:       text-only (no image)")
    else:
        if args.use_crop:
            image_path = (
                PROJECT_ROOT
                / "working"
                / "tfl_gold_regressions"
                / "tfl_handwritten_q4_N_or_K_and_W"
                / "crop_3.jpg"
            )
            print(f"Mode:       real crop image")
        else:
            image_path = create_tiny_test_image()
            print(f"Mode:       tiny test image")

        if not image_path.exists():
            print(f"[ERROR] Image not found: {image_path}")
            sys.exit(1)

        print(f"Image:      {image_path} ({image_path.stat().st_size:,} bytes)")

        # Import client for building messages
        from app.vlm_client import LMStudioQwenClient
        client = LMStudioQwenClient(model=model, timeout=args.timeout)
        messages = client.build_messages_for_prompt(image_path, prompt)

    # Send request
    import requests
    base_url = os.getenv("LMSTUDIO_BASE_URL", "http://127.0.0.1:1234/v1").rstrip("/")
    url = f"{base_url}/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.0,
    }

    print()
    print(f"[probe] Sending to {model} ...")
    t0 = time.time()
    try:
        resp = requests.post(url, json=payload, timeout=args.timeout)
        elapsed = time.time() - t0
        print(f"[probe] Response received in {elapsed:.1f}s")
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.Timeout:
        elapsed = time.time() - t0
        print(f"\n[TIMEOUT] No response after {elapsed:.1f}s (limit: {args.timeout}s)")
        print("This confirms the model hangs on image input via API.")
        sys.exit(1)
    except Exception as exc:
        elapsed = time.time() - t0
        print(f"\n[ERROR] Failed after {elapsed:.1f}s: {exc}")
        sys.exit(1)

    # Parse response
    if "choices" not in data:
        print(f"[ERROR] Unexpected response: {json.dumps(data, indent=2)[:500]}")
        sys.exit(1)

    message = data["choices"][0].get("message", {})
    content = str(message.get("content") or "")
    thinking = str(message.get("reasoning_content") or "")

    # Print thinking
    if thinking:
        print()
        print("=" * 72)
        print("THINKING TEXT")
        print("=" * 72)
        # Print first 500 chars
        print(thinking[:500])
        if len(thinking) > 500:
            print(f"... ({len(thinking)} chars total)")
        print()

    # Print response
    print()
    print("=" * 72)
    print("RESPONSE")
    print("=" * 72)
    print(content)
    print()

    # Print usage
    usage = data.get("usage", {})
    if usage:
        print(f"Usage: {json.dumps(usage)}")

    # Save result
    out_dir = PROJECT_ROOT / "working" / "thinking_captures"
    out_dir.mkdir(parents=True, exist_ok=True)
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model = model.replace("/", "_")
    tag = "text" if args.text_only else ("crop" if args.use_crop else "tiny")
    if args.no_think:
        tag += "_nothink"
    outfile = out_dir / f"diag_{safe_model}_{tag}_{ts}.json"

    result = {
        "model": model,
        "mode": tag,
        "no_think": args.no_think,
        "elapsed_s": round(elapsed, 1),
        "thinking_chars": len(thinking),
        "thinking_preview": thinking[:200] if thinking else "",
        "response": content,
        "usage": usage,
    }
    with open(outfile, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"[saved] {outfile}")


if __name__ == "__main__":
    main()