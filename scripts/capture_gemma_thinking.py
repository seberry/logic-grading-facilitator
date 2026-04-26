#!/usr/bin/env python
"""Capture Gemma 4 26b thinking on an image and save to file."""

import requests, json, base64, sys
from pathlib import Path
from datetime import datetime

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

img = Path("working/tfl_gold_regressions/tfl_handwritten_q4_N_or_K_and_W/crop_3.jpg")
b64 = base64.b64encode(img.read_bytes()).decode()
uri = f"data:image/jpeg;base64,{b64}"

payload = {
    "model": "google/gemma-4-26b-a4b",
    "messages": [
        {"role": "system", "content": "You are a careful vision model."},
        {"role": "user", "content": [
            {"type": "text", "text": "What formula is written on the first line of this image? Reply in 5 words or less."},
            {"type": "image_url", "image_url": {"url": uri}},
        ]},
    ],
    "temperature": 0.0,
    "max_tokens": 16384,
}

print("[capture] Sending image to Gemma 4 26b...")
r = requests.post("http://127.0.0.1:1234/v1/chat/completions", json=payload, timeout=300)
d = r.json()
msg = d["choices"][0]["message"]
thinking = msg.get("reasoning_content", "")
content = msg.get("content", "")

print(f"[capture] Got response — thinking: {len(thinking)} chars, content: {content!r}")
print()
print("=== THINKING ===")
print(thinking)
print()
print("=== CONTENT ===")
print(content)

# Save
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
out_txt = Path(f"working/thinking_captures/thinking_gemma-4-26b-a4b_IMAGE_{ts}.txt")
out_txt.write_text(
    f"=== USAGE ===\n{json.dumps(d.get('usage'), indent=2)}\n\n"
    f"=== THINKING ===\n{thinking}\n\n"
    f"=== CONTENT ===\n{content}\n",
    encoding="utf-8",
)
print(f"\nSaved to {out_txt}")