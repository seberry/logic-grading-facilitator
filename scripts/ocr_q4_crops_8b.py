"""OCR the Q4 non-context subproof crops with qwen3-vl-8b."""
import json
import sys
from pathlib import Path

# Ensure repo root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.vlm_client import LMStudioQwenClient
from proofs_ocr.pipelines.tfl_fitch import build_ocr_prompt

repo_root = Path(__file__).resolve().parent.parent
out_file = repo_root / "working" / "q4_8b_probe_results.txt"
client = LMStudioQwenClient(model="qwen/qwen3-vl-8b")
prompt = build_ocr_prompt(has_children=False, highlighted_box=False)

q4_dir = repo_root / "working" / "tfl_gold_regressions" / "tfl_handwritten_q4_N_or_K_and_W"
expected_path = repo_root / "proofs_ocr" / "regression" / "fixtures" / "tfl" / "handwritten" / "q4_N_or_K_and_W" / "expected_structure.json"
expected = json.loads(expected_path.read_text(encoding="utf-8"))

lines = []
lines.append("=== EXPECTED Q4 STRUCTURE ===")
lines.append(json.dumps(expected, indent=2, ensure_ascii=False))
lines.append("")
lines.append(f"PROMPT: {prompt}")
lines.append("")

for crop_name in ["crop_2.jpg", "crop_3.jpg"]:
    crop_path = q4_dir / crop_name
    if not crop_path.exists():
        lines.append(f"{crop_name}: NOT FOUND at {crop_path}")
        continue
    lines.append(f"=== {crop_name} (qwen3-vl-8b, leaf prompt) ===")
    print(f"[probe] Sending {crop_name} to 8b...", file=sys.stderr)
    try:
        response = client.chat_with_image_prompt(crop_path, prompt)
        lines.append(response)
    except Exception as e:
        lines.append(f"ERROR: {e}")
    lines.append("")

# Also run with 4b for comparison
client4 = LMStudioQwenClient(model="qwen/qwen3-vl-4b")
for crop_name in ["crop_2.jpg", "crop_3.jpg"]:
    crop_path = q4_dir / crop_name
    if not crop_path.exists():
        continue
    lines.append(f"=== {crop_name} (qwen3-vl-4b, leaf prompt) ===")
    print(f"[probe] Sending {crop_name} to 4b...", file=sys.stderr)
    try:
        response = client4.chat_with_image_prompt(crop_path, prompt)
        lines.append(response)
    except Exception as e:
        lines.append(f"ERROR: {e}")
    lines.append("")

result = "\n".join(lines)
out_file.parent.mkdir(parents=True, exist_ok=True)
out_file.write_text(result, encoding="utf-8")
print(result)