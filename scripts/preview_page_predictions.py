# scripts/preview_page_predictions.py

from pathlib import Path
from app.storage import load_roster
from app.vlm_client import LMStudioQwenClient
from app.vlm_classifier import classify_page
from app.utils import natural_key

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-scans-dir", type=Path, default=Path("data/raw_scans"))
    parser.add_argument("--roster", type=Path, default=Path("data/roster.csv"))
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--lmstudio-base-url", default="http://127.0.0.1:1234/v1")
    parser.add_argument("--lmstudio-model", default="qwen-vl")
    args = parser.parse_args()

    roster = load_roster(args.roster)
    client = LMStudioQwenClient(
        base_url=args.lmstudio_base_url,
        model=args.lmstudio_model,
    )

    exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    paths = [
        p for p in sorted(args.raw_scans_dir.iterdir(), key=natural_key)
        if p.is_file() and p.suffix in exts
    ][: args.limit]


    for i, path in enumerate(paths, start=1):
        pred = classify_page(path, roster, client)
        print(f"{i:02d}. {path.name}")
        print(f"    name_visible   = {pred.name_visible}")
        print(f"    raw_name       = {pred.raw_name!r}")
        print(f"    student_id     = {pred.student_id!r}")
        print(f"    confidence     = {pred.confidence:.3f}")
        print(f"    reasoning      = {pred.reasoning}")
        print()

if __name__ == "__main__":
    main()
