"""Prepare, run, and privately review a bounded Gemma pilot from a TF bundle."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from proofs_ocr.regression.private_fitch_pilot import discover_bundles, bundle_info, verify_bundle, prepare, run, serve, refresh_summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["inventory", "prepare", "run", "serve", "summary"])
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--quiz", type=int, default=7)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--port", type=int, default=5030)
    args = parser.parse_args()
    try:
        if args.action in {"inventory", "prepare"}:
            candidates = discover_bundles([Path.home() / "Downloads", Path.home() / "grading-data"])
            if args.action == "inventory":
                print(json.dumps([{"candidate": i, "quiz": c["quiz"], "submissions": c["students"],
                    "rubric_proof_hints": len(c["proof_ids"]),
                    "metadata_label_conflict": c["metadata_label_conflict"]} for i, c in enumerate(candidates)]))
                return
        if not args.output_dir:
            raise ValueError("An external output directory is required")
        if args.action == "prepare":
            if (args.output_dir / "manifest.json").exists():
                raise ValueError("This pilot already exists; choose another output folder")
            if args.bundle:
                info = bundle_info(args.bundle)
            else:
                choices = [c for c in candidates if c["quiz"] == args.quiz]
                if not choices:
                    raise ValueError("No matching proof bundle found")
                info = max(choices, key=lambda c: c["path"].stat().st_mtime)
            info["verification"] = verify_bundle(info, args.output_dir.parent / (args.output_dir.name + "_source_checks"))
            manifest = prepare(info, args.output_dir, samples=args.samples, seed=args.seed)
            print(json.dumps({"prepared": len(manifest["cases"]), "quiz": manifest["quiz"],
                "questions": len({c["problem_id"] for c in manifest["cases"]}),
                "policy": manifest["sample_policy"]}))
        elif args.action == "run":
            if args.limit is not None and args.limit < 1:
                raise ValueError("A positive limit is required")
            print(json.dumps(run(args.output_dir, limit=args.limit)))
        elif args.action == "serve":
            serve(args.output_dir, port=args.port)
        else:
            print(json.dumps(refresh_summary(args.output_dir)))
    except KeyboardInterrupt:
        pass
    except Exception as error:
        print(json.dumps({"error_type": type(error).__name__, "stopped": True,
            "message": "Private pilot stopped; no student content or paths are printed."}))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
