import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from proofs_ocr.domains.tfl import (
    canonicalize_formula_for_comparison as cf,
    canonicalize_justification_for_comparison as cj,
)
from proofs_ocr.regression.tfl_runner import compare_proof_structures_granular
from proofs_ocr.regression.runner import get_repo_root, load_manifest, case_expected_path

root = Path("working/handwritten_ocr_probe/tfl")
manifest = load_manifest()

for case_dir in sorted(root.iterdir()):
    ap = case_dir / "final_structure.json"
    if not ap.exists():
        continue
    actual = json.loads(ap.read_text("utf-8"))
    cid = case_dir.name
    mc = next((c for c in manifest.get("cases", []) if c.get("id") == cid), None)
    if not mc:
        continue
    ep = case_expected_path(mc, repo_root=get_repo_root())
    if not ep.exists():
        continue
    expected = json.loads(ep.read_text("utf-8"))
    m = compare_proof_structures_granular(actual, expected)
    print(f"=== {cid} ===")
    print(f"  F={m['formula_correct_line_count']}/{m['expected_line_count']}  "
          f"J={m['justification_correct_line_count']}/{m['expected_line_count']}  "
          f"B={m['both_correct_line_count']}/{m['expected_line_count']}")
    for r in m.get("line_records", []):
        if r.get("formula_matches") and r.get("justification_matches"):
            continue
        exp = r.get("expected", {})
        act = r.get("actual", {})
        fm = "Y" if r.get("formula_matches") else "N"
        jm = "Y" if r.get("justification_matches") else "N"
        print(f"  L{r['order_index']:2d} F:{fm} J:{jm}")
        if not r.get("formula_matches"):
            print(f"    formula: gold={cf(exp.get('wffstr',''))!r}  ocr={cf(act.get('wffstr',''))!r}")
        if not r.get("justification_matches"):
            print(f"    just:    gold={cj(exp.get('jstr',''))!r}  ocr={cj(act.get('jstr',''))!r}")
    print()