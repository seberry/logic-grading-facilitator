"""Private, instructor-reviewed whole-proof OCR pilot from a read-only TF ZIP.

Never prints student names, source paths, handwriting, predictions, or gold text.
All derived artifacts and human labels must remain outside the repository.
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import math
from pathlib import Path
import random
import re
import zipfile

from PIL import Image, ImageOps
from app.vlm_provider import create_vlm_client

from proofs_ocr.pipelines.tfl_whole_page import PROMPT, response_format, flatten_rows, rows_to_structure, transcribe_whole_proof
from proofs_ocr.regression.tfl_runner import compare_proof_structures, compare_proof_structures_granular

REPO = Path(__file__).resolve().parents[2]


def load_json(path: Path, default=None):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def external_root(path: Path) -> Path:
    path = path.resolve()
    if path == REPO or REPO in path.parents:
        raise ValueError("Private pilot output must be outside the repository")
    return path


def bundle_info(path: Path) -> dict:
    with zipfile.ZipFile(path) as archive:
        sessions = [n for n in archive.namelist() if n.endswith("/grades/session.json")]
        if len(sessions) != 1:
            raise ValueError("Expected one grading session")
        session_entry = sessions[0]
        session = json.loads(archive.read(session_entry))
        prefix = session_entry[:-len("grades/session.json")]
        key = session.get("grading_key") or {}
        filename_match = re.search(r"quiz[ _-]*(\d+)", path.stem, re.I)
        saved_match = re.search(r"quiz[ _-]*(\d+)", str(key.get("quiz_id", "")), re.I)
        match = filename_match or saved_match
        proof_ids = []
        for problem in key.get("problems", []):
            text = " ".join(str(problem.get(k) or "") for k in ("label", "prompt", "segmentation_hint"))
            text += " ".join(str(v.get(k) or "") for v in (problem.get("variants") or {}).values()
                             if isinstance(v, dict) for k in ("label", "prompt", "segmentation_hint"))
            if (not problem.get("extra_credit") and problem.get("id")
                    and re.search(r"proof|deriv|fitch|infer|deduc", text, re.I)
                    and not re.search(r"∀|∃|forall|exists", text)):
                proof_ids.append(problem["id"])
        return {"path": path.resolve(), "quiz": int(match.group(1)) if match else None,
                "metadata_label_conflict": bool(filename_match and saved_match and filename_match.group(1) != saved_match.group(1)),
                "students": len(session.get("work_units", [])), "proof_ids": proof_ids,
                "session_entry": session_entry, "prefix": prefix, "session": session}


def discover_bundles(roots: list[Path]) -> list[dict]:
    found, seen = [], set()
    for root in roots:
        files = root.glob("*.zip") if root.name == "Downloads" else root.rglob("*.zip")
        for path in files:
            if path.resolve() in seen or not re.search(r"tf.*(?:grad|bundle)|grad.*bundle", path.name, re.I):
                continue
            seen.add(path.resolve())
            try:
                found.append(bundle_info(path))
            except (OSError, ValueError, KeyError, zipfile.BadZipFile):
                continue
    return found


def valid_bbox(box) -> bool:
    return (isinstance(box, list) and len(box) == 4
            and all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) and 0 <= x <= 1 for x in box)
            and box[0] < box[2] and box[1] < box[3])


def classify_proof_crop(image_path: Path, *, client=None) -> dict:
    """Return layout categories only; never return student response text."""
    client = client or create_vlm_client("gemma")
    prompt = ("Classify the actual handwritten answer layout in this image. Do not transcribe, quote, "
        "solve, or summarize any student response or name. Ignore assignment titles and rubric claims. "
        "A Fitch proof has a proof spine/subproof bars or a clear sequence of formula lines with "
        "inference-rule justifications and cited line numbers. A list of independently symbolized "
        "English sentences is symbolization, not a Fitch proof. Blank/ambiguous writing is blank_or_unclear. "
        "Return only the requested classification fields. has_quantifiers means visible universal or existential quantifiers.")
    properties = {"kind": {"type": "string", "enum": ["fitch_proof", "symbolization", "truth_table", "other", "blank_or_unclear"]},
        "has_proof_bars": {"type": "boolean"}, "has_rule_citations": {"type": "boolean"},
        "has_quantifiers": {"type": "boolean"}}
    spec = {"type": "json_schema", "json_schema": {"name": "proof_layout", "strict": True,
        "schema": {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}}}
    with contextlib.redirect_stdout(io.StringIO()):
        raw = client.chat(client.build_messages_for_prompt(image_path, prompt), max_tokens=256,
                          extra_payload={"response_format": spec})
    value = json.loads(raw)
    if (not isinstance(value, dict) or value.get("kind") not in properties["kind"]["enum"]
            or any(not isinstance(value.get(key), bool) for key in properties if key != "kind")):
        raise ValueError("Invalid layout classification")
    return {key: value[key] for key in properties}


def verify_bundle(info: dict, output: Path) -> dict:
    """Verify actual answer layout privately, including bundles with stale rubrics."""
    output = external_root(output)
    output.mkdir(parents=True, exist_ok=True)
    candidates = [p["id"] for p in info["session"].get("grading_key", {}).get("problems", [])
                  if p.get("id") and not p.get("extra_credit")]
    checks, verified_ids = [], []
    with zipfile.ZipFile(info["path"]) as archive:
        for problem_id in candidates:
            provisional = {**info, "proof_ids": [problem_id]}
            try:
                examples = select_cases(provisional, samples=2, seed=42)
            except ValueError:
                continue
            for sample_index, sample in enumerate(examples, 1):
                with Image.open(io.BytesIO(archive.read(sample["image_entry"]))) as source:
                    page = ImageOps.exif_transpose(source).convert("RGB")
                    width, height = page.size
                    a,b,c,d = sample["bbox"]
                    crop = page.crop((int(a*width),int(b*height),math.ceil(c*width),math.ceil(d*height)))
                    image_path = output / f"check_{len(checks)+1:03d}.png"
                    crop.save(image_path)
                classification = classify_proof_crop(image_path)
                checks.append({"problem_id": problem_id, "sample": sample_index, **classification})
                if (classification["kind"] == "fitch_proof" and not classification["has_quantifiers"]
                        and (classification["has_proof_bars"] or classification["has_rule_citations"])):
                    verified_ids.append(problem_id)
                    break
                if classification["kind"] != "blank_or_unclear":
                    break
    evidence = {"verified_proof_ids": verified_ids, "checks": checks,
                "source_sha256": hashlib.sha256(info["path"].read_bytes()).hexdigest()}
    write_json(output / "verification.json", evidence)
    return evidence


def select_cases(info: dict, *, samples: int, seed: int) -> list[dict]:
    if samples < 1:
        raise ValueError("A positive sample size is required")
    session = info["session"]
    units = {u["unit_id"]: u for u in session.get("work_units", [])}
    groups = {}
    with zipfile.ZipFile(info["path"]) as archive:
        entries = set(archive.namelist())
        for problem_id in sorted(info["proof_ids"]):
            locations_entry = info["prefix"] + f"grades/segments_{problem_id}.json"
            if locations_entry not in entries:
                continue
            for location in json.loads(archive.read(locations_entry)):
                unit = units.get(location.get("unit_id"))
                if not unit or not location.get("present") or not valid_bbox(location.get("bbox")):
                    continue
                filename = location.get("page_filename")
                if not filename:
                    index = location.get("page_index", 0)
                    if not isinstance(index, int) or not 0 <= index < len(unit.get("page_filenames", [])):
                        continue
                    filename = unit["page_filenames"][index]
                if filename not in unit.get("page_filenames", []) or Path(filename).name != filename:
                    continue
                image_entry = info["prefix"] + "raw_scans/" + filename
                if image_entry not in entries:
                    continue
                group = (problem_id, str(unit.get("quiz_version") or "unknown"))
                groups.setdefault(group, []).append({"unit_id": unit["unit_id"],
                    "problem_id": problem_id, "version": unit.get("quiz_version"),
                    "image_entry": image_entry, "bbox": location["bbox"],
                    "student_key": unit.get("student_id") or f"unit:{unit['unit_id']}"})
    generator = random.Random(seed)
    for group in groups.values():
        generator.shuffle(group)
    selected, used = [], set()
    while len(selected) < samples:
        before = len(selected)
        for key in sorted(groups):
            group = groups[key]
            while group and group[-1]["student_key"] in used:
                group.pop()
            if group:
                case = group.pop()
                used.add(case.pop("student_key"))
                selected.append(case)
                if len(selected) == samples:
                    break
        if len(selected) == before:
            break
    if not selected:
        raise ValueError("No saved TFL proof regions found")
    return selected


def prepare(info: dict, output: Path, *, samples: int = 20, seed: int = 42) -> dict:
    output = external_root(output)
    if (output / "manifest.json").exists():
        raise ValueError("This pilot already exists; choose another output folder")
    verification = info.get("verification")
    if not verification or not verification.get("verified_proof_ids"):
        raise ValueError("Verify actual proof-image layout before preparing a pilot")
    if verification["source_sha256"] != hashlib.sha256(info["path"].read_bytes()).hexdigest():
        raise ValueError("The verified source bundle changed")
    info = {**info, "proof_ids": verification["verified_proof_ids"]}
    selected = select_cases(info, samples=samples, seed=seed)
    output.mkdir(parents=True, exist_ok=True)
    for name in ("crops", "pages", "results", "gold"):
        (output / name).mkdir(exist_ok=True)
    cases = []
    with zipfile.ZipFile(info["path"]) as archive:
        for index, source in enumerate(selected, 1):
            case_id = f"case_{index:04d}"
            image_bytes = archive.read(source["image_entry"])
            with Image.open(io.BytesIO(image_bytes)) as source_image:
                page = ImageOps.exif_transpose(source_image).convert("RGB")
                width, height = page.size
                x0, y0, x1, y1 = source["bbox"]
                pixel_box = (max(0, int((x0-.015)*width)), max(0, int((y0-.015)*height)),
                             min(width, math.ceil((x1+.015)*width)), min(height, math.ceil((y1+.015)*height)))
                crop = page.crop(pixel_box)
                crop.save(output / "crops" / f"{case_id}.png")
                page.thumbnail((1800, 2400))
                page.save(output / "pages" / f"{case_id}.jpg", quality=88)
            cases.append({"case_id": case_id, **source,
                "image_sha256": hashlib.sha256(image_bytes).hexdigest(),
                "crop_sha256": hashlib.sha256((output / "crops" / f"{case_id}.png").read_bytes()).hexdigest()})
    manifest = {"quiz": info["quiz"], "seed": seed, "sample_policy": "Balanced across main proof questions/version; at most one proof per student; selected before inference.",
                "source_verification": verification,
                "prompt_sha256": hashlib.sha256(PROMPT.encode()).hexdigest(),
                "schema_sha256": hashlib.sha256(json.dumps(response_format(), sort_keys=True).encode()).hexdigest(),
                "source_bundle": str(info["path"]),
                "source_sha256": hashlib.sha256(info["path"].read_bytes()).hexdigest(), "cases": cases}
    write_json(output / "manifest.json", manifest)
    return manifest


def run(output: Path, *, limit: int | None = None) -> dict:
    output = external_root(output)
    manifest = load_json(output / "manifest.json")
    if (manifest["prompt_sha256"] != hashlib.sha256(PROMPT.encode()).hexdigest()
            or manifest["schema_sha256"] != hashlib.sha256(json.dumps(response_format(), sort_keys=True).encode()).hexdigest()):
        raise ValueError("The pilot transcription contract changed; prepare a separate experiment")
    consecutive_failures = 0
    for case in manifest["cases"][:limit]:
        result_path = output / "results" / f"{case['case_id']}.json"
        if result_path.exists():
            continue
        crop = output / "crops" / f"{case['case_id']}.png"
        if hashlib.sha256(crop.read_bytes()).hexdigest() != case["crop_sha256"]:
            raise ValueError("A pilot crop changed")
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                result = transcribe_whole_proof(crop, model="gemma")
            result["status"] = "ok"
            consecutive_failures = 0
        except Exception as error:
            result = {"status": "error", "error_type": type(error).__name__}
            consecutive_failures += 1
        write_json(result_path, result)
        print(f"Proof {case['case_id']}: {result['status']}", flush=True)
        refresh_summary(output)
        if consecutive_failures >= 3:
            raise ValueError("Three consecutive requests failed; pilot stopped")
    return refresh_summary(output)


def comparison(output: Path) -> dict:
    manifest = load_json(output / "manifest.json")
    cases = []
    for case in manifest["cases"]:
        result = load_json(output / "results" / f"{case['case_id']}.json", {})
        cases.append({"case_id": case["case_id"], "problem_id": case["problem_id"],
            "version": case.get("version"), "status": result.get("status", "pending"),
            "rows": flatten_rows(result.get("proof", [])), "metadata": result.get("metadata", {})})
    return {"quiz": manifest["quiz"], "cases": cases, "reviews": load_json(output / "reviews.json", {})}


def save_review(output: Path, review: dict) -> dict:
    output = external_root(output)
    manifest = load_json(output / "manifest.json")
    case_id, status = review.get("case_id"), review.get("status")
    if case_id not in {c["case_id"] for c in manifest["cases"]} or status not in {"exact", "corrected", "needs_correction", "unreadable", "bad_crop", "unreviewed"}:
        raise ValueError("Invalid review")
    result = load_json(output / "results" / f"{case_id}.json", {})
    if status in {"exact", "corrected", "needs_correction"} and result.get("status") != "ok":
        raise ValueError("No successful transcription to review")
    if status in {"exact", "corrected"}:
        if result.get("status") != "ok":
            raise ValueError("No successful transcription to review")
        proof = result["proof"] if status == "exact" else rows_to_structure(review.get("rows", []))
        same, _ = compare_proof_structures(result["proof"], proof)
        old_numbers = [r["line"] for r in flatten_rows(result["proof"])]
        new_numbers = [r["line"] for r in flatten_rows(proof)]
        saved = {"status": "exact" if same and old_numbers == new_numbers else "corrected", "proof": proof}
        write_json(output / "gold" / f"{case_id}.json", proof)
    else:
        allowed = {"formula", "justification", "structure", "line_numbers", "missing_or_extra_lines"}
        tags = review.get("error_categories", [])
        if not isinstance(tags, list) or any(tag not in allowed for tag in tags):
            raise ValueError("Invalid error categories")
        saved = {"status": status, "error_categories": sorted(set(tags))}
    reviews = load_json(output / "reviews.json", {})
    if status == "unreviewed":
        reviews.pop(case_id, None)
    else:
        reviews[case_id] = saved
    write_json(output / "reviews.json", reviews)
    return refresh_summary(output)


def refresh_summary(output: Path) -> dict:
    data = comparison(output)
    reviews = data["reviews"]
    summary = {"sample_size": len(data["cases"]), "completed": 0, "failed": 0, "pending": 0,
        "reviewed": len(reviews), "exact": 0, "corrected": 0, "needs_correction": 0, "unreadable": 0, "bad_crop": 0,
        "error_categories": {"formula": 0, "justification": 0, "structure": 0, "line_numbers": 0, "missing_or_extra_lines": 0}}
    for case in data["cases"]:
        summary[{"ok": "completed", "error": "failed", "pending": "pending"}[case["status"]]] += 1
        review = reviews.get(case["case_id"])
        if not review:
            continue
        summary[review["status"]] += 1
        if review["status"] == "needs_correction":
            for category in review.get("error_categories", []):
                summary["error_categories"][category] += 1
        if review["status"] != "corrected":
            continue
        actual = load_json(output / "results" / f"{case['case_id']}.json")["proof"]
        metrics = compare_proof_structures_granular(actual, review["proof"])
        categories = summary["error_categories"]
        categories["formula"] += metrics["formula_correct_line_count"] < metrics["matched_line_count"]
        categories["justification"] += metrics["justification_correct_line_count"] < metrics["matched_line_count"]
        categories["structure"] += metrics["structure_mismatch_count"] > 0
        categories["missing_or_extra_lines"] += bool(metrics["missing_line_count"] or metrics["extra_line_count"])
        categories["line_numbers"] += ([r["line"] for r in flatten_rows(actual)] != [r["line"] for r in flatten_rows(review["proof"])])
    scored = summary["exact"] + summary["corrected"] + summary["needs_correction"]
    summary["scored"] = scored
    summary["exact_rate_on_scored"] = summary["exact"] / scored if scored else None
    write_json(output / "shareable_summary.json", summary)
    return summary


def serve(output: Path, *, port: int = 5030) -> None:
    from http.server import BaseHTTPRequestHandler, HTTPServer
    from urllib.parse import urlsplit
    output = external_root(output)
    valid_cases = {c["case_id"] for c in load_json(output / "manifest.json")["cases"]}
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *unused):
            pass

        def send(self, status, body, content_type="application/json"):
            self.send_response(status)
            for name, value in {"Content-Type": content_type, "Cache-Control": "no-store",
                "X-Content-Type-Options": "nosniff", "Cross-Origin-Resource-Policy": "same-origin",
                "Referrer-Policy": "no-referrer", "X-Frame-Options": "DENY"}.items():
                self.send_header(name, value)
            self.end_headers()
            self.wfile.write(body)

        def allowed(self):
            return self.headers.get("Host") in {f"127.0.0.1:{port}", f"localhost:{port}"}

        def do_GET(self):
            if not self.allowed():
                return self.send(403, b'{}')
            path = urlsplit(self.path).path
            if path == "/":
                self.send(200, (REPO / "app/templates/private_fitch_pilot.html").read_bytes(), "text/html; charset=utf-8")
            elif path in {"/data", "/summary"}:
                value = comparison(output) if path == "/data" else refresh_summary(output)
                self.send(200, json.dumps(value).encode())
            else:
                match = re.fullmatch(r"/image/(case_\d{4})/(crop|page)", path)
                if match and match[1] in valid_cases:
                    folder, suffix = ("crops", ".png") if match[2] == "crop" else ("pages", ".jpg")
                    self.send(200, (output / folder / (match[1] + suffix)).read_bytes(),
                              "image/png" if suffix == ".png" else "image/jpeg")
                else:
                    self.send(404, b'{}')

        def do_POST(self):
            origin = self.headers.get("Origin")
            if not self.allowed() or self.path != "/review" or (origin and origin != "http://" + self.headers.get("Host", "")):
                return self.send(403, b'{}')
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 256000 or self.headers.get("Content-Type") != "application/json":
                    raise ValueError("Invalid request")
                review = json.loads(self.rfile.read(length))
                summary = save_review(output, review)
                self.send(200, json.dumps({"saved": True, "summary": summary}).encode())
            except (ValueError, TypeError, KeyError):
                self.send(400, b'{"error":"Invalid review; check line numbers and nesting depths."}')
    print(f"Private proof review: http://127.0.0.1:{port}/", flush=True)
    HTTPServer(("127.0.0.1", port), Handler).serve_forever()
