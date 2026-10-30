"""Explicit, private REALLMS comparison; never updates the active grading data.

`run` snapshots the latest name cache, roster, and scan bytes outside the repo.
It reuses the production prompt, image encoder, parser, and roster validation.
`serve` provides a loopback-only review panel and saves independent human labels.
Only aggregate counts are printed. Model responses and names stay in the output.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
from dataclasses import asdict
import hashlib
from http.server import BaseHTTPRequestHandler
import io
import json
import os
from pathlib import Path
import shutil
import statistics
import sys
import time
from urllib.parse import urlsplit

from dotenv import dotenv_values
from PIL import Image, ImageOps
import requests

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from app.models import RosterEntry
from app.vlm_classifier import classify_page
from app.vlm_client import LMStudioQwenClient, NAME_SYSTEM_PROMPT, NAME_RESPONSE_FORMAT

BASE = "https://reallms.rescloud.iu.edu/direct/v1"
MODELS = ("qwen3.8-27b", "gemma-4-31B-it")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path, default=None):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def write_json(path: Path, data) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def private_output(path: Path) -> Path:
    path = path.resolve()
    if path == REPO or REPO in path.parents:
        raise ValueError("Private benchmark output must be outside the repository.")
    return path


def prepare(data_dir: Path, output: Path) -> dict:
    if (output / "inputs.json").exists():
        return read_json(output / "inputs.json")
    caches = list((data_dir / "name_review_cache").glob("*.json"))
    if not caches:
        raise ValueError("No saved name-recognition run exists.")
    cache = max(caches, key=lambda p: p.stat().st_mtime)
    baseline = read_json(cache)
    if not isinstance(baseline, list) or not baseline:
        raise ValueError("The saved name cache is empty or invalid.")
    with (data_dir / "roster.csv").open(encoding="utf-8-sig", newline="") as handle:
        roster = [RosterEntry(**row).model_dump() for row in csv.DictReader(handle)]
    raw_dir = (data_dir / "raw_scans").resolve()
    # A session can carry an outdated quiz label; match its page BYTES instead.
    session_path = data_dir / "grades" / "session.json"
    session = read_json(session_path, {})
    references = {}
    for unit in session.get("work_units", []):
        for filename in unit.get("page_filenames", []):
            path = (raw_dir / filename).resolve()
            if raw_dir not in path.parents or not path.is_file():
                continue
            references.setdefault(digest(path), set()).add(unit.get("student_id"))
    paths = []
    seen = set()
    for row in baseline:
        filename = row["filename"]
        path = (raw_dir / filename).resolve()
        if raw_dir not in path.parents or not path.is_file() or filename in seen:
            raise ValueError("Missing, duplicate, or unsafe cached scan; no requests sent.")
        seen.add(filename)
        paths.append(path)
    output.mkdir(parents=True, exist_ok=True)
    (output / "scans").mkdir(exist_ok=True)
    (output / "assets").mkdir(exist_ok=True)
    cases = []
    for index, (row, source) in enumerate(zip(baseline, paths), 1):
        case_id = f"case_{index:04d}"
        scan_name = case_id + source.suffix.lower()
        target = output / "scans" / scan_name
        shutil.copyfile(source, target)
        sha = digest(target)
        ref = references.get(sha, set())
        with Image.open(target) as image:
            preview = ImageOps.exif_transpose(image).convert("RGB")
            preview.thumbnail((1200, 1800))
            preview.save(output / "assets" / f"{case_id}.jpg", quality=88)
        cases.append({"case_id": case_id, "scan": scan_name, "sha256": sha,
                      "source_filename": row["filename"], "local": row,
                      "saved_session_student_id": next(iter(ref)) if len(ref) == 1 else None,
                      "session_reference_available": len(ref) == 1})
    inputs = {"models": list(MODELS), "roster": roster, "cases": cases,
              "source_cache": str(cache), "source_cache_sha256": digest(cache),
              "source_roster_sha256": digest(data_dir / "roster.csv"),
              "source_session_sha256": digest(session_path) if session_path.exists() else None,
              "prompt": NAME_SYSTEM_PROMPT, "response_format": NAME_RESPONSE_FORMAT,
              "baseline_model": "not recorded in original cache",
              "input_policy": "Same full-page scan encoder and complete roster as production; no image crops sent.",
              "max_tokens": 160}
    write_json(output / "inputs.json", inputs)
    return inputs


def api_key() -> str:
    key = os.environ.get("REALLMS_API_KEY", "").strip()
    if key:
        return key
    if os.name == "nt":
        import winreg
        for hive, location in ((winreg.HKEY_LOCAL_MACHINE,
                r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment"),
                (winreg.HKEY_CURRENT_USER, r"Environment")):
            try:
                with winreg.OpenKey(hive, location) as registry:
                    value, _ = winreg.QueryValueEx(registry, "REALLMS_API_KEY")
                if str(value).strip():
                    return str(value).strip()
            except OSError:
                continue
    raise ValueError("REALLMS_API_KEY is unavailable.")


class ReallmsClient(LMStudioQwenClient):
    def __init__(self, model: str, key: str):
        super().__init__(base_url=BASE, model=model, timeout=120)
        self.key = key
        self.responses = []

    def chat(self, messages, *, model=None, timeout=None, max_tokens=None, extra_payload=None):
        payload = {"model": model or self.model, "messages": messages, "temperature": 0.0}
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        payload.update(extra_payload or {})
        start = time.monotonic()
        response = requests.post(BASE + "/chat/completions", json=payload,
            headers={"Authorization": "Bearer " + self.key}, timeout=timeout or self.timeout,
            allow_redirects=False)
        elapsed = time.monotonic() - start
        if not 200 <= response.status_code < 300:
            # Keep any error body private, including provider echoes of prompts.
            self.responses.append({"status": response.status_code, "seconds": elapsed,
                                   "private_error_body": response.text.replace(self.key, "[REDACTED]")})
            error = requests.HTTPError("REALLMS request failed")
            error.response = response
            raise error
        data = response.json()
        choice = data["choices"][0]
        message = choice.get("message") or {}
        self.last_response_metadata = {"seconds": round(elapsed, 3),
            "returned_model": data.get("model"), "finish_reason": choice.get("finish_reason"),
            "usage": data.get("usage"), "thinking_chars": len(message.get("reasoning_content") or "")}
        self.responses.append({**self.last_response_metadata, "content": message.get("content")})
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Empty model completion")
        return content


def build_comparison(output: Path, inputs: dict, results: list) -> dict:
    by_case = {(r["case_id"], r["model"]): r for r in results}
    cases = [{**{k: c[k] for k in ("case_id", "local", "saved_session_student_id", "session_reference_available")},
              "predictions": {model: by_case.get((c["case_id"], model)) for model in MODELS}}
             for c in inputs["cases"]]
    comparison = {"roster": inputs["roster"], "models": ["local", *MODELS], "cases": cases}
    write_json(output / "comparison.json", comparison)
    summary = {"pages": len(cases), "local_matched": sum(bool(c["local"].get("student_id")) for c in cases),
               "saved_session_reference_pages": sum(c["session_reference_available"] for c in cases),
               "models": {}}
    for model in MODELS:
        rows = [r for r in results if r["model"] == model]
        ok = [r for r in rows if r["status"] == "ok"]
        summary["models"][model] = {"completed": len(ok), "failed": len(rows)-len(ok),
            "matched": sum(bool(r["prediction"].get("student_id")) for r in ok),
            "seconds": round(sum(r["elapsed_seconds"] for r in rows), 2),
            "median_seconds": round(statistics.median(r["elapsed_seconds"] for r in ok), 2) if ok else None,
            "http_requests": sum(len(r.get("responses", [])) for r in rows)}
    # Session comparisons are provisional; explicit panel reviews are the benchmark labels.
    reference_cases = [c for c in cases if c["session_reference_available"]]
    provisional = {}
    for model in ["local", *MODELS]:
        checked = []
        for case in reference_cases:
            result = case["predictions"].get(model) if model != "local" else None
            if model != "local" and (not result or result["status"] != "ok"):
                continue
            prediction = case["local"] if model == "local" else result["prediction"]
            checked.append((prediction.get("student_id"), case["saved_session_student_id"]))
        provisional[model] = {"compared": len(checked), "agrees": sum(a == b for a, b in checked),
                              "unresolved": sum(a is None and b is not None for a, b in checked),
                              "different_match": sum(a is not None and a != b for a, b in checked)}
    summary["provisional_vs_saved_session"] = provisional
    write_json(output / "summary.json", summary)
    return summary


def run(args) -> None:
    if args.limit is not None and args.limit < 1:
        raise ValueError("The page limit must be positive.")
    output = private_output(args.output_dir)
    config = dotenv_values(REPO / ".env")
    inputs = prepare(Path(args.data_dir or config["DATA_DIR"]), output)
    if inputs["prompt"] != NAME_SYSTEM_PROMPT or inputs["response_format"] != NAME_RESPONSE_FORMAT:
        raise ValueError("Production prompt changed since this benchmark was prepared.")
    results_path = output / "results.json"
    results = read_json(results_path, [])
    attempted = {(r["case_id"], r["model"]) for r in results}
    roster = [RosterEntry(**r) for r in inputs["roster"]]
    key = api_key()
    limit = args.limit or len(inputs["cases"])
    clients = {model: ReallmsClient(model, key) for model in MODELS}
    consecutive_errors = 0
    for case in inputs["cases"][:limit]:
        image = output / "scans" / case["scan"]
        if digest(image) != case["sha256"]:
            raise ValueError("A benchmark scan changed; refusing to continue.")
        for model, client in clients.items():
            if (case["case_id"], model) in attempted:
                continue
            client.responses = []
            start = time.monotonic()
            record = {"case_id": case["case_id"], "model": model}
            try:
                # The shared encoder can print sensitive filenames; suppress its diagnostics.
                with contextlib.redirect_stdout(io.StringIO()):
                    prediction = classify_page(image, roster, client)
                record.update(status="ok", prediction=asdict(prediction))
                consecutive_errors = 0
            except Exception as error:
                record.update(status="error", error_type=type(error).__name__)
                consecutive_errors += 1
            record.update(elapsed_seconds=round(time.monotonic()-start, 3), responses=client.responses)
            results.append(record)
            write_json(results_path, results)
            summary = build_comparison(output, inputs, results)
            print(f"{case['case_id']} {model}: {record['status']}", flush=True)
            if consecutive_errors >= 3:
                print(json.dumps(summary), flush=True)
                raise ValueError("Three consecutive failures; stopped without retrying requests.")
    summary = build_comparison(output, inputs, results)
    print(json.dumps(summary), flush=True)


def serve(args) -> None:
    output = private_output(args.output_dir)
    comparison = read_json(output / "comparison.json")
    valid_cases = {c["case_id"] for c in comparison["cases"]}
    valid_students = {r["student_id"] for r in comparison["roster"]} | {None}
    # A single-thread server serializes review updates; avoid lost simultaneous writes.
    from http.server import HTTPServer
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *unused):
            pass

        def send(self, status, body, content_type="application/json"):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Cross-Origin-Resource-Policy", "same-origin")
            self.send_header("Referrer-Policy", "no-referrer")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.headers.get("Host") not in {f"127.0.0.1:{args.port}", f"localhost:{args.port}"}:
                self.send(403, b'{}')
                return
            path = urlsplit(self.path).path
            if path == "/":
                self.send(200, (REPO / "app/templates/name_benchmark.html").read_bytes(), "text/html; charset=utf-8")
            elif path in {"/comparison.json", "/reviews.json"}:
                self.send(200, json.dumps(read_json(output / path[1:], {})).encode("utf-8"))
            elif path.startswith("/assets/") and path.removeprefix("/assets/").removesuffix(".jpg") in valid_cases and path.endswith(".jpg"):
                self.send(200, (output / path[1:]).read_bytes(), "image/jpeg")
            else:
                self.send(404, b'{}')

        def do_POST(self):
            origin = self.headers.get("Origin")
            if (self.headers.get("Host") not in {f"127.0.0.1:{args.port}", f"localhost:{args.port}"}
                    or self.path != "/reviews" or (origin and origin != "http://" + self.headers.get("Host", ""))):
                self.send(403, b'{}')
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 16384 or self.headers.get("Content-Type") != "application/json":
                    raise ValueError()
                review = json.loads(self.rfile.read(size))
                case_id, student_id = review["case_id"], review["student_id"]
                if case_id not in valid_cases or student_id not in valid_students:
                    raise ValueError()
                reviews = read_json(output / "reviews.json", {})
                if review.get("reviewed") is False:
                    reviews.pop(case_id, None)
                else:
                    reviews[case_id] = {"student_id": student_id}
                write_json(output / "reviews.json", reviews)
                self.send(200, b'{"saved":true}')
            except (ValueError, KeyError, TypeError):
                self.send(400, b'{}')
    print(f"Private review panel: http://127.0.0.1:{args.port}/", flush=True)
    HTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["run", "serve"])
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--port", type=int, default=5020)
    args = parser.parse_args()
    try:
        (run if args.action == "run" else serve)(args)
    except KeyboardInterrupt:
        pass
    except Exception as error:
        # Exception strings can contain scan paths, roster text, or provider echoes.
        print(f"Benchmark stopped ({type(error).__name__}); private artifacts retained.", flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
