"""Local vision-model narrator for private TFL/Fitch OCR cases.

The narrator is a privacy-boundary tool: it sends private case images only to a
local LM Studio endpoint and writes anonymized text observations that can be
summarized back to coding agents.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from proofs_ocr.regression.private_tfl_runner import (
    load_private_manifest,
    private_case_image_path,
    private_manifest_cases,
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}

DEFAULT_SYSTEM_PROMPT = """You are a local-only OCR failure analyst for handwritten TFL/Fitch proof crops.

Privacy rules:
- Do not transcribe or repeat visible student names, student ids, file names, page ids, timestamps, or local paths.
- If such identifiers are visible, write only "PII visible redacted".
- It is OK to discuss proof geometry, scope bars, line counts, formula/justification legibility, and logic symbols.
- Keep the answer concise and diagnostic.
""".strip()

DEFAULT_QUESTION = """Analyze this private TFL/Fitch OCR artifact.

Focus on why an OCR pipeline might fail on this case. Comment on:
- whether the crop cuts off the proof, starts/ends mid-line, or includes partial neighboring lines;
- visible Fitch scope bars and whether they are complete, slanted, faint, fragmented, or confused with handwriting;
- line count and row spacing;
- whether formulas and justifications are visually separated or crowded;
- clipped text, shadows, ruled-paper artifacts, scratch-outs, or neighboring content;
- dappled/noisy dark regions where there are no real letters, especially in corners or shadowed areas;
- which next diagnostic artifact would be most useful to inspect.

Do not include names, ids, file paths, or timestamps.
""".strip()


@dataclass(frozen=True)
class CaseArtifact:
    artifact_id: str
    kind: str
    path: Path
    source: str

    def to_redacted_json(self) -> Dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "kind": self.kind,
            "source": self.source,
            "filename": self.path.name,
        }


def _safe_console_print(text: str) -> None:
    try:
        print(text)
    except UnicodeEncodeError:
        encoded = text.encode(
            getattr(sys.stdout, "encoding", None) or "utf-8",
            errors="replace",
        )
        sys.stdout.buffer.write(encoded + b"\n")


def _slug(value: Any, *, fallback: str = "item") -> str:
    text = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(value or "").strip())
    while "__" in text:
        text = text.replace("__", "_")
    return text.strip("_") or fallback


def _manifest_dir(manifest: Mapping[str, Any]) -> Path:
    path_text = str(manifest.get("_manifest_path") or "").strip()
    if not path_text:
        raise ValueError("private manifest was not loaded by load_private_manifest")
    return Path(path_text).resolve().parent


def _case_by_id(manifest: Mapping[str, Any], case_id: str) -> Dict[str, Any]:
    normalized = str(case_id or "").strip()
    for case in private_manifest_cases(manifest):
        if str(case.get("id") or "").strip() == normalized:
            return case
    raise KeyError(f"Unknown private case id: {normalized}")


def _case_run_dir(*, run_root: Path | None, case_id: str) -> Path | None:
    if run_root is None:
        return None
    root = Path(run_root).expanduser().resolve()
    candidates = [root / case_id, root / _slug(case_id, fallback="case")]
    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate
    return candidates[0]


def list_case_artifacts(
    *,
    manifest: Mapping[str, Any],
    case_id: str,
    run_root: Path | None = None,
) -> List[CaseArtifact]:
    manifest_dir = _manifest_dir(manifest)
    case = _case_by_id(manifest, case_id)
    artifacts: List[CaseArtifact] = []

    image_path = private_case_image_path(case, manifest_dir=manifest_dir)
    if image_path.exists():
        artifacts.append(
            CaseArtifact(
                artifact_id="proof_crop",
                kind="source_crop",
                path=image_path,
                source="manifest",
            )
        )

    run_dir = _case_run_dir(run_root=run_root, case_id=case_id)
    if run_dir is not None and run_dir.exists():
        for child in sorted(run_dir.iterdir(), key=lambda p: p.name.lower()):
            if not child.is_file() or child.suffix not in IMAGE_EXTENSIONS:
                continue
            kind = "ocr_artifact"
            if child.name == "debug_bars.jpg":
                kind = "bar_overlay"
            elif child.name.startswith("crop_"):
                kind = "pipeline_crop"
            artifact_id = _slug(child.stem, fallback="artifact")
            artifacts.append(
                CaseArtifact(
                    artifact_id=artifact_id,
                    kind=kind,
                    path=child,
                    source="run_artifact",
                )
            )

    deduped: Dict[str, CaseArtifact] = {}
    for artifact in artifacts:
        key = artifact.artifact_id
        if key in deduped:
            key = f"{key}_{len(deduped) + 1}"
            artifact = CaseArtifact(
                artifact_id=key,
                kind=artifact.kind,
                path=artifact.path,
                source=artifact.source,
            )
        deduped[key] = artifact
    return list(deduped.values())


def render_artifact_list(artifacts: Sequence[CaseArtifact]) -> str:
    if not artifacts:
        return "No image artifacts found for this case."
    lines = ["Artifacts:"]
    for artifact in artifacts:
        lines.append(
            f"- {artifact.artifact_id}: kind={artifact.kind}, source={artifact.source}, filename={artifact.path.name}"
        )
    return "\n".join(lines)


def _artifact_by_id(artifacts: Sequence[CaseArtifact], artifact_id: str) -> CaseArtifact:
    normalized = str(artifact_id or "").strip()
    for artifact in artifacts:
        if artifact.artifact_id == normalized:
            return artifact
    known = ", ".join(artifact.artifact_id for artifact in artifacts) or "(none)"
    raise KeyError(f"Unknown artifact id {normalized!r}. Known artifacts: {known}")


def _load_shareable_summary(path: Path | None) -> Dict[str, Any]:
    if path is None:
        return {}
    parsed = json.loads(Path(path).read_text(encoding="utf-8"))
    return parsed if isinstance(parsed, dict) else {}


def suggest_cases_from_summary(
    summary: Mapping[str, Any],
    *,
    category: str = "",
    split: str = "",
    limit: int = 8,
) -> List[Dict[str, Any]]:
    raw_cases = summary.get("cases") if isinstance(summary.get("cases"), list) else []
    selected: List[Dict[str, Any]] = []
    for item in raw_cases:
        if not isinstance(item, dict):
            continue
        if item.get("passed"):
            continue
        categories = [str(value) for value in (item.get("categories") or [])]
        if category and category not in categories:
            continue
        if split and str(item.get("split") or "") != split:
            continue
        metrics = item.get("metrics") if isinstance(item.get("metrics"), dict) else {}
        severity = (
            10 * int(metrics.get("structure_mismatch_count", 0))
            + 6 * int(metrics.get("missing_line_count", 0))
            + 6 * int(metrics.get("extra_line_count", 0))
            + int(metrics.get("matched_line_count", 0))
            - int(metrics.get("both_correct_line_count", 0))
        )
        selected.append(
            {
                "case_id": item.get("case_id"),
                "split": item.get("split"),
                "categories": categories,
                "severity": severity,
                "metrics": {
                    "matched": int(metrics.get("matched_line_count", 0)),
                    "formula": int(metrics.get("formula_correct_line_count", 0)),
                    "justification": int(metrics.get("justification_correct_line_count", 0)),
                    "structure_mismatches": int(metrics.get("structure_mismatch_count", 0)),
                    "missing": int(metrics.get("missing_line_count", 0)),
                    "extra": int(metrics.get("extra_line_count", 0)),
                },
            }
        )
    selected.sort(key=lambda item: (-int(item.get("severity", 0)), str(item.get("case_id") or "")))
    return selected[: max(1, int(limit))]


def render_case_suggestions(suggestions: Sequence[Mapping[str, Any]]) -> str:
    if not suggestions:
        return "No matching failed cases found."
    lines = ["Suggested private cases:"]
    for item in suggestions:
        metrics = item.get("metrics") if isinstance(item.get("metrics"), dict) else {}
        categories = ",".join(str(value) for value in (item.get("categories") or [])) or "none"
        lines.append(
            f"- {item.get('case_id')} split={item.get('split')} categories={categories} "
            f"severity={item.get('severity')} "
            f"formula={metrics.get('formula')}/{metrics.get('matched')} "
            f"justification={metrics.get('justification')}/{metrics.get('matched')} "
            f"structure_mismatches={metrics.get('structure_mismatches')} "
            f"missing={metrics.get('missing')} extra={metrics.get('extra')}"
        )
    return "\n".join(lines)


def _default_model() -> str | None:
    return (
        os.getenv("PRIVATE_TFL_NARRATOR_MODEL_ID", "").strip()
        or os.getenv("LMSTUDIO_VERIFIER_MODEL_ID", "").strip()
        or os.getenv("LMSTUDIO_MODEL_ID", "").strip()
        or None
    )


def ask_local_model_about_artifact(
    *,
    image_path: Path,
    question: str,
    base_url: str | None = None,
    model: str | None = None,
    timeout: int | None = None,
    max_tokens: int | None = 2048,
    max_image_dim: int | None = 1200,
    disable_thinking: bool = False,
) -> tuple[str, Dict[str, Any]]:
    from app.vlm_client import LMStudioQwenClient

    client = LMStudioQwenClient(
        base_url=base_url,
        model=model or _default_model(),
        timeout=timeout,
    )
    messages = client.build_messages_for_prompt(
        image_path=Path(image_path),
        prompt=question or DEFAULT_QUESTION,
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        max_image_dim=max_image_dim,
    )
    extra_payload = {"chat_template_kwargs": {"enable_thinking": False}} if disable_thinking else None
    response = client.chat(
        messages,
        timeout=timeout,
        max_tokens=max_tokens,
        extra_payload=extra_payload,
    )
    return response, dict(client.last_response_metadata)


def write_narration_record(
    *,
    output_dir: Path,
    case_id: str,
    artifact: CaseArtifact,
    question: str,
    response: str,
    metadata: Mapping[str, Any],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "case_id": case_id,
        "artifact": artifact.to_redacted_json(),
        "question": question,
        "response": response,
        "response_metadata": dict(metadata),
        "privacy": {
            "contains_paths": False,
            "contains_images": False,
            "model_endpoint": "local_lmstudio",
        },
    }
    target = output_dir / f"narration_{_slug(case_id)}_{artifact.artifact_id}.json"
    target.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    (target.with_suffix(".txt")).write_text(response, encoding="utf-8")
    return target


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Ask a local vision model about private TFL/Fitch OCR cases.")
    subparsers = parser.add_subparsers(dest="command")

    suggest_parser = subparsers.add_parser("suggest-cases", help="Suggest failed anonymous cases from a shareable summary.")
    suggest_parser.add_argument("--summary", required=True, help="Path to shareable_summary.json from a private run.")
    suggest_parser.add_argument("--category", default="", help="Failure category filter, e.g. structure or justification.")
    suggest_parser.add_argument("--split", default="", help="Split filter, e.g. dev_regression.")
    suggest_parser.add_argument("--limit", type=int, default=8)

    list_parser = subparsers.add_parser("list-artifacts", help="List image artifacts available for one private case.")
    list_parser.add_argument("--manifest", required=True)
    list_parser.add_argument("--case", required=True)
    list_parser.add_argument("--run-root", default="", help="Private run root containing per-case OCR artifacts.")

    ask_parser = subparsers.add_parser("ask", help="Ask local 27B/Gemma about one artifact.")
    ask_parser.add_argument("--manifest", required=True)
    ask_parser.add_argument("--case", required=True)
    ask_parser.add_argument("--run-root", default="", help="Private run root containing per-case OCR artifacts.")
    ask_parser.add_argument("--artifact", default="proof_crop", help="Artifact id from list-artifacts.")
    ask_parser.add_argument("--question", default=DEFAULT_QUESTION)
    ask_parser.add_argument("--output-dir", default="", help="Where to write narration JSON/TXT. Defaults under run-root or manifest dir.")
    ask_parser.add_argument("--lmstudio-base-url", default="", help="LM Studio OpenAI-compatible base URL.")
    ask_parser.add_argument("--model", default="", help="Local vision model id. Defaults to PRIVATE_TFL_NARRATOR_MODEL_ID/LMSTUDIO_* env.")
    ask_parser.add_argument("--timeout", type=int, default=300)
    ask_parser.add_argument("--max-tokens", type=int, default=2048)
    ask_parser.add_argument("--max-image-dim", type=int, default=1200)
    ask_parser.add_argument(
        "--disable-thinking",
        action="store_true",
        help="Send chat_template_kwargs.enable_thinking=false. Useful for Qwen; may be counterproductive for Gemma.",
    )

    args = parser.parse_args()

    if args.command == "suggest-cases":
        summary = _load_shareable_summary(Path(args.summary))
        suggestions = suggest_cases_from_summary(
            summary,
            category=args.category,
            split=args.split,
            limit=args.limit,
        )
        _safe_console_print(render_case_suggestions(suggestions))
        return 0

    if args.command in {"list-artifacts", "ask"}:
        manifest = load_private_manifest(Path(args.manifest))
        run_root = Path(args.run_root).expanduser().resolve() if str(args.run_root or "").strip() else None
        artifacts = list_case_artifacts(manifest=manifest, case_id=args.case, run_root=run_root)

        if args.command == "list-artifacts":
            _safe_console_print(render_artifact_list(artifacts))
            return 0

        artifact = _artifact_by_id(artifacts, args.artifact)
        response, metadata = ask_local_model_about_artifact(
            image_path=artifact.path,
            question=args.question,
            base_url=str(args.lmstudio_base_url or "").strip() or None,
            model=str(args.model or "").strip() or None,
            timeout=args.timeout,
            max_tokens=args.max_tokens,
            max_image_dim=args.max_image_dim,
            disable_thinking=bool(args.disable_thinking),
        )
        output_dir = (
            Path(args.output_dir).expanduser().resolve()
            if str(args.output_dir or "").strip()
            else (
                (_case_run_dir(run_root=run_root, case_id=args.case) or (_manifest_dir(manifest) / "narrations" / args.case))
                / "narrations"
            )
        )
        target = write_narration_record(
            output_dir=output_dir,
            case_id=args.case,
            artifact=artifact,
            question=args.question,
            response=response,
            metadata=metadata,
        )
        _safe_console_print(response)
        _safe_console_print(f"\nSaved narration for {args.case} artifact {artifact.artifact_id}.")
        return 0

    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
