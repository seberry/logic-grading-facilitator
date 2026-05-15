from __future__ import annotations

import json
from pathlib import Path

from proofs_ocr.regression import private_tfl_narrator


def _write_manifest(root: Path) -> Path:
    (root / "crops" / "q6").mkdir(parents=True)
    (root / "crops" / "q6" / "exam1_q6_case_0001.jpg").write_bytes(b"fake image")
    manifest = root / "manifest.yaml"
    manifest.write_text(
        """
suite:
  name: private_fake_suite
cases:
  - id: exam1_q6_case_0001
    split: dev_regression
    image_path: crops/q6/exam1_q6_case_0001.jpg
    expected_structure_path: gold/q6/exam1_q6_case_0001.json
""".strip(),
        encoding="utf-8",
    )
    return manifest


def test_list_case_artifacts_includes_manifest_crop_and_run_artifacts(tmp_path: Path) -> None:
    manifest_path = _write_manifest(tmp_path)
    run_case_dir = tmp_path / "runs" / "exam1_q6_case_0001"
    run_case_dir.mkdir(parents=True)
    (run_case_dir / "debug_bars.jpg").write_bytes(b"bars")
    (run_case_dir / "crop_ROOT.jpg").write_bytes(b"root")
    (run_case_dir / "final_structure.json").write_text("[]", encoding="utf-8")

    manifest = private_tfl_narrator.load_private_manifest(manifest_path)
    artifacts = private_tfl_narrator.list_case_artifacts(
        manifest=manifest,
        case_id="exam1_q6_case_0001",
        run_root=tmp_path / "runs",
    )

    assert [(item.artifact_id, item.kind, item.source) for item in artifacts] == [
        ("proof_crop", "source_crop", "manifest"),
        ("crop_root", "pipeline_crop", "run_artifact"),
        ("debug_bars", "bar_overlay", "run_artifact"),
    ]
    rendered = private_tfl_narrator.render_artifact_list(artifacts)
    assert "proof_crop" in rendered
    assert str(tmp_path) not in rendered


def test_suggest_cases_from_summary_ranks_failures_without_text(tmp_path: Path) -> None:
    summary = {
        "cases": [
            {
                "case_id": "case_a",
                "split": "dev_regression",
                "passed": False,
                "categories": ["justification"],
                "metrics": {
                    "matched_line_count": 5,
                    "both_correct_line_count": 4,
                    "justification_correct_line_count": 4,
                },
            },
            {
                "case_id": "case_b",
                "split": "dev_regression",
                "passed": False,
                "categories": ["structure"],
                "metrics": {
                    "matched_line_count": 6,
                    "both_correct_line_count": 2,
                    "structure_mismatch_count": 2,
                },
            },
            {
                "case_id": "case_c",
                "split": "blind_holdout",
                "passed": False,
                "categories": ["structure"],
                "metrics": {
                    "matched_line_count": 6,
                    "both_correct_line_count": 1,
                    "structure_mismatch_count": 4,
                },
            },
        ]
    }

    suggestions = private_tfl_narrator.suggest_cases_from_summary(
        summary,
        category="structure",
        split="dev_regression",
    )

    assert [item["case_id"] for item in suggestions] == ["case_b"]
    rendered = private_tfl_narrator.render_case_suggestions(suggestions)
    assert "case_b" in rendered
    assert "structure_mismatches=2" in rendered
    assert str(tmp_path) not in rendered


def test_write_narration_record_is_redacted(tmp_path: Path) -> None:
    image_path = tmp_path / "exam1_q6_case_0001.jpg"
    image_path.write_bytes(b"fake")
    artifact = private_tfl_narrator.CaseArtifact(
        artifact_id="proof_crop",
        kind="source_crop",
        path=image_path,
        source="manifest",
    )

    target = private_tfl_narrator.write_narration_record(
        output_dir=tmp_path / "narrations",
        case_id="exam1_q6_case_0001",
        artifact=artifact,
        question="Why did OCR fail?",
        response="The left scope bar is faint. PII visible redacted.",
        metadata={"model": "local-model", "elapsed_s": 1.0},
    )

    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["case_id"] == "exam1_q6_case_0001"
    assert payload["artifact"] == {
        "artifact_id": "proof_crop",
        "kind": "source_crop",
        "source": "manifest",
        "filename": "exam1_q6_case_0001.jpg",
    }
    assert str(tmp_path) not in target.read_text(encoding="utf-8")
    assert payload["privacy"]["contains_paths"] is False
