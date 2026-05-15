from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from proofs_ocr.regression import private_tfl_runner


def _write_manifest(root: Path) -> Path:
    manifest = root / "private_manifest.yaml"
    manifest.write_text(
        """
suite:
  name: private_fake_suite
cases:
  - id: private_q6_case_0001
    split: dev_regression
    image_path: crops/q6/private_q6_case_0001.jpg
    expected_structure_path: gold/q6/private_q6_case_0001.json
    gold_status: human_checked
  - id: private_q6_case_0002
    split: blind_holdout
    image_path: crops/q6/private_q6_case_0002.jpg
    expected_structure_path: gold/q6/private_q6_case_0002.json
    gold_status: human_checked
""".strip(),
        encoding="utf-8",
    )
    return manifest


def _write_case_files(root: Path) -> None:
    (root / "crops" / "q6").mkdir(parents=True)
    (root / "gold" / "q6").mkdir(parents=True)
    for case_id in ("private_q6_case_0001", "private_q6_case_0002"):
        (root / "crops" / "q6" / f"{case_id}.jpg").write_bytes(b"fake image")
    (root / "gold" / "q6" / "private_q6_case_0001.json").write_text(
        json.dumps(
            {
                "lines": [
                    {"wffstr": "PRIVATE_NAME_SHOULD_NOT_LEAK", "jstr": "Pr"},
                    {"wffstr": "B", "jstr": "Pr"},
                ],
                "meta": {"num_premises": 2},
            }
        ),
        encoding="utf-8",
    )
    (root / "gold" / "q6" / "private_q6_case_0002.json").write_text(
        json.dumps(
            [
                {"wffstr": "A", "jstr": "Pr"},
                {"wffstr": "C", "jstr": "Pr"},
            ]
        ),
        encoding="utf-8",
    )


def test_private_manifest_summary_is_redacted(tmp_path: Path) -> None:
    _write_case_files(tmp_path)
    manifest_path = _write_manifest(tmp_path)

    manifest = private_tfl_runner.load_private_manifest(manifest_path)
    summary = private_tfl_runner.render_private_manifest_summary(manifest)

    assert "private_q6_case_0001" in summary
    assert "dev_regression" in summary
    assert str(tmp_path) not in summary
    assert "PRIVATE_NAME_SHOULD_NOT_LEAK" not in summary


def test_run_private_gold_regressions_writes_shareable_summary(tmp_path: Path) -> None:
    _write_case_files(tmp_path)
    manifest_path = _write_manifest(tmp_path)
    manifest = private_tfl_runner.load_private_manifest(manifest_path)
    output_root = tmp_path / "runs"

    def fake_transcribe(*, image_path: Path, output_dir: Path) -> Mapping[str, Any]:
        if image_path.name == "private_q6_case_0001.jpg":
            structure = [
                {"wffstr": "PRIVATE_NAME_SHOULD_NOT_LEAK", "jstr": "Pr"},
                {"wffstr": "B", "jstr": "Pr"},
            ]
        else:
            structure = [
                {"wffstr": "A", "jstr": "Pr"},
                {"wffstr": "D", "jstr": "Pr"},
            ]
        (output_dir / "final_structure.json").write_text(
            json.dumps(structure),
            encoding="utf-8",
        )
        return {"run_summary": "Fitch OCR pipeline: fake test run"}

    results = private_tfl_runner.run_private_tfl_gold_regressions(
        manifest,
        output_root=output_root,
        transcribe_fn=fake_transcribe,
        allow_repo_paths=True,
    )

    assert [result.passed for result in results] == [True, False]

    shareable_text = (output_root / "shareable_summary.txt").read_text(encoding="utf-8")
    shareable_json = json.loads((output_root / "shareable_summary.json").read_text(encoding="utf-8"))

    assert "Cases: 2" in shareable_text
    assert "Passed: 1" in shareable_text
    assert "Failed: 1" in shareable_text
    assert "formula: 1" in shareable_text
    assert str(tmp_path) not in shareable_text
    assert "PRIVATE_NAME_SHOULD_NOT_LEAK" not in shareable_text
    assert "D" not in shareable_text
    assert shareable_json["privacy"] == {
        "contains_paths": False,
        "contains_images": False,
        "contains_line_text": False,
        "contains_diffs": False,
    }

    local_diff = output_root / "private_q6_case_0002" / "private_gold_diff.local.txt"
    assert local_diff.exists()
    assert "D" in local_diff.read_text(encoding="utf-8")


def test_private_runner_can_filter_by_split(tmp_path: Path) -> None:
    _write_case_files(tmp_path)
    manifest_path = _write_manifest(tmp_path)
    manifest = private_tfl_runner.load_private_manifest(manifest_path)

    calls: list[str] = []

    def fake_transcribe(*, image_path: Path, output_dir: Path) -> Mapping[str, Any]:
        calls.append(image_path.name)
        (output_dir / "final_structure.json").write_text(
            json.dumps(
                [
                    {"wffstr": "A", "jstr": "Pr"},
                    {"wffstr": "C", "jstr": "Pr"},
                ]
            ),
            encoding="utf-8",
        )
        return {}

    results = private_tfl_runner.run_private_tfl_gold_regressions(
        manifest,
        output_root=tmp_path / "runs",
        splits=["blind_holdout"],
        transcribe_fn=fake_transcribe,
        allow_repo_paths=True,
    )

    assert [result.case_id for result in results] == ["private_q6_case_0002"]
    assert calls == ["private_q6_case_0002.jpg"]
