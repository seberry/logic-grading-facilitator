from __future__ import annotations

import argparse
import json
from pathlib import Path
from unittest.mock import patch

from PIL import Image
import pytest

from scripts import benchmark_name_recognition as benchmark


def source_data(root: Path) -> Path:
    data = root / "live"
    for folder in ["name_review_cache", "raw_scans", "grades"]:
        (data / folder).mkdir(parents=True)
    (data / "roster.csv").write_text("student_id,name\ns1,Example Ada\ns2,Example Ben\n", encoding="utf-8")
    scan = data / "raw_scans" / "original.png"
    Image.new("RGB", (40, 80), "white").save(scan)
    (data / "raw_scans" / "renamed.png").write_bytes(scan.read_bytes())
    benchmark.write_json(data / "name_review_cache" / "example.json", [{"filename": "original.png",
        "student_id": None, "raw_name": "unclear", "name_visible": True, "confidence": 0.1,
        "quiz_version": "A"}])
    benchmark.write_json(data / "grades" / "session.json", {"grading_key": {"quiz_id": "old_label"},
        "work_units": [{"student_id": "s1", "page_filenames": ["renamed.png"]}]})
    return data


def test_snapshot_matches_session_by_bytes_and_never_edits_live_files(tmp_path):
    data = source_data(tmp_path)
    before = {str(p.relative_to(data)): p.read_bytes() for p in data.rglob("*") if p.is_file()}
    output = tmp_path / "private-benchmark"
    inputs = benchmark.prepare(data, output)
    assert inputs["cases"][0]["saved_session_student_id"] == "s1"
    assert inputs["cases"][0]["session_reference_available"]
    copied = output / "scans" / inputs["cases"][0]["scan"]
    assert copied.read_bytes() == (data / "raw_scans" / "original.png").read_bytes()
    assert inputs["roster"][0]["name"] == "Example Ada"
    assert {str(p.relative_to(data)): p.read_bytes() for p in data.rglob("*") if p.is_file()} == before
    assert benchmark.prepare(data, output) == inputs


def test_missing_or_unsafe_scan_stops_before_output_is_created(tmp_path):
    data = source_data(tmp_path)
    benchmark.write_json(data / "name_review_cache" / "example.json", [{"filename": "../../escape.png"}])
    output = tmp_path / "private-benchmark"
    with pytest.raises(ValueError):
        benchmark.prepare(data, output)
    assert not output.exists()


def test_conflicting_session_labels_are_not_treated_as_reference(tmp_path):
    data = source_data(tmp_path)
    session_path = data / "grades" / "session.json"
    session = benchmark.read_json(session_path)
    session["work_units"].append({"student_id": "s2", "page_filenames": ["original.png"]})
    benchmark.write_json(session_path, session)
    inputs = benchmark.prepare(data, tmp_path / "private-benchmark")
    assert not inputs["cases"][0]["session_reference_available"]


def test_resume_does_not_repeat_inference_and_summary_contains_no_identity(tmp_path):
    data = source_data(tmp_path)
    output = tmp_path / "private-benchmark"
    args = argparse.Namespace(output_dir=output, data_dir=data, limit=None)
    from app.vlm_classifier import PagePrediction
    prediction = PagePrediction(filename="case_0001.png", student_id="s1", confidence=0.9,
                               name_visible=True, raw_name="Example Ada", reasoning="", quiz_version="A")
    with patch.object(benchmark, "api_key", return_value="fake-key"), patch.object(
        benchmark, "classify_page", return_value=prediction
    ) as infer:
        benchmark.run(args)
        benchmark.run(args)
    assert infer.call_count == 2
    summary = benchmark.read_json(output / "summary.json")
    for model in benchmark.MODELS:
        assert summary["models"][model]["completed"] == 1
        assert summary["provisional_vs_saved_session"][model]["agrees"] == 1
    text = json.dumps(summary)
    assert "Example Ada" not in text
    assert "original.png" not in text
    assert '"s1"' not in text
    assert "fake-key" not in (output / "inputs.json").read_text(encoding="utf-8")


def test_private_output_rejects_repository_paths():
    with pytest.raises(ValueError):
        benchmark.private_output(benchmark.REPO / "working" / "private-data")
