from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from unittest.mock import patch
import zipfile

from PIL import Image
import pytest

from proofs_ocr.pipelines.tfl_whole_page import flatten_rows, rows_to_structure, transcribe_whole_proof
from proofs_ocr.regression import private_fitch_pilot as pilot


PROOF = [
    {"line": 1, "wffstr": "P∨Q", "jstr": "Pr"},
    [{"line": 2, "wffstr": "P", "jstr": "Hyp"}, {"line": 3, "wffstr": "R", "jstr": "↔E 1,3"}],
    [{"line": 4, "wffstr": "Q", "jstr": "Hyp"}, {"line": 5, "wffstr": "R", "jstr": "↔E 1,5"}],
    {"line": 6, "wffstr": "R", "jstr": "∨E 1,2-3,4-5"},
]


def fake_bundle(root: Path) -> Path:
    path = root / "Quiz_7_tf_grading_bundle.zip"
    image = io.BytesIO()
    Image.new("RGB", (200, 300), "white").save(image, format="PNG")
    session = {"roster": [{"student_id": "private-id", "name": "Private Example"}],
        "grading_key": {"quiz_id": "Quiz 7", "problems": [
            {"id": f"Q{i}", "label": "Fitch proof", "answer_key": "NEVER SEND ANSWER KEY"} for i in range(1, 4)]},
        "work_units": [{"unit_id": i, "student_id": f"private-{i}", "quiz_version": "A",
            "page_filenames": [f"private_name_{i}.png"], "per_problem": {"Q1": 3}} for i in range(1, 5)]}
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("bundle/data/grades/session.json", json.dumps(session))
        for unit in session["work_units"]:
            archive.writestr("bundle/data/raw_scans/" + unit["page_filenames"][0], image.getvalue())
        for i in range(1, 4):
            locations = [{"unit_id": u["unit_id"], "page_filename": u["page_filenames"][0],
                          "bbox": [.1, .2, .9, .8], "present": True} for u in session["work_units"]]
            archive.writestr(f"bundle/data/grades/segments_Q{i}.json", json.dumps(locations))
    return path


def prepare_fake(tmp_path):
    bundle = fake_bundle(tmp_path)
    output = tmp_path / "pilot"
    info = pilot.bundle_info(bundle)
    info["verification"] = {"verified_proof_ids": info["proof_ids"], "checks": [],
                            "source_sha256": hashlib.sha256(bundle.read_bytes()).hexdigest()}
    manifest = pilot.prepare(info, output, samples=3, seed=42)
    return bundle, output, manifest


def run_fake(output):
    with patch.object(pilot, "transcribe_whole_proof", return_value={"proof": PROOF,
        "raw_response": json.dumps({"proof": PROOF}), "metadata": {"model": "fake-model", "seconds": 1}}) as infer:
        summary = pilot.run(output)
    return summary, infer


def test_nested_and_sibling_subproofs_survive_editor_round_trip():
    rows = flatten_rows(PROOF)
    assert rows[1]["start_subproof"] and rows[3]["start_subproof"]
    assert rows_to_structure(rows) == PROOF
    nested = [PROOF[0], [PROOF[0], [PROOF[0]]], PROOF[0]]
    assert rows_to_structure(flatten_rows(nested)) == nested


def test_shape_validation_does_not_repair_invalid_citations():
    class FakeClient:
        model = "fake"
        last_response_metadata = {}
        def build_messages_for_prompt(self, image_path, prompt):
            assert "NEVER SEND ANSWER KEY" not in prompt
            assert "Private Example" not in prompt
            return [{"role": "user", "content": prompt}]
        def chat(self, messages, **kwargs):
            assert kwargs["extra_payload"]["response_format"]["type"] == "json_schema"
            return json.dumps({"proof": PROOF})
    assert transcribe_whole_proof(Path("fake.png"), client=FakeClient())["proof"] == PROOF
    bad = [{"line": True, "depth": 0, "wffstr": "P", "jstr": "Pr"}]
    with pytest.raises(ValueError):
        rows_to_structure(bad)
    with pytest.raises(ValueError):
        rows_to_structure([{**bad[0], "line": 1, "depth": 2}])


def test_bundle_sampling_is_balanced_reproducible_and_read_only(tmp_path):
    bundle = fake_bundle(tmp_path)
    before = hashlib.sha256(bundle.read_bytes()).hexdigest()
    info = pilot.bundle_info(bundle)
    a = pilot.select_cases(info, samples=3, seed=42)
    assert a == pilot.select_cases(info, samples=3, seed=42)
    assert len({c["unit_id"] for c in a}) == 3
    assert len({c["problem_id"] for c in a}) == 3
    info["verification"] = {"verified_proof_ids": info["proof_ids"], "checks": [], "source_sha256": before}
    manifest = pilot.prepare(info, tmp_path / "pilot", samples=3)
    assert len(manifest["cases"]) == 3
    assert hashlib.sha256(bundle.read_bytes()).hexdigest() == before


def test_private_output_and_existing_pilots_are_protected(tmp_path):
    with pytest.raises(ValueError):
        pilot.external_root(pilot.REPO / "working" / "real-students")
    bundle, output, _ = prepare_fake(tmp_path)
    with pytest.raises(ValueError):
        pilot.prepare(pilot.bundle_info(bundle), output)


def test_only_images_go_to_inference_and_resume_skips_existing_results(tmp_path):
    _, output, _ = prepare_fake(tmp_path)
    summary, infer = run_fake(output)
    assert summary["completed"] == 3 and summary["reviewed"] == 0
    assert infer.call_count == 3
    for call in infer.call_args_list:
        assert call.kwargs == {"model": "gemma"}
        assert call.args[0].parent == output / "crops"
        assert call.args[0].name.startswith("case_")
    _, again = run_fake(output)
    again.assert_not_called()
    data = json.dumps(pilot.comparison(output))
    for value in ("Private Example", "private_name", "private-id", "NEVER SEND ANSWER KEY", str(output)):
        assert value not in data


def test_reviews_produce_anonymous_counts_without_changing_predictions(tmp_path):
    _, output, _ = prepare_fake(tmp_path)
    run_fake(output)
    before = (output / "results/case_0002.json").read_bytes()
    pilot.save_review(output, {"case_id": "case_0001", "status": "exact"})
    rows = flatten_rows(PROOF)
    rows[2]["jstr"] = "↔E 1,2"
    pilot.save_review(output, {"case_id": "case_0002", "status": "corrected", "rows": rows})
    summary = pilot.save_review(output, {"case_id": "case_0003", "status": "needs_correction", "error_categories": ["structure"]})
    assert summary["exact"] == 1 and summary["corrected"] == 1 and summary["needs_correction"] == 1
    assert summary["error_categories"]["justification"] == 1
    assert summary["error_categories"]["structure"] == 1
    assert summary["scored"] == 3 and summary["exact_rate_on_scored"] == 1/3
    assert (output / "results/case_0002.json").read_bytes() == before
    anonymous = json.dumps(summary)
    for value in ("case_", "Private Example", "private_name", "↔E", "private-id", str(output)):
        assert value not in anonymous
    summary = pilot.save_review(output, {"case_id": "case_0003", "status": "bad_crop"})
    assert summary["scored"] == 2 and summary["bad_crop"] == 1
    pilot.save_review(output, {"case_id": "case_0002", "status": "unreviewed"})
    assert pilot.refresh_summary(output)["reviewed"] == 2


def test_invalid_review_does_not_create_gold(tmp_path):
    _, output, _ = prepare_fake(tmp_path)
    run_fake(output)
    with pytest.raises(ValueError):
        pilot.save_review(output, {"case_id": "../../escape", "status": "exact"})
    assert not list((output / "gold").glob("*.json"))


def test_stale_proof_rubric_does_not_authorize_nonproof_images(tmp_path):
    bundle = fake_bundle(tmp_path)
    info = pilot.bundle_info(bundle)
    assert len(info["proof_ids"]) == 3
    classification = {"kind": "symbolization", "has_proof_bars": False,
                      "has_rule_citations": False, "has_quantifiers": False}
    with patch.object(pilot, "classify_proof_crop", return_value=classification) as classify:
        evidence = pilot.verify_bundle(info, tmp_path / "checks")
    assert classify.call_count == 3
    assert evidence["verified_proof_ids"] == []
    info["verification"] = evidence
    output = tmp_path / "pilot"
    with pytest.raises(ValueError, match="Verify actual"):
        pilot.prepare(info, output)
    assert not output.exists()


def test_preparation_requires_verification_of_this_exact_bundle(tmp_path):
    info = pilot.bundle_info(fake_bundle(tmp_path))
    with pytest.raises(ValueError, match="Verify actual"):
        pilot.prepare(info, tmp_path / "pilot")
    info["verification"] = {"verified_proof_ids": ["Q1"], "source_sha256": "wrong"}
    with pytest.raises(ValueError, match="changed"):
        pilot.prepare(info, tmp_path / "pilot")


def test_layout_classifier_emits_only_categorical_fields():
    class FakeClient:
        def build_messages_for_prompt(self, path, prompt):
            assert "Do not transcribe" in prompt
            return []
        def chat(self, *args, **kwargs):
            return json.dumps({"kind": "fitch_proof", "has_proof_bars": True,
                "has_rule_citations": True, "has_quantifiers": False, "unwanted_text": "Never display this"})
    result = pilot.classify_proof_crop(Path("fake.png"), client=FakeClient())
    assert set(result) == {"kind", "has_proof_bars", "has_rule_citations", "has_quantifiers"}
    assert "Never display" not in json.dumps(result)


def test_bundle_filename_takes_precedence_over_stale_session_quiz(tmp_path):
    old = fake_bundle(tmp_path)
    newer = tmp_path / "Quiz_9_tf_grading_bundle.zip"
    newer.write_bytes(old.read_bytes())
    info = pilot.bundle_info(newer)
    assert info["quiz"] == 9
    assert info["metadata_label_conflict"] is True
