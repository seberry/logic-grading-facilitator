"""Synthetic, offline coverage for rubric-driven FOL truth assistance."""
from copy import deepcopy
import hashlib
import html
import io
import json
from pathlib import Path
import re
from unittest.mock import patch
import zipfile

from fastapi.testclient import TestClient
from PIL import Image
import pytest

from app import server, ta_releases
from app.fol_truth_assistance import (assistance_schema, enabled, question_context,
                                      question_enabled, prepare_vision_reading)
from app.fol_truth_config import normalize_formula, object_token
from app.models import GradingKey, ProblemSpec, RosterEntry, Session, WorkUnit
from app.routes import exam_exports


CASES = json.loads((Path(__file__).parent / "fixtures/fol_truth_cases.json").read_text(encoding="utf-8"))


def config(case=0):
    case = CASES[case]
    return {"model": deepcopy(case["model"]), "formula": case["formula"],
            "asks_object": case.get("asks_object", True),
            "scoring": {"mode": "all_or_nothing", "points": 1}}


def key_with(configuration=None, **kwargs):
    return GradingKey(quiz_id="synthetic_mixed", problems=[
        ProblemSpec(id="Q1", label="Manual", points=2),
        ProblemSpec(id="EC", label="Model truth", points=1, extra_credit=True,
                    fol_truth=configuration if configuration is not None else config(), **kwargs)])


def session_with(key):
    return Session(roster=[RosterEntry(student_id="fake", name="Synthetic Student")],
                   grading_key=key, work_units=[WorkUnit(unit_id=1, student_id="fake",
                       page_filenames=["fake.jpg"], per_problem={p.id: None for p in key.problems},
                       comment=None)], unclaimed_pages=[])


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["name"])
def test_rubric_facts_check_with_python_and_round_trip(case):
    key = key_with({"model": case["model"], "formula": case["formula"],
                    "asks_object": case.get("asks_object", True)})
    restored = GradingKey.model_validate_json(key.model_dump_json())
    context = question_context(restored, "EC", None, None)
    assert context["truth"] is case["truth"]
    assert [c["value"] for c in context["object_choices"]] == [object_token(x) for x in case["model"]["domain"]]
    assert context["score_rule"] is None  # feedback does not require score automation
    assert enabled(restored) and assistance_schema(restored) == 2
    assert question_enabled(restored, "EC")
    assert not question_enabled(restored, "Q1")
    assert question_context(restored, "Q1", None, None) is None


def test_version_overrides_and_new_models_change_truth_and_fingerprint(tmp_path):
    key = key_with(variants={"A": {"prompt": "first"}, "B": {
        "prompt": "second", "fol_truth": {"model": CASES[1]["model"]}}})
    a = question_context(key, "EC", "A", {})
    b = question_context(key, "EC", "B", {})
    assert a["truth"] is True and b["truth"] is False
    assert a["fingerprint"] != b["fingerprint"]
    key.problems[1].fol_truth["model"]["predicates"]["R"] = []
    changed = question_context(key, "EC", "A", {})
    assert changed["truth"] is False and changed["fingerprint"] != a["fingerprint"]
    source = tmp_path / "synthetic.jpg"
    source.write_bytes(b"first revision")
    first = question_context(key, "EC", "A", {}, source)
    source.write_bytes(b"second revision")
    assert question_context(key, "EC", "A", {}, source)["fingerprint"] != first["fingerprint"]
    assert question_context(key, "EC", "A", {"bbox": [.1, .2, .3, .4]}, source)["fingerprint"] != first["fingerprint"]


@pytest.mark.parametrize("change", [
    {"formula": "∃x Undefined(x)"}, {"formula": "Px"}, {"formula": "∃x R(x)"},
    {"formula": "∃x Rxx\n∃x Rxx"}, {"formula": "Pa"}, {"formula": "∃u P(u)"},
    {"asks_object": "yes"}, {"sentence": "∃x Rxx"},
    {"scoring": {"truth": 1, "object": 1}},
    {"scoring": {"mode": "all_or_nothing", "points": True}},
    {"scoring": {"mode": [], "points": 1}},
    {"scoring": {"mode": "all_or_nothing", "points": 1, "blank_na_credit": "yes"}},
    {"model": {"domain": [2, 4], "predicates": {"R": [[2, 99]]}}},
    {"model": {"domain": [2, 4], "predicates": {"R": [[2, 4], [2]]}}},
    {"model": {"domain": [2, 2], "predicates": {"R": []}}},
    {"model": {"domain": [True], "predicates": {"R": []}}},
    {"model": {"domain": [2], "constants": {"a": False}, "predicates": {"R": []}}},
])
def test_invalid_assistance_rejected_when_loading_rubric(change):
    with pytest.raises(ValueError):
        key_with({**config(), **change})


def test_invalid_variant_rejected_even_when_another_variant_is_valid():
    with pytest.raises(ValueError, match="outside the domain"):
        key_with(variants={"A": {}, "B": {"fol_truth": {"model": {
            "domain": [2], "constants": {"a": 4}, "predicates": {"R": []}}}}})


def test_empty_configuration_is_rejected():
    with pytest.raises(ValueError, match="FOL model"):
        key_with({})


def test_assistance_on_only_one_version_leaves_other_versions_manual():
    key = GradingKey(quiz_id="synthetic_versions", problems=[ProblemSpec(
        id="EC", label="Truth", points=1, variants={
            "A": {"prompt": "Assisted", "fol_truth": config()}, "B": {"prompt": "Manual"}})])
    assert question_context(key, "EC", "A", None)["truth"] is True
    assert question_context(key, "EC", "B", None) is None
    session = session_with(key)
    session.work_units[0].quiz_version = "A"
    session.work_units.append(WorkUnit(unit_id=2, student_id=None, page_filenames=["fake2.jpg"],
                                      quiz_version="B", per_problem={"EC": None}, comment=None))
    with (patch.object(exam_exports, "get_session", return_value=session),
          patch.object(exam_exports, "_problem_location_cache_ready", return_value=True)):
        finish = TestClient(server.app).get("/finish?mode=tf-prep")
    assert finish.status_code == 200
    tasks = json.loads(html.unescape(re.search(r"data-tasks='([^']*)'", finish.text)[1]))
    assert tasks == [{"problem_id": "EC", "unit_id": 1}]


def test_unreasonably_large_trace_is_rejected_before_evaluation():
    with pytest.raises(ValueError, match="evaluation limit"):
        key_with({"model": {"domain": list(range(100)), "predicates": {"R": []}},
                  "formula": "∀x ∀y ∀z R(x,y,z)"})


def test_normalization_preserves_browser_precedence_and_scope():
    assert normalize_formula("~P(a) | Q(a) & P(a)") == "(¬Pa ∨ (Qa ∧ Pa))"
    assert normalize_formula("∀x P(x) -> Q(a)") == "(∀xPx → Qa)"
    assert normalize_formula("∀x (P(x) -> Q(a))") == "∀x(Px → Qa)"
    assert normalize_formula("P(a) -> Q(a) -> R(a)") == "(Pa → (Qa → Ra))"


def test_general_vision_reading_preserves_uncertainty_and_named_objects(tmp_path):
    image = tmp_path / "fake.jpg"
    Image.new("RGB", (300, 300), "white").save(image)

    class FakeClient:
        writing = "blue"
        def chat_with_image_prompt(self, image, prompt):
            assert "Do not solve" in prompt
            return json.dumps({"truth": "true", "written": self.writing})

    client = FakeClient()
    assert prepare_vision_reading(client, image, [0, 0, 1, 1], True, ["red", "blue"])["proposed_object"] == 'object:"blue"'
    client.writing = "4"
    assert prepare_vision_reading(client, image, [0, 0, 1, 1], True, [2, 4, 7])["proposed_object"] == "4"
    client.writing = "x = 4, perhaps 7"
    assert prepare_vision_reading(client, image, [0, 0, 1, 1], True, [2, 4, 7])["proposed_object"] == "unclear"


@pytest.mark.parametrize("mode", ["all_or_nothing", "half_for_truth"])
def test_mixed_strip_ui_and_manual_answer_save_without_vision(monkeypatch, mode):
    monkeypatch.setenv("TF_GRADER_MODE", "1")
    key = key_with({**config(), "scoring": {"mode": mode, "points": 1}})
    session = session_with(key)
    segment = {"unit_id": 1, "page_filename": "fake.jpg", "page_index": 0,
               "bbox": [.1, .2, .9, .5], "present": True, "confidence": 1}
    client = TestClient(server.app)
    with (patch("app.routes.problem_strips.get_session", return_value=session),
          patch("app.routes.problem_strips.ensure_segments_cached", return_value=[segment]),
          patch("app.routes.problem_strips.question_image_url", return_value=None),
          patch("app.routes.problem_strips.persist_session"),
          patch("app.routes.problem_strips.create_vlm_client") as vision):
        manual = client.get("/problem/Q1/strips")
        assisted = client.get("/problem/EC/strips")
        assert manual.status_code == assisted.status_code == 200
        assert 'data-fol-truth-question data-context=' not in manual.text
        assert 'data-fol-truth-question data-context=' in assisted.text
        assert 'data-fol-choice="object" data-value="4"' in assisted.text
        assert 'data-fol-choice="object" data-value="3"' not in assisted.text
        assert ('data-fol-choice="object" data-value="blank"' in assisted.text) == (mode == "half_for_truth")
        assert 'D = {2, 4, 7}; R = {(2, 4), (4, 4), (7, 2)}' in assisted.text
        review = {"confirmed_truth": "true", "confirmed_object": "4",
                  "fingerprint": question_context(key, "EC", None, segment)["fingerprint"]}
        saved = client.post("/problem/EC/strips", data={"fol_truth_review_1": json.dumps(review),
                            "score_1": "1", "comment_1": "Human approved", "action": "save_back"}, follow_redirects=False)
        assert saved.status_code == 303
        assert session.work_units[0].fol_truth_reviews["EC"] == review
        assert session.work_units[0].per_problem["EC"] == 1
        assert session.work_units[0].per_problem_comments["EC"] == "Human approved"
        vision.assert_not_called()


@pytest.mark.parametrize("mode, schema", [("all_or_nothing", 2), ("half_for_truth", 3)])
def test_finish_targets_assisted_questions_and_bundle_preserves_rubric(tmp_path, monkeypatch, mode, schema):
    monkeypatch.delenv("TF_GRADER_MODE", raising=False)
    session = session_with(key_with({**config(), "scoring": {"mode": mode, "points": 1}}))
    from app.fol_truth_preparation import definition_fingerprint
    session.fol_truth_definition_review = {'schema': 1, 'approved': True,
        'fingerprint': definition_fingerprint(session.grading_key)}
    scans, grades, assets = [tmp_path / name for name in ("scans", "grades", "assets")]
    for path in (scans, grades, assets):
        path.mkdir()
    Image.new("RGB", (300, 300), "white").save(scans / "fake.jpg")
    for pid in ("Q1", "EC"):
        (grades / f"segments_{pid}.json").write_text(json.dumps([{
            "unit_id": 1, "page_filename": "fake.jpg", "page_index": 0,
            "bbox": [.1, .2, .9, .5], "present": True, "confidence": 1}]))
    with (patch.object(exam_exports, "get_session", return_value=session),
          patch.object(exam_exports, "RAW_SCANS_DIR", scans),
          patch.object(exam_exports, "GRADES_DIR", grades),
          patch.object(exam_exports, "RUBRIC_ASSETS_DIR", assets)):
        client = TestClient(server.app)
        finish = client.get("/finish?mode=tf-prep")
        bundle = client.get("/export_tf_bundle")
    assert finish.status_code == bundle.status_code == 200
    tasks = json.loads(html.unescape(re.search(r"data-tasks='([^']*)'", finish.text)[1]))
    assert tasks == [{"problem_id": "EC", "unit_id": 1}]
    with zipfile.ZipFile(io.BytesIO(bundle.content)) as archive:
        root = "synthetic_mixed_tf_bundle"
        manifest = json.loads(archive.read(f"{root}/bundle_manifest.json"))
        assert manifest["fol_truth_assistance"]["schema"] == schema
        assert manifest["fol_truth_assistance"]["required_grader_feature"] == f"fol_truth_assistance_v{schema}"
        restored = Session.model_validate_json(archive.read(f"{root}/data/grades/session.json"))
        assert restored.grading_key.problems[1].fol_truth == session.grading_key.problems[1].fol_truth
        assert question_context(restored.grading_key, "EC", None, None)["truth"] is True
        assert "fol_truth_capabilities.json" in archive.read(f"{root}/START_TF_GRADER.ps1").decode()
        assert "fol_truth_capabilities.json" in archive.read(f"{root}/START_TF_GRADER.command").decode()


def test_finish_links_to_the_verified_compatible_code_zip(tmp_path, monkeypatch):
    monkeypatch.delenv("TF_GRADER_MODE", raising=False)
    monkeypatch.setattr(ta_releases, "TA_RELEASE_ROOT", tmp_path)
    key = key_with({**config(), "scoring": {"mode": "half_for_truth", "points": 1}})
    revision = ta_releases.explainer_revision()

    def register(folder, commit, schemas, pin):
        directory = tmp_path / folder
        directory.mkdir()
        path = directory / f"logic-grading-facilitator-{commit}.zip"
        with zipfile.ZipFile(path, "w") as archive:
            prefix = path.stem + "/"
            for name in ("app/server.py", "app/fol_truth_assistance.py", "app/assets/fol_truth_review.mjs",
                         "app/assets/fol_truth_scoring.mjs", "app/assets/vendor/fol-explanations/index.mjs"):
                archive.writestr(prefix + name, "")
            archive.writestr(prefix + ta_releases.CAPABILITIES, json.dumps({"schemas": schemas}))
            archive.writestr(prefix + ta_releases.EXPLAINER_SOURCE, json.dumps({"commit": pin}))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        path.with_suffix(".tested.json").write_text(json.dumps({"sha256": digest}), encoding="utf-8")
        return path

    match = register("01", "0000001", [1, 2, 3], revision)
    register("02", "0000002", [1, 2, 3], "different-explanation-revision")
    register("03", "0000003", [1, 2], revision)
    with patch.object(exam_exports, "get_session", return_value=session_with(key)):
        client = TestClient(server.app)
        path = f"/tf-grader-code/{match.name}"
        assert f'href="{path}"' in client.get("/finish?mode=tf-prep").text
        download = client.get(path)
        assert download.status_code == 200
        assert download.content == match.read_bytes()
        with match.open("ab") as stream:
            stream.write(b"this changes the tested checksum")
        assert "No tested grader ZIP is registered" in client.get("/finish?mode=tf-prep").text
        assert client.get(path).status_code == 404
