from app.fol_truth_assistance import (
    ASSESSMENT, MODELS, QUESTIONS, assert_supported, model_text, question_context,
    prepare_vision_reading,
    ANSWER_RESPONSE_FORMAT, VisionReadingFormatError,
)
from app.fol_model_checker import check_sentences
from app.models import GradingKey, Session, WorkUnit
from app.models import ProblemSpec, RosterEntry
from app import server
from fastapi.testclient import TestClient
from unittest.mock import patch
import json
from app.storage import load_grading_key
from app.exam_shared import _score_summary_for_work_unit
from pathlib import Path
import io
import zipfile
import fitz
import hashlib
import pytest
from app.routes import exam_exports as exam_exports_routes
from PIL import Image


EXPECTED = {
    "A": ([False, True, True, False, True, True, True], {"Q2": {1}, "Q3": {2}, "Q4": {2}, "Q5": set()}),
    "B": ([False, False, True, False, True, False, False], {"Q2": set(), "Q3": {1, 2, 3}, "Q4": {3}, "Q5": set()}),
    "C": ([True, False, True, True, False, True, True], {"Q2": set(), "Q3": {2}, "Q4": set(), "Q5": {1}}),
}


def _key():
    return GradingKey(quiz_id="fake", problems=[], fol_truth_assistance={"schema": 1, "assessment": ASSESSMENT})


def test_all_versions_match_independent_key_and_candidate_checks():
    key = _key()
    bodies = {"Q2": "((Pa ∧ Qa) ∧ Ra)", "Q3": "(Pa ∧ ¬Qa)",
              "Q4": "(Pa → (Qa ∨ Ra))", "Q5": "(Pa ∨ Qa)"}
    for version, model in MODELS.items():
        expected_truths, expected_candidates = EXPECTED[version]
        actual = [question_context(key, q, version, {})["truth"] for q in QUESTIONS]
        assert actual == expected_truths
        for question, body in bodies.items():
            valid = set()
            for obj in model["domain"]:
                trial = {**model, "constants": {**model["constants"], "a": obj}}
                result = check_sentences(model_text(trial), body).sentence_results[0]
                assert result.parsed_ok, result.error
                if (result.result if question in {"Q2", "Q3"} else not result.result):
                    valid.add(obj)
            assert valid == expected_candidates[question]


def test_fingerprint_changes_with_version_crop_config_and_source(tmp_path):
    key = _key()
    base = question_context(key, "Q2", "A", {"page_filename": "fake.jpg", "bbox": [0, 0, 1, 1]})
    assert base["fingerprint"] != question_context(key, "Q2", "B", {})["fingerprint"]
    assert base["fingerprint"] != question_context(key, "Q2", "A", {"page_filename": "fake.jpg", "bbox": [0, 0, .9, 1]})["fingerprint"]
    key.fol_truth_assistance["review_revision"] = 2
    assert base["fingerprint"] != question_context(key, "Q2", "A", {"page_filename": "fake.jpg", "bbox": [0, 0, 1, 1]})["fingerprint"]
    source = tmp_path / "fake.jpg"
    source.write_bytes(b"first")
    before = question_context(key, "Q2", "A", {}, source)["fingerprint"]
    source.write_bytes(b"second version")
    assert before != question_context(key, "Q2", "A", {}, source)["fingerprint"]


def test_review_state_survives_session_round_trip_and_unknown_schema_rejected():
    unit = WorkUnit(unit_id=1, page_filenames=["fake.jpg"], student_id="fake", per_problem={"Q2": None},
                    comment=None, fol_truth_reviews={"Q2": {"confirmed_truth": "true", "confirmed_object": "2"}})
    session = Session(roster=[], grading_key=_key(), work_units=[unit], unclaimed_pages=[])
    restored = Session.model_validate_json(session.model_dump_json())
    assert restored.work_units[0].fol_truth_reviews["Q2"]["confirmed_object"] == "2"
    restored.grading_key.fol_truth_assistance["schema"] = 99
    try:
        assert_supported(restored.grading_key)
    except ValueError as exc:
        assert "unsupported FOL truth assistance" in str(exc)
    else:
        raise AssertionError("unsupported assistance was accepted")


def test_sample_rubric_loads_opt_in_and_ten_regular_points():
    rubric = load_grading_key(Path("docs/quiz9_fol_truth_rubric.yaml"))
    assert rubric.fol_truth_assistance["schema"] == 1
    assert rubric.fol_truth_assistance["assessment"] == ASSESSMENT
    assert rubric.fol_truth_assistance["scoring"]["Q2"] == {"truth": 1, "object": 1, "blank_na_credit": False}
    assert sum(p.points for p in rubric.problems if not p.extra_credit) == 10
    assert [p.id for p in rubric.problems if p.extra_credit] == ["Q6", "Q7"]
    unit = WorkUnit(unit_id=1, page_filenames=[], student_id="fake", comment=None,
                    per_problem={p.id: p.points for p in rubric.problems})
    session = Session(roster=[], grading_key=rubric, work_units=[unit], unclaimed_pages=[])
    assert _score_summary_for_work_unit(session, unit)["total_score"] == 10


def test_vendored_shared_module_matches_pinned_source_hashes():
    root = Path("app/assets/vendor/fol-explanations")
    source = json.loads((root / "SOURCE.json").read_text(encoding="utf-8"))
    assert source["commit"] == "9d3dcdf744d4bbc1ac523a434c1cf633ffe71a9e"
    for name, expected in source["sha256"].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected


@pytest.mark.parametrize("legacy_header_context", [False, True])
def test_opted_in_strip_review_saves_transcription_and_human_comment(legacy_header_context):
    key = GradingKey(quiz_id="fake", problems=[ProblemSpec(id="Q4", label="Q4", points=2,
                         variants={v: {"prompt": "printed Q4"} for v in MODELS})],
                     fol_truth_assistance={"schema": 1, "assessment": ASSESSMENT})
    unit = WorkUnit(unit_id=1, page_filenames=["fake.jpg"], student_id="s1", quiz_version="A",
                    per_problem={"Q4": None}, comment=None)
    session = Session(roster=[RosterEntry(student_id="s1", name="Fake Student")],
                      grading_key=key, work_units=[unit], unclaimed_pages=[])
    segment = {"unit_id": 1, "page_filename": "fake.jpg", "page_index": 0,
               "bbox": [.1, .2, .9, .5], "present": True, "confidence": 1}
    client = TestClient(server.app)
    from app.routes.problem_strips import templates
    render_template = templates.TemplateResponse

    def render_with_header_context(name, context):
        if legacy_header_context:
            context.pop("fol_truth_question", None)
        return render_template(name, context)

    with (patch("app.routes.problem_strips.get_session", return_value=session),
          patch("app.routes.problem_strips.ensure_segments_cached", return_value=[segment]),
          patch("app.routes.problem_strips.question_image_url", return_value=None),
          patch("app.routes.problem_strips.persist_session"),
          patch.object(templates, "TemplateResponse", side_effect=render_with_header_context)):
        page = client.get("/problem/Q4/strips")
        assert page.status_code == 200
        assert "Interpreted student answer" in page.text
        assert page.text.count('data-fol-truth-question data-context=') == 1
        assert page.text.count('data-fol-truth-expected') == 1
        assert '<summary>Why?</summary>' in page.text
        assert '<details class="fol-answer-reveal">' in page.text
        assert "Optional scratchpad" in page.text
        assert "Check the suggested answer" in page.text
        assert "data-fol-key-confirm" in page.text
        assert "data-fol-key-reject" in page.text
        assert "Grade manually instead" in page.text
        assert 'data-fol-choice="truth" data-value="true"' in page.text
        assert 'src="/proof_viewer_assets/fol_truth_review.mjs"' in page.text
        fingerprint = question_context(key, "Q4", "A", segment)["fingerprint"]
        review = {"proposed_truth": "true", "confirmed_truth": "false",
                  "raw_object": "x = 2", "confirmed_object": "2", "fingerprint": fingerprint}
        saved = client.post("/problem/Q4/strips", data={"active_version": "A", "action": "save_back",
                      "fol_truth_review_1": json.dumps(review), "score_1": "1", "comment_1": "Human approved feedback"},
                      follow_redirects=False)
        assert saved.status_code == 303
        assert unit.fol_truth_reviews["Q4"] == review
        assert unit.per_problem["Q4"] == 1
        assert unit.per_problem_comments["Q4"] == "Human approved feedback"


def test_opted_in_bundle_and_accepted_feedback_pdf(tmp_path):
    key = GradingKey(quiz_id="fake", problems=[ProblemSpec(id="Q1", label="Truth value", points=2,
                         variants={"A": {"prompt": "¬P(c) ∨ Q(c)"}})],
                     fol_truth_assistance={"schema": 1, "assessment": ASSESSMENT})
    unit = WorkUnit(unit_id=1, page_filenames=["fake.pdf"], student_id="s1", quiz_version="A",
                    per_problem={"Q1": 1}, per_problem_comments={"Q1": "TF approved: Q(c) is false."},
                    fol_truth_reviews={"Q1": {"proposed_truth": "true", "confirmed_truth": "false"}}, comment=None)
    session = Session(roster=[RosterEntry(student_id="s1", name="Fake Student")],
                      grading_key=key, work_units=[unit], unclaimed_pages=[])
    scans = tmp_path / "scans"
    grades = tmp_path / "grades"
    assets = tmp_path / "assets"
    for path in (scans, grades, assets):
        path.mkdir()
    pdf = fitz.open()
    pdf.new_page()
    pdf.save(scans / "fake.pdf")
    pdf.close()
    (grades / "segments_Q1.json").write_text(json.dumps([{"unit_id": 1, "page_filename": "fake.pdf",
        "page_index": 0, "bbox": [.1, .2, .9, .5], "present": True, "confidence": 1}]))
    client = TestClient(server.app)
    with (patch.object(exam_exports_routes, "get_session", return_value=session),
          patch.object(exam_exports_routes, "RAW_SCANS_DIR", scans),
          patch.object(exam_exports_routes, "GRADES_DIR", grades),
          patch.object(exam_exports_routes, "RUBRIC_ASSETS_DIR", assets)):
        bundle = client.get("/export_tf_bundle")
        feedback = client.get("/export_feedback_zip")
    assert bundle.status_code == 200
    with zipfile.ZipFile(io.BytesIO(bundle.content)) as archive:
        manifest = json.loads(archive.read("fake_tf_bundle/bundle_manifest.json"))
        state = json.loads(archive.read("fake_tf_bundle/data/grades/session.json"))
        assert manifest["fol_truth_assistance"]["schema"] == 1
        assert manifest["deterministic_grading_assistance_included"] is True
        assert state["work_units"][0]["fol_truth_reviews"]["Q1"]["confirmed_truth"] == "false"
    assert feedback.status_code == 200
    with zipfile.ZipFile(io.BytesIO(feedback.content)) as archive:
        pdf_name = next(name for name in archive.namelist() if name.endswith("_graded.pdf"))
        output = fitz.open(stream=archive.read(pdf_name), filetype="pdf")
        text = "\n".join(page.get_text() for page in output)
        assert "TF approved: Q(c) is false." in text


def test_vision_proposal_is_strict_and_never_confirms_complex_writing(tmp_path):
    image = tmp_path / "fake.jpg"
    Image.new("RGB", (600, 400), "white").save(image)

    class FakeClient:
        def __init__(self, payload):
            self.payload = payload
            self.seen = False

        def chat_with_image_prompt(self, crop_path, prompt):
            self.seen = crop_path.is_file() and "Do not solve" in prompt
            return self.payload

    client = FakeClient('{"truth":"false","written":"x = 2, maybe 3"}')
    result = prepare_vision_reading(client, image, [.1, .2, .9, .6], True)
    assert client.seen
    assert result == {"proposed_truth": "false", "proposed_object": "unclear",
                      "proposed_raw_object": "x = 2, maybe 3"}
    assert prepare_vision_reading(FakeClient('{"truth":"true","written":"2"}'),
                                  image, [.1, .2, .9, .6], True)["proposed_object"] == "2"
    assert prepare_vision_reading(FakeClient('```json\n{"truth":"true","written":"2"}\n```'),
                                  image, [.1, .2, .9, .6], True)["proposed_object"] == "2"


def test_vision_endpoint_is_forbidden_in_tf_mode(monkeypatch):
    monkeypatch.setenv("TF_GRADER_MODE", "1")
    response = TestClient(server.app).post("/problem/Q2/fol-truth/prepare-reading", data={"unit_id": "1"})
    assert response.status_code == 403


@pytest.mark.parametrize('truth, expected', [(False, 'false'), (True, 'true'), ('false', 'false')])
def test_json_boolean_false_is_a_clear_reading_not_unclear(tmp_path, truth, expected):
    image = tmp_path / 'synthetic.jpg'
    Image.new('RGB', (100, 100), 'white').save(image)
    class Client:
        def chat_with_image_prompt(self, path, prompt):
            return json.dumps({'truth': truth, 'written': ''})
    assert prepare_vision_reading(Client(), image, [0, 0, 1, 1], False)['proposed_truth'] == expected


def test_structured_answer_reading_uses_json_schema(tmp_path):
    image = tmp_path / 'synthetic.jpg'
    Image.new('RGB', (100, 100), 'white').save(image)
    class Client:
        def chat_with_image_json(self, path, prompt, response_format, max_tokens):
            assert response_format == ANSWER_RESPONSE_FORMAT and max_tokens == 512
            assert '{"truth":"false","written":""}' in prompt
            return '{"truth":"false","written":"N/A"}'
        def chat_with_image_prompt(self, *args):
            pytest.fail('Structured reading should not use the unconstrained path.')
    assert prepare_vision_reading(Client(), image, [0, 0, 1, 1], True)['proposed_object'] == 'na'


@pytest.mark.parametrize('reply', [
    'False', 'The answer is {"truth":"false","written":""}',
    '{"truth":"false","written":""', '[{"truth":"false","written":""}]',
    '{"truth":"false"}', '{"truth":"false","written":2}',
])
def test_malformed_readings_are_rejected_instead_of_inferred(tmp_path, reply):
    image = tmp_path / 'synthetic.jpg'
    Image.new('RGB', (100, 100), 'white').save(image)
    class Client:
        def chat_with_image_prompt(self, path, prompt):
            return reply
    with pytest.raises(VisionReadingFormatError):
        prepare_vision_reading(Client(), image, [0, 0, 1, 1], True)


def test_format_failure_is_retryable_and_saved_reading_does_not_call_a_model(monkeypatch):
    monkeypatch.delenv('TF_GRADER_MODE', raising=False)
    key = GradingKey(quiz_id='fake', problems=[ProblemSpec(id='Q1', label='Q1', points=2)],
                    fol_truth_assistance={'schema': 1, 'assessment': ASSESSMENT})
    unit = WorkUnit(unit_id=1, student_id=None, page_filenames=['fake.jpg'], quiz_version='A',
                    per_problem={'Q1': None}, comment=None)
    session = Session(roster=[], grading_key=key, work_units=[unit], unclaimed_pages=[])
    segment = {'unit_id': 1, 'page_filename': 'fake.jpg', 'bbox': [.1, .2, .9, .5], 'present': True}
    with (patch('app.routes.problem_strips.get_session', return_value=session),
          patch('app.routes.problem_strips.ensure_segments_cached', return_value=[segment]),
          patch('app.routes.problem_strips._fol_source_path', return_value=None),
          patch('app.routes.problem_strips.create_vlm_client') as client,
          patch('app.routes.problem_strips.prepare_vision_reading', side_effect=VisionReadingFormatError('Invalid JSON.')),
          patch('app.routes.problem_strips.persist_session') as save):
        api = TestClient(server.app)
        failed = api.post('/problem/Q1/fol-truth/prepare-reading', data={'unit_id': 1})
        assert failed.status_code == 502 and failed.json()['code'] == 'vision_reading_format'
        assert unit.fol_truth_reviews == {} and unit.per_problem['Q1'] is None
        save.assert_not_called()
        prior = {'proposed_truth': 'false', 'proposed_object': None, 'proposed_raw_object': None,
                 'fingerprint': question_context(key, 'Q1', 'A', segment)['fingerprint']}
        unit.fol_truth_reviews['Q1'] = prior
        client.reset_mock()
        cached = api.post('/problem/Q1/fol-truth/prepare-reading', data={'unit_id': 1})
        assert cached.status_code == 200 and cached.json()['status'] == 'cached'
        client.assert_not_called()
        assert unit.fol_truth_reviews['Q1'] == prior


def test_instructor_vision_endpoint_saves_only_proposal(tmp_path, monkeypatch):
    monkeypatch.delenv("TF_GRADER_MODE", raising=False)
    Image.new("RGB", (600, 400), "white").save(tmp_path / "fake.jpg")
    key = GradingKey(quiz_id="fake", problems=[ProblemSpec(id="Q2", label="Q2", points=2)],
                     fol_truth_assistance={"schema": 1, "assessment": ASSESSMENT})
    unit = WorkUnit(unit_id=1, page_filenames=["fake.jpg"], student_id="fake", quiz_version="A",
                    per_problem={"Q2": None}, comment=None)
    session = Session(roster=[], grading_key=key, work_units=[unit], unclaimed_pages=[])
    segment = {"unit_id": 1, "page_filename": "fake.jpg", "bbox": [.1, .2, .9, .5], "present": True}

    class FakeClient:
        def chat_with_image_prompt(self, crop_path, prompt):
            return '{"truth":"true","written":"x = 1"}'

    with (patch("app.routes.problem_strips.get_session", return_value=session),
          patch("app.routes.problem_strips.ensure_segments_cached", return_value=[segment]),
          patch("app.routes.problem_strips.create_vlm_client", return_value=FakeClient()),
          patch("app.routes.problem_strips.RAW_SCANS_DIR", tmp_path),
          patch("app.routes.problem_strips.persist_session")):
        response = TestClient(server.app).post("/problem/Q2/fol-truth/prepare-reading", data={"unit_id": "1"})
    assert response.status_code == 200
    review = unit.fol_truth_reviews["Q2"]
    assert review["proposed_truth"] == "true"
    assert review["proposed_object"] == "unclear"
    assert review["proposed_raw_object"] == "x = 1"
    assert "confirmed_truth" not in review
    assert unit.per_problem["Q2"] is None
