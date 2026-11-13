from copy import deepcopy
import json
from unittest.mock import patch

from bs4 import BeautifulSoup
from fastapi.testclient import TestClient
import fitz
import pytest
import yaml

from app import server
from app.fol_truth_assistance import question_context
from app.fol_truth_preparation import (definition_fingerprint, definitions_approved, key_from_rows,
    preparation_rows, transcribe_blank_assessment)
from app.instructor_lock import InstructorDataLock
from app.models import GradingKey, ProblemSpec, Session, WorkUnit
from app.routes import fol_truth_preparation as routes
from app.routes.exam_exports import _validate_tf_bundle


def sample():
    key = GradingKey(quiz_id='synthetic_review', problems=[
        ProblemSpec(id='Q1', label='Model truth', points=2,
            fol_truth={'asks_object': True, 'scoring': {'truth': 1, 'object': 1}},
            variants={v: {'fol_truth': {'formula': '∃x Px', 'model': {
                'domain': [1, 2], 'constants': {}, 'predicates': {'P': extension}}}}
                for v, extension in [('D', [1]), ('E', [])]}),
        ProblemSpec(id='Q2', label='Manual', points=2, variants={'D': {}, 'E': {}})])
    return Session(roster=[], grading_key=key, unclaimed_pages=[], work_units=[
        WorkUnit(unit_id=1, student_id=None, page_filenames=['synthetic.jpg'], quiz_version='D',
                 per_problem={'Q1': 1.5, 'Q2': 2}, per_problem_comments={'Q1': 'Keep this comment'}, comment=None)])


def form_fields(html):
    form = BeautifulSoup(html, 'html.parser').find('form', id='definitions-form')
    fields = {}
    for element in form.select('input[name],textarea[name],select[name]'):
        if element.name == 'textarea':
            fields[element['name']] = element.text
        elif element.name == 'select':
            selected = element.find('option', selected=True) or element.find('option')
            fields[element['name']] = selected['value']
        elif element.get('type') == 'checkbox':
            if element.has_attr('checked'):
                fields[element['name']] = 'on'
        else:
            fields[element['name']] = element.get('value', '')
    return fields


def test_input_review_approval_and_correction_preserve_grades(tmp_path, monkeypatch):
    monkeypatch.delenv('TF_GRADER_MODE', raising=False)
    session = sample()
    before = deepcopy(session.work_units)
    with (patch.object(routes, 'get_session', return_value=session),
          patch.object(routes, 'GRADES_DIR', tmp_path), patch.object(routes, 'RUBRIC_DIR', tmp_path),
          patch.object(routes, 'question_image_url', return_value=None), patch.object(routes, 'persist_session') as save):
        client = TestClient(server.app)
        page = client.get('/prepare/fol-truth')
        assert page.status_code == 200
        assert 'Approve definitions and enable feedback' in page.text
        assert 'data-definition-preview' in page.text
        assert not definitions_approved(session)
        fields = form_fields(page.text)
        fields['formula_0'] = '∀x Px'
        refused = client.post('/prepare/fol-truth', data={**fields, 'action': 'approve'})
        assert refused.status_code == 400
        save.assert_not_called()
        assert not definitions_approved(session)
        checked = client.post('/prepare/fol-truth', data={**fields, 'action': 'preview'})
        assert checked.status_code == 200
        accepted = client.post('/prepare/fol-truth', data={**form_fields(checked.text), 'action': 'approve'}, follow_redirects=False)
        assert accepted.status_code == 303
        save.assert_called_once()
        assert definitions_approved(session)
        assert session.work_units == before
        assert question_context(session.grading_key, 'Q1', 'D', None)['truth'] is False
        assert len(list((tmp_path / 'fol_truth_definition_backups').glob('*.json'))) == 1
        session.work_units[0].per_problem['Q1'] = .5
        assert definitions_approved(session)
        session.grading_key.problems[0].variants['D'].fol_truth['model']['predicates']['P'] = [1, 2]
        assert not definitions_approved(session)


def test_invalid_logical_inputs_never_change_active_key(tmp_path, monkeypatch):
    monkeypatch.delenv('TF_GRADER_MODE', raising=False)
    session = sample()
    before = session.model_dump()
    with (patch.object(routes, 'get_session', return_value=session), patch.object(routes, 'GRADES_DIR', tmp_path),
          patch.object(routes, 'RUBRIC_DIR', tmp_path), patch.object(routes, 'question_image_url', return_value=None),
          patch.object(routes, 'persist_session') as save):
        client = TestClient(server.app)
        fields = form_fields(client.get('/prepare/fol-truth').text)
        fields['model_text_0'] = 'DOMAIN: 1,2\nCONSTANTS:\na: 99\nPREDICATES:\nP: 1'
        response = client.post('/prepare/fol-truth', data={**fields, 'action': 'preview'})
        assert response.status_code == 400
        assert 'not in the domain' in response.text
        assert session.model_dump() == before
        save.assert_not_called()


def test_named_source_can_recover_stripped_definitions_for_instructor_review(tmp_path, monkeypatch):
    monkeypatch.delenv('TF_GRADER_MODE', raising=False)
    session = sample()
    (tmp_path / 'synthetic_review.yaml').write_text(yaml.safe_dump(session.grading_key.model_dump()), encoding='utf-8')
    for problem in session.grading_key.problems:
        problem.fol_truth = None
        for variant in problem.variants.values():
            variant.fol_truth = None
    with (patch.object(routes, 'get_session', return_value=session), patch.object(routes, 'RUBRIC_DIR', tmp_path),
          patch.object(routes, 'question_image_url', return_value=None)):
        page = TestClient(server.app).get('/prepare/fol-truth')
    assert page.status_code == 200
    assert len(BeautifulSoup(page.text, 'html.parser').select('[data-definition-preview]')) == 2
    assert session.grading_key.problems[0].fol_truth is None  # Preview is not approval.


def pdf_bytes(text=''):
    with fitz.open() as doc:
        page = doc.new_page()
        if text:
            page.insert_text((50, 50), text)
        return doc.tobytes()


def extracted():
    return {'questions': [{'problem_id': 'Q1', 'version': 'D', 'formula': 'exists x P(x)',
        'model': {'domain': [1, 2], 'predicates': {'P': [2]}}, 'asks_object': True}]}


class FakeReader:
    def __init__(self, payload=None):
        self.payload = payload if payload is not None else extracted()
        self.messages = None
        self.encoded = 0
    def chat(self, messages, **kwargs):
        self.messages = messages
        return json.dumps(self.payload)
    def _encode_image_to_data_uri(self, path):
        assert path.is_file()
        self.encoded += 1
        return 'data:image/png;base64,synthetic-blank-page'


def test_transcription_reads_embedded_blank_text_and_proposes_no_grades():
    session = sample()
    reader = FakeReader()
    before = session.model_dump()
    rows = transcribe_blank_assessment(pdf_bytes('Blank quiz Version D. Domain 1,2. P is the set containing 2. Q1 exists x Px.'), session.grading_key, reader)
    assert reader.encoded == 0
    assert 'Do not solve questions' in reader.messages[0]['content']
    assert 'BLANK ASSESSMENT SOURCE:' in reader.messages[0]['content']
    candidate = key_from_rows(session.grading_key, rows)
    assert question_context(candidate, 'Q1', 'D', None)['model']['predicates']['P'] == [2]
    assert question_context(candidate, 'Q1', 'E', None) is None
    assert session.model_dump() == before


def test_image_only_blank_pdf_uses_iu_vision():
    reader = FakeReader()
    transcribe_blank_assessment(pdf_bytes(), sample().grading_key, reader)
    assert reader.encoded == 1
    assert reader.messages[0]['content'][1]['type'] == 'image_url'


@pytest.mark.parametrize('payload', [
    {'questions': [{'problem_id': 'Q99', 'version': 'D', 'formula': 'Pa', 'model': {}, 'asks_object': False}]},
    {'questions': [{**extracted()['questions'][0], 'truth': True}]},
    {'questions': [extracted()['questions'][0], extracted()['questions'][0]]},
])
def test_extraction_rejects_unexpected_questions_and_generated_answers(payload):
    with pytest.raises(ValueError):
        transcribe_blank_assessment(pdf_bytes('Blank finite-model assessment with sufficient embedded text.'), sample().grading_key, FakeReader(payload))


def test_blank_upload_displays_review_source_without_approving(tmp_path, monkeypatch):
    monkeypatch.delenv('TF_GRADER_MODE', raising=False)
    session = sample()
    with (patch.object(routes, 'get_session', return_value=session), patch.object(routes, 'GRADES_DIR', tmp_path),
          patch.object(routes, 'RUBRIC_DIR', tmp_path), patch.object(routes, 'question_image_url', return_value=None),
          patch.object(routes, 'create_vlm_client', return_value=FakeReader()), patch.object(routes, 'persist_session') as save):
        client = TestClient(server.app)
        page = client.get('/prepare/fol-truth')
        revision = form_fields(page.text)['revision']
        response = client.post('/prepare/fol-truth/read-blank', data={'revision': revision, 'model': 'gemma'},
            files={'blank_pdf': ('blank.pdf', pdf_bytes('Blank Version D. Domain 1,2. P contains 2. Evaluate Q1 exists x Px.'), 'application/pdf')})
        assert response.status_code == 200
        iframe = BeautifulSoup(response.text, 'html.parser').find('iframe')
        source = client.get(iframe['src'])
        assert source.status_code == 200 and source.content.startswith(b'%PDF')
        restored = client.get('/prepare/fol-truth')
        restored_fields = form_fields(restored.text)
        assert '2' in restored_fields['model_text_0']
        assert BeautifulSoup(restored.text, 'html.parser').find('iframe')['src'] == iframe['src']
        save.assert_not_called()
        assert not definitions_approved(session)


def test_export_requires_definition_approval_for_general_assistance():
    session = sample()
    errors, _ = _validate_tf_bundle(session)
    assert any('approve the models and sentences' in error for error in errors)
    session.fol_truth_definition_review = {'schema': 1, 'approved': True,
        'fingerprint': definition_fingerprint(session.grading_key)}
    errors, _ = _validate_tf_bundle(session)
    assert not any('approve the models and sentences' in error for error in errors)


def test_tf_mode_cannot_read_or_approve_definitions(monkeypatch):
    monkeypatch.setenv('TF_GRADER_MODE', '1')
    client = TestClient(server.app)
    assert client.get('/prepare/fol-truth').status_code == 403
    assert client.post('/prepare/fol-truth', data={}).status_code == 403
    assert client.get('/prepare/fol-truth/source/' + 'a' * 64).status_code == 403


def test_one_instructor_process_per_data_directory(tmp_path):
    with InstructorDataLock(tmp_path):
        with pytest.raises(RuntimeError, match='Another instructor grader'):
            with InstructorDataLock(tmp_path):
                pytest.fail('A competing instructor acquired the same data directory.')
        with InstructorDataLock(tmp_path / 'independent'):
            pass
    with InstructorDataLock(tmp_path):
        pass
