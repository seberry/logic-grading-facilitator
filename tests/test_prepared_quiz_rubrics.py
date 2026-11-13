"""Reviewed blank-assessment definitions, without student data or PDFs."""
import json
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
import pytest
import yaml

from app import server
from app.fol_truth_assistance import question_context
from app.models import GradingKey, RosterEntry
from app.quiz_variants import available_quiz_versions
from app.vlm_client import NAME_RESPONSE_FORMAT
from app.vlm_classifier import classify_page


ROOT = Path(__file__).resolve().parents[1]


def rubric(filename):
    return GradingKey(**yaml.safe_load((ROOT / 'docs' / filename).read_text(encoding='utf-8')))


def test_retake_models_and_existing_scoring_policy():
    key = rubric('quiz9_retakes_DE_rubric.yaml')
    assert available_quiz_versions(key) == ['D', 'E']
    assert [p.points for p in key.problems] == [2, 2, 2, 2, 2, 1, 1]
    assert [p.id for p in key.problems if p.extra_credit] == ['Q6', 'Q7']
    expected = {'D': [False, True, True, False, False, True, True],
                'E': [True, False, False, True, False, False, True]}
    for version in expected:
        contexts = [question_context(key, p.id, version, None) for p in key.problems]
        assert [c['truth'] for c in contexts] == expected[version]
        assert contexts[1]['score_rule'] == {'mode': 'components', 'truth': 1, 'object': 1, 'blank_na_credit': False}
        assert contexts[0]['model']['constants']['c'] == (1 if version == 'D' else 2)
    assert question_context(key, 'Q1', 'E', None)['model']['predicates']['P'] == []


def test_quiz10_is_manual_and_preserves_printed_points_and_versions():
    key = rubric('quiz10_polyadic_symbolization_rubric.yaml')
    assert available_quiz_versions(key) == ['A', 'B', 'C']
    assert [p.points for p in key.problems] == [2, 2, 2, 2, 2, 1]
    assert [p.id for p in key.problems if p.extra_credit] == ['Q6']
    for version in ['A', 'B', 'C']:
        assert all(question_context(key, p.id, version, None) is None for p in key.problems)
    assert 'Every dog is comfortable around some cat.' in key.problems[2].variants['B'].prompt
    assert 'Everyone admires Lena.' in key.problems[1].variants['C'].prompt


@pytest.mark.parametrize('observed', ['D', 'E', 'Z'])
def test_name_review_offers_retake_versions_and_retains_any_observed_label(tmp_path, observed):
    key = rubric('quiz9_retakes_DE_rubric.yaml')
    rubrics, scans = tmp_path / 'rubric', tmp_path / 'scans'
    rubrics.mkdir()
    scans.mkdir()
    (rubrics / f'{key.quiz_id}.yaml').write_text(yaml.safe_dump(key.model_dump()), encoding='utf-8')
    (scans / 'synthetic.jpg').write_bytes(b'fake scan not decoded by cached review')
    cache = tmp_path / 'names.json'
    cache.write_text(json.dumps([{'filename': 'synthetic.jpg', 'student_id': 'fake',
        'confidence': 1, 'name_visible': True, 'quiz_version': observed}]), encoding='utf-8')
    with (patch.object(server, 'RAW_SCANS_DIR', scans), patch.object(server, 'RUBRIC_DIR', rubrics),
          patch.object(server, 'name_review_cache_path', return_value=cache),
          patch.object(server, 'load_roster', return_value=[RosterEntry(student_id='fake', name='Synthetic Student')])):
        response = TestClient(server.app).get('/name_review', params={'quiz': key.quiz_id})
    assert response.status_code == 200
    assert f'option value="{observed}" selected' in response.text
    assert 'option value="D"' in response.text and 'option value="E"' in response.text
    assert 'option value="A"' not in response.text


@pytest.mark.parametrize('version', ['D', 'E'])
def test_classifier_preserves_explicit_retake_version(version):
    class FakeClient:
        def call_for_page(self, *args, **kwargs):
            return json.dumps({'name_visible': True, 'student_id': 'fake', 'raw_name': None,
                               'quiz_version': f'Version {version}', 'confidence': 1})
    prediction = classify_page(Path('synthetic.jpg'), [RosterEntry(student_id='fake', name='Synthetic Student')], FakeClient())
    assert prediction.quiz_version == version
    # The provider's structured response must permit the labels the UI preserves.
    field = NAME_RESPONSE_FORMAT['json_schema']['schema']['properties']['quiz_version']
    assert 'enum' not in field or version in field['enum']
