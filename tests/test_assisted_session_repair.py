from copy import deepcopy
import json

import pytest

from app.models import GradingKey, ProblemSpec, Session, WorkUnit
from scripts.repair_assisted_quiz_session import parse_version_reply, repaired_session


def sample():
    key = GradingKey(quiz_id='synthetic_retake', problems=[ProblemSpec(
        id='Q1', label='Truth', points=2, variants={
            'D': {'fol_truth': {'model': {'domain': [1, 2], 'predicates': {'P': [1]}},
                                'formula': '∃x Px', 'scoring': {'truth': 2}}},
            'E': {'fol_truth': {'model': {'domain': [1, 2], 'predicates': {'P': []}},
                                'formula': '∃x Px', 'scoring': {'truth': 2}}}})])
    session = Session(grading_key=key, roster=[], unclaimed_pages=[], work_units=[
        WorkUnit(unit_id=1, student_id=None, page_filenames=['synthetic.jpg'],
                 per_problem={'Q1': 1.5}, per_problem_comments={'Q1': 'Keep this'}, comment='Keep this too')]).model_dump()
    for variant in session['grading_key']['problems'][0]['variants'].values():
        variant.pop('fol_truth')
    session['extra_future_metadata'] = {'keep': True}
    return key, session


def test_repair_changes_only_assistance_and_missing_version_preserving_all_other_state():
    key, original = sample()
    untouched = deepcopy(original)
    repaired = repaired_session(original, key, {'1': {'version': 'D', 'confidence': .99}})
    assert original == untouched
    assert repaired['work_units'][0]['quiz_version'] == 'D'
    assert repaired['work_units'][0]['per_problem'] == {'Q1': 1.5}
    assert repaired['work_units'][0]['per_problem_comments'] == {'Q1': 'Keep this'}
    assert repaired['work_units'][0]['comment'] == 'Keep this too'
    assert repaired['extra_future_metadata'] == {'keep': True}
    assert repaired['grading_key'] == key.model_dump()


def test_repair_refuses_to_change_points_or_reinterpret_unclear_versions():
    key, original = sample()
    with pytest.raises(ValueError, match='still needs review'):
        repaired_session(original, key, {'1': {'version': 'D', 'confidence': .5}})
    with pytest.raises(ValueError, match='still needs review'):
        repaired_session(original, key, {})
    original['grading_key']['problems'][0]['points'] = 3
    with pytest.raises(ValueError, match='content/points differ'):
        repaired_session(original, key, {'1': {'version': 'D', 'confidence': 1}})


def test_existing_version_is_preserved_even_if_a_model_proposal_disagrees():
    key, original = sample()
    original['work_units'][0]['quiz_version'] = 'E'
    repaired = repaired_session(original, key, {'1': {'version': 'D', 'confidence': 1}})
    assert repaired['work_units'][0]['quiz_version'] == 'E'


@pytest.mark.parametrize('reply', [
    {'version': 'A', 'confidence': 1}, {'version': 'D', 'confidence': True},
    {'version': 'D', 'confidence': float('nan')}, {'version': 'D', 'confidence': 1, 'student_name': 'not allowed'},
])
def test_version_reader_rejects_unexpected_or_extra_content(reply):
    with pytest.raises(ValueError):
        parse_version_reply(json.dumps(reply), ['D', 'E'])


def test_version_reader_accepts_only_printed_version_and_confidence():
    assert parse_version_reply('```json\n{"version":"E","confidence":0.99}\n```', ['D', 'E']) == {
        'version': 'E', 'confidence': .99}
