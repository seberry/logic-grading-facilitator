from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
from uuid import uuid4

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool

from app.fol_truth_assistance import HALF_FOR_TRUTH, enabled, question_context
from app.fol_truth_preparation import (definition_fingerprint, definitions_approved, key_from_rows,
    key_revision, preparation_rows, transcribe_blank_assessment)
from app.rubric_assets import question_image_url
from app.session_state import _session_lock, get_session, persist_session
from app.storage import GRADES_DIR, RUBRIC_DIR, load_grading_key
from app.vlm_provider import create_vlm_client

router = APIRouter()
templates = Jinja2Templates(directory=str(Path(__file__).resolve().parents[1] / 'templates'))


def instructor_only():
    if os.getenv('TF_GRADER_MODE', '').strip().lower() in {'1', 'true', 'yes', 'on'}:
        raise HTTPException(403, 'Logical-input preparation is instructor-only.')


def _source_key(session):
    key = session.grading_key
    if not enabled(key):
        # A named reviewed source can recover an active copy stripped by old code.
        filename = RUBRIC_DIR / (Path(key.quiz_id).name + '.yaml')
        if filename.is_file():
            candidate = load_grading_key(filename)
            if candidate.quiz_id == key.quiz_id and enabled(candidate):
                return candidate
    return key


def _review_error(exc):
    if isinstance(exc, ValidationError):
        return exc.errors(include_url=False, include_context=False, include_input=False)[0]['msg'].removeprefix('Value error, ')
    return str(exc)


def _render(request, session, rows=None, source='', error=None, status=200, score_policy=None):
    source_key = _source_key(session)
    rows = rows if rows is not None else preparation_rows(source_key)
    if score_policy is None:
        modes = {row['scoring_mode'] for row in rows if row['enabled'] and not row.get('manual_evidence')}
        score_policy = HALF_FOR_TRUTH if modes == {HALF_FOR_TRUTH} else 'keep'
    fingerprint = ''
    try:
        candidate = key_from_rows(session.grading_key, rows, score_policy)
        fingerprint = definition_fingerprint(candidate)
    except (ValueError, TypeError, KeyError) as exc:
        candidate = None
        error = error or _review_error(exc)
    specs = {p.id: p for p in source_key.problems}
    display = []
    for row in rows:
        context = question_context(candidate, row['problem_id'], row['version'], None) if candidate and row['enabled'] else None
        display.append({**row, 'context': context,
            'question_image_url': question_image_url(source_key, specs[row['problem_id']], row['version'])})
    return templates.TemplateResponse('fol_truth_preparation.html', {
        'request': request, 'quiz_id': session.grading_key.quiz_id, 'rows': display, 'score_policy': score_policy,
        'revision': key_revision(session.grading_key), 'preview_fingerprint': fingerprint,
        'approved': definitions_approved(session) and enabled(session.grading_key)
                    and fingerprint == definition_fingerprint(session.grading_key),
        'source': source if re.fullmatch(r'[a-f0-9]{64}', source) else '', 'error': error,
    }, status_code=status)


def _active_draft(session):
    path = GRADES_DIR / 'fol_truth_preparation_draft.json'
    try:
        draft = json.loads(path.read_text(encoding='utf-8'))
        if isinstance(draft, dict) and draft.get('revision') == key_revision(session.grading_key) and 'rows' in draft:
            return draft
    except (OSError, ValueError):
        pass
    return {}


@router.get('/prepare/fol-truth')
def get_review(request: Request):
    instructor_only()
    session = get_session()
    draft = _active_draft(session)
    if draft:
        return _render(request, session, draft['rows'], draft.get('source', ''),
                       score_policy=draft.get('score_policy'))
    return _render(request, session)


def _save_draft(session, rows, source, score_policy=None):
    GRADES_DIR.mkdir(parents=True, exist_ok=True)
    path = GRADES_DIR / 'fol_truth_preparation_draft.json'
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps({'revision': key_revision(session.grading_key), 'rows': rows,
                                     'source': source, 'score_policy': score_policy}, ensure_ascii=False), encoding='utf-8')
    temporary.replace(path)


@router.post('/prepare/fol-truth/read-blank')
async def read_blank(request: Request, blank_pdf: UploadFile = File(...), model: str = Form('gemma'),
                     revision: str = Form(...)):
    instructor_only()
    session = get_session()
    if revision != key_revision(session.grading_key):
        raise HTTPException(409, 'The assessment changed. Reload the review page.')
    if model not in {'gemma', 'qwen'}:
        raise HTTPException(400, 'Choose IU Gemma or IU Qwen.')
    data = await blank_pdf.read(12 * 1024 * 1024 + 1)
    try:
        rows = await run_in_threadpool(transcribe_blank_assessment, data, _source_key(session), create_vlm_client(model))
    except Exception as exc:
        detail = _review_error(exc) if isinstance(exc, ValueError) else 'IU could not read the blank quiz. Retry with Qwen or edit the definitions below.'
        return _render(request, session, error=detail, status=400)
    if revision != key_revision(get_session().grading_key):
        raise HTTPException(409, 'The assessment changed during reading. Reload the review page.')
    token = hashlib.sha256(data).hexdigest()
    directory = GRADES_DIR / 'blank_fol_sources'
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / (token + '.pdf')
    if not target.exists():
        target.write_bytes(data)
    score_policy = _active_draft(session).get('score_policy')
    _save_draft(session, rows, token, score_policy)
    return _render(request, session, rows, token, score_policy=score_policy)


@router.get('/prepare/fol-truth/source/{token}')
def blank_source(token: str):
    instructor_only()
    if not re.fullmatch(r'[a-f0-9]{64}', token):
        raise HTTPException(404)
    path = GRADES_DIR / 'blank_fol_sources' / (token + '.pdf')
    if not path.is_file():
        raise HTTPException(404)
    return FileResponse(path, media_type='application/pdf')


@router.post('/prepare/fol-truth')
async def post_review(request: Request):
    instructor_only()
    form = await request.form()
    session = get_session()
    if form.get('revision') != key_revision(session.grading_key):
        raise HTTPException(409, 'The assessment changed. Reload the review page before approving.')
    score_policy = form.get('score_policy', 'keep')
    try:
        count = int(form.get('row_count', 0))
        if count != len(preparation_rows(session.grading_key)):
            raise ValueError('The question/version list changed. Reload the page.')
        rows = []
        for i in range(count):
            row = {name: form.get(f'{name}_{i}', '') for name in (
                'problem_id', 'version', 'formula', 'model_text', 'scoring_mode', 'truth_points', 'object_points')}
            row['version'] = row['version'] or None
            row.update({name: form.get(f'{name}_{i}') == 'on' for name in (
                'enabled', 'asks_object', 'manual_evidence', 'blank_na_credit')})
            spec = next(p for p in session.grading_key.problems if p.id == row['problem_id'])
            row.update(label=spec.label, points=spec.points)
            rows.append(row)
        candidate = key_from_rows(session.grading_key, rows, score_policy)
    except (ValueError, TypeError, StopIteration, KeyError) as exc:
        return _render(request, session, locals().get('rows') or None, form.get('source', ''),
                       error=_review_error(exc), status=400, score_policy=score_policy)
    if form.get('action') != 'approve':
        _save_draft(session, rows, form.get('source', ''), score_policy)
        return _render(request, session, rows, form.get('source', ''), score_policy=score_policy)
    if form.get('preview_fingerprint') != definition_fingerprint(candidate):
        return _render(request, session, rows, form.get('source', ''),
                       error='Check the changed definitions before approving them.', status=400, score_policy=score_policy)
    with _session_lock:
        current = get_session()
        if form.get('revision') != key_revision(current.grading_key):
            raise HTTPException(409, 'The assessment changed. Reload and review the latest definitions.')
        backup = GRADES_DIR / 'fol_truth_definition_backups'
        backup.mkdir(parents=True, exist_ok=True)
        (backup / (uuid4().hex + '.json')).write_text(current.model_dump_json(indent=2), encoding='utf-8')
        current.grading_key = candidate
        current.fol_truth_definition_review = {'schema': 1, 'approved': True,
                                               'fingerprint': definition_fingerprint(candidate)}
        persist_session()
    return RedirectResponse('/finish?mode=tf-prep', status_code=303)
