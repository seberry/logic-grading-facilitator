"""Recover rubric assistance lost by a stale instructor process.

Version proposals come only from IU REALLMS reading printed version labels.
No scans or handwritten responses are displayed or copied into the repository.
Stop the instructor server before --apply. Both replaced files are backed up,
and an exact session hash prevents replacing concurrent grading changes.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml
from app.models import GradingKey, Session
from app.quiz_variants import available_quiz_versions
from app.storage import DATA_DIR
from app.vlm_provider import create_vlm_client


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def parse_version_reply(raw: str, versions: list[str]) -> dict:
    text = raw.strip()
    if text.startswith('```') and text.endswith('```'):
        text = '\n'.join(text.splitlines()[1:-1])
    try:
        value = json.loads(text)
    except (ValueError, TypeError):
        raise ValueError('Printed-version reading was not valid JSON.') from None
    if not isinstance(value, dict) or set(value) != {'version', 'confidence'}:
        raise ValueError('Printed-version reading had invalid fields.')
    confidence = value['confidence']
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
        raise ValueError('Printed-version confidence was invalid.')
    if value['version'] not in [None, *versions]:
        raise ValueError('Printed-version reading used an unexpected label.')
    return value


def repaired_session(original: dict, key: GradingKey, proposals: dict) -> dict:
    """Allow only assistance-field restoration and missing version recovery."""
    old_key = original['grading_key']
    if old_key['quiz_id'] != key.quiz_id:
        raise ValueError('Reviewed rubric does not match the active quiz.')
    replacement = key.model_dump()

    def strip_assistance(value):
        clean = json.loads(json.dumps(value))
        clean.pop('fol_truth_assistance', None)
        for problem in clean['problems']:
            problem.pop('fol_truth', None)
            for variant in problem.get('variants', {}).values():
                variant.pop('fol_truth', None)
        return clean

    if strip_assistance(GradingKey(**old_key).model_dump()) != strip_assistance(replacement):
        raise ValueError('Rubric content/points differ beyond assistance fields; manual review is needed.')
    candidate = json.loads(json.dumps(original))
    candidate['grading_key'] = replacement
    versions = available_quiz_versions(key)
    for unit in candidate['work_units']:
        if unit.get('quiz_version'):
            if unit['quiz_version'] not in versions:
                raise ValueError('An existing version is outside the reviewed rubric.')
            continue
        proposal = proposals.get(str(unit['unit_id']), {})
        if proposal.get('version') not in versions or proposal.get('confidence', 0) < .95:
            raise ValueError('A printed version still needs review; no session was written.')
        unit['quiz_version'] = proposal['version']
    Session(**candidate)
    # This comparison protects names, scores, comments, scans, crops, and all
    # unrelated session state, including fields a newer model might ignore.
    restored = json.loads(json.dumps(candidate))
    restored['grading_key'] = original['grading_key']
    for before, after in zip(original['work_units'], restored['work_units']):
        if 'quiz_version' in before:
            after['quiz_version'] = before['quiz_version']
        else:
            after.pop('quiz_version', None)
    if restored != original:
        raise ValueError('Session preservation check failed.')
    return candidate


def run(rubric: Path, read_versions: bool, apply: bool, limit: int | None):
    key = GradingKey(**yaml.safe_load(rubric.read_text(encoding='utf-8')))
    path = DATA_DIR / 'grades/session.json'
    session_bytes = path.read_bytes()
    original = json.loads(session_bytes)
    if original['grading_key']['quiz_id'] != key.quiz_id:
        raise ValueError('The active quiz is different; nothing changed.')
    job = DATA_DIR / 'grades' / ('assistance_repair_' + digest(session_bytes)[:16])
    proposal_path = job / 'version_proposals.json'
    proposals = json.loads(proposal_path.read_text(encoding='utf-8')) if proposal_path.is_file() else {}
    versions = available_quiz_versions(key)
    if read_versions:
        job.mkdir(exist_ok=True)
        completed = 0
        for unit in original['work_units']:
            token = str(unit['unit_id'])
            if unit.get('quiz_version') or proposals.get(token, {}).get('confidence', 0) >= .95:
                continue
            if len(unit['page_filenames']) != 1:
                raise ValueError('Version recovery currently requires one page per submission.')
            filename = unit['page_filenames'][0]
            source = (DATA_DIR / 'raw_scans' / filename).resolve()
            if not source.is_relative_to((DATA_DIR / 'raw_scans').resolve()):
                raise ValueError('Invalid source path.')
            prompt = ('Read ONLY the explicitly printed quiz VERSION label in the header of this page. '
                      'Do not transcribe any student name, handwriting, answers, or other page content. '
                      f'The allowed labels are {", ".join(versions)}; use null if the printed label is not clear. '
                      'Do not infer a label from the model or answers. Return ONLY JSON with exactly '
                      'version (a printed label or null) and confidence (a number from 0 to 1).')
            for model in ('gemma', 'qwen'):
                client = create_vlm_client(model)
                try:
                    proposal = parse_version_reply(client.chat_with_image_prompt(source, prompt), versions)
                except ValueError:
                    proposal = {'version': None, 'confidence': 0}
                if proposal['version'] is not None and proposal['confidence'] >= .95:
                    break
            proposals[token] = proposal
            temporary = proposal_path.with_suffix('.tmp')
            temporary.write_text(json.dumps(proposals, indent=2), encoding='utf-8')
            temporary.replace(proposal_path)
            completed += 1
            print(json.dumps({'new_readings': completed, 'stored_readings': len(proposals),
                              'clear_readings': sum(v.get('version') in versions and v.get('confidence', 0) >= .95 for v in proposals.values())}), flush=True)
            if limit and completed >= limit:
                break
    if apply:
        repaired = repaired_session(original, key, proposals)
        pointer = DATA_DIR / 'rubric/quiz1.yaml'
        pointer_bytes = pointer.read_bytes()
        if GradingKey(**yaml.safe_load(pointer_bytes)).quiz_id != key.quiz_id:
            raise ValueError('The selected setup rubric changed; nothing was written.')
        if path.read_bytes() != session_bytes:
            raise ValueError('The session changed during repair; nothing was written.')
        job.mkdir(exist_ok=True)
        for name, content in [('session.before.json', session_bytes), ('selected_rubric.before.yaml', pointer_bytes)]:
            backup = job / name
            if backup.exists() and backup.read_bytes() != content:
                raise ValueError('Existing backup differs; nothing was written.')
            if not backup.exists():
                with backup.open('xb') as handle:
                    handle.write(content)
        new_session = path.with_suffix('.repair.tmp')
        new_session.write_text(json.dumps(repaired, indent=2, ensure_ascii=False), encoding='utf-8')
        new_pointer = pointer.with_suffix('.repair.tmp')
        new_pointer.write_bytes(rubric.read_bytes())
        new_pointer.replace(pointer)
        new_session.replace(path)
        print(json.dumps({'applied': True, 'submissions': len(repaired['work_units']),
                          'version_counts': dict(Counter(w['quiz_version'] for w in repaired['work_units'])),
                          'names_scores_comments_preserved': True, 'backups_saved': True}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rubric', required=True, type=Path)
    parser.add_argument('--read-versions', action='store_true')
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    try:
        run(args.rubric, args.read_versions, args.apply, args.limit)
    except Exception as exc:
        # No raw provider replies, scan paths, or student records in tool output.
        print(json.dumps({'error_type': type(exc).__name__, 'detail': str(exc) if isinstance(exc, ValueError) else 'Private repair stopped; saved progress was preserved.'}))
        raise SystemExit(1)
