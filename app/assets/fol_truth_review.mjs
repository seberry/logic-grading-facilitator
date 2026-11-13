import { explainTruth, explainCandidate } from './vendor/fol-explanations/index.mjs';
import { scoreConfirmed } from './fol_truth_scoring.mjs';
import { candidateValue, clearObjectAnswer, objectLabel, readingFromWriting } from './fol_truth_answers.mjs';

// One answer key for this question/version; student readings stay visually separate.
const keyReviews = [];
for (const question of document.querySelectorAll('[data-fol-truth-question]')) {
  const context = JSON.parse(question.dataset.context);
  const review = {context, state: 'pending', listeners: []};
  keyReviews.push(review);
  const status = question.querySelector('[data-fol-key-status]');
  const confirm = question.querySelector('[data-fol-key-confirm]');
  const reject = question.querySelector('[data-fol-key-reject]');
  const manual = question.querySelector('[data-fol-key-manual]');
  const setState = state => {
    review.state = state;
    question.dataset.keyState = state;
    question.querySelector('[data-fol-key-step]').textContent = {
      confirmed: 'Next, review the student readings below.',
      rejected: 'Pause here: check with the instructor about this answer key.',
      manual: 'Grade the responses below manually.',
    }[state];
    question.querySelector('[data-fol-key-reveal]').textContent = state === 'confirmed'
      ? 'Answer key confirmed — view again' : 'Check the suggested answer';
    if (state === 'confirmed' || state === 'manual') question.querySelector('.fol-answer-reveal').open = false;
    status.dataset.state = state;
    status.textContent = {
      confirmed: context.score_rule
        ? 'Key confirmed. Now compare each scan with its bold reading; scores and explanations update as you correct choices.'
        : 'Key confirmed. Now compare each scan with its bold reading; explanations update as you correct choices. Enter scores manually.',
      rejected: 'Email the instructor with this question and version before using this key. Automatic grades and explanations are off; manual grading is available.',
      manual: 'Manual grading: enter scores and comments yourself. Automatic grades and explanations are off.',
    }[state];
    confirm.textContent = state === 'confirmed' ? 'Key confirmed' : context.score_rule
      ? 'Answer is correct — prefill grades' : 'Answer is correct — suggest feedback';
    confirm.disabled = state === 'confirmed';
    for (const listener of review.listeners) listener();
  };
  try {
    const truth = explainTruth(context.formula, context.model);
    if (truth.value !== context.truth) throw new Error('Python and browser truth checks disagree.');
    const candidates = context.asks_object && truth.requiredEvidence
      ? context.model.domain.filter(value => explainCandidate(context.formula, context.model, value).valid).join(', ')
      : 'N/A';
    question.querySelector('[data-fol-truth-expected]').textContent =
      `${truth.value ? 'True' : 'False'}${context.asks_object ? '; written response: ' + candidates : ''}`;
    question.querySelector('[data-fol-truth-explanation]').textContent = truth.text;
    confirm.addEventListener('click', () => setState('confirmed'));
    reject.addEventListener('click', () => setState('rejected'));
  } catch (error) {
    review.state = 'unavailable';
    confirm.disabled = true;
    question.querySelector('[data-fol-truth-expected]').textContent = 'Check manually';
    question.querySelector('[data-fol-truth-explanation]').textContent = error.message;
    status.textContent = 'The answer key could not be checked. Email the instructor; manual grading is available.';
  }
  manual.addEventListener('click', () => setState('manual'));
}

for (const panel of document.querySelectorAll('[data-fol-truth-panel]')) {
  const context = JSON.parse(panel.dataset.context);
  const saved = JSON.parse(panel.dataset.review || '{}');
  const keyReview = keyReviews.find(review => review.context.version === context.version &&
    review.context.formula === context.formula && JSON.stringify(review.context.model) === JSON.stringify(context.model));
  const truthSelect = panel.querySelector('[data-fol-truth-confirmed]');
  const objectSelect = panel.querySelector('[data-fol-truth-object]');
  const rawInput = panel.querySelector('[data-fol-truth-raw]');
  const storage = panel.querySelector('[data-fol-truth-storage]');
  const verdict = panel.querySelector('[data-fol-truth-verdict]');
  const stale = panel.querySelector('[data-fol-truth-stale]');
  const proposal = panel.querySelector('[data-fol-truth-proposal]');
  const choiceButtons = panel.querySelectorAll('[data-fol-choice]');
  const missing = panel.querySelector('[data-fol-truth-missing]');
  const details = panel.querySelector('[data-fol-truth-details]');
  const prepare = panel.querySelector('[data-fol-truth-prepare]');
  const strip = panel.closest('.strip');
  const versionSelect = strip.querySelector('select[name^="quiz_version_"]');
  const scoreInput = strip.querySelector('.score-input');
  const commentInput = strip.querySelector('.comment-input');
  const bboxUpdate = strip.querySelector('input[name^="bbox_update_"]');
  const pageUpdate = strip.querySelector('input[name^="page_update_"]');
  let fingerprint = saved.fingerprint || context.fingerprint;
  let appliedScore = null;

  function setObjectReading(value) {
    if (!objectSelect) return;
    if (value && objectSelect.options && ![...objectSelect.options].some(option => option.value === value)) {
      objectSelect.add(new Option(objectLabel(value), value));
    }
    objectSelect.value = value || '';
  }

  proposal.textContent = saved.proposed_truth || saved.proposed_object
    ? `Truth: ${saved.proposed_truth || 'unclear'}; written: ${saved.proposed_raw_object || saved.proposed_object || 'unclear'}`
    : 'None prepared';
  const fresh = !saved.fingerprint || saved.fingerprint === context.fingerprint;
  truthSelect.value = saved.confirmed_truth ?? (fresh ? saved.proposed_truth : '') ?? '';
  setObjectReading(saved.confirmed_object ?? (fresh ? saved.proposed_object : '') ?? '');
  if (rawInput) rawInput.value = saved.raw_object ?? (fresh ? saved.proposed_raw_object : '') ?? '';

  let previousDraft = null;
  let initial = true;
  function render(updateFields = false) {
    const changedVersion = versionSelect && (versionSelect.value || null) !== (context.version || null);
    const sourceChanged = Boolean(bboxUpdate?.value.trim()) ||
      (pageUpdate && pageUpdate.value !== context.source_filename);
    const isStale = fingerprint !== context.fingerprint || changedVersion || sourceChanged;
    if (prepare) prepare.disabled = Boolean(changedVersion || sourceChanged);
    stale.hidden = !isStale;
    const needsReading = !['true', 'false', 'blank'].includes(truthSelect.value) ||
      (context.asks_object && !clearObjectAnswer(objectSelect.value));
    panel.dataset.needsReading = String(Boolean(isStale || needsReading));
    missing.hidden = isStale || !needsReading;
    const display = value => ({true: 'True', false: 'False', na: 'N/A', blank: 'Blank', unclear: 'Other / unclear'}[value] || objectLabel(value) || 'Needs reading');
    panel.querySelector('[data-fol-truth-interpreted]').textContent =
      !truthSelect.value && context.asks_object && !objectSelect.value ? 'Needs reading' :
        `${display(truthSelect.value)}${context.asks_object ? ' · ' + display(objectSelect.value) : ''}`;
    for (const button of choiceButtons) {
      const value = button.dataset.folChoice === 'truth' ? truthSelect.value : objectSelect?.value;
      button.setAttribute('aria-pressed', String(button.dataset.value === value ||
        (button.dataset.value === 'unclear' && value === 'blank')));
    }
    storage.value = JSON.stringify({
      proposed_truth: saved.proposed_truth || null,
      proposed_object: saved.proposed_object || null,
      proposed_raw_object: saved.proposed_raw_object || null,
      confirmed_truth: truthSelect.value || null,
      confirmed_object: objectSelect?.value || null,
      raw_object: rawInput?.value || null,
      fingerprint,
    });
    if (isStale || keyReview?.state !== 'confirmed') {
      if (previousDraft && commentInput.value.startsWith(previousDraft)) {
        commentInput.value = commentInput.value.slice(previousDraft.length).trimStart();
      }
      if (appliedScore !== null && scoreInput?.value === appliedScore) {
        scoreInput.value = '';
        appliedScore = null;
      }
      verdict.textContent = isStale ? 'Reconfirm this reading after saving the corrected version or crop.' :
        'Automatic scores and explanations wait for answer-key confirmation above. Manual grading is available.';
      return;
    }
    try {
      const truth = explainTruth(context.formula, context.model);
      if (truth.value !== context.truth) throw new Error('Python and browser truth checks disagree.');
      const notes = [];
      const messages = [];
      const truthAnswer = truthSelect.value;
      if (truthAnswer === 'true' || truthAnswer === 'false') {
        const correct = (truthAnswer === 'true') === truth.value;
        notes.push(`Truth value: ${correct ? 'correct' : 'incorrect'}.`);
        if (!correct) messages.push(explainTruth(context.formula, context.model, {claimedTruth: truthAnswer === 'true'}).text);
      } else {
        notes.push(`Truth value: ${truthAnswer || 'unreviewed'}; human review needed.`);
      }
      if (context.asks_object) {
        const expected = truth.requiredEvidence;
        const objectAnswer = objectSelect.value;
        if (!objectAnswer || objectAnswer === 'unclear') {
          notes.push('Written response needs review.');
        } else if (objectAnswer === 'blank') {
          notes.push('Written response blank: 0 written-response points.');
          messages.push(expected ? `A ${expected} is required for this sentence.` : 'N/A is expected here; a blank response receives no written-response credit.');
        } else if (!expected) {
          notes.push(objectAnswer === 'na' ? 'N/A: correct.' : 'Written response: N/A expected.');
          if (objectAnswer !== 'na') messages.push('N/A is expected here because no witness or counterexample is required.');
        } else if (objectAnswer === 'na') {
          notes.push(`${expected}: N/A is incorrect.`);
          messages.push(`A ${expected} is required for this sentence.`);
        } else {
          const candidate = explainCandidate(context.formula, context.model, candidateValue(objectAnswer).value);
          notes.push(`${expected}: ${candidate.valid ? 'correct' : 'incorrect'}.`);
          if (!candidate.valid) messages.push(candidate.text);
        }
        const raw = rawInput.value.trim();
        if (raw && readingFromWriting(context.model, raw) === 'unclear') {
          notes.push('The written form needs TF interpretation; check the confirmed choice against the scan.');
        }
      }
      verdict.textContent = notes.join(' ');
      const explanationText = messages.join('\n\n');
      const draft = explanationText ? `[${explanationText}]` : '';
      const result = scoreConfirmed(context, truth, truthAnswer, objectSelect?.value);
      if (initial || updateFields) {
        if (scoreInput && context.score_rule && (updateFields || scoreInput.value === '')) {
          scoreInput.value = result ? String(result.points) : '';
          appliedScore = scoreInput.value;
          scoreInput.dispatchEvent(new Event('input', {bubbles: true}));
        }
        // Replace only our generated prefix; retain anything the TF added.
        let extra = commentInput.value;
        if (previousDraft && extra.startsWith(previousDraft)) extra = extra.slice(previousDraft.length).trimStart();
        // Recognize the earlier unlabelled default without relabelling custom feedback.
        const legacyPrefixes = explanationText ? [`[Default explainer (autogenerated)]\n${explanationText}`, explanationText] : [];
        const legacyPrefix = initial && legacyPrefixes.find(prefix => extra.startsWith(prefix));
        const legacyDefault = Boolean(legacyPrefix);
        if (legacyPrefix) extra = extra.slice(legacyPrefix.length).trimStart();
        if (extra.startsWith('[TF comment]\n')) extra = extra.slice('[TF comment]\n'.length).trimStart();
        if (updateFields || !extra || legacyDefault) {
          commentInput.value = [draft, extra].filter(Boolean).join('\n\n');
        }
      }
      previousDraft = draft;
      initial = false;
    } catch (error) {
      panel.dataset.needsReading = 'true';
      missing.hidden = false;
      missing.textContent = `Check manually: ${error.message}`;
      verdict.textContent = missing.textContent;
    }
  }
  for (const input of [truthSelect, objectSelect]) {
    input?.addEventListener('input', () => { fingerprint = context.fingerprint; render(true); });
  }
  rawInput?.addEventListener('input', () => {
    setObjectReading(readingFromWriting(context.model, rawInput.value));
    fingerprint = context.fingerprint;
    render(true);
  });
  versionSelect?.addEventListener('change', () => render());
  for (const button of choiceButtons) {
    button.addEventListener('click', () => {
      const input = button.dataset.folChoice === 'truth' ? truthSelect : objectSelect;
      input.value = button.dataset.value;
      if (button.dataset.value === 'unclear') details.open = true;
      input.dispatchEvent(new Event('input', {bubbles: true}));
    });
  }
  strip.closest('form')?.addEventListener('submit', () => render());
  keyReview?.listeners.push(() => {
    if (keyReview.state === 'confirmed') initial = true;
    render();
  });
  prepare?.addEventListener('click', async () => {
    prepare.disabled = true;
    verdict.textContent = 'Reading this one crop with vision…';
    try {
      const body = new FormData();
      body.set('unit_id', prepare.dataset.unitId);
      body.set('force', 'true');
      const response = await fetch(`/problem/${encodeURIComponent(prepare.dataset.problemId)}/fol-truth/prepare-reading`,
        {method: 'POST', body});
      const result = await response.json();
      if (!response.ok) throw new Error(result.detail || 'Vision reading failed.');
      Object.assign(saved, result.proposal);
      truthSelect.value = saved.proposed_truth || '';
      setObjectReading(saved.proposed_object || '');
      if (rawInput) rawInput.value = saved.proposed_raw_object || '';
      fingerprint = context.fingerprint;
      proposal.textContent = `Truth: ${saved.proposed_truth || 'unclear'}; written: ${saved.proposed_raw_object || saved.proposed_object || 'unclear'}`;
      render(true);
      prepare.disabled = false;
    } catch (error) {
      verdict.textContent = error.message;
      prepare.disabled = false;
    }
  });
  render();
}
