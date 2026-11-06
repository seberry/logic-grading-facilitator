import { explainTruth, explainCandidate } from './vendor/fol-explanations/index.mjs';
import { scoreConfirmed } from './fol_truth_scoring.mjs';

for (const panel of document.querySelectorAll('[data-fol-truth-panel]')) {
  const context = JSON.parse(panel.dataset.context);
  const saved = JSON.parse(panel.dataset.review || '{}');
  const truthSelect = panel.querySelector('[data-fol-truth-confirmed]');
  const objectSelect = panel.querySelector('[data-fol-truth-object]');
  const rawInput = panel.querySelector('[data-fol-truth-raw]');
  const storage = panel.querySelector('[data-fol-truth-storage]');
  const verdict = panel.querySelector('[data-fol-truth-verdict]');
  const explanation = panel.querySelector('[data-fol-truth-explanation]');
  const draft = panel.querySelector('[data-fol-truth-draft]');
  const stale = panel.querySelector('[data-fol-truth-stale]');
  const proposal = panel.querySelector('[data-fol-truth-proposal]');
  const proposalScore = panel.querySelector('[data-fol-truth-proposal-score]');
  const acceptProposal = panel.querySelector('[data-fol-truth-accept-proposal]');
  const apply = panel.querySelector('[data-fol-truth-apply]');
  const scoreDisplay = panel.querySelector('[data-fol-truth-score]');
  const applyScore = panel.querySelector('[data-fol-truth-apply-score]');
  const prepare = panel.querySelector('[data-fol-truth-prepare]');
  const strip = panel.closest('.strip');
  const versionSelect = strip.querySelector('select[name^="quiz_version_"]');
  const scoreInput = strip.querySelector('.score-input');
  const commentInput = strip.querySelector('.comment-input');
  const bboxUpdate = strip.querySelector('input[name^="bbox_update_"]');
  const pageUpdate = strip.querySelector('input[name^="page_update_"]');
  let fingerprint = saved.fingerprint || context.fingerprint;
  let appliedDraft = null;
  let suggestedScore = null;
  let appliedScore = null;

  proposal.textContent = saved.proposed_truth || saved.proposed_object
    ? `Truth: ${saved.proposed_truth || 'unclear'}; written: ${saved.proposed_raw_object || saved.proposed_object || 'unclear'}`
    : 'None prepared';
  truthSelect.value = saved.confirmed_truth || '';
  if (objectSelect) objectSelect.value = saved.confirmed_object || '';
  if (rawInput) rawInput.value = saved.raw_object || '';

  function render() {
    const changedVersion = versionSelect && versionSelect.value !== context.version;
    const sourceChanged = Boolean(bboxUpdate?.value.trim()) ||
      (pageUpdate && pageUpdate.value !== context.source_filename);
    const isStale = fingerprint !== context.fingerprint || changedVersion || sourceChanged;
    if (prepare) prepare.disabled = Boolean(changedVersion || sourceChanged);
    stale.hidden = !isStale;
    storage.value = JSON.stringify({
      proposed_truth: saved.proposed_truth || null,
      proposed_object: saved.proposed_object || null,
      proposed_raw_object: saved.proposed_raw_object || null,
      confirmed_truth: truthSelect.value || null,
      confirmed_object: objectSelect?.value || null,
      raw_object: rawInput?.value || null,
      fingerprint,
    });
    if (isStale) {
      if (proposalScore) proposalScore.textContent = 'Stale reading';
      if (acceptProposal) acceptProposal.disabled = true;
      if (appliedDraft !== null && commentInput.value === appliedDraft) {
        commentInput.value = '';
        appliedDraft = null;
      }
      if (appliedScore !== null && scoreInput?.value === appliedScore) {
        scoreInput.value = '';
        appliedScore = null;
      }
      verdict.textContent = 'Reconfirm this reading after saving the corrected version or crop.';
      if (scoreDisplay) scoreDisplay.textContent = 'Stale reading';
      if (applyScore) applyScore.disabled = true;
      explanation.textContent = '';
      draft.value = '';
      apply.disabled = true;
      return;
    }
    try {
      const truth = explainTruth(context.formula, context.model);
      if (truth.value !== context.truth) throw new Error('Python and browser truth checks disagree.');
      const preliminary = scoreConfirmed(context, truth, saved.proposed_truth, saved.proposed_object);
      if (proposalScore) proposalScore.textContent = preliminary
        ? `${preliminary.points} / ${preliminary.maximum} (unconfirmed)` : 'Needs a clear reading';
      if (acceptProposal) acceptProposal.disabled = !preliminary;
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
          notes.push('Written response blank; score policy pending.');
        } else if (!expected) {
          notes.push(objectAnswer === 'na' ? 'N/A: correct.' : 'Written response: N/A expected.');
          if (objectAnswer !== 'na') messages.push('N/A is expected here because no witness or counterexample is required.');
        } else if (objectAnswer === 'na') {
          notes.push(`${expected}: N/A is incorrect.`);
          messages.push(`A ${expected} is required for this sentence.`);
        } else {
          const candidate = explainCandidate(context.formula, context.model, Number(objectAnswer));
          notes.push(`${expected}: ${candidate.valid ? 'correct' : 'incorrect'}.`);
          if (!candidate.valid) messages.push(candidate.text);
        }
        const raw = rawInput.value.trim();
        if (raw && !/^(?:[123]|n\s*\/?\s*a)$/i.test(raw)) {
          notes.push('The written form needs TF interpretation; check the confirmed choice against the scan.');
        }
      }
      verdict.textContent = notes.join(' ');
      if (context.score_rule && scoreDisplay && applyScore) {
        const result = scoreConfirmed(context, truth, truthAnswer, objectSelect?.value);
        if (result) {
          suggestedScore = String(result.points);
          scoreDisplay.textContent = `${result.points} / ${result.maximum}`;
          applyScore.disabled = false;
        } else {
          suggestedScore = null;
          scoreDisplay.textContent = 'Needs confirmed reading';
          applyScore.disabled = true;
        }
      }
      explanation.textContent = truth.text;
      draft.value = messages.join('\n\n');
      apply.disabled = !draft.value || !['true', 'false'].includes(truthAnswer) ||
        (context.asks_object && !objectSelect.value) ||
        (context.asks_object && ['unclear', 'blank'].includes(objectSelect.value));
    } catch (error) {
      if (proposalScore) proposalScore.textContent = 'Assistance unavailable';
      if (acceptProposal) acceptProposal.disabled = true;
      verdict.textContent = `Assistance unavailable: ${error.message}`;
      explanation.textContent = '';
      draft.value = '';
      apply.disabled = true;
      suggestedScore = null;
      if (scoreDisplay) scoreDisplay.textContent = 'Assistance unavailable';
      if (applyScore) applyScore.disabled = true;
    }
  }
  for (const input of [truthSelect, objectSelect, rawInput]) {
    input?.addEventListener('input', () => { fingerprint = context.fingerprint; render(); });
  }
  versionSelect?.addEventListener('change', render);
  apply.addEventListener('click', () => {
    if (apply.disabled) return;
    commentInput.value = draft.value;
    appliedDraft = draft.value;
    commentInput.dispatchEvent(new Event('input', {bubbles: true}));
    scoreInput?.focus();
  });
  applyScore?.addEventListener('click', () => {
    if (applyScore.disabled || suggestedScore === null) return;
    scoreInput.value = suggestedScore;
    appliedScore = suggestedScore;
    scoreInput.dispatchEvent(new Event('input', {bubbles: true}));
  });
  acceptProposal?.addEventListener('click', () => {
    if (acceptProposal.disabled) return;
    truthSelect.value = saved.proposed_truth;
    if (objectSelect) objectSelect.value = saved.proposed_object || '';
    if (rawInput) rawInput.value = saved.proposed_raw_object || '';
    fingerprint = context.fingerprint;
    render();
    applyScore?.click();
  });
  strip.closest('form')?.addEventListener('submit', render);
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
      proposal.textContent = `Truth: ${saved.proposed_truth || 'unclear'}; written: ${saved.proposed_raw_object || saved.proposed_object || 'unclear'}`;
      render();
      prepare.disabled = false;
    } catch (error) {
      verdict.textContent = error.message;
      prepare.disabled = false;
    }
  });
  render();
}
