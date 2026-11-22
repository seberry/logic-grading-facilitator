import {explainTruth, explainCandidate} from './vendor/fol-explanations/index.mjs';

const approve = document.getElementById('approve-definitions');
const status = document.getElementById('definition-status');
for (const panel of document.querySelectorAll('[data-definition-preview]')) {
  const summary = panel.closest('[data-question-id]').querySelector('[data-summary-result]');
  try {
    const context = JSON.parse(panel.dataset.context);
    const truth = explainTruth(context.formula, context.model);
    if (truth.value !== context.truth) throw new Error('The independent truth checks disagree.');
    const candidates = context.asks_object && truth.requiredEvidence
      ? context.model.domain.filter(value => explainCandidate(context.formula, context.model, value).valid).join(', ')
      : 'N/A';
    const verdict = truth.value ? 'True' : 'False';
    const evidence = truth.requiredEvidence ? ' · ' + truth.requiredEvidence : '';
    summary.textContent = ' · ' + verdict + (context.asks_object ? evidence : '');
    panel.querySelector('[data-definition-result]').textContent = verdict +
      (context.asks_object ? ' · Written response: ' + candidates : '');
    panel.querySelector('[data-definition-explanation]').textContent = truth.text;
  } catch (error) {
    summary.textContent = ' · Needs correction';
    panel.querySelector('[data-definition-result]').textContent = 'Needs correction';
    panel.querySelector('[data-definition-explanation]').textContent = error.message;
    approve.disabled = true;
    status.textContent = 'A definition could not be checked. Correct it before approving.';
  }
}
const form = document.getElementById('definitions-form');
const policy = document.getElementById('score-policy');
for (const event of ['input', 'change']) form.addEventListener(event, () => {
  approve.disabled = true;
  status.textContent = 'Definitions changed. Click Check definitions before approving.';
  for (const panel of document.querySelectorAll('[data-definition-preview]')) panel.hidden = true;
  for (const summary of form.querySelectorAll('[data-summary-result]')) summary.textContent = '';
});
policy.addEventListener('change', () => {
  for (const settings of form.querySelectorAll('[data-question-scoring]')) {
    settings.hidden = policy.value !== 'keep';
  }
});
for (const button of form.querySelectorAll('[data-apply-formula]')) {
  button.addEventListener('click', () => {
    const source = button.closest('[data-question-id]');
    const formula = source.querySelector('[data-question-formula]').value;
    for (const row of form.querySelectorAll('[data-question-id]')) {
      if (row.dataset.questionId !== source.dataset.questionId ||
          !row.querySelector('input[name^="enabled_"]').checked) continue;
      const input = row.querySelector('[data-question-formula]');
      input.value = formula;
      input.dispatchEvent(new Event('input', {bubbles: true}));
    }
  });
}
form.addEventListener('input', event => {
  if (!event.target.matches('[data-question-formula]')) return;
  const summary = event.target.closest('[data-question-id]').querySelector('[data-summary-formula]');
  if (summary) summary.textContent = event.target.value;
});
document.getElementById('read-blank-form').addEventListener('submit', event => {
  event.target.querySelector('button').disabled = true;
  document.getElementById('reading-status').textContent = 'Reading the blank quiz with IU…';
});
