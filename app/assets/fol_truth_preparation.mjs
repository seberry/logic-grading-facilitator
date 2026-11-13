import {explainTruth, explainCandidate} from './vendor/fol-explanations/index.mjs';

const approve = document.getElementById('approve-definitions');
const status = document.getElementById('definition-status');
for (const panel of document.querySelectorAll('[data-definition-preview]')) {
  try {
    const context = JSON.parse(panel.dataset.context);
    const truth = explainTruth(context.formula, context.model);
    if (truth.value !== context.truth) throw new Error('The independent truth checks disagree.');
    const candidates = context.asks_object && truth.requiredEvidence
      ? context.model.domain.filter(value => explainCandidate(context.formula, context.model, value).valid).join(', ')
      : 'N/A';
    panel.querySelector('[data-definition-result]').textContent = `${truth.value ? 'True' : 'False'}${context.asks_object ? '; written response: ' + candidates : ''}`;
    panel.querySelector('[data-definition-explanation]').textContent = truth.text;
  } catch (error) {
    panel.querySelector('[data-definition-result]').textContent = 'Needs correction';
    panel.querySelector('[data-definition-explanation]').textContent = error.message;
    approve.disabled = true;
    status.textContent = 'A definition could not be checked. Correct it before approving.';
  }
}
for (const event of ['input', 'change']) document.getElementById('definitions-form').addEventListener(event, () => {
  approve.disabled = true;
  status.textContent = 'Definitions changed. Click Check definitions before approving.';
  for (const panel of document.querySelectorAll('[data-definition-preview]')) panel.hidden = true;
});
document.getElementById('read-blank-form').addEventListener('submit', event => {
  event.target.querySelector('button').disabled = true;
  document.getElementById('reading-status').textContent = 'Reading the blank quiz with IU…';
});
