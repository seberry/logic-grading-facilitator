import test from 'node:test';
import assert from 'node:assert/strict';
import { versions, questions } from '../app/assets/vendor/fol-explanations/examples/quiz9.mjs';

class Field extends EventTarget {
  value = ''; textContent = ''; dataset = {}; hidden = false; disabled = false;
  attributes = {};
  setAttribute(name, value) { this.attributes[name] = value; }
}

globalThis.Option = class {
  constructor(text, value) { this.text = text; this.value = value; }
};

async function mount(saved, score = '', comment = '', confirmKey = true, contextOverride = {}) {
  const fields = new Map();
  const get = selector => {
    if (!fields.has(selector)) fields.set(selector, new Field());
    return fields.get(selector);
  };
  get('.score-input').value = score;
  get('.comment-input').value = comment;
  get('[data-fol-truth-object]').options = ['', '1', '2', '3', 'na', 'blank', 'unclear']
    .map(value => new Option(value, value));
  get('[data-fol-truth-object]').add = option => get('[data-fol-truth-object]').options.push(option);
  get('select[name^="quiz_version_"]').value = 'version' in contextOverride ? contextOverride.version || '' : 'A';
  get('input[name^="page_update_"]').value = 'fake.jpg';
  const form = new Field();
  const strip = {querySelector: get, closest: () => form};
  const buttons = ['truth:true', 'truth:false', 'truth:unclear', 'object:1', 'object:2', 'object:3', 'object:na', 'object:unclear'].map(key => {
    const button = new Field();
    [button.dataset.folChoice, button.dataset.value] = key.split(':');
    return button;
  });
  const panel = {
    dataset: {context: JSON.stringify({formula: questions[1].formula, model: versions.A,
      truth: true, asks_object: true, version: 'A', fingerprint: 'fresh', source_filename: 'fake.jpg',
      score_rule: {truth: 1, object: 1, blank_na_credit: false}, ...contextOverride}), review: JSON.stringify(saved)},
    querySelector: get, querySelectorAll: () => buttons, closest: () => strip,
  };
  const question = {dataset: panel.dataset, querySelector: get};
  globalThis.document = {querySelectorAll: selector => selector === '[data-fol-truth-question]' ? [question] : [panel]};
  try {
    await import(`../app/assets/fol_truth_review.mjs?test=${Math.random()}`);
  } finally {
    delete globalThis.document;
  }
  if (confirmKey) get('[data-fol-key-confirm]').dispatchEvent(new Event('click'));
  return {get, form, panel, buttons, change(selector, value) {
    get(selector).value = value;
    get(selector).dispatchEvent(new Event('input'));
  }};
}

test('proposals prefill; corrections update scores and feedback while preserving added text', async () => {
  const ui = await mount({fingerprint: 'fresh', proposed_truth: 'false', proposed_object: '2'});
  assert.equal(ui.get('.score-input').value, '0');
  assert.ok(ui.get('.comment-input').value.length > 0);
  assert.ok(ui.get('.comment-input').value.startsWith('['));
  assert.match(ui.get('[data-fol-truth-expected]').textContent, /True; written response: 1/);
  ui.get('.comment-input').value += '\n\nPlease come to office hours.';
  ui.change('[data-fol-truth-confirmed]', 'true');
  assert.equal(ui.get('.score-input').value, '1');
  assert.ok(ui.get('.comment-input').value.endsWith('Please come to office hours.'));
  ui.change('[data-fol-truth-object]', '1');
  assert.equal(ui.get('.score-input').value, '2');
  assert.equal(ui.get('.comment-input').value, 'Please come to office hours.');
  ui.form.dispatchEvent(new Event('submit'));
  const stored = JSON.parse(ui.get('[data-fol-truth-storage]').value);
  assert.equal(stored.confirmed_truth, 'true');
  assert.equal(stored.confirmed_object, '1');
});

test('reload preserves a saved manual score and comment', async () => {
  const ui = await mount({fingerprint: 'fresh', confirmed_truth: 'true', confirmed_object: '1'}, '1.5', 'Human adjustment');
  assert.equal(ui.get('.score-input').value, '1.5');
  assert.equal(ui.get('.comment-input').value, 'Human adjustment');
});

test('existing default feedback gains a label while keeping the TF addition', async () => {
  const saved = {fingerprint: 'fresh', confirmed_truth: 'false', confirmed_object: '2'};
  const original = await mount(saved);
  const unlabelled = original.get('.comment-input').value.slice(1, -1);
  const restored = await mount(saved, '0', unlabelled + '\n\nPlease come to office hours.');
  const comment = restored.get('.comment-input').value;
  assert.ok(comment.startsWith('['));
  assert.ok(comment.endsWith('Please come to office hours.'));
  restored.change('[data-fol-truth-confirmed]', 'true');
  restored.change('[data-fol-truth-object]', '1');
  assert.equal(restored.get('.comment-input').value, 'Please come to office hours.');
});

test('stale and unclear readings do not prefill scores', async () => {
  for (const saved of [
    {fingerprint: 'old', proposed_truth: 'true', proposed_object: '1'},
    {fingerprint: 'fresh', proposed_truth: 'true', proposed_object: 'unclear'},
  ]) {
    const ui = await mount(saved);
    assert.equal(ui.get('.score-input').value, '');
  }
});


test('quick choices mark selection, and Other exposes interpretation details', async () => {
  const ui = await mount({fingerprint: 'fresh', proposed_truth: 'true', proposed_object: '1'});
  assert.equal(ui.panel.dataset.needsReading, 'false');
  assert.equal(ui.get('[data-fol-truth-missing]').hidden, true);
  const other = ui.buttons.find(b => b.dataset.folChoice === 'object' && b.dataset.value === 'unclear');
  other.dispatchEvent(new Event('click'));
  assert.equal(ui.panel.dataset.needsReading, 'true');
  assert.equal(ui.get('[data-fol-truth-details]').open, true);
  assert.equal(ui.get('.score-input').value, '');
  assert.equal(other.attributes['aria-pressed'], 'true');
  ui.buttons.find(b => b.dataset.folChoice === 'object' && b.dataset.value === '2').dispatchEvent(new Event('click'));
  assert.equal(ui.get('.score-input').value, '1');
  // A clear wrong answer has neutral reading styling, not an OCR warning.
  assert.equal(ui.panel.dataset.needsReading, 'false');
  assert.equal(other.attributes['aria-pressed'], 'false');
});


test('answer-key confirmation gates automatic grades while OCR review remains usable', async () => {
  const ui = await mount({fingerprint: 'fresh', proposed_truth: 'false', proposed_object: '2'}, '', '', false);
  assert.equal(ui.get('.score-input').value, '');
  assert.equal(ui.get('.comment-input').value, '');
  assert.equal(ui.get('[data-fol-truth-interpreted]').textContent, 'False · 2');
  ui.change('[data-fol-truth-confirmed]', 'true');
  assert.equal(ui.get('.score-input').value, '');
  ui.get('[data-fol-key-confirm]').dispatchEvent(new Event('click'));
  assert.equal(ui.get('.score-input').value, '1');
  assert.match(ui.get('.comment-input').value, /^\[[\s\S]+\]$/);
  ui.get('.comment-input').value += '\n\nMy added feedback.';
  ui.get('[data-fol-key-reject]').dispatchEvent(new Event('click'));
  assert.equal(ui.get('.score-input').value, '');
  assert.equal(ui.get('.comment-input').value, 'My added feedback.');
  assert.match(ui.get('[data-fol-key-status]').textContent, /Email the instructor/);
  ui.change('[data-fol-truth-object]', '1');
  assert.equal(ui.get('.score-input').value, '');
  ui.get('.score-input').value = '1.5';
  ui.get('[data-fol-key-manual]').dispatchEvent(new Event('click'));
  ui.form.dispatchEvent(new Event('submit'));
  assert.equal(ui.get('.score-input').value, '1.5');
});

test('manually filled general-model readings generate feedback and all-or-nothing scores', async () => {
  const ui = await mount({}, '', '', true, {
    formula: '∃x Rxx', truth: true, model: {domain: [2, 4, 7], predicates: {R: [[4, 4]]}},
    score_rule: {mode: 'all_or_nothing', points: 1},
  });
  assert.equal(ui.get('.score-input').value, '');
  ui.change('[data-fol-truth-confirmed]', 'true');
  ui.change('[data-fol-truth-object]', '7');
  assert.equal(ui.get('.score-input').value, '0');
  assert.match(ui.get('.comment-input').value, /7 is not a witness/);
  ui.get('.comment-input').value += '\n\nTF addition.';
  ui.change('[data-fol-truth-raw]', '4');
  assert.equal(ui.get('[data-fol-truth-object]').value, '4');
  assert.equal(ui.get('.score-input').value, '1');
  assert.equal(ui.get('.comment-input').value, 'TF addition.');
});

test('named witnesses can be filled manually, including reserved reading words', async () => {
  const ui = await mount({}, '', '', true, {
    formula: '∃x Px', truth: true, model: {domain: ['na', 'blank'], predicates: {P: ['na']}},
    score_rule: {mode: 'all_or_nothing', points: 1},
  });
  ui.change('[data-fol-truth-confirmed]', 'true');
  ui.change('[data-fol-truth-raw]', 'blank');
  assert.equal(ui.get('.score-input').value, '0');
  assert.equal(ui.get('[data-fol-truth-interpreted]').textContent, 'True · blank');
  ui.change('[data-fol-truth-raw]', 'na');
  assert.equal(ui.get('.score-input').value, '1');
  assert.equal(ui.get('[data-fol-truth-interpreted]').textContent, 'True · na');
  ui.change('[data-fol-truth-raw]', 'na or blank');
  assert.equal(ui.panel.dataset.needsReading, 'true');
  assert.equal(ui.get('.score-input').value, '');
});

test('feedback-only assistance never clears or replaces a manual score', async () => {
  const ui = await mount({}, '0.5', '', true, {
    formula: '∃x Rxx', truth: true, model: {domain: [2, 4], predicates: {R: [[4, 4]]}},
    score_rule: null,
  });
  ui.change('[data-fol-truth-confirmed]', 'false');
  ui.change('[data-fol-truth-object]', '2');
  assert.match(ui.get('.comment-input').value, /You answered false/);
  assert.equal(ui.get('.score-input').value, '0.5');
  ui.change('[data-fol-truth-object]', '4');
  ui.get('[data-fol-key-manual]').dispatchEvent(new Event('click'));
  assert.equal(ui.get('.score-input').value, '0.5');
});

test('an entered object outside the domain is a clear wrong answer and survives the select control', async () => {
  const ui = await mount({}, '', '', true, {
    formula: '∃x Rxx', truth: true, model: {domain: [2, 4], predicates: {R: [[4, 4]]}},
    score_rule: {mode: 'all_or_nothing', points: 1},
  });
  ui.change('[data-fol-truth-confirmed]', 'true');
  ui.change('[data-fol-truth-raw]', '99');
  assert.equal(ui.get('.score-input').value, '0');
  assert.match(ui.get('.comment-input').value, /not an object in the domain/);
  assert.ok(ui.get('[data-fol-truth-object]').options.some(option => option.value === '99'));
  assert.equal(JSON.parse(ui.get('[data-fol-truth-storage]').value).confirmed_object, '99');
});

test('an assessment without printed versions can use assistance', async () => {
  const ui = await mount({}, '', '', true, {version: null});
  ui.change('[data-fol-truth-confirmed]', 'true');
  ui.change('[data-fol-truth-object]', '1');
  assert.equal(ui.get('.score-input').value, '2');
  assert.equal(ui.panel.dataset.needsReading, 'false');
});
