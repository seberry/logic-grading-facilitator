import test from 'node:test';
import assert from 'node:assert/strict';
import { versions, questions } from '../app/assets/vendor/fol-explanations/examples/quiz9.mjs';

class Field extends EventTarget {
  value = ''; textContent = ''; dataset = {}; hidden = false; disabled = false;
  attributes = {};
  setAttribute(name, value) { this.attributes[name] = value; }
}

async function mount(saved, score = '', comment = '', confirmKey = true) {
  const fields = new Map();
  const get = selector => {
    if (!fields.has(selector)) fields.set(selector, new Field());
    return fields.get(selector);
  };
  get('.score-input').value = score;
  get('.comment-input').value = comment;
  get('select[name^="quiz_version_"]').value = 'A';
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
      score_rule: {truth: 1, object: 1, blank_na_credit: false}}), review: JSON.stringify(saved)},
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
