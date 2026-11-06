import assert from 'node:assert/strict';
import test from 'node:test';
import { explainTruth } from '../app/assets/vendor/fol-explanations/index.mjs';
import { versions, questions } from '../app/assets/vendor/fol-explanations/examples/quiz9.mjs';
import { scoreConfirmed } from '../app/assets/fol_truth_scoring.mjs';

function check(version, number, truthAnswer, objectAnswer) {
  const q = questions[number - 1];
  const model = versions[version];
  const truth = explainTruth(q.formula, model);
  const context = {formula: q.formula, model, asks_object: q.objectRequested,
    score_rule: q.objectRequested
      ? {truth: 1, object: 1, blank_na_credit: false}
      : {truth: number === 1 ? 2 : 1}};
  return scoreConfirmed(context, truth, truthAnswer, objectAnswer)?.points ?? null;
}

test('approved Quiz 9 policy separates truth and written credit', () => {
  assert.equal(check('A', 1, 'false'), 2);
  assert.equal(check('A', 1, 'true'), 0);
  assert.equal(check('A', 2, 'true', '1'), 2);
  assert.equal(check('A', 2, 'false', '1'), 1);
  assert.equal(check('A', 2, 'true', '2'), 1);
  assert.equal(check('A', 2, 'true', 'na'), 1);
  assert.equal(check('A', 2, 'true', 'blank'), 1);
  assert.equal(check('B', 2, 'false', 'na'), 2);
  assert.equal(check('B', 2, 'false', 'blank'), 1);
  assert.equal(check('C', 5, 'false', '1'), 2);
  assert.equal(check('C', 5, 'false', '2'), 1);
  assert.equal(check('C', 4, 'true', 'na'), 2);
  assert.equal(check('B', 3, 'true', '1'), 2);
  assert.equal(check('B', 3, 'true', '2'), 2);
  assert.equal(check('B', 3, 'true', '3'), 2);
  assert.equal(check('A', 6, 'true'), 1);
  assert.equal(check('B', 7, 'false'), 1);
});

test('unclear or unconfirmed fields produce no score suggestion', () => {
  assert.equal(check('A', 2, 'unclear', '1'), null);
  assert.equal(check('A', 2, 'true', 'unclear'), null);
  assert.equal(check('A', 2, 'true', ''), null);
  assert.equal(check('A', 2, '', '1'), null);
});
