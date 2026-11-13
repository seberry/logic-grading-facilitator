import assert from 'node:assert/strict';
import test from 'node:test';
import { readFileSync } from 'node:fs';
import { explainTruth, explainCandidate } from '../app/assets/vendor/fol-explanations/index.mjs';
import { scoreConfirmed } from '../app/assets/fol_truth_scoring.mjs';
import { candidateValue, objectToken, readingFromWriting } from '../app/assets/fol_truth_answers.mjs';

// The Python checks exercise the same source formulas/models and independent expected results.
const cases = JSON.parse(readFileSync(new URL('./fixtures/fol_truth_cases.json', import.meta.url)));
for (const fixture of cases) {
  test(`general FOL truth and evidence: ${fixture.name}`, () => {
    const {formula, model} = fixture;
    const result = explainTruth(formula, model);
    assert.equal(result.value, fixture.truth);
    const candidates = fixture.asks_object === false ? [] : model.domain.filter(value => explainCandidate(formula, model, value).valid);
    assert.deepEqual(candidates, fixture.candidates);
    const context = {formula, model, asks_object: fixture.asks_object !== false,
      score_rule: {mode: 'all_or_nothing', points: 1}};
    const truthAnswer = String(fixture.truth);
    const evidence = result.requiredEvidence ? objectToken(candidates[0]) : 'na';
    assert.equal(scoreConfirmed(context, result, truthAnswer, evidence).points, 1);
    assert.equal(scoreConfirmed(context, result, String(!fixture.truth), evidence).points, 0);
    assert.equal(scoreConfirmed(context, result, 'unclear', evidence), null);
    if (context.asks_object) {
      assert.equal(scoreConfirmed(context, result, truthAnswer, 'blank').points, 0);
      assert.equal(scoreConfirmed(context, result, truthAnswer, 'unclear'), null);
      if (result.requiredEvidence) {
        assert.equal(scoreConfirmed(context, result, truthAnswer, '999').points, 0);
        assert.match(explainCandidate(formula, model, 999).text, /not an object in the domain/);
      }
    }
  });
}

test('component scoring and feedback-only mode are independent of the model', () => {
  const {formula, model} = cases[0];
  const truth = explainTruth(formula, model);
  const context = {formula, model, asks_object: true, score_rule: {truth: 0.5, object: 1.5}};
  assert.equal(scoreConfirmed(context, truth, 'false', '4').points, 1.5);
  assert.equal(scoreConfirmed(context, truth, 'true', '7').points, 0.5);
  assert.equal(scoreConfirmed(context, truth, 'true', '4').points, 2);
  context.score_rule = null;
  assert.equal(scoreConfirmed(context, truth, 'true', '4'), null);
  assert.equal(explainTruth(formula, model, {claimedTruth: false}).value, true);
});

test('written readings distinguish named objects from N/A, blank, and numeric objects', () => {
  const model = {domain: ['na', 'blank', 'red', -3, 0]};
  assert.equal(readingFromWriting(model, 'na'), 'object:"na"');
  assert.equal(readingFromWriting(model, 'blank'), 'object:"blank"');
  assert.equal(readingFromWriting(model, 'N/A'), 'na');
  assert.equal(readingFromWriting(model, ''), 'blank');
  assert.equal(readingFromWriting(model, '-3'), '-3');
  assert.equal(readingFromWriting(model, '99'), '99');
  assert.equal(readingFromWriting(model, 'red or blue'), 'unclear');
  assert.equal(readingFromWriting({domain: [1, '1']}, '1'), 'unclear');
  assert.deepEqual(candidateValue('object:"blank"'), {clear: true, value: 'blank'});
  assert.deepEqual(candidateValue('blank'), {clear: false});
  assert.deepEqual(candidateValue('9007199254740992'), {clear: false});
});
