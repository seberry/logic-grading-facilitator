import { explainCandidate } from './vendor/fol-explanations/index.mjs';
import { candidateValue, clearObjectAnswer } from './fol_truth_answers.mjs';

/** Return null until both requested fields have a clear TF-confirmed reading. */
export function scoreConfirmed(context, truth, truthAnswer, objectAnswer) {
  const rule = context.score_rule;
  if (!rule || !['true', 'false', 'blank'].includes(truthAnswer)) return null;
  if (context.asks_object && !clearObjectAnswer(objectAnswer)) return null;
  const truthCorrect = truthAnswer !== 'blank' && (truthAnswer === 'true') === truth.value;
  let objectCorrect = true;
  if (context.asks_object) {
    let correct = false;
    if (!truth.requiredEvidence) {
      correct = objectAnswer === 'na' ||
        (objectAnswer === 'blank' && rule.blank_na_credit === true);
    } else if (candidateValue(objectAnswer).clear) {
      correct = explainCandidate(context.formula, context.model, candidateValue(objectAnswer).value).valid;
    }
    objectCorrect = correct;
  }
  if (rule.mode === 'all_or_nothing') {
    return {points: truthCorrect && objectCorrect ? rule.points : 0, maximum: rule.points};
  }
  const points = (truthCorrect ? rule.truth : 0) + (context.asks_object && objectCorrect ? (rule.object || 0) : 0);
  return {points, maximum: rule.truth + (rule.object || 0)};
}
