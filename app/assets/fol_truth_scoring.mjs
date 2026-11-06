import { explainCandidate } from './vendor/fol-explanations/index.mjs';

/** Return null until both requested fields have a clear TF-confirmed reading. */
export function scoreConfirmed(context, truth, truthAnswer, objectAnswer) {
  const rule = context.score_rule;
  if (!rule || !['true', 'false', 'blank'].includes(truthAnswer)) return null;
  if (context.asks_object && (!objectAnswer || objectAnswer === 'unclear')) return null;
  let points = truthAnswer !== 'blank' && (truthAnswer === 'true') === truth.value
    ? rule.truth : 0;
  if (context.asks_object) {
    let correct = false;
    if (!truth.requiredEvidence) {
      correct = objectAnswer === 'na' ||
        (objectAnswer === 'blank' && rule.blank_na_credit === true);
    } else if (['1', '2', '3'].includes(objectAnswer)) {
      correct = explainCandidate(context.formula, context.model, Number(objectAnswer)).valid;
    }
    if (correct) points += rule.object;
  }
  return {points, maximum: rule.truth + (rule.object || 0)};
}
