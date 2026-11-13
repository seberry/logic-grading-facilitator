// Reading tokens are distinct from domain objects (including an object named "na").
export function objectToken(value) {
  return typeof value === 'number' ? String(value) : 'object:' + JSON.stringify(value);
}

export function candidateValue(answer) {
  if (typeof answer !== 'string') return {clear: false};
  if (/^-?(?:0|[1-9]\d*)$/.test(answer) && Number.isSafeInteger(Number(answer))) {
    return {clear: true, value: Number(answer)};
  }
  if (answer.startsWith('object:')) {
    try {
      const value = JSON.parse(answer.slice(7));
      if (typeof value === 'string' && value) return {clear: true, value};
    } catch { /* An invalid token needs human interpretation. */ }
  }
  return {clear: false};
}

export function clearObjectAnswer(answer) {
  return ['na', 'blank'].includes(answer) || candidateValue(answer).clear;
}

export function readingFromWriting(model, writing) {
  const text = writing.trim();
  if (!text) return 'blank';
  const matches = model.domain.filter(value => String(value) === text);
  if (matches.length === 1) return objectToken(matches[0]);
  if (matches.length > 1) return 'unclear';
  if (/^n\s*\/?\s*a$/i.test(text)) return 'na';
  if (candidateValue(text).clear) return text;
  // Quoted strings can explicitly represent an object outside a named domain.
  if (candidateValue('object:' + text).clear) return 'object:' + text;
  return 'unclear';
}

export function objectLabel(answer) {
  const candidate = candidateValue(answer);
  return candidate.clear ? String(candidate.value) : answer;
}
