/**
 * Browser equivalent of the fitted sklearn ColumnTransformer and the 14
 * cumulative-threshold LogisticRegression models in Assignment 2.
 * No training, refitting, approximation, or network inference occurs here.
 */

const MISSING = '__MISSING__';

export function encodeResponses(model, responses) {
  const features = new Map();
  const numericValues = {};

  for (const spec of model.numeric) {
    let raw;
    if (spec.name === 'WorkExpSquared') {
      const work = responses.WorkExp;
      raw = work === null || work === '' || work === undefined || !Number.isFinite(Number(work))
        ? null : Number(work) ** 2;
    } else {
      raw = responses[spec.name];
    }
    const missing = raw === null || raw === '' || raw === undefined || !Number.isFinite(Number(raw));
    numericValues[spec.name] = { missing };
    const number = missing ? spec.median : Number(raw);
    features.set(spec.index, (number - spec.mean) / spec.scale);
  }

  for (const spec of model.numericMissing) {
    if (numericValues[spec.name]?.missing) features.set(spec.index, 1);
  }

  for (const spec of model.single) {
    const answer = responses[spec.name];
    const category = answer === null || answer === '' || answer === undefined ? MISSING : String(answer);
    const i = spec.options.indexOf(category);
    // sklearn OneHotEncoder(handle_unknown="ignore") drops unseen categories.
    if (i !== -1) features.set(spec.indices[i], 1);
  }

  for (const spec of model.multi) {
    let values = responses[spec.name];
    if (!Array.isArray(values)) {
      values = values === null || values === undefined || values === ''
        ? [] : String(values).split(';').map(v => v.trim()).filter(Boolean);
    }
    const tokens = values.length ? [...new Set(values.map(String))] : [MISSING];
    const indices = new Map(spec.options.map(option => [option.value, option.index]));
    for (const token of tokens) {
      const index = indices.get(token);
      if (index !== undefined) features.set(index, 1);
    }
  }
  return features;
}

function sigmoid(z) {
  // Stable equivalent to sklearn's expit.
  if (z >= 0) return 1 / (1 + Math.exp(-z));
  const e = Math.exp(z);
  return e / (1 + e);
}

export function predict(model, responses) {
  const active = [...encodeResponses(model, responses).entries()];
  const cumulative = [];
  let previous = 0;
  for (let k = 0; k < model.intercepts.length; k += 1) {
    const weights = model.weights[k];
    let logit = model.intercepts[k];
    for (const [index, value] of active) logit += weights[index] * value;
    const adjusted = Math.max(previous, Math.min(1, Math.max(0, sigmoid(logit))));
    cumulative.push(adjusted);
    previous = adjusted;
  }

  const probabilities = [];
  let lower = 0;
  for (const boundary of [...cumulative, 1]) {
    probabilities.push(Math.max(0, boundary - lower));
    lower = boundary;
  }
  const total = probabilities.reduce((sum, p) => sum + p, 0);
  const normalized = probabilities.map(p => p / total);
  const median = normalized.findIndex((_, idx) => normalized.slice(0, idx + 1).reduce((a,b) => a+b,0) >= 0.5 - 1e-14);
  const mode = normalized.indexOf(Math.max(...normalized));
  const expected = normalized.reduce((sum, p, i) => sum + i*p, 0);
  const above100k = normalized.slice(7).reduce((a,b) => a+b, 0);
  const percentileBucket = quantile => {
    let cdf = 0;
    for (let i = 0; i < normalized.length; i++) {
      cdf += normalized[i];
      if (cdf >= quantile - 1e-14) return i;
    }
    return normalized.length - 1;
  };
  return {
    probabilities: normalized,
    medianBucket: median,
    modalBucket: mode,
    expectedBucket: expected,
    above100k,
    interval80: [percentileBucket(0.10), percentileBucket(0.90)],
  };
}

/**
 * Exactly the Q5.5 "one added tool" experiment: hold all other answers fixed;
 * require at least 30 training respondents; rank by expected ordinal bucket.
 */
export function recommend(model, responses, baseline = predict(model, responses)) {
  const result = [];
  const excluded = ['none of the above', 'none', 'other, please specify',
                    'prefer not', 'not applicable', "i don't know", 'not sure'];
  for (const name of model.recommendationQuestions) {
    const feature = model.multi.find(spec => spec.name === name);
    const selected = new Set(Array.isArray(responses[name]) ? responses[name] :
      (responses[name] ? String(responses[name]).split(';').map(s => s.trim()) : []));
    for (const option of feature.options) {
      if (option.support < model.minimumSupport || selected.has(option.value)) continue;
      if (!option.value || excluded.some(term => option.value.toLowerCase().includes(term))) continue;
      const updated = { ...responses, [name]: [...selected, option.value].sort() };
      const next = predict(model, updated);
      result.push({
        feature: name, name: option.value, support: option.support,
        change: next.expectedBucket - baseline.expectedBucket,
        nextExpected: next.expectedBucket,
        changeHigh: next.above100k - baseline.above100k,
      });
    }
  }
  result.sort((a,b) => (b.change - a.change) || (b.support - a.support));
  return result;
}
