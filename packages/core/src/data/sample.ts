import type { DataConfig, DataInfo, DatasetName } from '../common.js';
import { makeRng } from '../rng.js';
import { SETS, generate } from './sets.js';
import { realRows } from './real.js';

export type Standardization = { mean: number[]; std: number[] };

export type Dataset = {
  info: DataInfo;
  /** Row-major [nTrain, d]. Standardised when `standardization` is not null. */
  xTrain: Float32Array;
  yTrain: Float32Array;
  nTrain: number;
  /** Row-major [nVal, d], standardised with the training mean and std when on. */
  xVal: Float32Array;
  yVal: Float32Array;
  nVal: number;
  /** Mean and population std of the raw training columns; null when standardisation is off. */
  standardization: Standardization | null;
  /** Per column, in raw units (also when standardisation is on). */
  domain: [lo: number, hi: number][];
};

/** Applies (x - mean) / std column-wise to a row-major array with d columns. */
export function standardizeRows(x: Float32Array, d: number, st: Standardization): Float32Array {
  const out = new Float32Array(x.length);
  for (let i = 0; i < x.length; i++) {
    const c = i % d;
    out[i] = ((x[i] ?? 0) - (st.mean[c] ?? 0)) / (st.std[c] ?? 1);
  }
  return out;
}

/** Inverse of `standardizeRows`. */
export function unstandardizeRows(x: Float32Array, d: number, st: Standardization): Float32Array {
  const out = new Float32Array(x.length);
  for (let i = 0; i < x.length; i++) {
    const c = i % d;
    out[i] = (x[i] ?? 0) * (st.std[c] ?? 1) + (st.mean[c] ?? 0);
  }
  return out;
}

/** Number of validation rows: round(n * val); the last ones of the n generated rows. */
export function valCount(n: number, val: number): number {
  return Math.min(n, Math.max(0, Math.round(n * val)));
}

/** True for the generators (they have a clean truth); false for real datasets. */
export function hasTruth(set: DatasetName): boolean {
  return !SETS[set].real;
}

/** Regression 1, binary 2, multiclass k. */
export function classCount(set: DatasetName): number {
  const spec = SETS[set];
  return spec.kind === 'regression' ? 1 : spec.kind === 'binary' ? 2 : (spec.numClasses ?? 3);
}

export function sample(config: DataConfig): Dataset {
  const spec = SETS[config.set];
  const d = spec.d;
  const rng = makeRng(config.seed);
  // A real set has fixed rows: n is its size, noise does not apply, the seed shuffles the split.
  const real = spec.real ? realRows(config.set, rng) : null;
  const rows = real ? real.rows : generate(config.set, config.n, config.noise, rng);
  const n = rows.length;
  const nVal = valCount(n, config.val);
  const nTrain = n - nVal;

  const xTrain = new Float32Array(nTrain * d);
  const yTrain = new Float32Array(nTrain);
  const xVal = new Float32Array(nVal * d);
  const yVal = new Float32Array(nVal);
  rows.forEach((row, i) => {
    const train = i < nTrain;
    const j = train ? i : i - nTrain;
    const xs = train ? xTrain : xVal;
    for (let c = 0; c < d; c++) xs[j * d + c] = row.x[c] ?? 0;
    (train ? yTrain : yVal)[j] = row.y;
  });

  const cats = spec.categories ?? {};
  const isCat = (c: number): boolean => cats[c] !== undefined;
  let standardization: Standardization | null = null;
  let xt: Float32Array = xTrain;
  let xv: Float32Array = xVal;
  // standardization stays null only when nTrain is 0, which the schema (n >= 10, val <= 0.5) makes unreachable.
  if (config.standardize && nTrain > 0) {
    const mean: number[] = [];
    const std: number[] = [];
    for (let c = 0; c < d; c++) {
      if (isCat(c)) {
        // a category index is not a quantity: leave it alone (mean 0, std 1)
        mean.push(0);
        std.push(1);
        continue;
      }
      let sum = 0;
      for (let i = 0; i < nTrain; i++) sum += xTrain[i * d + c] ?? 0;
      const m = sum / nTrain;
      let ss = 0;
      for (let i = 0; i < nTrain; i++) ss += ((xTrain[i * d + c] ?? 0) - m) ** 2;
      const sd = Math.sqrt(ss / nTrain);
      mean.push(m);
      std.push(sd > 1e-12 ? sd : 1); // constant column: leave the scale alone
    }
    standardization = { mean, std };
    xt = standardizeRows(xTrain, d, standardization);
    xv = standardizeRows(xVal, d, standardization);
  }

  return {
    info: {
      columns: Array.from({ length: d }, (_, c) => `x${c + 1}`),
      ...(spec.featureNames ? { featureNames: spec.featureNames } : {}),
      ...(Object.keys(cats).length > 0
        ? {
            columnKinds: Array.from({ length: d }, (_, c) =>
              isCat(c) ? ('categorical' as const) : ('numeric' as const),
            ),
            categories: Object.fromEntries(
              Object.entries(cats).map(([c, names]) => [`x${Number(c) + 1}`, [...names]]),
            ),
          }
        : {}),
      kind: spec.kind,
      numClasses: classCount(config.set),
      targetName: spec.targetName ?? 'y',
      ...(spec.kind === 'regression'
        ? {}
        : {
            classNames:
              spec.classNames ??
              Array.from({ length: classCount(config.set) }, (_, i) => `class ${i}`),
          }),
    },
    xTrain: xt,
    yTrain,
    nTrain,
    xVal: xv,
    yVal,
    nVal,
    standardization,
    domain: (real ? real.domain : spec.domain).map(([lo, hi]) => [lo, hi]),
  };
}

/**
 * Regression: f*(x). Binary: P(y=1|x) of the clean generator. Multiclass: the class index of the
 * clean generator.
 * `xRaw` is row-major raw (unstandardised) inputs with the set's d columns.
 */
export function truth(set: DatasetName, xRaw: Float32Array): Float32Array {
  const spec = SETS[set];
  if (spec.real) throw new Error('A real dataset has no true function.');
  const d = spec.d;
  const n = Math.floor(xRaw.length / d);
  const out = new Float32Array(n);
  const row: number[] = new Array<number>(d).fill(0);
  for (let i = 0; i < n; i++) {
    for (let c = 0; c < d; c++) row[c] = xRaw[i * d + c] ?? 0;
    out[i] = spec.f(row);
  }
  return out;
}

/**
 * Smooth field for drawing the true class boundary: the zero level set is the boundary and the
 * sign is positive on class 1. Regression sets return f*(x) (no boundary).
 */
export function truthMargin(set: DatasetName, xRaw: Float32Array): Float32Array {
  const spec = SETS[set];
  const g = spec.margin ?? ((x: readonly number[]) => spec.f(x) - 0.5);
  const d = spec.d;
  const n = Math.floor(xRaw.length / d);
  const out = new Float32Array(n);
  const row: number[] = new Array<number>(d).fill(0);
  for (let i = 0; i < n; i++) {
    for (let c = 0; c < d; c++) row[c] = xRaw[i * d + c] ?? 0;
    out[i] = g(row);
  }
  return out;
}

/**
 * What a model's output should look like on `xRaw`: the same as `truth` for regression and binary
 * sets; for multiclass sets row-major [n, k] one-hot of the true class.
 */
export function truthProbs(set: DatasetName, xRaw: Float32Array): Float32Array {
  const spec = SETS[set];
  const t = truth(set, xRaw);
  if (spec.kind !== 'multiclass') return t;
  const k = classCount(set);
  const out = new Float32Array(t.length * k);
  t.forEach((c, i) => {
    out[i * k + c] = 1;
  });
  return out;
}
