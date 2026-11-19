import type { DataConfig, DataInfo, DatasetName } from '../common.js';
import { makeRng } from '../rng.js';
import { SETS, generate } from './sets.js';

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

export function sample(config: DataConfig): Dataset {
  const spec = SETS[config.set];
  const d = spec.d;
  const n = config.n;
  const rows = generate(config.set, n, config.noise, makeRng(config.seed));
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

  let standardization: Standardization | null = null;
  let xt: Float32Array = xTrain;
  let xv: Float32Array = xVal;
  // standardization stays null only when nTrain is 0, which the schema (n >= 10, val <= 0.5) makes unreachable.
  if (config.standardize && nTrain > 0) {
    const mean: number[] = [];
    const std: number[] = [];
    for (let c = 0; c < d; c++) {
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
      kind: spec.kind,
      numClasses: spec.kind === 'binary' ? 2 : 1,
    },
    xTrain: xt,
    yTrain,
    nTrain,
    xVal: xv,
    yVal,
    nVal,
    standardization,
    domain: spec.domain.map(([lo, hi]) => [lo, hi]),
  };
}

/**
 * Regression: f*(x). Classification: P(y=1|x) of the clean generator.
 * `xRaw` is row-major raw (unstandardised) inputs with the set's d columns.
 */
export function truth(set: DatasetName, xRaw: Float32Array): Float32Array {
  const spec = SETS[set];
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
