import { standardizeRows, unstandardizeRows, type Standardization } from './sample.js';

/**
 * The raw value each column is held at when a plot shows a 2-D slice of d > 2 columns: the training
 * mean of a numeric column, the most common training category (lowest index on a tie) of a
 * categorical one. The fit plot and the resample overlay both use it.
 */
export function heldValues(ds: {
  info: { columns: string[]; columnKinds?: ('numeric' | 'categorical')[] };
  xTrain: Float32Array;
  nTrain: number;
  standardization: Standardization | null;
}): number[] {
  const d = ds.info.columns.length;
  const xt = ds.standardization ? unstandardizeRows(ds.xTrain, d, ds.standardization) : ds.xTrain;
  return Array.from({ length: d }, (_, c) => {
    if (ds.info.columnKinds?.[c] === 'categorical') {
      const counts = new Map<number, number>();
      for (let i = 0; i < ds.nTrain; i++) {
        const v = xt[i * d + c] ?? 0;
        counts.set(v, (counts.get(v) ?? 0) + 1);
      }
      let best = 0;
      let bestN = -1;
      for (const [v, k] of counts) {
        if (k > bestN || (k === bestN && v < best)) {
          best = v;
          bestN = k;
        }
      }
      return best;
    }
    let sum = 0;
    for (let i = 0; i < ds.nTrain; i++) sum += xt[i * d + c] ?? 0;
    return ds.nTrain > 0 ? sum / ds.nTrain : 0;
  });
}

export type Grid = {
  /** Number of grid points (rows). */
  n: number;
  /** Columns per row. */
  d: number;
  /** Row-major [n, d], raw units. */
  xRaw: Float32Array;
  /** Row-major [n, d], standardised with `standardization`; equals xRaw when it is null. */
  xStd: Float32Array;
};

function linspace(lo: number, hi: number, m: number): number[] {
  if (m <= 1) return [(lo + hi) / 2];
  return Array.from({ length: m }, (_, i) => lo + ((hi - lo) * i) / (m - 1));
}

function finish(xRaw: Float32Array, d: number, st: Standardization | null): Grid {
  return {
    n: xRaw.length / d,
    d,
    xRaw,
    xStd: st ? standardizeRows(xRaw, d, st) : xRaw,
  };
}

/** `m` points spanning the domain inclusive of both ends. Default 200. */
export function grid1d(
  domain: readonly [lo: number, hi: number][],
  m = 200,
  standardization: Standardization | null = null,
): Grid {
  const [lo, hi] = domain[0] ?? [0, 1];
  return finish(Float32Array.from(linspace(lo, hi, m)), 1, standardization);
}

/**
 * An m x m grid (default 64). Row index = iy * m + ix, where ix runs along x1 (fastest) and
 * iy along x2, so rows read left to right, bottom to top, like an image with a flipped y axis.
 */
export function grid2d(
  domain: readonly [lo: number, hi: number][],
  m = 64,
  standardization: Standardization | null = null,
): Grid {
  const [lo1, hi1] = domain[0] ?? [0, 1];
  const [lo2, hi2] = domain[1] ?? [0, 1];
  const a = linspace(lo1, hi1, m);
  const b = linspace(lo2, hi2, m);
  const x = new Float32Array(m * m * 2);
  let k = 0;
  for (const y2 of b) {
    for (const y1 of a) {
      x[k++] = y1;
      x[k++] = y2;
    }
  }
  return finish(x, 2, standardization);
}

/**
 * `grid2d` for data with d >= 2 columns: an m x m grid over columns `ia` (fastest) and `ib`, every
 * other column held at `fixedRaw[c]` (raw units; its length is d). Rows have d columns; row index
 * as in `grid2d`. With d = 2, ia = 0 and ib = 1 it is `grid2d`.
 */
export function grid2dSlice(
  domain: readonly [lo: number, hi: number][],
  fixedRaw: readonly number[],
  ia: number,
  ib: number,
  m = 64,
  standardization: Standardization | null = null,
): Grid {
  const d = fixedRaw.length;
  const [lo1, hi1] = domain[ia] ?? [0, 1];
  const [lo2, hi2] = domain[ib] ?? [0, 1];
  const a = linspace(lo1, hi1, m);
  const b = linspace(lo2, hi2, m);
  const x = new Float32Array(m * m * d);
  let k = 0;
  for (const y2 of b) {
    for (const y1 of a) {
      for (let c = 0; c < d; c++) x[k++] = c === ia ? y1 : c === ib ? y2 : (fixedRaw[c] ?? 0);
    }
  }
  return finish(x, d, standardization);
}
