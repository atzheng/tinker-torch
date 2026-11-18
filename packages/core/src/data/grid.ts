import { standardizeRows, type Standardization } from './sample.js';

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
