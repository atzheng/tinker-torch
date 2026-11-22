/** Pure helpers for the 1-D loss curve panel (tinker-6t7.2): the parameter list, range, tangent. No DOM. */
import type { HostParams, NodeId } from '@tinker/core';

/** Points of the swept curve. */
export const CURVE_N = 160;
/** Minimum half-width of the range around the current value. */
export const CURVE_MIN_HALF = 3;
/** Largest half-width the range follows the visited values to; beyond it the last range is kept. */
export const CURVE_MAX_HALF = 100;
/** The picker lists at most this many parameters. */
export const MAX_PARAMS = 200;

export type CurveAxis = { node: NodeId; tensor: 'W' | 'b'; index: number };
export type CurveParam = { axis: CurveAxis; key: string; value: number; label: string };

export const axisKey = (a: CurveAxis): string => `${a.node}:${a.tensor}:${a.index}`;

/**
 * Every scalar entry of the model, in the order of `order` (weights then biases of each piece).
 * The label is the piece's name and the parameter's symbol, as `$...$` text: `bias $b$`,
 * `lin1 $W_{2}$` (subscript only when the tensor has several entries).
 */
export function listParams(host: HostParams, order: readonly NodeId[]): CurveParam[] {
  const out: CurveParam[] = [];
  for (const node of order) {
    const p = Object.hasOwn(host, node) ? host[node] : undefined;
    if (!p) continue;
    const name = node.replace(/\$/g, '');
    for (const tensor of ['W', 'b'] as const) {
      const arr = tensor === 'W' ? p.W : p.b;
      if (!arr) continue;
      for (let index = 0; index < arr.length; index++) {
        const sym = arr.length > 1 ? `${tensor}_{${index + 1}}` : tensor;
        const axis: CurveAxis = { node, tensor, index };
        out.push({ axis, key: axisKey(axis), value: arr[index] ?? 0, label: `${name} $${sym}$` });
        if (out.length >= MAX_PARAMS) return out;
      }
    }
  }
  return out;
}

export type Visited = { min: number; max: number };

/** Adds `v` to the visited interval; a non-finite value is ignored. Returns a new object only when it grows. */
export function visit(prev: Visited | null, v: number): Visited | null {
  if (!Number.isFinite(v)) return prev;
  if (!prev) return { min: v, max: v };
  if (v >= prev.min && v <= prev.max) return prev;
  return { min: Math.min(prev.min, v), max: Math.max(prev.max, v) };
}

/**
 * Axis range: `prev` while it still holds the visited interval and the current value; otherwise a
 * window centred on `cur` with half-width `max(CURVE_MIN_HALF, 1.25 x` the farthest visited value`)`.
 * A run that runs off past `CURVE_MAX_HALF` keeps the last range (the dot may then leave it).
 */
export function nextAxisRange(
  prev: [number, number] | null,
  visited: Visited,
  cur: number,
): [number, number] | null {
  if (!Number.isFinite(cur)) return prev;
  if (prev && visited.min >= prev[0] && visited.max <= prev[1] && cur >= prev[0] && cur <= prev[1])
    return prev;
  const half = Math.max(
    CURVE_MIN_HALF,
    1.25 * Math.max(Math.abs(visited.min - cur), Math.abs(visited.max - cur)),
  );
  if (half > CURVE_MAX_HALF && prev) return prev;
  return [cur - half, cur + half];
}

/** `n` evenly spaced values from lo to hi inclusive. */
export function sweep(lo: number, hi: number, n: number): Float32Array {
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = n > 1 ? lo + ((hi - lo) * i) / (n - 1) : lo;
  return out;
}

/**
 * Values to evaluate: all of the range, or with `exploredOnly` only the part of it inside the
 * visited interval (none while a single value has been visited).
 */
export function sweepValues(
  range: [number, number],
  visited: Visited,
  exploredOnly: boolean,
): Float32Array {
  if (!exploredOnly) return sweep(range[0], range[1], CURVE_N);
  const lo = Math.max(range[0], visited.min);
  const hi = Math.min(range[1], visited.max);
  if (!(hi > lo)) return new Float32Array(0);
  const frac = (hi - lo) / (range[1] - range[0]);
  return sweep(lo, hi, Math.max(2, Math.round(CURVE_N * frac)));
}

export type YRange = { lo: number; hi: number };

/**
 * Vertical range: the finite curve values and the current loss. A very steep tail (the log loss
 * far from its minimum) is cut at three times the 90th-percentile height above the minimum.
 */
export function yRange(curve: Float32Array, now: number | null): YRange {
  const v: number[] = [];
  for (const x of curve) if (Number.isFinite(x)) v.push(x);
  const nowOk = now !== null && Number.isFinite(now);
  if (nowOk) v.push(now);
  if (v.length === 0) return { lo: 0, hi: 1 };
  const sorted = v.sort((p, q) => p - q);
  const min = sorted[0] ?? 0;
  const max = sorted[sorted.length - 1] ?? min;
  const p90 = sorted[Math.min(sorted.length - 1, Math.floor(0.9 * sorted.length))] ?? max;
  let hi = Math.min(max, min + 3 * Math.max(p90 - min, 1e-6));
  if (nowOk) hi = Math.max(hi, now);
  // a flat curve (a parameter the loss barely reads) still gets a readable axis and tangent
  hi = Math.max(hi, min + 0.05 * Math.max(1, Math.abs(min)));
  const pad = 0.05 * (hi - min);
  return { lo: min - pad, hi: hi + pad };
}

/**
 * Tangent segment at (x0, y0) with `slope`, over x0 +- `halfWidth`, in data units; null when the
 * slope or the point is not finite.
 */
export function tangent(
  x0: number,
  y0: number,
  slope: number,
  halfWidth: number,
): { x1: number; y1: number; x2: number; y2: number } | null {
  if (![x0, y0, slope, halfWidth].every(Number.isFinite)) return null;
  return {
    x1: x0 - halfWidth,
    y1: y0 - slope * halfWidth,
    x2: x0 + halfWidth,
    y2: y0 + slope * halfWidth,
  };
}
