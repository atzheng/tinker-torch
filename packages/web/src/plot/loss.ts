import type { HistoryPoint } from '../state/trainSlice.js';

export type LossRange = { stepMax: number; lo: number; hi: number };

/**
 * Axis ranges for the loss curves in one pass over the history (no allocation). Steps start at 0.
 * Linear: [0, max * 1.05]. Log: [min positive / 1.25, max * 1.25]; non-positive and non-finite
 * values are ignored. `extra` (for example the best validation loss) is included as a value.
 */
export function lossRange(
  history: readonly HistoryPoint[],
  log: boolean,
  extra?: number,
): LossRange {
  let stepMax = 1;
  let min = Infinity;
  let max = -Infinity;
  const see = (v: number | null | undefined) => {
    if (v === null || v === undefined || !Number.isFinite(v)) return;
    if (log && v <= 0) return;
    if (v < min) min = v;
    if (v > max) max = v;
  };
  for (const p of history) {
    if (p.step > stepMax) stepMax = p.step;
    see(p.train);
    see(p.val);
  }
  see(extra);
  if (!(max >= min)) return { stepMax, lo: log ? 0.1 : 0, hi: 1 };
  if (log) {
    const lo = min / 1.25;
    const hi = max * 1.25;
    return { stepMax, lo, hi: hi > lo * 1.5 ? hi : lo * 2 };
  }
  return { stepMax, lo: 0, hi: max > 0 ? max * 1.05 : 1 };
}
