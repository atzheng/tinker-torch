import type { HistoryPoint } from '../state/trainSlice.js';

export const EM_DASH = '—';

/** 4 significant figures; exponent form below 1e-3 or from 1e4 (3.142e-5). Non-finite: an em dash. */
export function formatLoss(x: number | null | undefined): string {
  if (x === null || x === undefined || !Number.isFinite(x)) return EM_DASH;
  if (x === 0) return '0';
  const a = Math.abs(x);
  if (a < 1e-3 || a >= 1e4) return x.toExponential(3);
  return String(Number(x.toPrecision(4)));
}

export type LossReadout = { train: string; val: string; diverged: boolean };

/**
 * What the loss readout shows. The big number is the latest `train` (full batch) or the latest
 * non-null `trainFull` (mini-batch, not the noisy batch loss); val is the latest non-null `val`.
 */
export function lossReadout(
  history: readonly HistoryPoint[],
  miniBatch: boolean,
  diverged: boolean,
): LossReadout {
  let train: number | null | undefined = null;
  let val: number | null = null;
  for (let i = history.length - 1; i >= 0 && (train == null || val === null); i--) {
    const p = history[i];
    if (!p) continue;
    if (train == null) train = miniBatch ? p.trainFull : p.train;
    if (val === null) val = p.val;
  }
  return { train: formatLoss(train), val: formatLoss(val), diverged };
}
