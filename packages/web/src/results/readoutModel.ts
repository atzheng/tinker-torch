import type { LossMode } from '@tinker/core';
import type { ExampleStatsPair } from '../state/lossMode.js';
import type { HistoryPoint, TrainSlice } from '../state/trainSlice.js';

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

/** What the readout needs beyond the history: the live values, the mode and the split sizes. */
export type ReadoutLive = {
  live: TrainSlice['liveLoss'];
  stats: ExampleStatsPair | null;
  mode: LossMode | null;
  nTrain: number;
  nVal: number;
};

/**
 * What the loss readout shows. With `live` (the objective of the current parameters on the whole
 * train and validation sets, refreshed on every step and hand edit) the numbers follow the
 * parameters; without it they come from the history: the latest `train` (full batch) or the latest
 * non-null `trainFull` (mini-batch, not the noisy batch loss), and the latest non-null `val`.
 * Mode `total` is n x mean; `likelihood` is the float32 product from `stats` (an em dash before
 * it exists); `mean` and a null mode show the mean.
 */
export function lossReadout(
  history: readonly HistoryPoint[],
  miniBatch: boolean,
  diverged: boolean,
  opts?: ReadoutLive,
): LossReadout {
  let train: number | null | undefined = opts?.live ? opts.live.train : null;
  let val: number | null = opts?.live ? opts.live.val : null;
  if (!opts?.live) {
    for (let i = history.length - 1; i >= 0 && (train == null || val === null); i--) {
      const p = history[i];
      if (!p) continue;
      if (train == null) train = miniBatch ? p.trainFull : p.train;
      if (val === null) val = p.val;
    }
  }
  const mode = opts?.mode ?? 'mean';
  const show = (mean: number | null | undefined, n: number, lik: number | undefined): string => {
    if (mode === 'likelihood') return formatLoss(lik);
    if (mode === 'total') return formatLoss(mean == null ? null : mean * n);
    return formatLoss(mean);
  };
  return {
    train: show(train, opts?.nTrain ?? 0, opts?.stats?.train?.likelihood),
    val: show(val, opts?.nVal ?? 0, opts?.stats?.val?.likelihood),
    diverged,
  };
}
