import { makeRng } from '../rng.js';

/** CONTRACTS.md section 7. x is row-major [n, d]; y has one value per row. */
export type Batch = { x: Float32Array; y: Float32Array; n: number };

/** Copies the rows `idx` of the training arrays into a fresh Batch (x is already standardised if on). */
export function gatherBatch(
  xTrain: Float32Array,
  yTrain: Float32Array,
  d: number,
  idx: Int32Array,
): Batch {
  const n = idx.length;
  const x = new Float32Array(n * d);
  const y = new Float32Array(n);
  for (let k = 0; k < n; k++) {
    const i = idx[k] ?? 0;
    for (let c = 0; c < d; c++) x[k * d + c] = xTrain[i * d + c] ?? 0;
    y[k] = yTrain[i] ?? 0;
  }
  return { x, y, n };
}

/** Steps per epoch for a batch setting (the last batch of an epoch may be smaller). */
export function stepsPerEpoch(nTrain: number, batch: number | 'full'): number {
  if (nTrain <= 0) return 0;
  if (batch === 'full') return 1;
  return Math.ceil(nTrain / Math.max(1, Math.min(batch, nTrain)));
}

/**
 * Infinite sequence of index arrays into the training rows, one per optimiser step.
 * Each epoch e shuffles 0..nTrain-1 with `makeRng(seed).fork('epoch' + e)` and cuts it into
 * batches, so every row appears exactly once per epoch. `'full'` yields all rows in order
 * every step. Each yielded array is fresh and may be kept.
 */
export function* batchSchedule(
  nTrain: number,
  batch: number | 'full',
  seed: number,
): Generator<Int32Array, never, undefined> {
  const base = makeRng(seed);
  const size = batch === 'full' ? nTrain : Math.max(1, Math.min(Math.floor(batch), nTrain));
  for (let epoch = 0; ; epoch++) {
    if (nTrain <= 0) {
      yield new Int32Array(0);
      continue;
    }
    const order = Int32Array.from({ length: nTrain }, (_, i) => i);
    if (batch !== 'full') {
      const rng = base.fork('epoch' + epoch);
      for (let i = nTrain - 1; i > 0; i--) {
        const j = rng.int(i + 1);
        const t = order[i] ?? 0;
        order[i] = order[j] ?? 0;
        order[j] = t;
      }
    }
    for (let start = 0; start < nTrain; start += size) {
      yield order.slice(start, Math.min(nTrain, start + size));
    }
  }
}
