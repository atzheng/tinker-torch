/**
 * The probability bar on a Link that carries a ŷ pin (tinker-6t7.17). Pure: the numbers come from
 * the engine's `predict` readback (one `[n, width]` row-major array per pin), nothing is kept.
 */
import { parsePort, type Graph } from '@tinker/core';

/** Class shares (each 0..1, summing to about 1 for softmax) or the lone sigmoid value. */
export type ProbBar = {
  /** One entry for a sigmoid, one per class for a softmax. */
  values: number[];
  /** True when the value differs between examples, so `values` is a mean over the training examples. */
  mean: boolean;
};

/** Examples whose value differs by less than this count as the same. */
const SAME = 1e-6;
/** At most this many training rows are fed to `predict` for the mean. */
export const PROB_BAR_MAX_ROWS = 128;
/** While training runs, the bars are recomputed at most this often (ms). */
export const PROB_BAR_EVERY_MS = 250;

/**
 * Whether new parameters should be read now: always when not running or when the value is exact and
 * cheap (a no-input model, one row), else at most every `PROB_BAR_EVERY_MS` while running.
 */
export function probBarDue(now: number, last: number, running: boolean, cheap: boolean): boolean {
  return !running || cheap || now - last >= PROB_BAR_EVERY_MS;
}

/**
 * The bar for one pin's readback `out` (`n` rows of `width` columns). The value of a no-input model
 * (the same for every example) is shown as is; otherwise the column means, flagged `mean`. Null when
 * the shape is wrong or a value is not a finite number in [0, 1] (the engine refuses those steps).
 */
export function probBarOf(out: Float32Array | undefined, n: number, width: number): ProbBar | null {
  if (!out || n < 1 || width < 1 || out.length !== n * width) return null;
  const values: number[] = [];
  let mean = false;
  for (let c = 0; c < width; c++) {
    let sum = 0;
    let lo = Infinity;
    let hi = -Infinity;
    for (let i = 0; i < n; i++) {
      const v = out[i * width + c] as number;
      sum += v;
      lo = Math.min(lo, v);
      hi = Math.max(hi, v);
    }
    if (!Number.isFinite(lo) || !Number.isFinite(hi) || lo < -1e-6 || hi > 1 + 1e-6) return null;
    if (hi - lo > SAME) mean = true;
    values.push(Math.min(1, Math.max(0, sum / n)));
  }
  return { values, mean };
}

/**
 * Which pins are Links (sigmoid or softmax) with a known width: node id -> index into
 * `graph.predictions` and the width of its output.
 */
export function probBarTargets(
  graph: Pick<Graph, 'nodes' | 'predictions'>,
  outWidth: Readonly<Record<string, number | undefined>>,
): { id: string; pin: number; width: number }[] {
  const out: { id: string; pin: number; width: number }[] = [];
  const pinNodes = graph.predictions.map((p) => parsePort(p).node);
  for (const node of graph.nodes) {
    if (node.type !== 'link') continue;
    const pin = pinNodes.indexOf(node.id);
    const width = outWidth[node.id];
    if (pin < 0 || width === undefined || width < 1) continue;
    if (node.fn === 'sigmoid' && width !== 1) continue;
    out.push({ id: node.id, pin, width });
  }
  return out;
}

/** Bars for every pinned Link, from `predict`'s per-pin arrays over `n` rows. */
export function probBars(
  graph: Pick<Graph, 'nodes' | 'predictions'>,
  outWidth: Readonly<Record<string, number | undefined>>,
  outs: readonly Float32Array[],
  n: number,
): Record<string, ProbBar> {
  const bars: Record<string, ProbBar> = {};
  for (const t of probBarTargets(graph, outWidth)) {
    const bar = probBarOf(outs[t.pin], n, t.width);
    if (bar) bars[t.id] = bar;
  }
  return bars;
}
