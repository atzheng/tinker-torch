// Score readout modes (tinker-6t7.5): likelihood, total negative log-likelihood and mean of a
// per-example loss. Pure; the per-example values come from `Engine.exampleLosses`.
import { resolveEdge, type Graph, type Node } from './graph.js';
import type { Analysis } from './analyse.js';

export type LossMode = 'likelihood' | 'total' | 'mean';
export const LOSS_MODES: readonly LossMode[] = ['likelihood', 'total', 'mean'];
/** The UI id of each mode (`UI_IDS`, group panel-controls). */
export const LOSS_MODE_IDS: Record<LossMode, string> = {
  likelihood: 'loss-mode-likelihood',
  total: 'loss-mode-total',
  mean: 'loss-mode-mean',
};

/**
 * The node whose per-example output the loss averages: the node feeding the `in` port of the L node
 * when the L node is itself a Mean and that node gives one value per example (width 1). Null in
 * every other case (a penalty added to the Mean, a Mean pooling inside the model, a wider source).
 * Same rule as the engine's `exampleLosses`.
 */
export function exampleSource(graph: Graph, analysis: Analysis): Node | null {
  const { lossNode } = analysis;
  if (lossNode === null) return null;
  if (graph.nodes.find((n) => n.id === lossNode)?.type !== 'mean') return null;
  for (const e of graph.edges) {
    const r = resolveEdge(graph.nodes, e);
    if (!r.ok || r.to.node !== lossNode || r.to.port !== 'in') continue;
    const sh = analysis.shape[r.from.node];
    if (sh?.kind !== 'example' || sh.width !== 1) return null;
    return graph.nodes.find((n) => n.id === r.from.node) ?? null;
  }
  return null;
}

/**
 * Modes the readout can offer. Likelihood only when L is directly a Mean over a log loss or
 * cross-entropy (so likelihood = exp(-total) with total = n x L); otherwise total and mean.
 */
export function lossModesFor(graph: Graph, analysis: Analysis): LossMode[] {
  const src = exampleSource(graph, analysis);
  if (src === null) return ['total', 'mean'];
  return src.type === 'log_loss' || src.type === 'cross_entropy'
    ? ['likelihood', 'total', 'mean']
    : ['total', 'mean'];
}

/**
 * The mode to use: of the `offered` modes keep those the preset shows (`shown(id)`); a single one is
 * fixed; else the user's `choice` when valid, else mean, else the first shown. Null when none.
 */
export function effectiveLossMode(
  offered: readonly LossMode[],
  shown: (id: string) => boolean,
  choice: LossMode,
): LossMode | null {
  const ok = offered.filter((m) => shown(LOSS_MODE_IDS[m]));
  if (ok.length === 0) return null;
  if (ok.length === 1) return ok[0] ?? null;
  if (ok.includes(choice)) return choice;
  return ok.includes('mean') ? 'mean' : (ok[0] ?? null);
}

export type ExampleStats = {
  n: number;
  /** Sum of the per-example losses (float64): the total negative log-likelihood. */
  total: number;
  /**
   * Product of exp(-loss_i), the probabilities of what happened, in float32 like the engine: each
   * factor and each running product is rounded with Math.fround, so it underflows to exactly 0
   * (about 150 fair coin flips).
   */
  likelihood: number;
};

export function exampleStats(losses: Float32Array): ExampleStats {
  let total = 0;
  let likelihood = 1;
  for (const l of losses) {
    total += l;
    likelihood = Math.fround(likelihood * Math.fround(Math.exp(-l)));
  }
  return { n: losses.length, total, likelihood };
}
