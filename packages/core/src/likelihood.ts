// Score readout modes (tinker-6t7.5): likelihood, total negative log-likelihood and mean of a
// per-example loss. Pure; the per-example values come from `Engine.exampleLosses`.
import { resolveEdge, type Graph, type Node, type NodeId } from './graph.js';
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
 * when that is a Mean, else of the first Mean (in evaluation order) among its ancestors. Null when
 * there is none. Same rule as the engine's `exampleLosses`.
 */
export function exampleSource(graph: Graph, analysis: Analysis): Node | null {
  const { lossNode } = analysis;
  if (lossNode === null) return null;
  const feeds = new Map<NodeId, NodeId[]>();
  const inPort = new Map<NodeId, NodeId>();
  for (const e of graph.edges) {
    const r = resolveEdge(graph.nodes, e);
    if (!r.ok) continue;
    feeds.set(r.to.node, [...(feeds.get(r.to.node) ?? []), r.from.node]);
    if (r.to.port === 'in' && !inPort.has(r.to.node)) inPort.set(r.to.node, r.from.node);
  }
  const anc = new Set<NodeId>();
  const todo = [lossNode];
  while (todo.length > 0) {
    const id = todo.pop() as NodeId;
    if (anc.has(id)) continue;
    anc.add(id);
    todo.push(...(feeds.get(id) ?? []));
  }
  const byId = new Map(graph.nodes.map((n) => [n.id, n]));
  const isMean = (id: NodeId) => byId.get(id)?.type === 'mean';
  const mean = isMean(lossNode) ? lossNode : analysis.order.find((id) => anc.has(id) && isMean(id));
  const src = mean === undefined ? undefined : inPort.get(mean);
  return src === undefined ? null : (byId.get(src) ?? null);
}

/** Modes the readout can offer: likelihood only for log loss and cross-entropy. Empty with no Mean. */
export function lossModesFor(graph: Graph, analysis: Analysis): LossMode[] {
  const src = exampleSource(graph, analysis);
  if (src === null) return [];
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
