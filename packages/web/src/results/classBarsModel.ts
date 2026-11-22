// Predicted-versus-observed class bars (tinker-6t7.16): pure model. Nothing engine-side is kept.
import { parsePort, type Graph } from '@tinker/core';
import { probBarOf, probBarTargets } from '../canvas/probBarPure.js';

export type ClassBar = {
  /** Class name (the data's `classNames`, else "class k"). */
  name: string;
  /** Mean predicted probability of the class over the examples. */
  predicted: number;
  /** Share of the examples whose label is the class. */
  observed: number;
};

/** Rows of the training set fed to `predict` (both columns use the same rows). */
export const CLASS_BARS_MAX_ROWS = 2000;

/**
 * The bars for one prediction readback `out` (`n` rows of `width` columns) and the labels `y`
 * (first `nObs` used for the observed shares; `n` rows of predictions, which is 1 for a model with no input). Width 1 is a sigmoid: P(class 1) = value, P(class 0) = 1 - value (two
 * classes). Width k is a softmax: one bar per column. Null when the shape does not fit the
 * kind, or a value is not a probability.
 */
export function classBars(
  out: Float32Array | undefined,
  n: number,
  width: number,
  y: ArrayLike<number>,
  numClasses: number,
  classNames?: readonly string[],
  nObs: number = n,
): ClassBar[] | null {
  if (n < 1 || nObs < 1 || y.length < nObs) return null;
  const bar = probBarOf(out, n, width);
  if (!bar) return null;
  let pred: number[];
  if (width === 1 && numClasses === 2) {
    const p1 = bar.values[0] as number;
    pred = [1 - p1, p1];
  } else if (width === numClasses && width >= 2) pred = bar.values;
  else return null;
  const counts = new Array<number>(numClasses).fill(0);
  for (let i = 0; i < nObs; i++) {
    const c = Math.round(y[i] as number);
    if (c >= 0 && c < numClasses) counts[c] = (counts[c] as number) + 1;
  }
  return pred.map((predicted, k) => ({
    name: classNames?.[k] ?? `class ${k}`,
    predicted,
    observed: (counts[k] as number) / nObs,
  }));
}

/**
 * Whether the node's output can be read as class probabilities: a sigmoid/softmax Link, or (tinker-ss1)
 * a width-1 output of any other piece that feeds a Log loss `pred` input, whose values are
 * probabilities by construction (a Bias wired straight into Log loss). A width-1 output that feeds
 * only an MSE-style loss is refused.
 */
export function isClassProbNode(
  graph: Pick<Graph, 'nodes' | 'edges' | 'predictions'>,
  outWidth: Readonly<Record<string, number | undefined>>,
  node: string,
): boolean {
  if (probBarTargets(graph, outWidth).some((t) => t.id === node)) return true;
  if (outWidth[node] !== 1) return false;
  const logLoss = new Set(graph.nodes.filter((n) => n.type === 'log_loss').map((n) => n.id));
  return graph.edges.some(([from, to]) => {
    const t = parsePort(to);
    return parsePort(from).node === node && t.port === 'pred' && logLoss.has(t.node);
  });
}
