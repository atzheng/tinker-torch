/**
 * Ready-made groups (CONTRACTS.md section 3): the five classic losses and the weight penalty, built
 * as ordinary primitive nodes and edges. A group is never a node of its own; inserting one adds
 * these nodes to the graph, and v1 migration uses the same builders.
 */
import type { Node } from './graph.js';
import type { Edge, NodeId, PortRef } from './common.js';

/** Makes a fresh, unused node id from a short suffix (e.g. "diff" -> "loss_diff"). */
export type Namer = (suffix: string) => NodeId;

export type Subgraph = {
  nodes: Node[];
  edges: Edge[];
  /** Destination ports (as "node.port") the caller wires into, by role. */
  inputs: Record<string, PortRef>;
  /** The node whose `out` is the group's result, or null for an empty group. */
  out: NodeId | null;
};

export const LOSS_KINDS = [
  'squared_error',
  'absolute_error',
  'huber',
  'log_loss',
  'cross_entropy',
] as const;
export type LossKind = (typeof LOSS_KINDS)[number];

/**
 * pred - target, then square / abs / huber, then Mean; or log loss / cross-entropy, then Mean.
 * `inputs` has `pred` and `target`. The Mean gets `meanId` (default `name('loss')`).
 */
export function lossSubgraph(
  kind: LossKind,
  opts: { delta?: number; name: Namer; meanId?: NodeId },
): Subgraph {
  const err = opts.name('err');
  const mean = opts.meanId ?? opts.name('loss');
  const nodes: Node[] = [];
  const edges: Edge[] = [];
  let inputs: Record<string, PortRef>;
  if (kind === 'log_loss' || kind === 'cross_entropy') {
    nodes.push({ id: err, type: kind });
    inputs = { pred: `${err}.pred`, target: `${err}.target` };
  } else {
    const diff = opts.name('diff');
    nodes.push({ id: diff, type: 'subtract' });
    edges.push([diff, err]);
    inputs = { pred: `${diff}.a`, target: `${diff}.b` };
    nodes.push(
      kind === 'squared_error'
        ? { id: err, type: 'square' }
        : kind === 'absolute_error'
          ? { id: err, type: 'abs' }
          : { id: err, type: 'huber', delta: opts.delta ?? 1 },
    );
  }
  nodes.push({ id: mean, type: 'mean' });
  edges.push([err, mean]);
  return { nodes, edges, inputs, out: mean };
}

/**
 * Weights -> norm for each Linear in `linears`, summed, times lambda, then added to an existing
 * total. `inputs.total` is where the existing loss total is wired. The final Add gets `finalId`
 * (default `name('total')`). With no Linear the group is empty (`out` null).
 */
export function weightPenaltySubgraph(opts: {
  kind: 'l2' | 'l1';
  lambda: number;
  linears: readonly NodeId[];
  name: Namer;
  finalId?: NodeId;
}): Subgraph {
  if (opts.linears.length === 0) return { nodes: [], edges: [], inputs: {}, out: null };
  const nodes: Node[] = [];
  const edges: Edge[] = [];
  let acc: NodeId | null = null;
  let k = 0;
  for (const lin of opts.linears) {
    const w = opts.name(`w_${lin}`);
    const n = opts.name(`n_${lin}`);
    nodes.push({ id: w, type: 'weights', of: lin }, { id: n, type: 'norm', kind: opts.kind });
    edges.push([w, n]);
    if (acc === null) acc = n;
    else {
      k += 1;
      const sum = opts.name(`sum${k}`);
      nodes.push({ id: sum, type: 'add' });
      edges.push([acc, `${sum}.a`], [n, `${sum}.b`]);
      acc = sum;
    }
  }
  const pen = opts.name('pen');
  const final = opts.finalId ?? opts.name('total');
  nodes.push({ id: pen, type: 'scale', c: opts.lambda }, { id: final, type: 'add' });
  edges.push([acc as NodeId, pen], [pen, `${final}.b`]);
  return { nodes, edges, inputs: { total: `${final}.a` }, out: final };
}
