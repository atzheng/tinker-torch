/** Analysis (CONTRACTS.md section 4): order, widths, parameter counts and the issues that disable Train. */
import type { DataInfo, Graph, LossFn, Node, NodeId } from './graph.js';
import { resolveEdge } from './graph.js';
import { CORE_PIECES } from './registry.js';

export type Issue = { at: { node?: NodeId; edge?: number }; message: string };
export type Analysis = {
  /** Topological order of the nodes on the path to the loss (all acyclic nodes when there is no single loss). */
  order: NodeId[];
  /** Width entering each node. Absent when unknown, for pieces with no input, and for Add when its inputs disagree. */
  inWidth: Record<NodeId, number>;
  /** Width leaving each node. Absent when unknown; a loss has none. */
  outWidth: Record<NodeId, number>;
  /** Per `graph.edges` index: the width the wire carries, null where unknown. */
  edgeWidth: (number | null)[];
  /** Parameters per node whose input width is known (0 for pieces with none). */
  params: Record<NodeId, number>;
  /** Sum of `params` over `order`. */
  totalParams: number;
  predNode: NodeId | null;
  lossNode: NodeId | null;
  issues: Issue[];
  trainable: boolean;
};

export const MAX_PARAMS = 10_000;

const LOSS_LABEL: Record<LossFn, string> = {
  squared_error: 'Squared error',
  absolute_error: 'Absolute error',
  huber: 'Huber loss',
  log_loss: 'Log loss',
  cross_entropy: 'Cross-entropy',
};

function isWidthError(v: number | { error: string }): v is { error: string } {
  return typeof v !== 'number';
}

export function analyse(graph: Graph, data: DataInfo): Analysis {
  const issues: Issue[] = [];
  // Defensive: `parseGraph` rejects duplicate ids, but an unparsed graph may have them. The first wins.
  const seen = new Set<NodeId>();
  const nodes: Node[] = [];
  for (const n of graph.nodes) {
    if (seen.has(n.id)) {
      issues.push({ at: { node: n.id }, message: `Two nodes are both called "${n.id}".` });
      continue;
    }
    seen.add(n.id);
    nodes.push(n);
  }
  const byId = new Map<NodeId, Node>(nodes.map((n) => [n.id, n]));

  // ---- resolve edges; each input port keeps its first wire
  type Used = { index: number; from: NodeId };
  const inEdges = new Map<NodeId, Map<string, Used>>();
  const used: { index: number; from: NodeId; to: NodeId; port: string }[] = [];
  graph.edges.forEach((edge, index) => {
    const r = resolveEdge(nodes, edge);
    if (!r.ok) {
      issues.push({ at: { edge: index }, message: r.error });
      return;
    }
    let ports = inEdges.get(r.to.node);
    if (!ports) inEdges.set(r.to.node, (ports = new Map()));
    const prior = ports.get(r.to.port);
    if (prior) {
      issues.push({
        at: { edge: index, node: r.to.node },
        message: `"${r.to.node}" has two wires into its ${r.to.port} input; each input takes only one.`,
      });
      return;
    }
    ports.set(r.to.port, { index, from: r.from.node });
    used.push({ index, from: r.from.node, to: r.to.node, port: r.to.port });
  });

  // ---- topological sort (Kahn, in node order so the result is stable)
  const indeg = new Map<NodeId, number>(nodes.map((n) => [n.id, 0]));
  const succ = new Map<NodeId, NodeId[]>(nodes.map((n) => [n.id, []]));
  for (const e of used) {
    indeg.set(e.to, (indeg.get(e.to) ?? 0) + 1);
    succ.get(e.from)?.push(e.to);
  }
  const topo: NodeId[] = [];
  const ready = nodes.filter((n) => indeg.get(n.id) === 0).map((n) => n.id);
  while (ready.length > 0) {
    const id = ready.shift();
    if (id === undefined) break;
    topo.push(id);
    for (const s of succ.get(id) ?? []) {
      const d = (indeg.get(s) ?? 0) - 1;
      indeg.set(s, d);
      if (d === 0) ready.push(s);
    }
  }
  const done = new Set(topo);
  if (done.size < nodes.length) {
    // Peel off blocked nodes with no blocked successor; what remains lies on a cycle.
    const stuck = new Set(nodes.filter((n) => !done.has(n.id)).map((n) => n.id));
    let changed = true;
    while (changed) {
      changed = false;
      for (const id of [...stuck]) {
        if (!(succ.get(id) ?? []).some((s) => stuck.has(s))) {
          stuck.delete(id);
          changed = true;
        }
      }
    }
    const members = nodes.filter((n) => stuck.has(n.id)).map((n) => n.id);
    const first = members[0];
    if (first !== undefined) {
      const self = members.length === 1 && succ.get(first)?.includes(first);
      issues.push({
        at: { node: first },
        message: self
          ? `"${first}" is wired to itself; remove that wire.`
          : `The wires form a loop through ${members.map((m) => `"${m}"`).join(', ')}; remove one wire to break it.`,
      });
    }
  }

  // ---- widths, in topological order (unknown stays absent)
  const ctx = { dataColumns: data.columns.length, numClasses: data.numClasses };
  const inWidth: Record<NodeId, number> = Object.create(null) as Record<NodeId, number>; // no prototype: ids like "constructor" are safe
  const outWidth: Record<NodeId, number> = Object.create(null) as Record<NodeId, number>; // no prototype: ids like "constructor" are safe
  const params: Record<NodeId, number> = Object.create(null) as Record<NodeId, number>; // no prototype: ids like "constructor" are safe

  for (const id of topo) {
    const node = byId.get(id);
    if (!node) continue;
    if (node.type === 'loss') continue; // checked below
    const ports = CORE_PIECES[node.type].inputPorts;
    const inW: Record<string, number> = {};
    let unknownUpstream = false;
    const missing: string[] = [];
    for (const port of ports) {
      const edge = inEdges.get(id)?.get(port);
      if (!edge) {
        missing.push(port);
        continue;
      }
      const w = outWidth[edge.from];
      if (w === undefined) unknownUpstream = true;
      else inW[port] = w;
    }
    if (node.type === 'input') {
      const dup = node.columns.filter((c, i) => node.columns.indexOf(c) !== i);
      if (dup.length > 0) {
        issues.push({
          at: { node: id },
          message: `Input lists ${[...new Set(dup)].map((c) => `"${c}"`).join(', ')} more than once; switch each column on only once.`,
        });
      }
      const unknown = node.columns.filter((c) => !data.columns.includes(c));
      if (unknown.length > 0) {
        issues.push({
          at: { node: id },
          message: `Input uses ${unknown.map((c) => `"${c}"`).join(', ')}, which this dataset does not have.`,
        });
      }
    }
    if (missing.length > 0) {
      // The piece's own sentence ("Linear needs an input.") from the registry.
      const r = (CORE_PIECES[node.type].outWidth as WidthFn)(node, inW, ctx);
      issues.push({
        at: { node: id },
        message: isWidthError(r) ? r.error : `${node.type} has an input that is not connected.`,
      });
      continue;
    }
    if (unknownUpstream) continue;
    const r = (CORE_PIECES[node.type].outWidth as WidthFn)(node, inW, ctx);
    if (isWidthError(r)) {
      const at: Issue['at'] = { node: id };
      if (node.type === 'add') {
        const b = inEdges.get(id)?.get('b');
        if (b) at.edge = b.index;
      }
      issues.push({ at, message: r.error });
      continue;
    }
    outWidth[id] = r;
    const single =
      ports.length === 1 ? inW[ports[0] ?? ''] : node.type === 'add' ? inW['a'] : undefined;
    if (single !== undefined) inWidth[id] = single;
  }

  // ---- parameters
  for (const n of nodes) {
    const w = inWidth[n.id];
    if (w === undefined) continue;
    const shapes = (CORE_PIECES[n.type].paramShapes as ShapeFn)(n, w);
    params[n.id] = (shapes.W ? shapes.W[0] * shapes.W[1] : 0) + (shapes.b ? shapes.b[0] : 0);
  }

  // ---- loss, output and target
  const losses = nodes.filter((n) => n.type === 'loss');
  let lossNode: NodeId | null = null;
  let predNode: NodeId | null = null;
  const firstLoss = losses[0];
  if (losses.length === 0) {
    issues.push({ at: {}, message: 'Add a loss to the graph so there is something to train.' });
  } else if (losses.length > 1) {
    for (const l of losses.slice(1)) {
      issues.push({
        at: { node: l.id },
        message: `The graph has ${losses.length} losses; keep only one.`,
      });
    }
  } else if (firstLoss?.type === 'loss') {
    lossNode = firstLoss.id;
    const loss = firstLoss;
    const label = LOSS_LABEL[loss.fn];
    const pred = inEdges.get(loss.id)?.get('pred');
    const target = inEdges.get(loss.id)?.get('target');
    const predSrc = pred ? byId.get(pred.from) : undefined;
    const targetSrc = target ? byId.get(target.from) : undefined;
    if (!pred) {
      issues.push({ at: { node: loss.id }, message: 'Connect the output to the loss.' });
    } else if (predSrc?.type !== 'output') {
      issues.push({
        at: { edge: pred.index, node: loss.id },
        message: `${label} compares the Output piece with the target, so connect the Output to it.`,
      });
    } else {
      predNode = predSrc.id;
    }
    if (!target) {
      issues.push({ at: { node: loss.id }, message: 'Connect the target to the loss.' });
    } else if (targetSrc?.type !== 'target') {
      issues.push({
        at: { edge: target.index, node: loss.id },
        message: `${label} needs the Target piece on its target input.`,
      });
    }
    const predWidth = pred ? outWidth[pred.from] : undefined;
    if (predWidth !== undefined) inWidth[loss.id] = predWidth;

    // Output/loss/target table (CONTRACTS.md section 4).
    if (predSrc?.type === 'output') {
      const link = predSrc.link;
      const w = outWidth[predSrc.id];
      if (loss.fn === 'log_loss') {
        if (link !== 'sigmoid') {
          issues.push({
            at: { node: predSrc.id },
            message: 'Log loss needs a sigmoid output.',
          });
        }
      } else if (loss.fn === 'cross_entropy') {
        if (link !== 'softmax') {
          issues.push({
            at: { node: predSrc.id },
            message: 'Cross-entropy needs a softmax output.',
          });
        }
      } else if (link === 'softmax') {
        issues.push({
          at: { node: predSrc.id },
          message: `${label} needs an identity or sigmoid output, not softmax.`,
        });
      }
      if (w !== undefined) {
        const want = loss.fn === 'cross_entropy' ? data.numClasses : 1;
        if (w !== want) {
          issues.push({
            at: { node: predSrc.id },
            message:
              loss.fn === 'cross_entropy'
                ? `Cross-entropy needs an output of width ${want}, one per class, but this output has width ${w}.`
                : `${label} needs an output of width 1, but this output has width ${w}.`,
          });
        }
      }
    }
    if ((loss.fn === 'log_loss' || loss.fn === 'cross_entropy') && data.kind !== 'binary') {
      issues.push({
        at: { node: loss.id },
        message: `${label} needs a dataset with two classes, but this dataset is regression.`,
      });
    }
  }

  // ---- wiring rules: the target only feeds the loss; the output only feeds the loss
  for (const e of used) {
    const from = byId.get(e.from);
    const to = byId.get(e.to);
    if (!from || !to) continue;
    if (from.type === 'target' && !(to.type === 'loss' && e.port === 'target')) {
      issues.push({
        at: { edge: e.index },
        message: 'The target can only be connected to the loss.',
      });
    } else if (from.type === 'output' && !(to.type === 'loss' && e.port === 'pred')) {
      issues.push({
        at: { edge: e.index },
        message: 'The output can only be connected to the loss.',
      });
    }
  }

  // ---- order: ancestors of the single loss, else every acyclic node
  let order = topo;
  if (lossNode !== null && done.has(lossNode)) {
    const keep = new Set<NodeId>([lossNode]);
    const stack = [lossNode];
    while (stack.length > 0) {
      const cur = stack.pop();
      if (cur === undefined) break;
      for (const used of inEdges.get(cur)?.values() ?? []) {
        if (!keep.has(used.from)) {
          keep.add(used.from);
          stack.push(used.from);
        }
      }
    }
    order = topo.filter((id) => keep.has(id));
  }

  // ---- parameter limit
  let totalParams = 0;
  for (const id of order) totalParams += params[id] ?? 0;
  if (totalParams > MAX_PARAMS) {
    issues.push({
      at: {},
      message: `This model has ${totalParams.toLocaleString('en-US')} parameters; the limit is ${MAX_PARAMS.toLocaleString('en-US')}.`,
    });
  }

  // ---- wire widths
  const edgeWidth: (number | null)[] = graph.edges.map(() => null);
  for (const e of used) edgeWidth[e.index] = outWidth[e.from] ?? null;

  return {
    order,
    inWidth,
    outWidth,
    edgeWidth,
    params,
    totalParams,
    predNode,
    lossNode,
    issues,
    trainable: issues.length === 0,
  };
}

// The registry's `{ [T]: CorePiece<NodeOf<T>> }` does not distribute over a `Node` we looked up by id, so
// these two casts re-state the shape. Safe: each call passes the node whose own `type` selects the piece.
type WidthFn = (
  node: Node,
  inW: Record<string, number>,
  ctx: { dataColumns: number; numClasses: number },
) => number | { error: string };
type ShapeFn = (node: Node, inW: number) => { W?: [number, number]; b?: [number] };
