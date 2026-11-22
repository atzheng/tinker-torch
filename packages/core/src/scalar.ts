/**
 * Scalar view <-> Linear (CONTRACTS.md sections 5 and 8): "Group into Linear" and "Expand to scalars".
 *
 * Each is one pure function of (graph, host params) that returns the new graph and the new params
 * together, so the web applies it as one undo step. Values are copied exactly (never re-initialised):
 * a Linear's `W[k][0]` is the Weight of the k-th column and its `b[0]` is the Bias.
 *
 * Pattern (`findLinearGroups`): a Sum S whose every input wire comes from a distinct Weight W_i, each
 * W_i fed by its own Column X_i that names a dataset column (x1..xd of the graph's data set). W_i and
 * S have no wires other than the pattern's. An optional Bias B joins the pattern when it is S's only
 * consumer and takes only S. Only Columns are recognised as feeders (another width-1 source is not
 * yet). A Column with other wires out or a pin stays in the graph; otherwise it is removed.
 *
 * Choices: grouping always makes a fresh Input (smallest free `input<k>`) and a fresh Linear
 * (`linear<k>`), with the columns sorted by dataset index (x1 < x2 ...) and the weights permuted to
 * match, so any wiring order gives the same Linear. Expanding walks the Input's columns in order, so
 * Sum's inputs (and its fold order) follow the Input's column order.
 */
import { categoryOf, type Analysis } from './analyse.js';
import { lookupCategories } from './layout.js';
import { freshId, LOCKED_MESSAGE, type EditOptions } from './edit.js';
import { SETS } from './data/sets.js';
import {
  parseGraph,
  type DataInfo,
  parsePort,
  resolveEdge,
  type Edge,
  type Graph,
  type Node,
  type NodeId,
  type PortRef,
} from './graph.js';
import type { HostLayer, HostParams } from './params.js';

export type ScalarEditResult =
  { ok: true; graph: Graph; host: HostParams; select: NodeId } | { ok: false; error: string };

const fail = (error: string): ScalarEditResult => ({ ok: false, error });

const MISSING = 'The values of these pieces are not known yet, so they cannot be carried over.';
const INVALID = 'That change would leave an invalid graph, so it was not made.';
const NOT_A_GROUP =
  'These pieces are not exactly one set of Columns, Weights and a Sum that a Linear can replace.';

type Match = {
  sum: NodeId;
  /** Column ids, one per Weight, in Sum's input order. */
  columns: NodeId[];
  weights: NodeId[];
  bias: NodeId | null;
  /** The node whose consumers the Linear takes over: the Bias, else the Sum. */
  last: NodeId;
};

type Wire = { index: number; from: NodeId; to: NodeId; port: string };

function wiresOf(g: Graph): Wire[] {
  const out: Wire[] = [];
  g.edges.forEach((e, index) => {
    const r = resolveEdge(g.nodes, e);
    if (r.ok) out.push({ index, from: r.from.node, to: r.to.node, port: r.to.port });
  });
  return out;
}

const pinNode = (ref: PortRef): NodeId => parsePort(ref).node;

const isPinned = (g: Graph, id: NodeId): boolean =>
  g.predictions.some((p) => pinNode(p) === id) || (g.loss !== null && pinNode(g.loss) === id);

/** The pattern around Sum `sumId`, or null when it is not one. */
function matchAt(g: Graph, sumId: NodeId, wires: Wire[]): Match | null {
  const byId = new Map(g.nodes.map((n) => [n.id, n]));
  if (byId.get(sumId)?.type !== 'sum') return null;
  const d = SETS[g.data.set].d;
  const into = (id: NodeId) => wires.filter((x) => x.to === id);
  const outOf = (id: NodeId) => wires.filter((x) => x.from === id);
  const ins = into(sumId);
  if (ins.length === 0) return null;
  const weights: NodeId[] = [];
  const columns: NodeId[] = [];
  const names = new Set<string>();
  for (const w of ins) {
    const wn = byId.get(w.from);
    if (wn?.type !== 'scalew' || weights.includes(w.from)) return null;
    // A Weight has one wire in (from its Column) and one out (to the Sum).
    const feeds = into(w.from);
    const col = feeds.length === 1 ? byId.get(feeds[0]?.from ?? '') : undefined;
    if (col?.type !== 'column' || outOf(w.from).length !== 1) return null;
    const k = /^x(\d+)$/.exec(col.name);
    if (!k || Number(k[1]) < 1 || Number(k[1]) > d || names.has(col.name)) return null;
    names.add(col.name);
    weights.push(w.from);
    columns.push(col.id);
  }
  const sOut = outOf(sumId);
  const only = sOut[0];
  const bias =
    sOut.length === 1 && only && byId.get(only.to)?.type === 'bias' && into(only.to).length === 1
      ? only.to
      : null;
  return { sum: sumId, columns, weights, bias, last: bias ?? sumId };
}

const idsOf = (g: Graph, m: Match): NodeId[] => {
  const all = new Set<NodeId>([...m.columns, ...m.weights, m.sum, ...(m.bias ? [m.bias] : [])]);
  return g.nodes.filter((n) => all.has(n.id)).map((n) => n.id);
};

/**
 * Every Sum that heads a "Columns -> Weights -> Sum (-> Bias)" pattern (see the file comment), as the
 * ids of the pieces it would replace (Columns, Weights, Sum, Bias if any) in node order. A structural
 * check: a group with an analysis issue on one of its pieces is skipped; locks and pins are checked by
 * `groupIntoLinear`.
 */
export function findLinearGroups(graph: Graph, analysis: Analysis): NodeId[][] {
  const wires = wiresOf(graph);
  const out: NodeId[][] = [];
  for (const n of graph.nodes) {
    const m = n.type === 'sum' ? matchAt(graph, n.id, wires) : null;
    if (m === null) continue;
    const ids = idsOf(graph, m);
    const bad = analysis.issues.some((i) => i.at.node !== undefined && ids.includes(i.at.node));
    if (!bad) out.push(ids);
  }
  return out;
}

function locked(opts: EditOptions | undefined, ids: Iterable<NodeId>): boolean {
  const l = opts?.locked;
  if (!l || l.length === 0) return false;
  for (const id of ids) if (l.includes(id)) return true;
  return false;
}

/** A copy of `host` without `drop` and with `add` (other layers are the same objects). */
function newHost(
  host: HostParams,
  drop: Iterable<NodeId>,
  add: Record<NodeId, HostLayer>,
): HostParams {
  const out = Object.create(null) as HostParams;
  const gone = new Set(drop);
  for (const id of Object.keys(host)) {
    const l = host[id];
    if (l && !gone.has(id)) out[id] = l;
  }
  for (const id of Object.keys(add)) {
    const l = add[id];
    if (l) out[id] = l;
  }
  return out;
}

/**
 * Replaces the pattern named by `ids` (as returned by `findLinearGroups`) by Input(columns) ->
 * Linear(d, 1), bias iff the pattern has a Bias. W[k][0] is the Weight of the k-th column by dataset
 * index, b is the Bias. Consumers of the Bias (else the Sum) read the Linear, their wires keeping their
 * place in the edge list. A ŷ or L pin on that last node moves to the Linear. Refused (one sentence)
 * when `ids` is not exactly one pattern, a pin sits on a Weight or on a Sum that a Bias follows, a
 * piece that is removed or changed is locked (a Column that stays does not count), or the host has no
 * values for a Weight or the Bias. `select` is the Linear.
 */
export function groupIntoLinear(
  graph: Graph,
  host: HostParams,
  ids: readonly NodeId[],
  opts?: EditOptions,
): ScalarEditResult {
  const wires = wiresOf(graph);
  const sums = ids.filter((id) => graph.nodes.find((n) => n.id === id)?.type === 'sum');
  const sumId = sums[0];
  const m = sums.length === 1 && sumId !== undefined ? matchAt(graph, sumId, wires) : null;
  if (m === null) return fail(NOT_A_GROUP);
  const want = idsOf(graph, m);
  if (want.length !== new Set(ids).size || !want.every((id) => ids.includes(id))) {
    return fail(NOT_A_GROUP);
  }
  const inner = [...m.weights, ...(m.bias ? [m.sum] : [])];
  if (inner.some((id) => isPinned(graph, id))) {
    return fail('A piece inside the group carries a ŷ or L pin; move the pin first.');
  }
  const stays = (c: NodeId) =>
    isPinned(graph, c) || wires.some((x) => x.from === c && !m.weights.includes(x.to));
  const removed = new Set<NodeId>([
    ...m.weights,
    m.sum,
    ...(m.bias ? [m.bias] : []),
    ...m.columns.filter((c) => !stays(c)),
  ]);
  if (locked(opts, removed)) return fail(LOCKED_MESSAGE);

  const wVals = m.weights.map((w) => host[w]?.['W']?.[0]);
  const biasVals = m.bias === null ? undefined : host[m.bias]?.['b'];
  if (wVals.some((v) => v === undefined) || (m.bias !== null && biasVals?.length !== 1)) {
    return fail(MISSING);
  }

  // Dataset order: x<k> by k, the weights permuted with their columns.
  const order = m.columns
    .map((c, i) => {
      const n = graph.nodes.find((x) => x.id === c);
      return { name: n?.type === 'column' ? n.name : '', w: wVals[i] ?? 0 };
    })
    .sort((p, q) => Number(p.name.slice(1)) - Number(q.name.slice(1)));

  const inputId = freshId(graph, 'input');
  const input: Node = { id: inputId, type: 'input', columns: order.map((o) => o.name) };
  const linearId = freshId({ ...graph, nodes: [...graph.nodes, input] }, 'linear');
  const nodes: Node[] = [
    ...graph.nodes.filter((n) => !removed.has(n.id)),
    input,
    { id: linearId, type: 'linear', out: 1, bias: m.bias !== null },
  ];
  const edges: Edge[] = [];
  const wireAt = new Map(wires.map((w) => [w.index, w]));
  graph.edges.forEach((e, i) => {
    const w = wireAt.get(i);
    if (!w) edges.push(e);
    else if (removed.has(w.to)) return;
    else edges.push(w.from === m.last ? [linearId, e[1]] : e);
  });
  edges.push([inputId, linearId]);
  const move = (ref: PortRef): PortRef => (pinNode(ref) === m.last ? linearId : ref);
  const parsed = parseGraph({
    ...graph,
    nodes,
    edges,
    predictions: graph.predictions.map(move),
    loss: graph.loss === null ? null : move(graph.loss),
  });
  if (!parsed.ok) return fail(INVALID);
  const layer: HostLayer = { W: Float32Array.from(order.map((o) => o.w)) };
  if (biasVals) layer['b'] = Float32Array.from(biasVals);
  return {
    ok: true,
    graph: parsed.graph,
    host: newHost(host, removed, { [linearId]: layer }),
    select: linearId,
  };
}

/**
 * Replaces a Linear with one output, fed by an Input of d >= 1 columns, by d Columns (the Input's
 * columns, in its order), d Weights, one Sum and, when the Linear has a bias, a Bias. Ids are the
 * smallest free `x<k>`, `w<k>`, `sum<k>`, `bias<k>`. The Input is removed when nothing else reads it
 * and no pin sits on it. Weight k takes W[k][0], the Bias takes b. Consumers of the Linear read the
 * last node (Bias, else Sum), keeping their place in the edge list; a ŷ or L pin on the Linear moves
 * there. Refused (one sentence) for another kind of piece, out > 1, a feeder that is not an Input,
 * an Input with no columns, a Weights tap reading the Linear, a locked Linear (or a locked Input that
 * would be removed) or missing values. `select` is the last node.
 */
export function expandToScalars(
  graph: Graph,
  host: HostParams,
  linearId: NodeId,
  opts?: EditOptions,
): ScalarEditResult {
  const lin = graph.nodes.find((n) => n.id === linearId);
  if (lin?.type !== 'linear') return fail('Expand to scalars works on a Linear piece.');
  if (lin.out !== 1) return fail('Expand to scalars works on a Linear with one output.');
  const wires = wiresOf(graph);
  const feed = wires.filter((x) => x.to === linearId);
  const inp = feed.length === 1 ? graph.nodes.find((n) => n.id === feed[0]?.from) : undefined;
  if (inp?.type !== 'input') return fail('Expand to scalars works on a Linear fed by an Input.');
  if (inp.columns.length === 0) return fail('The Input has no columns to expand.');
  if (graph.nodes.some((n) => n.type === 'weights' && n.of === linearId)) {
    return fail('A Weights piece reads this Linear, so it cannot be expanded.');
  }
  const d = inp.columns.length;
  const W = host[linearId]?.['W'];
  const b = host[linearId]?.['b'];
  if (W?.length !== d || (lin.bias && b?.length !== 1)) return fail(MISSING);
  const keepInput =
    isPinned(graph, inp.id) || wires.some((x) => x.from === inp.id && x.to !== linearId);
  const removed = new Set<NodeId>([linearId, ...(keepInput ? [] : [inp.id])]);
  if (locked(opts, removed)) return fail(LOCKED_MESSAGE);

  const added: Node[] = [];
  const add = (node: Node extends infer N ? (N extends Node ? Omit<N, 'id'> : never) : never) => {
    const id = freshId({ ...graph, nodes: [...graph.nodes, ...added] }, node.type);
    added.push({ id, ...node } as Node);
    return id;
  };
  const cols = inp.columns.map((c) => add({ type: 'column', name: c }));
  const ws = cols.map(() => add({ type: 'scalew' }));
  const sumId = add({ type: 'sum' });
  const biasId = lin.bias ? add({ type: 'bias' }) : null;
  const last = biasId ?? sumId;

  const edges: Edge[] = [];
  const wireAt = new Map(wires.map((w) => [w.index, w]));
  graph.edges.forEach((e, i) => {
    const w = wireAt.get(i);
    if (!w) edges.push(e);
    else if (removed.has(w.to)) return;
    else edges.push(w.from === linearId ? [last, e[1]] : e);
  });
  cols.forEach((c, k) => {
    const w = ws[k] ?? '';
    edges.push([c, w], [w, sumId]);
  });
  if (biasId) edges.push([sumId, biasId]);
  const move = (ref: PortRef): PortRef => (pinNode(ref) === linearId ? last : ref);
  const parsed = parseGraph({
    ...graph,
    nodes: [...graph.nodes.filter((n) => !removed.has(n.id)), ...added],
    edges,
    predictions: graph.predictions.map(move),
    loss: graph.loss === null ? null : move(graph.loss),
  });
  if (!parsed.ok) return fail(INVALID);
  const layers: Record<NodeId, HostLayer> = {};
  ws.forEach((w, k) => {
    layers[w] = { W: Float32Array.of(W[k] ?? 0) };
  });
  if (biasId && b) layers[biasId] = { b: Float32Array.from(b) };
  return { ok: true, graph: parsed.graph, host: newHost(host, removed, layers), select: last };
}

/**
 * One draggable per-category value at neuron grain (tinker-4w2). `target` is the parameter the
 * dial sets: `W[i][j]` of `node` (a Lookup's table row `i`, or row `i` of the Linear that a
 * One-hot feeds alone). Null when the piece has no single scalar parameter per category (a One-hot
 * with no Linear of out 1 after it, or with other consumers), so the dial is only drawn.
 */
export type CategoryDial = {
  /** The One-hot or Lookup the dial is drawn on. */
  node: NodeId;
  /** Category index. */
  index: number;
  /** Category name, `category k` while the data has no names. */
  name: string;
  target: { node: NodeId; i: number; j: number } | null;
};

/**
 * The dials of every One-hot and every Lookup of width 1, in graph order and category order. A
 * Lookup of width > 1 has no scalar per category: it gets none (it stays a layer box). Pure: the
 * values live in the host params, `W[i * out + j]`.
 */
export function categoryDials(graph: Graph, analysis: Analysis, data: DataInfo): CategoryDial[] {
  const dials: CategoryDial[] = [];
  for (const n of graph.nodes) {
    if (n.type !== 'onehot' && !(n.type === 'lookup' && n.width === 1)) continue;
    const k = n.type === 'onehot' ? analysis.outWidth[n.id] : lookupCategories(analysis, n.id);
    if (k === undefined || k < 1) continue;
    const names = categoryOf(analysis, data, n.id)?.names;
    let target: ((i: number) => CategoryDial['target']) | null = null;
    if (n.type === 'lookup') target = (i) => ({ node: n.id, i, j: 0 });
    else {
      const out = graph.edges.filter((e) => parsePort(e[0]).node === n.id);
      const first = out[0];
      const to =
        out.length === 1 && first
          ? graph.nodes.find((m) => m.id === parsePort(first[1]).node)
          : undefined;
      if (to?.type === 'linear' && to.out === 1) target = (i) => ({ node: to.id, i, j: 0 });
    }
    for (let i = 0; i < k; i++) {
      dials.push({
        node: n.id,
        index: i,
        name: names?.[i] ?? `category ${i}`,
        target: target ? target(i) : null,
      });
    }
  }
  return dials;
}
