/**
 * Analysis for graph v2 (CONTRACTS.md section 4): shapes, roles, regions, order, parameter counts and
 * the issues that disable Train. Pure; no DOM.
 */
import type { DataInfo, Graph, Node, NodeId } from './graph.js';
import { isVariadicPort, resolveEdge, resolvePin } from './graph.js';
import {
  CORE_PIECES,
  describeShape,
  ownsParams,
  sameShape,
  type ParamInit,
  type ParamShapes,
  type ShapeCtx,
  type Shape,
} from './registry.js';

export type Pin = { role: 'loss' } | { role: 'prediction'; index: number };
export type Issue = {
  /** `edge` is an index into graph.edges; `pin` names the pin (its node is in `node` too). */
  at: { node?: NodeId; edge?: number; pin?: Pin };
  message: string;
};
export type Region = 'model' | 'objective';

export type Analysis = {
  /** Topological order of what the roles need (ancestors of L and of the ŷ pins); every acyclic node when no role resolves. */
  order: NodeId[];
  /** What each node outputs. Absent where unknown. */
  shape: Record<NodeId, Shape>;
  /** Example width entering each node (its first input), absent when unknown or the node has no per-example input. */
  inWidth: Record<NodeId, number>;
  /** Example width leaving each node; absent for scalars, parameter matrices and unknowns. */
  outWidth: Record<NodeId, number>;
  /**
   * Per node, per input port: the wires that feed it, as `graph.edges` indices with their source, in
   * edge order. A plain port has at most one entry; a variadic port (CONTRACTS.md section 3) has
   * all its wires, which the engine folds in this order. Only wires that resolved are listed.
   */
  inputs: Record<NodeId, Record<string, { edge: number; from: NodeId }[]>>;
  /** Per graph.edges index: the shape the wire carries, null where unknown. */
  edgeShape: (Shape | null)[];
  /** Per graph.edges index: the example width, null for scalars, matrices and unknowns. */
  edgeWidth: (number | null)[];
  /** Parameters per node whose input width is known (0 for pieces with none). */
  params: Record<NodeId, number>;
  /** Named tensor shapes of each parameter-owning node (Linear: W, b) whose input width is known. */
  paramShapes: Record<NodeId, ParamShapes>;
  /** The init of each node in `paramShapes` (from the piece's `paramInit`). */
  paramInit: Record<NodeId, ParamInit>;
  /** The piece type of each node in `paramShapes`, so `reconcileParams` can tell a reused id from the same node. */
  paramType: Record<NodeId, string>;
  /** Parameter-owning nodes whose parameters are live: in `order`, or a Linear read by a Weights tap in `order`. In node order. */
  paramNodes: NodeId[];
  /** Sum of `params` over `paramNodes`. */
  totalParams: number;
  /** The nodes carrying the ŷ pins, in pin order (pins that name no node are skipped). */
  predictions: NodeId[];
  /** The node carrying the L pin, or null. */
  lossNode: NodeId | null;
  /** Membership: absent = neither. Model = ancestors of the ŷ pins; objective = ancestors of L not in the model. */
  region: Record<NodeId, Region>;
  /** `order` restricted to the model region: what predict() runs (dropout off). */
  modelOrder: NodeId[];
  /** Weights taps: the dashed parameter wire from a Linear to the Weights node reading it. */
  paramWires: { linear: NodeId; weights: NodeId }[];
  /** Link nodes that feed a matching loss (sigmoid -> log_loss, softmax -> cross_entropy): the engine may fuse them. */
  fusable: { link: NodeId; loss: NodeId }[];
  issues: Issue[];
  /** True when there are no issues. */
  trainable: boolean;
};

export const MAX_PARAMS = 10_000;

// The registry's `{ [T]: CorePiece<NodeOf<T>> }` does not distribute over a `Node` we looked up by id, so
// these casts re-state the shape. Safe: each call passes the node whose own `type` selects the piece.
type ShapeFn = (
  node: Node,
  inShapes: Record<string, Shape>,
  ctx: ShapeCtx,
) => Shape | { error: string };
type ParamFn = (node: Node, inW: number) => ParamShapes;
type InitFn = (node: Node, inW: number, data: DataInfo) => ParamInit;

const record = <T>() => Object.create(null) as Record<NodeId, T>; // no prototype: ids like "constructor" are safe

export function analyse(graph: Graph, data: DataInfo): Analysis {
  const issues: Issue[] = [];
  // Defensive: parseGraph rejects duplicate ids, but an unparsed graph may have them. The first wins.
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

  // ---- resolve edges; a plain input port keeps its first wire, a variadic port keeps all of them
  type Used = { index: number; from: NodeId };
  const inEdges = new Map<NodeId, Map<string, Used[]>>();
  const wireOf = (id: NodeId, port: string): Used | undefined => inEdges.get(id)?.get(port)?.[0];
  const used: { index: number; from: NodeId; to: NodeId; port: string }[] = [];
  graph.edges.forEach((edge, index) => {
    const r = resolveEdge(nodes, edge);
    if (!r.ok) {
      issues.push({ at: { edge: index }, message: r.error });
      return;
    }
    let ports = inEdges.get(r.to.node);
    if (!ports) inEdges.set(r.to.node, (ports = new Map()));
    const list = ports.get(r.to.port);
    const toType = byId.get(r.to.node)?.type;
    if (list && !(toType !== undefined && isVariadicPort(toType, r.to.port))) {
      issues.push({
        at: { edge: index, node: r.to.node },
        message: `"${r.to.node}" has two wires into its ${r.to.port} input; each input takes only one.`,
      });
      return;
    }
    if (list?.some((x) => x.from === r.from.node)) {
      issues.push({
        at: { edge: index, node: r.to.node },
        message: `"${r.from.node}" is wired into the ${r.to.port} input of "${r.to.node}" twice; remove one of the two wires.`,
      });
      return;
    }
    const w = { index, from: r.from.node };
    if (list) list.push(w);
    else ports.set(r.to.port, [w]);
    used.push({ index, from: r.from.node, to: r.to.node, port: r.to.port });
  });

  // ---- topological sort (Kahn, node order). Data wires only: `weights.of` is no dependency.
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

  // ---- shapes. Memoised and demand-driven, because a Weights tap needs the size of its Linear's input,
  // which may come later in topological order (there is no data wire between them).
  const ctx: ShapeCtx = {
    dataColumns: data.columns.length,
    numClasses: data.numClasses,
    weightShape: (linearId) => {
      const lin = byId.get(linearId);
      if (lin?.type !== 'linear') return undefined;
      const src = wireOf(linearId, 'in');
      const s = src ? shapeOf(src.from) : undefined;
      return s?.kind === 'example' ? [s.width, lin.out] : undefined;
    },
  };
  const shape = record<Shape>();
  const inWidth = record<number>();
  const outWidth = record<number>();
  const params = record<number>();
  const paramShapes = record<ParamShapes>();
  const paramInit = record<ParamInit>();
  const paramType = record<string>();
  const computed = new Set<NodeId>();
  const visiting = new Set<NodeId>();
  let loopHit = false; // set when a shape lookup re-enters a node that is still being computed

  function shapeOf(id: NodeId): Shape | undefined {
    if (computed.has(id)) return shape[id];
    const node = byId.get(id);
    if (!node || !done.has(id)) return undefined;
    if (visiting.has(id)) {
      loopHit = true;
      return undefined;
    }
    visiting.add(id);
    compute(node);
    visiting.delete(id);
    computed.add(id);
    return shape[id];
  }

  function compute(node: Node): void {
    const id = node.id;
    const piece = CORE_PIECES[node.type];
    const ports = piece.inputPorts;
    const inS: Record<string, Shape> = {};
    let unknownUpstream = false;
    let missing = false;
    let mismatch: { edge: number; message: string } | undefined;
    for (const port of ports) {
      const wires = inEdges.get(id)?.get(port) ?? [];
      if (wires.length === 0) {
        missing = true;
        continue;
      }
      const shapes = wires.map((w) => shapeOf(w.from));
      const s = shapes[0];
      if (shapes.some((x) => x === undefined) || s === undefined) unknownUpstream = true;
      else {
        inS[port] = s;
        if (isVariadicPort(node.type, port)) {
          shapes.forEach((x, k) => {
            if (k > 0 && x !== undefined && !sameShape(s, x)) {
              mismatch ??= {
                edge: (wires[k] as Used).index,
                message: `"${id}" needs every wire into its ${port} input to carry the same kind of value: the first is ${describeShape(s)}, but this one is ${describeShape(x)}.`,
              };
            }
          });
        }
      }
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
    if (node.type === 'weights') {
      const target = byId.get(node.of);
      if (!target) {
        issues.push({
          at: { node: id },
          message: `Weights reads "${node.of}", which is not a piece in this graph.`,
        });
        return;
      }
      if (target.type !== 'linear') {
        issues.push({
          at: { node: id },
          message: `Weights must read a Linear piece, but "${node.of}" is not one.`,
        });
        return;
      }
      loopHit = false;
      if (ctx.weightShape(node.of) === undefined) {
        // Otherwise the root cause is at the Linear's input and has its own issue.
        if (loopHit) {
          issues.push({
            at: { node: id },
            message: `"${id}" reads the weights of "${node.of}", which depends on "${id}" itself.`,
          });
        }
        return;
      }
    }
    const r = (piece.outShape as ShapeFn)(node, inS, ctx);
    if (missing) {
      // The piece's own sentence ("Linear needs an input.") from the registry.
      issues.push({
        at: { node: id },
        message: 'error' in r ? r.error : `${node.type} has an input that is not connected.`,
      });
      return;
    }
    if (unknownUpstream) return;
    if (mismatch) {
      issues.push({ at: { node: id, edge: mismatch.edge }, message: mismatch.message });
      return;
    }
    if ('error' in r) {
      const at: Issue['at'] = { node: id };
      if (node.type === 'add' || node.type === 'subtract') {
        const b = wireOf(id, 'b');
        if (b) at.edge = b.index;
      }
      issues.push({ at, message: r.error });
      return;
    }
    shape[id] = r;
    if (r.kind === 'example') outWidth[id] = r.width;
    const firstPort = ports[0];
    const firstIn = firstPort === undefined ? undefined : inS[firstPort];
    if (firstIn?.kind === 'example') inWidth[id] = firstIn.width;
    // A piece with no input port that owns parameters (e.g. one learnable scalar) needs no width.
    const w = firstIn?.kind === 'example' ? firstIn.width : ports.length === 0 ? 0 : undefined;
    if (w !== undefined) {
      const ps = (piece.paramShapes as ParamFn)(node, w);
      let count = 0;
      for (const k of Object.keys(ps)) count += (ps[k] ?? []).reduce((a, d) => a * d, 1);
      params[id] = count;
      if (Object.keys(ps).length > 0) {
        paramShapes[id] = ps;
        paramInit[id] = (piece.paramInit as InitFn)(node, w, data);
        paramType[id] = node.type;
      }
    }
  }

  for (const id of topo) shapeOf(id);

  // ---- ancestors over data wires (the first wire into each port)
  const ancestors = (roots: NodeId[]): Set<NodeId> => {
    const stack = roots.filter((r) => done.has(r));
    const keep = new Set<NodeId>(stack);
    while (stack.length > 0) {
      const cur = stack.pop();
      if (cur === undefined) break;
      for (const list of inEdges.get(cur)?.values() ?? []) {
        for (const u of list) {
          if (!keep.has(u.from)) {
            keep.add(u.from);
            stack.push(u.from);
          }
        }
      }
    }
    return keep;
  };

  // ---- pins: the ŷ pins and L
  const predictions: NodeId[] = [];
  const predPins: { index: number; node: NodeId }[] = [];
  graph.predictions.forEach((ref, index) => {
    const r = resolvePin(nodes, ref);
    if (!r.ok) {
      issues.push({ at: { pin: { role: 'prediction', index } }, message: r.error });
      return;
    }
    if (predictions.includes(r.at.node)) {
      issues.push({
        at: { node: r.at.node, pin: { role: 'prediction', index } },
        message: `"${r.at.node}" is pinned as a prediction twice.`,
      });
      return;
    }
    predictions.push(r.at.node);
    predPins.push({ index, node: r.at.node });
  });
  let lossNode: NodeId | null = null;
  if (graph.loss !== null) {
    const r = resolvePin(nodes, graph.loss);
    if (r.ok) lossNode = r.at.node;
    else issues.push({ at: { pin: { role: 'loss' } }, message: r.error });
  } else {
    issues.push({
      at: {},
      message:
        'No loss is pinned; pin a single-number output as the loss so there is something to train.',
    });
  }

  // ---- region sets and order
  const roots = [...predictions, ...(lossNode !== null ? [lossNode] : [])].filter((r) =>
    done.has(r),
  );
  const modelSet = ancestors(predictions);
  const lossSet = lossNode !== null ? ancestors([lossNode]) : new Set<NodeId>();
  const needed = new Set<NodeId>([...modelSet, ...lossSet]);
  const order = roots.length > 0 ? topo.filter((id) => needed.has(id)) : topo;
  const region = record<Region>();
  for (const id of order) {
    if (modelSet.has(id)) region[id] = 'model';
    else if (lossSet.has(id)) region[id] = 'objective';
  }
  const modelOrder = order.filter((id) => region[id] === 'model');
  const inOrder = new Set(order);
  // Safety net: trainable must never hold with a missing shape.
  if (issues.length === 0) {
    const bare = order.find((id) => shape[id] === undefined);
    if (bare !== undefined) {
      issues.push({
        at: { node: bare },
        message: `The size of "${bare}" could not be worked out.`,
      });
    }
  }

  // ---- L: a single number that depends on a parameter
  if (lossNode !== null) {
    const pin: Pin = { role: 'loss' };
    const s = shape[lossNode];
    if (s !== undefined && s.kind !== 'scalar') {
      issues.push({
        at: { node: lossNode, pin },
        message:
          s.kind === 'example'
            ? `The loss must be a single number, but "${lossNode}" gives ${describeShape(s)}; add a Mean to average over the examples.`
            : `The loss must be a single number, but "${lossNode}" is ${describeShape(s)}.`,
      });
    }
    if (done.has(lossNode)) {
      const hasParam = [...lossSet].some((id) => {
        const n = byId.get(id);
        return (
          (n !== undefined && ownsParams(n)) ||
          (n?.type === 'weights' && byId.get(n.of)?.type === 'linear')
        );
      });
      if (!hasParam) {
        issues.push({
          at: { node: lossNode, pin },
          message: `The loss "${lossNode}" does not depend on any parameter, so there is nothing to train.`,
        });
      }
    }
  }

  // ---- ŷ pins: width equals the dataset target; a softmax prediction needs a classification dataset
  for (const { index, node } of predPins) {
    const pin: Pin = { role: 'prediction', index };
    const s = shape[node];
    if (s === undefined) continue;
    if ([...ancestors([node])].some((id) => byId.get(id)?.type === 'target')) {
      issues.push({
        at: { node, pin },
        message: `A prediction cannot depend on the target, but "${node}" does.`,
      });
    }
    const n = byId.get(node);
    const softmax = n?.type === 'link' && n.fn === 'softmax';
    const want = softmax ? data.numClasses : 1;
    if (s.kind !== 'example') {
      issues.push({
        at: { node, pin },
        message: `A prediction must be one row per example, but "${node}" is ${describeShape(s)}.`,
      });
    } else if (s.width !== want) {
      issues.push({
        at: { node, pin },
        message: softmax
          ? `This dataset has ${want} classes, so the softmax prediction needs width ${want}, but "${node}" has width ${s.width}.`
          : `This dataset's target has width 1, so the prediction needs width 1, but "${node}" has width ${s.width}.`,
      });
    }
    if (softmax && data.kind !== 'binary') {
      issues.push({
        at: { node, pin },
        message:
          'A softmax prediction needs a dataset with classes, but this dataset is regression.',
      });
    }
  }

  // ---- log loss and cross-entropy take probabilities from a matching link
  const fusable: Analysis['fusable'] = [];
  for (const id of order) {
    const n = byId.get(id);
    if (n?.type !== 'log_loss' && n?.type !== 'cross_entropy') continue;
    const label = n.type === 'log_loss' ? 'Log loss' : 'Cross-entropy';
    if (data.kind !== 'binary') {
      issues.push({
        at: { node: id },
        message: `${label} needs a dataset with two classes, but this dataset is regression.`,
      });
    }
    const pred = wireOf(id, 'pred');
    if (!pred) continue;
    const src = byId.get(pred.from);
    const wantFn = n.type === 'log_loss' ? 'sigmoid' : 'softmax';
    if (src?.type === 'link' && src.fn === wantFn) fusable.push({ link: src.id, loss: id });
    else {
      issues.push({
        at: { edge: pred.index, node: id },
        message: `${label} takes probabilities, so wire a ${wantFn === 'sigmoid' ? 'Sigmoid' : 'Softmax'} link straight into its pred input.`,
      });
    }
    const w = shape[pred.from];
    if (n.type === 'cross_entropy' && w?.kind === 'example' && w.width !== data.numClasses) {
      issues.push({
        at: { edge: pred.index, node: id },
        message: `Cross-entropy needs one probability per class (${data.numClasses}), but its pred input has width ${w.width}.`,
      });
    }
  }

  // ---- parameter wires and the live Linears
  const paramWires: Analysis['paramWires'] = [];
  const live = new Set<NodeId>();
  for (const id of order) {
    const n = byId.get(id);
    if (n && ownsParams(n)) live.add(id);
  }
  for (const n of nodes) {
    if (n.type !== 'weights') continue;
    const t = byId.get(n.of);
    if (t?.type !== 'linear') continue;
    paramWires.push({ linear: t.id, weights: n.id });
    if (inOrder.has(n.id)) live.add(t.id);
  }
  const paramNodes = nodes.filter((n) => live.has(n.id)).map((n) => n.id);

  // ---- parameter limit
  let totalParams = 0;
  for (const id of paramNodes) totalParams += params[id] ?? 0;
  if (totalParams > MAX_PARAMS) {
    issues.push({
      at: {},
      message: `This model has ${totalParams.toLocaleString('en-US')} parameters; the limit is ${MAX_PARAMS.toLocaleString('en-US')}.`,
    });
  }

  // ---- wire shapes
  const edgeShape: (Shape | null)[] = graph.edges.map(() => null);
  const edgeWidth: (number | null)[] = graph.edges.map(() => null);
  for (const e of used) {
    edgeShape[e.index] = shape[e.from] ?? null;
    edgeWidth[e.index] = outWidth[e.from] ?? null;
  }

  const inputs: Analysis['inputs'] = record();
  for (const [id, ports] of inEdges) {
    const byPort: Record<string, { edge: number; from: NodeId }[]> = {};
    for (const [port, list] of ports)
      byPort[port] = list.map((w) => ({ edge: w.index, from: w.from }));
    inputs[id] = byPort;
  }

  return {
    order,
    inputs,
    shape,
    inWidth,
    outWidth,
    edgeShape,
    edgeWidth,
    params,
    paramShapes,
    paramInit,
    paramType,
    paramNodes,
    totalParams,
    predictions,
    lossNode,
    region,
    modelOrder,
    paramWires,
    fusable,
    issues,
    trainable: issues.length === 0,
  };
}
