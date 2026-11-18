// Graph + Analysis -> Model: the graph walked into jitted predict / loss / step functions.
//
// Ownership (jax-js: functions consume their array arguments, `.ref` adds a reference). Every
// jitted function below CONSUMES all its array arguments; callers pass `tree.ref(...)` for what
// they keep. Jitted functions are built lazily on first use and cached on the model.
import { jit, numpy as np, tree, valueAndGrad } from '@jax-js/jax';
import type { Array as JArray } from '@jax-js/jax';
import * as optax from '@jax-js/optax';
import { analyse, resolveEdge } from '@tinker/core';
import type { Analysis, Batch, DataInfo, Graph, Node, NodeId, OutputLink } from '@tinker/core';
import { CORE_PIECES } from '@tinker/core';
import type { LayerShape, ParamTree } from './convert.js';
import { LOSSES } from './losses.js';
import { OPTIMIZERS } from './optim.js';
import { PENALTY_IMPLEMENTED } from './penalty.js';
import { ENGINE_PIECES } from './pieces/index.js';
import type { PieceCtx } from './pieces/index.js';
import type { Model } from './types.js';

type Mode = 'predict' | 'loss';
type Step = { node: Node; ports: [port: string, src: NodeId][] };

type PredictFn = (params: ParamTree, x: JArray) => JArray;
type LossFn = (params: ParamTree, x: JArray, y: JArray) => JArray;
type StepFn = (
  params: ParamTree,
  opt: optax.OptState,
  x: JArray,
  y: JArray,
) => [ParamTree, optax.OptState, JArray, JArray];

/** Private compiled state behind a Model. */
export type Internals = {
  graph: Graph;
  analysis: Analysis;
  dataColumns: readonly string[];
  /** Shapes of the linear nodes on the loss path. */
  shapes: Map<NodeId, LayerShape>;
  predNode: NodeId;
  lossNode: NodeId;
  link: OutputLink;
  steps: Step[];
  /** Consumer counts per node, by mode (the loss, or the final result in predict mode, counts too). */
  uses: Record<Mode, Map<NodeId, number>>;
  predictFn(): PredictFn;
  evalFn(): LossFn;
  stepFn(): StepFn;
  optimizer(): optax.GradientTransformation;
};

const internalsOf = new WeakMap<object, Internals>();

export function internals(model: Model): Internals {
  const m = internalsOf.get(model);
  if (!m) throw new Error('Not a Model made by this engine.');
  return m;
}

/** Throws one plain sentence when a piece, loss, penalty or optimizer is still a placeholder. */
function assertImplemented(graph: Graph, nodes: Node[]): void {
  for (const node of nodes) {
    if (!ENGINE_PIECES[node.type].implemented) {
      throw new Error(`The ${node.type} piece is not implemented yet.`);
    }
    if (node.type === 'loss') {
      if (!LOSSES[node.fn].implemented)
        throw new Error(`The ${node.fn} loss is not implemented yet.`);
      if (node.penalty && !PENALTY_IMPLEMENTED) {
        throw new Error('The weight penalty is not implemented yet.');
      }
    }
  }
  if (!OPTIMIZERS[graph.train.optimizer].implemented) {
    throw new Error(`The ${graph.train.optimizer} optimizer is not implemented yet.`);
  }
}

/** Throws Error(first issue) when the graph is not trainable. Borrows its arguments. */
export function compile(graph: Graph, data: DataInfo): Model {
  const analysis = analyse(graph, data);
  if (!analysis.trainable || analysis.lossNode === null || analysis.predNode === null) {
    throw new Error(analysis.issues[0]?.message ?? 'This graph cannot be trained.');
  }
  const byId = new Map<NodeId, Node>();
  for (const n of graph.nodes) if (!byId.has(n.id)) byId.set(n.id, n);
  const nodes: Node[] = [];
  for (const id of analysis.order) {
    const n = byId.get(id);
    if (n) nodes.push(n);
  }
  assertImplemented(graph, nodes);

  const inOrder = new Set(analysis.order);
  const wired = new Map<NodeId, Map<string, NodeId>>();
  for (const e of graph.edges) {
    const r = resolveEdge(graph.nodes, e);
    if (!r.ok || !inOrder.has(r.from.node) || !inOrder.has(r.to.node)) continue;
    const ports = wired.get(r.to.node) ?? new Map<string, NodeId>();
    // analysis uses the lowest edge index for a doubly-fed port; an untrainable graph never gets here
    if (!ports.has(r.to.port)) ports.set(r.to.port, r.from.node);
    wired.set(r.to.node, ports);
  }
  const steps: Step[] = nodes.map((node) => ({
    node,
    ports: [...(wired.get(node.id) ?? new Map<string, NodeId>())],
  }));

  const usesFor = (mode: Mode): Map<NodeId, number> => {
    const u = new Map<NodeId, number>();
    for (const s of steps) {
      if (mode === 'predict' && (s.node.type === 'loss' || s.node.type === 'target')) continue;
      for (const [, src] of s.ports) u.set(src, (u.get(src) ?? 0) + 1);
    }
    if (mode === 'predict')
      u.set(analysis.predNode ?? '', (u.get(analysis.predNode ?? '') ?? 0) + 1);
    return u;
  };

  const shapes = new Map<NodeId, LayerShape>();
  for (const node of nodes) {
    const w = analysis.inWidth[node.id];
    if (node.type !== 'linear' || w === undefined) continue;
    const s = CORE_PIECES.linear.paramShapes(node, w);
    if (s.W) shapes.set(node.id, { W: s.W, ...(s.b ? { b: s.b } : {}) });
  }

  const pred = byId.get(analysis.predNode);
  const link: OutputLink = pred?.type === 'output' ? pred.link : 'identity';

  let predictFn: PredictFn | null = null;
  let evalFn: LossFn | null = null;
  let stepFn: StepFn | null = null;
  let opt: optax.GradientTransformation | null = null;

  const self: Internals = {
    graph,
    analysis,
    dataColumns: data.columns,
    shapes,
    predNode: analysis.predNode,
    lossNode: analysis.lossNode,
    link,
    steps,
    uses: { predict: usesFor('predict'), loss: usesFor('loss') },
    predictFn() {
      // jit consumes params and x
      predictFn ??= jit((params: ParamTree, x: JArray) =>
        walk(self, params, x, null, 'predict'),
      ) as unknown as PredictFn;
      return predictFn;
    },
    evalFn() {
      evalFn ??= jit((params: ParamTree, x: JArray, y: JArray) =>
        walk(self, params, x, y, 'loss'),
      ) as unknown as LossFn;
      return evalFn;
    },
    optimizer() {
      opt ??= OPTIMIZERS[graph.train.optimizer].make(graph.train.lr);
      return opt;
    },
    stepFn() {
      if (stepFn) return stepFn;
      const optimizer = self.optimizer();
      // jit consumes params, opt, x and y. valueAndGrad consumes its arguments, so params is
      // `.ref`ed for it and consumed again by applyUpdates.
      stepFn = jit((params: ParamTree, optState: optax.OptState, x: JArray, y: JArray) => {
        const [loss, grads] = valueAndGrad((p: ParamTree, xx: JArray, yy: JArray) =>
          walk(self, p, xx, yy, 'loss'),
        )(tree.ref(params), x, y);
        const [updates, newOpt] = optimizer.update(grads, optState);
        const newParams = optax.applyUpdates(params, updates) as ParamTree;
        // One scalar that is non-finite when any updated parameter is (read next to the loss).
        let check = np.array(0);
        for (const leaf of tree.leaves(tree.ref(newParams)) as JArray[])
          check = check.add(leaf.sum());
        return [newParams, newOpt, loss, check];
      }) as unknown as StepFn;
      return stepFn;
    },
  };
  const model: Model = Object.freeze({ graph, analysis });
  internalsOf.set(model, self);
  return model;
}

/**
 * Forward pass in topological order. Consumes `params`, `x` and `y` (y may be null in predict
 * mode). 'predict' returns the post-link prediction [n, width]; 'loss' returns the mean loss (no
 * penalty yet: tinker-hj4.2 adds it here, taking `tree.ref(params)` before the walk).
 * A value feeding several consumers is `.ref`ed for all but the last.
 */
function walk(m: Internals, params: ParamTree, x: JArray, y: JArray | null, mode: Mode): JArray {
  const values = new Map<NodeId, JArray>();
  const remaining = new Map(m.uses[mode]);
  let xLeft = m.steps.filter((s) => s.node.type === 'input').length;
  let yLeft = mode === 'loss' ? m.steps.filter((s) => s.node.type === 'target').length : 0;
  const ctx: PieceCtx = {
    train: mode === 'loss',
    rawOutput: mode === 'loss',
    link: m.link,
    dataColumns: m.dataColumns,
    takeX: () => (--xLeft > 0 ? x.ref : x),
    takeY: () => {
      if (!y) throw new Error('This pass has no target.');
      return --yLeft > 0 ? y.ref : y;
    },
  };
  const take = (id: NodeId): JArray => {
    const v = values.get(id);
    if (!v) throw new Error(`Internal error: "${id}" has no value yet.`);
    const left = remaining.get(id) ?? 1;
    if (left > 1) {
      remaining.set(id, left - 1);
      return v.ref;
    }
    values.delete(id);
    return v;
  };
  let result: JArray | null = null;
  for (const s of m.steps) {
    const { node } = s;
    if (mode === 'predict' && (node.type === 'target' || node.type === 'loss')) continue;
    const inputs: Record<string, JArray> = {};
    for (const [port, src] of s.ports) inputs[port] = take(src);
    const layer = node.type === 'linear' ? params[node.id] : undefined;
    // The piece table is indexed by node.type; TypeScript cannot correlate the two here.
    const piece = ENGINE_PIECES[node.type] as unknown as {
      forward(
        n: Node,
        i: Record<string, JArray>,
        l: ParamTree[string] | undefined,
        c: PieceCtx,
      ): JArray;
    };
    const out = piece.forward(node, inputs, layer, ctx);
    if (node.type === 'loss' || (mode === 'predict' && node.id === m.predNode)) {
      result = out;
      break;
    }
    values.set(node.id, out);
  }
  if (!result) throw new Error('Internal error: the graph produced no result.');
  for (const v of values.values()) v.dispose();
  if (xLeft > 0) x.dispose();
  if (y && yLeft > 0) y.dispose();
  return result;
}

/** Throws a plain sentence when a batch does not fit the model (before anything is consumed). */
export function checkBatch(m: Internals, x: Float32Array, y: Float32Array | null, n: number): void {
  const d = m.dataColumns.length;
  if (!Number.isInteger(n) || n < 1) throw new Error('The batch needs at least one row.');
  if (x.length !== n * d)
    throw new Error(`The batch x has ${x.length} values, expected ${n} x ${d}.`);
  if (y && y.length !== n) throw new Error(`The batch y has ${y.length} values, expected ${n}.`);
}

/** Builds the jax arrays of a batch (copies). The caller passes them to a jitted function, which consumes them. */
export function batchArrays(m: Internals, b: Batch): { x: JArray; y: JArray } {
  const d = m.dataColumns.length;
  return {
    x: np.array(Float32Array.from(b.x), { shape: [b.n, d], dtype: np.float32 }),
    y: np.array(Float32Array.from(b.y), { shape: [b.n], dtype: np.float32 }),
  };
}
