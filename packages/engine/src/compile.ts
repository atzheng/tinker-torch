// Graph v2 + Analysis -> Model: the DAG walked into jitted predict / loss / step functions.
//
// Ownership (jax-js: functions consume their array arguments, `.ref` adds a reference). Every
// jitted function below CONSUMES all its array arguments; callers pass `tree.ref(...)` for what
// they keep. Jitted functions are built lazily on first use and cached on the model.
//
// Two plans are made at compile time from the analysis:
//   - predict: `analysis.modelOrder`, outputs = the nodes carrying the ŷ pins (dropout off);
//   - loss: `analysis.order` restricted to what the L pin needs, output = the L node (a scalar).
// Stable fusion happens while making the loss plan: for every pair in `analysis.fusable` (a
// sigmoid link feeding a log loss, a softmax link feeding a cross-entropy) the loss node reads the
// link's INPUT (the logits) and computes the logsumexp / log-softmax form; the link itself is only
// evaluated in the loss pass if something else uses it. The graph and the predict plan stay semantic.
import { jit, numpy as np, random, tree, valueAndGrad } from '@jax-js/jax';
import type { Array as JArray } from '@jax-js/jax';
import * as optax from '@jax-js/optax';
import {
  analyse,
  isVariadicPort,
  ownsParams,
  type Analysis,
  type Graph,
  type Node,
  resolveEdge,
} from '@tinker/core';
import type { Batch, DataInfo, HostParams } from '@tinker/core';
import { treeToHost } from './convert.js';
import type { Layer, LayerShape, ParamTree } from './convert.js';
import { OPTIMIZERS } from './optim.js';
import { ENGINE_PIECES } from './pieces/index.js';
import type { PieceCtx } from './pieces/index.js';
import type { Model } from './types.js';

type NodeId = string;
type Mode = 'predict' | 'loss';
/** One node of a plan: the node and, per input port, the node that feeds it. */
type Step = { node: Node; ports: [port: string, src: NodeId][] };

/** What one pass runs. */
type Plan = {
  steps: Step[];
  /** Consumer counts per node (a node that is an output counts once more). */
  uses: Map<NodeId, number>;
  /** The nodes whose values the pass returns, in order. */
  outputs: NodeId[];
  /** Loss nodes whose `pred` is the logits of a fused link. */
  fused: Set<NodeId>;
  /** How many times each Linear's parameters are read (its own forward pass plus Weights taps). */
  layerUses: Map<NodeId, number>;
};

type PredictFn = (params: ParamTree, x: JArray) => JArray[];
type LossFn = (params: ParamTree, x: JArray, y: JArray) => JArray;
type StepFn = (
  params: ParamTree,
  opt: optax.OptState,
  x: JArray,
  y: JArray,
  key: JArray,
) => [ParamTree, optax.OptState, JArray, JArray, JArray];

/** Private compiled state behind a Model. */
export type Internals = {
  graph: Graph;
  analysis: Analysis;
  dataColumns: readonly string[];
  /** Shapes of the Linears with live parameters (`analysis.paramNodes`). */
  shapes: Map<NodeId, LayerShape>;
  lossNode: NodeId;
  plans: Record<Mode, Plan>;
  predictFn(): PredictFn;
  evalFn(): LossFn;
  stepFn(): StepFn;
  optimizer(): optax.GradientTransformation;
  /** Jitted functions built for this model (here and in many.ts / surface.ts); freed by `disposeModel`. */
  disposables: Set<{ dispose(): void }>;
  disposed: boolean;
};

const internalsOf = new WeakMap<object, Internals>();

export function internals(model: Model): Internals {
  const m = internalsOf.get(model);
  if (!m) throw new Error('Not a Model made by this engine.');
  if (m.disposed) throw new Error('This model was disposed.');
  return m;
}

/**
 * Frees every jitted function (and the constants they captured) built for `model`. Safe to call
 * twice; afterwards `internals(model)` throws.
 */
export function disposeCompiled(model: Model): void {
  const m = internalsOf.get(model);
  if (!m) throw new Error('Not a Model made by this engine.');
  if (m.disposed) return;
  m.disposed = true;
  for (const d of m.disposables) {
    try {
      d.dispose();
    } catch {
      // already freed
    }
  }
  m.disposables.clear();
}

/** The nodes of `steps` that `outputs` depend on, following `ports` backwards. */
function neededFrom(outputs: NodeId[], ports: Map<NodeId, Map<string, NodeId>>): Set<NodeId> {
  const need = new Set<NodeId>();
  const todo = [...outputs];
  while (todo.length > 0) {
    const id = todo.pop() as NodeId;
    if (need.has(id)) continue;
    need.add(id);
    for (const src of ports.get(id)?.values() ?? []) todo.push(src);
  }
  return need;
}

function makePlan(
  order: NodeId[],
  byId: Map<NodeId, Node>,
  ports: Map<NodeId, Map<string, NodeId>>,
  outputs: NodeId[],
  fused: Set<NodeId>,
): Plan {
  const need = neededFrom(outputs, ports);
  const steps: Step[] = [];
  const uses = new Map<NodeId, number>();
  const layerUses = new Map<NodeId, number>();
  for (const id of order) {
    const node = byId.get(id);
    if (!node || !need.has(id)) continue;
    const own = [...(ports.get(id) ?? new Map<string, NodeId>())];
    steps.push({ node, ports: own });
    for (const [, src] of own) uses.set(src, (uses.get(src) ?? 0) + 1);
    if (ownsParams(node)) layerUses.set(id, (layerUses.get(id) ?? 0) + 1);
    if (node.type === 'weights') layerUses.set(node.of, (layerUses.get(node.of) ?? 0) + 1);
  }
  for (const o of outputs) uses.set(o, (uses.get(o) ?? 0) + 1);
  return { steps, uses, outputs, fused, layerUses };
}

/** Throws Error(first issue) when the graph is not trainable. Borrows its arguments. */
export function compile(graph: Graph, data: DataInfo): Model {
  const analysis = analyse(graph, data);
  if (!analysis.trainable || analysis.lossNode === null) {
    throw new Error(analysis.issues[0]?.message ?? 'This graph cannot be trained.');
  }
  if (!OPTIMIZERS[graph.train.optimizer].implemented) {
    throw new Error(`The ${graph.train.optimizer} optimizer is not implemented yet.`);
  }
  const byId = new Map<NodeId, Node>();
  for (const n of graph.nodes) if (!byId.has(n.id)) byId.set(n.id, n);

  // Who feeds each input port. analyse keeps the first wire of a doubly fed port; an untrainable
  // graph never gets here.
  const wired = new Map<NodeId, Map<string, NodeId>>();
  for (const e of graph.edges) {
    const r = resolveEdge(graph.nodes, e);
    if (!r.ok) continue;
    const ports = wired.get(r.to.node) ?? new Map<string, NodeId>();
    const toType = byId.get(r.to.node)?.type;
    if (toType !== undefined && isVariadicPort(toType, r.to.port)) {
      // a variadic port keeps every wire, in edge order: `in`, `in#1`, `in#2`, ... (a repeated source is an issue)
      let key = r.to.port;
      for (let k = 1; ports.has(key); k++) key = `${r.to.port}#${k}`;
      ports.set(key, r.from.node);
    } else if (!ports.has(r.to.port)) ports.set(r.to.port, r.from.node);
    wired.set(r.to.node, ports);
  }

  // Loss plan: fuse each link into the loss it feeds (the loss reads the link's input).
  const lossPorts = new Map<NodeId, Map<string, NodeId>>();
  for (const [id, p] of wired) lossPorts.set(id, new Map(p));
  const fused = new Set<NodeId>();
  for (const { link, loss } of analysis.fusable) {
    const logits = wired.get(link)?.get('in');
    if (logits === undefined) continue;
    lossPorts.get(loss)?.set('pred', logits);
    fused.add(loss);
  }
  const plans: Record<Mode, Plan> = {
    predict: makePlan(analysis.modelOrder, byId, wired, analysis.predictions, new Set()),
    loss: makePlan(analysis.order, byId, lossPorts, [analysis.lossNode], fused),
  };

  const shapes = new Map<NodeId, LayerShape>();
  for (const id of analysis.paramNodes) {
    const s = analysis.paramShapes[id];
    if (s) shapes.set(id, s);
  }

  let predictFn: PredictFn | null = null;
  let evalFn: LossFn | null = null;
  let stepFn: StepFn | null = null;
  let opt: optax.GradientTransformation | null = null;

  const self: Internals = {
    graph,
    analysis,
    dataColumns: data.columns,
    shapes,
    lossNode: analysis.lossNode,
    plans,
    disposables: new Set(),
    disposed: false,
    predictFn() {
      // jit consumes params and x
      predictFn ??= jit((params: ParamTree, x: JArray) =>
        walk(self, 'predict', params, x, null, { train: false, masks: null }),
      ) as unknown as PredictFn;
      self.disposables.add(predictFn as unknown as { dispose(): void });
      return predictFn;
    },
    evalFn() {
      evalFn ??= jit(
        (params: ParamTree, x: JArray, y: JArray) =>
          walk(self, 'loss', params, x, y, { train: false, masks: null })[0] as JArray,
      ) as unknown as LossFn;
      self.disposables.add(evalFn as unknown as { dispose(): void });
      return evalFn;
    },
    optimizer() {
      opt ??= OPTIMIZERS[graph.train.optimizer].make(graph.train.lr);
      return opt;
    },
    stepFn() {
      if (stepFn) return stepFn;
      const optimizer = self.optimizer();
      // jit consumes params, opt, x, y and key (the new key is returned). valueAndGrad consumes its arguments, so params is
      // `.ref`ed for it and consumed again by applyUpdates.
      stepFn = jit(
        (params: ParamTree, optState: optax.OptState, x: JArray, y: JArray, key: JArray) => {
          // The masks are sampled outside valueAndGrad: random ops traced under it leave one
          // constant buffer per compiled model that jit never frees.
          const [newKey, masks] = sampleMasks(self, key, x.shape[0] ?? 0);
          const [loss, grads] = valueAndGrad(
            (p: ParamTree, xx: JArray, yy: JArray, mm: Masks) =>
              walk(self, 'loss', p, xx, yy, { train: true, masks: mm })[0] as JArray,
          )(tree.ref(params), x, y, masks);
          const [updates, newOpt] = optimizer.update(grads, optState);
          const newParams = optax.applyUpdates(params, updates) as ParamTree;
          // One scalar: how many updated parameters are non-finite (0 = all finite), read next to the loss.
          // (jax-js folds x * 0 to 0, NaN included, so x * 0 cannot be used; isfinite is exact, also at 3e38.)
          let check = np.array(0);
          for (const leaf of tree.leaves(tree.ref(newParams)) as JArray[])
            check = check.add(np.isfinite(leaf).astype(np.float32).mul(-1).add(1).sum());
          return [newParams, newOpt, loss, check, newKey];
        },
      ) as unknown as StepFn;
      self.disposables.add(stepFn as unknown as { dispose(): void });
      return stepFn;
    },
  };
  const model: Model = Object.freeze({ graph, analysis });
  internalsOf.set(model, self);
  return model;
}

/** Dropout keep masks (0/1, [n, width]) by Dropout node id. */
type Masks = Record<NodeId, JArray>;

/** `train`: dropout on. `masks`: one per Dropout node when training (consumed; unused ones are disposed). */
type WalkOpts = { train: boolean; masks: Masks | null };

/**
 * Splits `key` (consumed) into the next key and one subkey per Dropout node of the loss plan, and
 * samples the keep masks for a batch of `n` rows. Returns [next key, masks].
 */
function sampleMasks(m: Internals, key: JArray, n: number): [JArray, Masks] {
  const masks: Masks = {};
  let k = key;
  for (const s of m.plans.loss.steps) {
    if (s.node.type !== 'dropout' || s.node.p <= 0) continue;
    const width = m.analysis.inWidth[s.node.id] ?? 0;
    // np.unstack consumes the parent key (destructuring the split would leak it, docs/spike.md)
    const [next, sub] = np.unstack(random.split(k, 2)) as [JArray, JArray];
    k = next;
    masks[s.node.id] = random.uniform(sub, [n, width]).greaterEqual(s.node.p).astype(np.float32);
  }
  const [next] = np.unstack(random.split(k, 2)) as [JArray, JArray];
  return [next, masks];
}

/**
 * Forward pass of one plan, in topological order. Consumes `params`, `x`, `y` and `opts.masks`
 * (`y` is null in a predict pass: a Target then reads zeros). Returns one array per plan output:
 * the ŷ values `[n, width]` in pin order (predict), or the scalar L (loss). A value feeding several
 * consumers is `.ref`ed for all but the last; a Linear's parameters likewise per read (its own
 * forward pass and each Weights tap); parameters the pass never reads are disposed.
 */
function walk(
  m: Internals,
  mode: Mode,
  params: ParamTree,
  x: JArray,
  y: JArray | null,
  opts: WalkOpts,
): JArray[] {
  const plan = m.plans[mode];
  const n = x.shape[0] ?? 0;
  const masks = new Map(Object.entries(opts.masks ?? {}));
  const values = new Map<NodeId, JArray>();
  const remaining = new Map(plan.uses);
  const layerLeft = new Map(plan.layerUses);
  let xLeft = plan.steps.filter((s) => s.node.type === 'input' || s.node.type === 'column').length;
  let yLeft = plan.steps.filter((s) => s.node.type === 'target').length;
  if (xLeft === 0) x.dispose();
  if (y && yLeft === 0) y.dispose();
  const ctx: PieceCtx = {
    train: opts.train,
    dataColumns: m.dataColumns,
    fused: (id) => plan.fused.has(id),
    takeX: () => (--xLeft > 0 ? x.ref : x),
    takeY: () => {
      if (!y) return np.zeros([n]);
      return --yLeft > 0 ? y.ref : y;
    },
    dropoutMask: (id) => {
      const mask = masks.get(id);
      if (!mask) throw new Error('Internal error: this pass has no dropout mask.');
      masks.delete(id);
      return mask;
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
  const takeLayer = (id: NodeId): Layer => {
    const l = Object.hasOwn(params, id) ? params[id] : undefined;
    const left = layerLeft.get(id);
    if (!l || left === undefined) throw new Error(`Internal error: "${id}" has no parameters.`);
    if (left > 1) {
      layerLeft.set(id, left - 1);
      return Object.fromEntries(Object.entries(l).map(([k, a]) => [k, a.ref]));
    }
    layerLeft.delete(id);
    return l;
  };
  for (const s of plan.steps) {
    const { node } = s;
    const inputs: Record<string, JArray> = {};
    for (const [port, src] of s.ports) inputs[port] = take(src);
    const layer = ownsParams(node)
      ? takeLayer(node.id)
      : node.type === 'weights'
        ? takeLayer(node.of)
        : undefined;
    // The piece table is indexed by node.type; TypeScript cannot correlate the two here.
    const piece = ENGINE_PIECES[node.type] as unknown as {
      forward(n: Node, i: Record<string, JArray>, l: Layer | undefined, c: PieceCtx): JArray;
    };
    values.set(node.id, piece.forward(node, inputs, layer, ctx));
  }
  const out = plan.outputs.map(take);
  for (const v of values.values()) v.dispose();
  if (xLeft > 0) x.dispose();
  if (y && yLeft > 0) y.dispose();
  for (const mk of masks.values()) mk.dispose();
  // parameters this pass never reads (e.g. a Linear only the loss uses, in a predict pass)
  for (const [id, l] of Object.entries(params)) {
    if (plan.layerUses.has(id)) continue;
    tree.dispose(l);
  }
  return out;
}

/**
 * Test-only (CONTRACTS.md section 11, "Dropout fixtures"; not part of `Engine`): the training
 * forward and backward with dropout ON and the given keep masks instead of sampling. Borrows
 * `params`. Returns the first ŷ pin's values (post-link, empty with no ŷ), the objective L and the gradients.
 */
export function debugTrainPass(
  model: Model,
  params: ParamTree,
  batch: Batch,
  masks: Record<NodeId, number[][]>,
): { pred: Float32Array; loss: number; grads: HostParams } {
  const m = internals(model);
  const hostMasks = (): Masks => {
    const out: Masks = {};
    for (const [id, rows] of Object.entries(masks)) {
      out[id] = np.array(Float32Array.from(rows.flat()), {
        shape: [rows.length, rows[0]?.length ?? 0],
        dtype: np.float32,
      });
    }
    return out;
  };
  const first = batchArrays(m, batch);
  first.y.dispose();
  const preds = walk(m, 'predict', tree.ref(params), first.x, null, {
    train: true,
    masks: hostMasks(),
  });
  const [pred, ...rest] = preds;
  for (const r of rest) r.dispose();
  const predHost = pred ? Float32Array.from(pred.dataSync() as Float32Array) : new Float32Array(0);
  const { x, y } = batchArrays(m, batch);
  const [loss, grads] = valueAndGrad(
    (p: ParamTree, xx: JArray, yy: JArray) =>
      walk(m, 'loss', p, xx, yy, { train: true, masks: hostMasks() })[0] as JArray,
  )(tree.ref(params), x, y);
  const lossValue = loss.item();
  const host = treeToHost(grads as ParamTree);
  tree.dispose(grads);
  return { pred: predHost, loss: lossValue, grads: host };
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

/** Throws a plain sentence when `t` does not have the parameter shapes of the model. Borrows `t`. */
export function checkTree(m: Internals, t: ParamTree): void {
  const same = (a: number[], b: number[]) => a.length === b.length && a.every((v, i) => v === b[i]);
  const bad = new Error('These parameters were made for a different model.');
  const ids = Object.keys(t);
  if (ids.length !== m.shapes.size) throw bad;
  for (const [id, s] of m.shapes) {
    const l = Object.hasOwn(t, id) ? t[id] : undefined;
    if (!l) throw bad;
    const names = Object.keys(s);
    if (Object.keys(l).length !== names.length) throw bad;
    for (const k of names) {
      const a = Object.hasOwn(l, k) ? l[k] : undefined;
      if (!a || !same(a.shape, s[k] ?? [])) throw bad;
    }
  }
}

/**
 * Runs `fn` (which hands `held` to a jitted function, consuming it). If `fn` throws, disposes
 * everything in `held` that is still alive, so a failing call leaks no buffer, then rethrows.
 * `held` holds jax arrays and trees of them.
 */
export function guarded<T>(held: unknown[], fn: () => T): T {
  try {
    return fn();
  } catch (e) {
    for (const h of held) {
      try {
        tree.dispose(h as ParamTree);
      } catch {
        // already consumed by the failed call
      }
    }
    throw e;
  }
}
