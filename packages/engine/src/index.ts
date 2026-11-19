// The engine (CONTRACTS.md section 7): the only module that imports jax-js. Every exported
// function says which arguments it consumes; everything not listed is borrowed.
import { init, numpy as np, random, tree } from '@jax-js/jax';
import type { Array as JArray } from '@jax-js/jax';
import type * as optax from '@jax-js/optax';
import { CORE_NAME, type Graph, initParams } from '@tinker/core';
import type { Batch, DataInfo, HostParams, NodeId } from '@tinker/core';
import {
  batchArrays,
  checkBatch,
  checkTree,
  compile,
  disposeCompiled,
  guarded,
  internals,
} from './compile.js';
import type { ParamTree } from './convert.js';
import {
  cloneTree,
  disposeParams,
  hostToTree,
  peekParams,
  takeParams,
  treeToHost,
  wrapParams,
} from './convert.js';
import { trainMany } from './many.js';
import { probeLiveArrays } from './probe.js';
import { lossGrid } from './surface.js';
import type { Engine, Model, Params, StepResult, TrainState } from './types.js';

export type {
  Engine,
  ManyResult,
  Model,
  ParamAxis,
  Params,
  StepResult,
  TrainState,
} from './types.js';

export const ENGINE_NAME = '@tinker/engine';
export const ENGINE_CORE = CORE_NAME;

/** TrainStates that were consumed by `step` or disposed. */
const deadStates = new WeakSet<object>();
/** Optimizer name a TrainState was started with (the state trees of the optimizers differ). */
const stateOptimizer = new WeakMap<object, string>();

function liveState(s: TrainState): void {
  if (deadStates.has(s)) throw new Error('This TrainState was already consumed or disposed.');
}

let ready: Promise<void> | null = null;

/** Awaits jax-js `init` once (Wasm device; never WebGPU in v1). */
export function createEngine(): Promise<Engine> {
  ready ??= init('wasm').then(() => undefined);
  return ready.then(makeEngine);
}

function makeEngine(): Engine {
  const engine: Engine = {
    compile(graph: Graph, data: DataInfo): Model {
      return compile(graph, data);
    },

    init(model, seed) {
      const m = internals(model);
      return engine.fromHost(model, initParams(m.analysis, seed));
    },

    /** Borrows `host`: the values are copied. Throws when it does not fit the model. */
    fromHost(model, host) {
      return wrapParams(hostToTree(internals(model).shapes, host));
    },

    /** Borrows `params`. Always allocates new arrays. */
    toHost(params) {
      return treeToHost(peekParams(params));
    },

    /** CONSUMES `params`; returns the Params to use from now on. Throws (params stay valid) on a bad node or length. */
    write(params, nodeId: NodeId, values) {
      const t = peekParams(params);
      const layer = Object.hasOwn(t, nodeId) ? t[nodeId] : undefined;
      if (!layer) throw new Error(`There are no parameters for "${nodeId}".`);
      const size = (a: JArray): number => a.shape.reduce((p, q) => p * q, 1);
      for (const [k, v] of Object.entries(values)) {
        const a = Object.hasOwn(layer, k) ? layer[k] : undefined;
        if (!a) throw new Error(`"${nodeId}" has no "${k}" parameters.`);
        if (v.length !== size(a)) {
          throw new Error(`The "${k}" parameters of "${nodeId}" need ${size(a)} values.`);
        }
      }
      const owned = takeParams(params);
      const old = owned[nodeId];
      if (!old) throw new Error('Internal error: the layer vanished.');
      const next: ParamTree[string] = {};
      for (const [k, a] of Object.entries(old)) {
        const v = Object.hasOwn(values, k) ? values[k] : undefined;
        if (v) {
          next[k] = np.array(Float32Array.from(v), { shape: a.shape, dtype: np.float32 });
          a.dispose();
        } else {
          next[k] = a;
        }
      }
      owned[nodeId] = next;
      return wrapParams(owned);
    },

    /** CONSUMES `params` (the arrays move into the returned state, whose `params` is a new Params). */
    startTraining(model, params, seed) {
      const m = internals(model);
      const t = peekParams(params);
      checkTree(m, t);
      const opt = m.optimizer().init(tree.ref(t));
      takeParams(params);
      const started: TrainState = { params: wrapParams(t), opt, step: 0, key: random.key(seed) };
      stateOptimizer.set(started, m.graph.train.optimizer);
      return started;
    },

    /**
     * CONSUMES `state` and its `params`. Returns a new state. On a non-finite loss or non-finite
     * updated parameters it returns `diverged: true` and a state equal to the old one (same
     * arrays); it never throws for divergence. The loss is read every step (jax-js is lazy). If the
     * batch does not fit the model, or `state` was started for a different optimizer (switching
     * optimizer means compile + startTraining again), it throws before consuming anything.
     * The first call with a new batch size retraces the jitted step (about 10-30 ms measured,
     * against about 0.5 ms when cached); repeated sizes reuse the trace.
     */
    step(model, state, batch): StepResult {
      const m = internals(model);
      liveState(state);
      if (stateOptimizer.get(state) !== m.graph.train.optimizer) {
        throw new Error(
          'This training state was started for a different optimizer; start training again.',
        );
      }
      const oldTree = peekParams(state.params);
      const oldOpt = state.opt as optax.OptState;
      checkBatch(m, batch.x, batch.y, batch.n);
      checkTree(m, oldTree);
      const { x, y } = batchArrays(m, batch);
      // The jitted step consumes what it is given; `state` keeps its own references until we know
      // the outcome, so it stays valid if anything throws and a diverged step can return it.
      const pRef = tree.ref(oldTree);
      const oRef = tree.ref(oldOpt);
      const kRef = (state.key as JArray).ref;
      const [newTree, newOpt, lossArr, checkArr, newKey] = guarded([pRef, oRef, x, y, kRef], () =>
        m.stepFn()(pRef, oRef, x, y, kRef),
      );
      const loss = lossArr.item();
      const check = checkArr.item();
      deadStates.add(state);
      if (!Number.isFinite(loss) || check !== 0) {
        tree.dispose(newTree);
        tree.dispose(newOpt);
        newKey.dispose();
        // Same arrays, same Params object, new TrainState object.
        const same: TrainState = {
          params: state.params,
          opt: state.opt,
          step: state.step,
          key: state.key,
        };
        stateOptimizer.set(same, m.graph.train.optimizer);
        return { state: same, loss, diverged: true };
      }
      disposeParams(state.params);
      tree.dispose(oldOpt);
      (state.key as JArray).dispose();
      const next: TrainState = {
        params: wrapParams(newTree),
        opt: newOpt,
        step: state.step + 1,
        key: newKey,
      };
      stateOptimizer.set(next, m.graph.train.optimizer);
      return { state: next, loss, diverged: false };
    },

    /**
     * Borrows `model` and `params`. Dropout off. One `[n, width]` row-major array per ŷ pin, in pin
     * order, each the value at the pinned node (after its link, if it is one); empty when the graph
     * has no ŷ pin.
     */
    predict(model, params, x, n) {
      const m = internals(model);
      const t = peekParams(params);
      checkBatch(m, x, null, n);
      checkTree(m, t);
      if (m.plans.predict.outputs.length === 0) return [];
      const xa = np.array(Float32Array.from(x), {
        shape: [n, m.dataColumns.length],
        dtype: np.float32,
      });
      const pRef = tree.ref(t);
      const outs = guarded([pRef, xa], () => m.predictFn()(pRef, xa));
      return outs.map((o) => Float32Array.from(o.dataSync() as Float32Array));
    },

    /** Borrows `params`. The L pin's value (a single number), dropout off. */
    evaluate(model, params, batch) {
      const m = internals(model);
      const t = peekParams(params);
      checkBatch(m, batch.x, batch.y, batch.n);
      checkTree(m, t);
      const { x, y } = batchArrays(m, batch);
      const pRef = tree.ref(t);
      return guarded([pRef, x, y], () => m.evalFn()(pRef, x, y)).item();
    },

    /** Borrows `model` and `params`. [a.length, b.length] row-major; dropout off. */
    lossGrid(model, params, batch, axes, a, b) {
      return lossGrid(model, peekParams(params), batch, axes, a, b);
    },

    trainMany(model, seeds, data, steps) {
      return trainMany(model, seeds, data, steps);
    },

    /**
     * Frees the model's jitted functions (predict, evaluate, step, lossGrid, trainMany). Safe to call
     * twice. Afterwards every method that takes the model throws "This model was disposed.". Params
     * and TrainStates made for it stay valid arrays: `toHost`, `write`, `clone` and `dispose` still work,
     * but nothing can run them (no `step`/`predict`) without a new model.
     */
    disposeModel(model) {
      disposeCompiled(model);
    },

    /** Borrows `params`; the clone shares the immutable arrays by reference count and is independent to dispose. */
    clone(params) {
      return wrapParams(cloneTree(peekParams(params)));
    },

    /** CONSUMES its argument. Safe to call twice. */
    dispose(x: Params | TrainState) {
      if ('step' in x) {
        if (deadStates.has(x)) return;
        deadStates.add(x);
        disposeParams(x.params);
        if (x.opt !== null && x.opt !== undefined) tree.dispose(x.opt as optax.OptState);
        (x.key as JArray).dispose();
      } else {
        disposeParams(x);
      }
    },

    /** Test-only: live Wasm buffers, 0 unless `./probe.js` was imported before jax-js. */
    liveArrays() {
      return probeLiveArrays();
    },
  };
  return engine;
}

export type { Batch, HostParams };
