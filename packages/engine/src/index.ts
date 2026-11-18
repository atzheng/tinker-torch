// The engine (CONTRACTS.md section 7): the only module that imports jax-js. Every exported
// function says which arguments it consumes; everything not listed is borrowed.
import { init, numpy as np, random, tree } from '@jax-js/jax';
import type { Array as JArray } from '@jax-js/jax';
import type * as optax from '@jax-js/optax';
import { CORE_NAME, initParams } from '@tinker/core';
import type { Batch, DataInfo, Graph, HostParams, NodeId } from '@tinker/core';
import { batchArrays, checkBatch, compile, internals } from './compile.js';
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

export type { Engine, Model, ParamAxis, Params, StepResult, TrainState } from './types.js';

export const ENGINE_NAME = '@tinker/engine';
export const ENGINE_CORE = CORE_NAME;

/** TrainStates that were consumed by `step` or disposed. */
const deadStates = new WeakSet<object>();

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
      return engine.fromHost(model, initParams(m.graph, m.analysis, seed));
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
      if (values.W && values.W.length !== size(layer.W)) {
        throw new Error(`The weights of "${nodeId}" need ${size(layer.W)} values.`);
      }
      if (values.b && (!layer.b || values.b.length !== size(layer.b))) {
        throw new Error(
          layer.b
            ? `The bias of "${nodeId}" needs ${size(layer.b)} values.`
            : `"${nodeId}" has no bias.`,
        );
      }
      const owned = takeParams(params);
      const old = owned[nodeId];
      if (!old) throw new Error('Internal error: the layer vanished.');
      const next: ParamTree[string] = { W: old.W };
      if (old.b) next.b = old.b;
      if (values.W) {
        next.W = np.array(Float32Array.from(values.W), { shape: old.W.shape, dtype: np.float32 });
        old.W.dispose();
      }
      if (values.b && old.b) {
        next.b = np.array(Float32Array.from(values.b), { shape: old.b.shape, dtype: np.float32 });
        old.b.dispose();
      }
      owned[nodeId] = next;
      return wrapParams(owned);
    },

    /** CONSUMES `params` (the arrays move into the returned state, whose `params` is a new Params). */
    startTraining(model, params, seed) {
      const m = internals(model);
      const t = peekParams(params);
      const opt = m.optimizer().init(tree.ref(t));
      takeParams(params);
      return { params: wrapParams(t), opt, step: 0, key: random.key(seed) };
    },

    /**
     * CONSUMES `state` and its `params`. Returns a new state. On a non-finite loss or non-finite
     * updated parameters it returns `diverged: true` and a state equal to the old one (same
     * arrays); it never throws for divergence. The loss is read every step (jax-js is lazy). If the
     * batch does not fit the model it throws before consuming anything.
     */
    step(model, state, batch): StepResult {
      const m = internals(model);
      liveState(state);
      const oldTree = peekParams(state.params);
      const oldOpt = state.opt as optax.OptState;
      checkBatch(m, batch.x, batch.y, batch.n);
      const { x, y } = batchArrays(m, batch);
      // The jitted step consumes what it is given; `state` keeps its own references until we know
      // the outcome, so it stays valid if anything throws and a diverged step can return it.
      const [newTree, newOpt, lossArr, checkArr] = m.stepFn()(
        tree.ref(oldTree),
        tree.ref(oldOpt),
        x,
        y,
      );
      const loss = lossArr.item();
      const check = checkArr.item();
      deadStates.add(state);
      if (!Number.isFinite(loss) || !Number.isFinite(check)) {
        tree.dispose(newTree);
        tree.dispose(newOpt);
        // Same arrays, same Params object, new TrainState object.
        return {
          state: { params: state.params, opt: state.opt, step: state.step, key: state.key },
          loss,
          diverged: true,
        };
      }
      disposeParams(state.params);
      tree.dispose(oldOpt);
      return {
        state: { params: wrapParams(newTree), opt: newOpt, step: state.step + 1, key: state.key },
        loss,
        diverged: false,
      };
    },

    /** Borrows `model` and `params`. Dropout off; post-link; [n, width] row-major. */
    predict(model, params, x, n) {
      const m = internals(model);
      const t = peekParams(params);
      checkBatch(m, x, null, n);
      const xa = np.array(Float32Array.from(x), {
        shape: [n, m.dataColumns.length],
        dtype: np.float32,
      });
      const out = m.predictFn()(tree.ref(t), xa);
      return Float32Array.from(out.dataSync() as Float32Array);
    },

    /** Borrows `params`. Mean loss plus penalty, dropout off. */
    evaluate(model, params, batch) {
      const m = internals(model);
      const t = peekParams(params);
      checkBatch(m, batch.x, batch.y, batch.n);
      const { x, y } = batchArrays(m, batch);
      return m.evalFn()(tree.ref(t), x, y).item();
    },

    lossGrid() {
      return lossGrid();
    },

    trainMany() {
      return trainMany();
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
