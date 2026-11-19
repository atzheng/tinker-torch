import type { StateCreator, StoreApi } from 'zustand';
import {
  analyse,
  batchSchedule,
  gatherBatch,
  initParams,
  makeRng,
  reconcileParams,
  sample,
  type Analysis,
  type Batch,
  type Graph,
  type HostParams,
} from '@tinker/core';
import type { Engine, Model, ParamAxis, Params, TrainState } from '@tinker/engine';
import { getEngine, onEngineError, onEngineReady } from '../engine.js';
import { initialGraph } from './graphSlice.js';
import { plotIndex } from './plotPrediction.js';
import type { Store } from './store.js';
import { bumpWeightGeneration } from './weightHistory.js';

/**
 * Train slice (tinker-27m.2).
 *
 * The store holds plain numbers and immutable HostParams only. The engine handles (Model,
 * TrainState) live in the module-level runtime `rt` below and are disposed whenever they are
 * replaced. HostParams arrays are never written into (they are shared with undo history).
 *
 * Policy on edits (see `resync`):
 * - a structural change (nodes, edges, data) keeps the weights of unchanged layers
 *   (`reconcileParams`), clears the loss history, the best snapshot and the step counter, and pauses;
 * - a change to `graph.train` only (learning rate, optimizer, batch) recompiles and keeps the
 *   weights, the history and the best; a running loop keeps running; a diverged run may continue
 *   (the engine leaves the parameters unchanged on a diverged step); after a divergence any edit
 *   starts a new run from fresh weights (the old ones are blown up), with diverged cleared;
 * - an invalid graph disables training and shows its first issue; the weights are kept so that
 *   fixing the graph brings the model back.
 */

/** A batch size at or above nTrain (after a data change) is the same as full batch. */
export function isMiniBatch(batch: 'full' | number, nTrain: number): boolean {
  return batch !== 'full' && batch < nTrain;
}

export type HistoryPoint = {
  step: number;
  /** Loss of the batch used for this update, before the update. */
  train: number;
  /** Validation loss after the update; null on steps where it was not evaluated (see VAL_EVERY). */
  val: number | null;
  /** Loss on the whole training set, evaluated on the validation cadence; only when mini-batching. */
  trainFull?: number | null;
};
export type Best = { step: number; val: number; params: HostParams };

export type TrainSlice = {
  engineStatus: 'loading' | 'ready' | 'error';
  /** Analysis of the current graph and dataset. */
  analysis: Analysis;
  /** Why training is disabled (first graph issue, a not-yet-implemented piece, or an engine failure); null when it can run. */
  trainIssue: string | null;
  /** The model compiled and a TrainState exists. */
  canTrain: boolean;
  running: boolean;
  diverged: boolean;
  /** Parameter updates done since the last Reset or structural edit. */
  step: number;
  trainLoss: number | null;
  valLoss: number | null;
  history: HistoryPoint[];
  best: Best | null;
  /** Steps per frame while running (12 ms budget per frame). */
  speed: number;
  /** Immutable snapshot of the current parameters, published at most once per frame. */
  host: HostParams | null;
  stepOnce: () => void;
  run: () => void;
  pause: () => void;
  reset: () => void;
  restoreBest: () => void;
  /**
   * Install saved weights after `loadGraph` (tinker-ysl.1). `host` belongs to the current graph
   * (already filtered by `paramsFromJson`); nodes it lacks keep their fresh initialisation.
   * Starts a new run (history and best cleared) from these weights. No-op while the graph is invalid.
   * Returns false when the engine is ready but the model did not compile or start (weights not
   * installed); true otherwise (also when the engine is still loading: the weights are kept).
   */
  loadHost: (host: HostParams) => boolean;
  /** Set the run seed (init of new layers, batch order, dropout) before applying a loaded graph. */
  setRunSeed: (seed: number) => void;
  /**
   * Predictions of the current parameters (dropout off, after the link) for the rows of `x`
   * (row-major [n, dataColumns], already standardised if on); [n, width], or null when there is
   * no compiled model. Allocates only the result. Every distinct `n` costs one retrace in the
   * engine (about 2 ms), so callers keep n fixed per dataset (a constant grid). The output is the ŷ pin
   * picked in the fit plot header (`plotPrediction`, default the first).
   */
  predictGrid: (x: Float32Array, n: number) => Float32Array | null;
  /**
   * Training loss (mean loss plus penalty over the whole training set, dropout off) of the current
   * model with the two parameters `axes` set to a[i] and b[j] and every other parameter as it is;
   * [a.length, b.length] row-major, or null when there is no compiled model or the call fails.
   * Used by the loss-surface view (tinker-ran.3); callers keep the grid size fixed.
   */
  lossSurface: (
    axes: [ParamAxis, ParamAxis],
    a: Float32Array,
    b: Float32Array,
  ) => Float32Array | null;
  /**
   * Hand-set weights (tinker-ran.2): overwrite the W and/or b of one Linear in the live model
   * through `engine.write` (the optimizer state is kept) and republish `host`. Does not touch the
   * graph, the history or the step counter. False when there is no live model or the values do
   * not fit (nothing changes then). Callers refuse while running and for locked pieces.
   */
  writeParams: (node: string, values: { W?: Float32Array; b?: Float32Array }) => boolean;
  setSpeed: (n: number) => void;
  /** Commit a learning rate (one undo step). Committed on release or blur by the UI. */
  setLearningRate: (lr: number) => void;
  /** Switching optimizer keeps the parameters and restarts the optimizer state. */
  setOptimizer: (optimizer: Graph['train']['optimizer']) => void;
  /** Batch size, or 'full'. The batch order restarts from the run seed. */
  setBatch: (batch: Graph['train']['batch']) => void;
};

export const SPEEDS = [1, 2, 5, 10, 25, 50];
export const DEFAULT_SPEED = 10;
export const FRAME_BUDGET_MS = 12;
export const MAX_HISTORY = 20000;

/**
 * Validation loss is evaluated after every k-th update, with k = 1 up to 100 validation points and
 * ceil(nVal / 100) above (so at most 3 for the largest dataset, n = 500, val = 0.5).
 */
export function valEvery(nVal: number): number {
  return nVal <= 100 ? 1 : Math.ceil(nVal / 100);
}

type Runtime = {
  engine: Engine | null;
  model: Model | null;
  state: TrainState | null;
  batches: Generator<Int32Array, never, undefined> | null;
  raf: number | null;
  seed: number;
  /** Parameters between engine states, with the graph and analysis they belong to. */
  host: HostParams | null;
  hostGraph: Graph | null;
  hostAnalysis: Analysis | null;
  lastGraph: Graph | null;
  lastDataset: unknown;
  history: HistoryPoint[];
  best: Best | null;
  step: number;
  trainLoss: number | null;
  valLoss: number | null;
  valBatch: Batch | null;
  /** The whole training set as one batch (fixed nTrain, so one trace). */
  trainBatch: Batch | null;
};

const rt: Runtime = {
  engine: null,
  model: null,
  state: null,
  batches: null,
  raf: null,
  seed: 0,
  host: null,
  hostGraph: null,
  hostAnalysis: null,
  lastGraph: null,
  lastDataset: null,
  history: [],
  best: null,
  step: 0,
  trainLoss: null,
  valLoss: null,
  valBatch: null,
  trainBatch: null,
};

/** Whether `b` is `a` with only its piece positions changed (same node, edge, pin, data and train objects). */
function onlyMoved(a: Graph, b: Graph): boolean {
  return (
    a.positions !== b.positions &&
    a.nodes === b.nodes &&
    a.edges === b.edges &&
    a.predictions === b.predictions &&
    a.loss === b.loss &&
    a.data === b.data &&
    a.train === b.train
  );
}

function structuralKey(g: Graph): string {
  return JSON.stringify([g.nodes, g.edges, g.data]);
}

function initialAnalysis(): Analysis {
  const g = initialGraph();
  return analyse(g, sample(g.data).info);
}

function disposeState(): void {
  if (rt.state && rt.engine) rt.engine.dispose(rt.state); // the TrainState, never state.params
  rt.state = null;
}

function stopLoop(): void {
  if (rt.raf !== null) cancelAnimationFrame(rt.raf);
  rt.raf = null;
}

export const createTrainSlice: StateCreator<Store, [], [], TrainSlice> = (set, get) => {
  /** Publish the runtime to the store: once per frame while running, once per action otherwise. */
  function publish(extra: Partial<TrainSlice> = {}): void {
    const engine = rt.engine;
    const host = rt.state && engine ? engine.toHost(rt.state.params) : rt.host;
    set({
      step: rt.step,
      trainLoss: rt.trainLoss,
      valLoss: rt.valLoss,
      history: rt.history.slice(),
      best: rt.best,
      host,
      ...extra,
    });
  }

  function clearRun(): void {
    rt.history = [];
    rt.best = null;
    rt.step = 0;
    rt.trainLoss = null;
    rt.valLoss = null;
  }

  /** Validation loss at step 0 so that "Restore best" can return to the initial parameters. */
  function evaluateInitial(): void {
    const { engine, model, state, valBatch } = rt;
    if (!engine || !model || !state || !valBatch || valBatch.n === 0) return;
    const val = engine.evaluate(model, state.params, valBatch);
    if (Number.isFinite(val)) {
      rt.valLoss = val;
      rt.best = { step: 0, val, params: engine.toHost(state.params) };
    }
  }

  /**
   * Brings model, TrainState and params in line with the current graph and dataset. `fresh`
   * discards the parameters (Reset).
   */
  function resync(freshArg = false): void {
    const { graph, dataset, dataInfo } = get();
    // A diverged run leaves huge (finite) weights behind, so no lr, optimizer, batch or data
    // change could recover from them: the edit that follows a divergence starts from fresh weights.
    const fresh = freshArg || get().diverged;
    if (fresh) bumpWeightGeneration(); // hand-edit undo records no longer fit these weights
    const engine = rt.engine;
    const structural =
      fresh || !rt.lastGraph || structuralKey(graph) !== structuralKey(rt.lastGraph);
    const wasRunning = get().running;
    rt.lastGraph = graph;
    rt.lastDataset = dataset;

    // 1. Take the parameters out of the live state, then free it.
    if (rt.state && engine) rt.host = engine.toHost(rt.state.params);
    disposeState();
    if (rt.model && engine) engine.disposeModel(rt.model); // frees its jitted functions
    rt.model = null;
    stopLoop();

    const analysis = analyse(graph, dataInfo);
    if (fresh) rt.host = null;
    if (structural) clearRun();

    // 2. Invalid graph: training off, weights kept for later.
    if (!analysis.trainable) {
      set({
        analysis,
        trainIssue: analysis.issues[0]?.message ?? 'The graph cannot be trained yet.',
        canTrain: false,
        running: false,
        diverged: false,
        host: null, // rt.host belongs to an earlier graph; the canvas must not see it with this one
        step: rt.step,
        trainLoss: rt.trainLoss,
        valLoss: rt.valLoss,
        history: rt.history.slice(),
        best: rt.best,
      });
      return;
    }

    // 3. Weights: keep what still fits, draw the rest.
    rt.host =
      rt.host && rt.hostGraph && rt.hostAnalysis
        ? reconcileParams(rt.host, rt.hostAnalysis, analysis, rt.seed)
        : initParams(analysis, rt.seed);
    rt.hostGraph = graph;
    rt.hostAnalysis = analysis;
    if (structural) {
      rt.best = null; // old best params do not fit a changed architecture
    }
    rt.batches = batchSchedule(dataset.nTrain, graph.train.batch, rt.seed);
    rt.valBatch = { x: dataset.xVal, y: dataset.yVal, n: dataset.nVal };
    rt.trainBatch = { x: dataset.xTrain, y: dataset.yTrain, n: dataset.nTrain };

    // 4. Compile and start. Not-yet-implemented pieces surface here as an ordinary sentence.
    let issue: string | null = null;
    if (!engine) {
      issue = get().engineStatus === 'error' ? 'The training engine could not be loaded.' : null;
    } else {
      let model: Model | null = null;
      let params: Params | null = null;
      try {
        model = engine.compile(graph, dataInfo);
        params = engine.fromHost(model, rt.host);
        rt.state = engine.startTraining(model, params, rt.seed);
        rt.model = model;
        if (structural) evaluateInitial();
      } catch (e) {
        if (params && rt.state === null) engine.dispose(params); // startTraining throws before consuming
        if (model && rt.model !== model) engine.disposeModel(model); // compiled but never used
        issue = e instanceof Error ? e.message : 'The model could not be compiled.';
      }
    }
    const canTrain = rt.state !== null;
    const keepRunning = !structural && wasRunning && canTrain;
    // One store update, so a reader never sees the new analysis with the old host.
    publish({ analysis, trainIssue: issue, canTrain, running: false, diverged: false });
    if (keepRunning) startLoop();
  }

  function doStep(): boolean {
    const { engine, model, state, batches } = rt;
    const { dataset, dataInfo } = get();
    if (!engine || !model || !state || !batches) return false;
    const idx = batches.next().value;
    const batch = gatherBatch(dataset.xTrain, dataset.yTrain, dataInfo.columns.length, idx);
    const r = engine.step(model, state, batch);
    rt.state = r.state;
    if (r.diverged) return false;
    rt.step += 1;
    rt.trainLoss = r.loss;
    let val: number | null = null;
    let trainFull: number | null = null;
    const vb = rt.valBatch;
    if (vb && vb.n > 0 && rt.step % valEvery(vb.n) === 0) {
      const v = engine.evaluate(model, r.state.params, vb);
      if (Number.isFinite(v)) {
        val = v;
        rt.valLoss = v;
        if (rt.trainBatch && isMiniBatch(get().graph.train.batch, rt.trainBatch.n)) {
          const t = engine.evaluate(model, r.state.params, rt.trainBatch);
          if (Number.isFinite(t)) trainFull = t;
        }
        if (!rt.best || v < rt.best.val) {
          rt.best = { step: rt.step, val: v, params: engine.toHost(r.state.params) };
        }
      }
    }
    rt.history.push({ step: rt.step, train: r.loss, val, trainFull });
    if (rt.history.length > MAX_HISTORY) {
      const last = rt.history.length - 1;
      rt.history = rt.history.filter((_, i) => i % 2 === 0 || i === last);
    }
    return true;
  }

  /** Runs doStep and turns divergence or an engine error into store state. Returns whether to go on. */
  function guardedStep(): boolean {
    try {
      if (doStep()) return true;
      set({ diverged: true, running: false });
      return false;
    } catch (e) {
      set({
        trainIssue: e instanceof Error ? e.message : 'Training stopped on an error.',
        running: false,
      });
      return false;
    }
  }

  function frame(): void {
    rt.raf = null;
    if (!get().running) return;
    const t0 = performance.now();
    const speed = get().speed;
    let ok = true;
    for (let i = 0; i < speed && ok; i++) {
      ok = guardedStep();
      if (performance.now() - t0 >= FRAME_BUDGET_MS) break;
    }
    try {
      publish();
    } catch (e) {
      set({
        trainIssue: e instanceof Error ? e.message : 'Training stopped on an error.',
        running: false,
      });
      return;
    }
    if (ok && get().running) rt.raf = requestAnimationFrame(frame);
  }

  function startLoop(): void {
    set({ running: true });
    stopLoop();
    rt.raf = requestAnimationFrame(frame);
  }

  resyncRef = () => resync();

  // Engine loading: compile as soon as it is ready.
  onEngineReady((e) => {
    rt.engine = e;
    set({ engineStatus: 'ready' });
    resync();
  });
  onEngineError(() => {
    set({ engineStatus: 'error', trainIssue: 'The training engine could not be loaded.' });
  });

  return {
    engineStatus: getEngine() ? 'ready' : 'loading',
    analysis: initialAnalysis(),
    trainIssue: null,
    canTrain: false,
    running: false,
    diverged: false,
    step: 0,
    trainLoss: null,
    valLoss: null,
    history: [],
    best: null,
    speed: DEFAULT_SPEED,
    host: null,
    stepOnce: () => {
      const s = get();
      if (!s.canTrain || s.running || s.diverged) return;
      guardedStep();
      publish();
    },
    run: () => {
      const s = get();
      if (!s.canTrain || s.running || s.diverged) return;
      startLoop();
    },
    pause: () => {
      stopLoop();
      set({ running: false });
    },
    reset: () => {
      const s = get();
      if (!s.analysis.trainable) return;
      rt.seed = makeRng(rt.seed)
        .fork('reset')
        .int(2 ** 31);
      resync(true);
    },
    restoreBest: () => {
      const { engine, model, best } = rt;
      if (!engine || !model || !best) return;
      stopLoop();
      bumpWeightGeneration();
      let params: Params | null = null;
      try {
        params = engine.fromHost(model, best.params);
        disposeState();
        rt.state = engine.startTraining(model, params, rt.seed);
        params = null; // consumed
        rt.batches = batchSchedule(get().dataset.nTrain, get().graph.train.batch, rt.seed);
        rt.valLoss = best.val;
        rt.trainLoss = null; // no loss was computed for the restored parameters yet
        publish({ running: false, diverged: false });
      } catch (e) {
        if (params) engine.dispose(params); // not consumed by startTraining
        set({
          trainIssue: e instanceof Error ? e.message : 'The best parameters could not be restored.',
          running: false,
        });
      }
    },
    loadHost: (loaded) => {
      const { analysis, graph } = get();
      if (!analysis.trainable || !rt.host) return false;
      const merged = Object.assign(Object.create(null) as HostParams, rt.host, loaded);
      bumpWeightGeneration();
      stopLoop();
      disposeState();
      rt.host = merged;
      rt.hostGraph = graph;
      rt.hostAnalysis = analysis;
      clearRun();
      const { engine, model } = rt;
      let issue: string | null = null;
      if (engine && model) {
        let params: Params | null = null;
        try {
          params = engine.fromHost(model, merged);
          rt.state = engine.startTraining(model, params, rt.seed);
          params = null; // consumed
          evaluateInitial();
        } catch (e) {
          if (params) engine.dispose(params); // not consumed by startTraining
          issue = e instanceof Error ? e.message : 'The saved weights could not be loaded.';
        }
      }
      publish({ trainIssue: issue, canTrain: rt.state !== null, running: false, diverged: false });
      return !engine || rt.state !== null;
    },
    setRunSeed: (seed) => {
      rt.seed = seed;
    },
    predictGrid: (x, n) => {
      const { engine, model, state } = rt;
      if (!engine || !model || !state) return null;
      try {
        const { graph, plotPrediction } = get();
        return engine.predict(model, state.params, x, n)[plotIndex(graph, plotPrediction)] ?? null;
      } catch {
        return null;
      }
    },
    lossSurface: (axes, a, b) => {
      const { engine, model, state } = rt;
      if (!engine || !model || !state) return null;
      const { dataset } = get();
      try {
        const batch = { x: dataset.xTrain, y: dataset.yTrain, n: dataset.nTrain };
        return engine.lossGrid(model, state.params, batch, axes, a, b);
      } catch {
        return null;
      }
    },
    writeParams: (node, values) => {
      const { engine, state } = rt;
      if (!engine || !state) return false;
      try {
        state.params = engine.write(state.params, node, values); // consumes the old params
      } catch {
        return false; // the engine leaves the params valid when it throws
      }
      publish();
      return true;
    },
    setSpeed: (n) => set({ speed: Math.max(1, Math.round(n)) }),
    setLearningRate: (lr) => {
      const { graph } = get();
      if (!(lr > 0) || lr === graph.train.lr) return;
      get().apply({ ok: true, graph: { ...graph, train: { ...graph.train, lr } } });
    },
    setOptimizer: (optimizer) => {
      const { graph } = get();
      if (optimizer === graph.train.optimizer) return;
      get().apply({ ok: true, graph: { ...graph, train: { ...graph.train, optimizer } } });
    },
    setBatch: (batch) => {
      const { graph } = get();
      if (batch === graph.train.batch) return;
      get().apply({ ok: true, graph: { ...graph, train: { ...graph.train, batch } } });
    },
  };
};

/**
 * Re-sync the model whenever the graph or the sampled dataset changes. Called from store.ts after
 * the data subscription, so the dataset is already the one for the new `graph.data`. The seed
 * starts at `graph.train.seed`.
 */
export function attachTrainLoop(store: StoreApi<Store>): void {
  rt.seed = store.getState().graph.train.seed;
  store.subscribe(() => {
    const s = store.getState(); // the listener argument can be stale inside nested set() calls
    if (s.graph !== rt.lastGraph || s.dataset !== rt.lastDataset) {
      // Moving a piece changes only `positions`: nothing to recompile, the run goes on.
      if (rt.lastGraph && s.dataset === rt.lastDataset && onlyMoved(rt.lastGraph, s.graph)) {
        rt.lastGraph = s.graph;
        return;
      }
      resyncRef?.();
    }
  });
  resyncRef?.();
}

let resyncRef: (() => void) | null = null;
