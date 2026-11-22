import type { StateCreator, StoreApi } from 'zustand';
import {
  analyse,
  batchSchedule,
  exampleStats,
  gatherBatch,
  initParams,
  makeRng,
  reconcileParams,
  withStartParams,
  sample,
  type Analysis,
  type Batch,
  type Graph,
  type HostParams,
  type LossMode,
} from '@tinker/core';
import type { Engine, Model, ParamAxis, Params, TrainState } from '@tinker/engine';
import { getEngine, onEngineError, onEngineReady } from '../engine.js';
import { listParams } from '../plot/lossCurve1d.js';
import { initialGraph } from './graphSlice.js';
import { currentLossMode, type ExampleStatsPair } from './lossMode.js';
import { plotIndex } from './plotPrediction.js';
import { probabilitySentence } from './probRange.js';
import type { Store } from './store.js';
import { bumpWeightGeneration, scalarHostFor, type ScalarCarry } from './weightHistory.js';

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
/**
 * One Step of the step log (tinker-6t7.10). `values` are the parameters AFTER the step (in
 * `listParams` order), `prediction` the first prediction pin at those parameters (first training
 * example when the model has inputs), `slopes` the derivative of the full-batch loss at the
 * parameters BEFORE the step (null unless plain SGD, full batch and no dropout, where it is not the step's
 * gradient) and `loss` the loss the step measured before the update.
 */
export type StepLogRow = {
  step: number;
  values: number[];
  prediction: number | null;
  slopes: number[] | null;
  loss: number;
};
/** The log is kept only for models with at most this many scalar parameter entries. */
export const STEP_LOG_MAX_PARAMS = 4;
/** Rows kept. */
export const STEP_LOG_MAX_ROWS = 50;

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
  /**
   * Plain sentence when an unfused log loss / cross-entropy gets a prediction outside (0, 1)
   * (tinker-ddg), else null. Set when a step stops on a non-finite loss for that reason, and
   * re-checked after a hand edit, a reset and a compile; the run is stopped but not "diverged":
   * the parameters are the last finite ones and a hand edit back into range clears it.
   */
  probIssue: string | null;
  /** Parameter updates done since the last Reset or structural edit. */
  step: number;
  trainLoss: number | null;
  valLoss: number | null;
  history: HistoryPoint[];
  best: Best | null;
  /** Rows of the step log; only Step adds rows (and only when the preset shows `panel-step-log`). */
  stepLog: StepLogRow[];
  /** The readout mode the user picked (tinker-6t7.5); `currentLossMode` applies the preset's disclosure. */
  lossMode: LossMode;
  setLossMode: (m: LossMode) => void;
  /**
   * The objective L (dropout off) of the current parameters on the whole train and validation
   * sets, null for a split that is empty or non-finite, and null as a whole with no compiled
   * model. Republished with `host` (at most once per frame while running, on every hand edit):
   * the loss readout shows it in every mode, so it follows a dragged parameter.
   */
  liveLoss: { train: number | null; val: number | null } | null;
  /**
   * Likelihood (product of the per-example probabilities, float32) of the current parameters on
   * each split; null unless the current readout mode is likelihood. Republished with `liveLoss`.
   */
  exampleStats: ExampleStatsPair | null;
  /**
   * Per-example values that the L pin's Mean averages (`Engine.exampleLosses`, dropout off) of the
   * current parameters on the whole train or validation split, one per row; null with no compiled
   * model or no Mean. For the likelihood readout and the per-example strip.
   */
  exampleLosses: (split: 'train' | 'val') => Float32Array | null;
  /** Steps per frame while running (12 ms budget per frame). */
  speed: number;
  /** Immutable snapshot of the current parameters, published at most once per frame. */
  host: HostParams | null;
  /**
   * The exact parameters right now, read from the engine (`host` is at most one frame old while a
   * run is going). Null when there is no TrainState.
   */
  currentHost: () => HostParams | null;
  stepOnce: () => void;
  run: () => void;
  pause: () => void;
  reset: () => void;
  /** Like Reset, but ignores the start values (except for locked pieces): a new seed and fresh initialisation. */
  randomStart: () => void;
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
   * 1-D loss curve (tinker-6t7.2): training loss (as `lossSurface`) with the one parameter `axis`
   * set to each of `values`, every other parameter as it is; one number per value, or null when
   * there is no compiled model or the call fails.
   */
  lossCurve: (axis: ParamAxis, values: Float32Array) => Float32Array | null;
  /**
   * Derivative of the training loss (whole training set, dropout off) with respect to the one
   * parameter `axis` at its current value: the gradient Step applies for it with full batch and no
   * dropout. Null when there is no compiled model or the call fails.
   */
  lossSlope: (axis: ParamAxis) => number | null;
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
  stepLog: StepLogRow[];
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
  stepLog: [],
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
      stepLog: rt.stepLog,
      host,
      liveLoss: computeLiveLoss(),
      exampleStats: computeExampleStats(),
      ...extra,
    });
  }

  /**
   * The sentence for a probability loss whose prediction is outside (0, 1) on the training set, or
   * null. Non-jitted engine call, so it is made only when a loss is non-finite or after an edit.
   */
  function probSentence(): string | null {
    const { engine, model, state, trainBatch } = rt;
    if (!engine || !model || !state || !trainBatch || trainBatch.n === 0) return null;
    try {
      return probabilitySentence(
        get().graph,
        engine.probabilityRange(model, state.params, trainBatch),
      );
    } catch {
      return null;
    }
  }

  function exampleLossesOf(split: 'train' | 'val'): Float32Array | null {
    const { engine, model, state } = rt;
    const batch = split === 'train' ? rt.trainBatch : rt.valBatch;
    if (!engine || !model || !state || !batch || batch.n === 0) return null;
    try {
      return engine.exampleLosses(model, state.params, batch);
    } catch {
      return null;
    }
  }

  /** The stats the readout needs now: only for the likelihood and total modes, else null. */
  function computeExampleStats(): ExampleStatsPair | null {
    const mode = currentLossMode(get());
    if (mode !== 'likelihood') return null;
    const t = exampleLossesOf('train');
    const v = exampleLossesOf('val');
    return { train: t && exampleStats(t), val: v && exampleStats(v) };
  }

  function computeLiveLoss(): TrainSlice['liveLoss'] {
    const { engine, model, state, trainBatch, valBatch } = rt;
    if (!engine || !model || !state) return null;
    const one = (b: Batch | null): number | null => {
      if (!b || b.n === 0) return null;
      try {
        const v = engine.evaluate(model, state.params, b);
        return Number.isFinite(v) ? v : null;
      } catch {
        return null;
      }
    };
    return { train: one(trainBatch), val: one(valBatch) };
  }

  /** The start values of locked pieces only: a random start never changes what a preset locks. */
  function lockedStart(start: HostParams | null): HostParams | null {
    if (!start) return null;
    const out = Object.create(null) as HostParams;
    for (const id of get().preset?.locked ?? [])
      if (Object.hasOwn(start, id)) out[id] = start[id] as HostParams[string];
    return out;
  }

  function clearRun(): void {
    rt.history = [];
    rt.best = null;
    rt.stepLog = [];
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
  function resync(freshArg = false, carried: ScalarCarry | null = null, randomStart = false): void {
    const { graph, dataset, dataInfo } = get();
    // A diverged run leaves huge (finite) weights behind, so no lr, optimizer, batch or data
    // change could recover from them: the edit that follows a divergence starts from fresh weights.
    const fresh = freshArg || get().diverged;
    if (fresh) bumpWeightGeneration(); // hand-edit undo records no longer fit these weights
    const engine = rt.engine;
    // Group / expand (and their undo and redo): the same function under new nodes. The weights
    // are carried over exactly and the run (history, step, loss) goes on; only the optimizer's
    // own state (momentum, Adam moments) restarts, as it does for any recompile.
    const carry = fresh ? null : carried;
    const structural =
      !carry && (fresh || !rt.lastGraph || structuralKey(graph) !== structuralKey(rt.lastGraph));
    const wasRunning = get().running;
    rt.lastGraph = graph;
    rt.lastDataset = dataset;

    // 1. Take the parameters out of the live state, then free it.
    if (rt.state && engine) {
      // The engine holds only live owners; inert owners' weights stay frozen in the old host (reconcileParams).
      const live = engine.toHost(rt.state.params);
      rt.host = Object.assign(Object.create(null) as HostParams, rt.host ?? {}, live);
    }
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
        probIssue: null,
        liveLoss: null,
        exampleStats: null,
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
    rt.host = carry
      ? carry.host
      : rt.host && rt.hostGraph && rt.hostAnalysis
        ? reconcileParams(rt.host, rt.hostAnalysis, analysis, rt.seed)
        : withStartParams(
            initParams(analysis, rt.seed),
            randomStart ? lockedStart(get().startParams) : get().startParams,
            analysis,
          ); // Reset: the loaded values, the rest from the seed; New random start: the seed only (locked pieces keep theirs)
    rt.hostGraph = graph;
    rt.hostAnalysis = analysis;
    if (structural || carry) {
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
        if (carry && carry.step !== rt.step) {
          // Undo / redo went back to the weights of an earlier step: the run's numbers go back with
          // them, and the losses are measured on these weights (full train set, validation set).
          rt.step = carry.step;
          rt.history = rt.history.filter((h) => h.step <= carry.step);
          rt.stepLog = rt.stepLog.filter((r) => r.step <= carry.step);
          const { trainBatch, valBatch } = rt;
          const t =
            trainBatch && trainBatch.n > 0
              ? engine.evaluate(model, rt.state.params, trainBatch)
              : NaN;
          const v =
            valBatch && valBatch.n > 0 ? engine.evaluate(model, rt.state.params, valBatch) : NaN;
          rt.trainLoss = Number.isFinite(t) ? t : null;
          rt.valLoss = Number.isFinite(v) ? v : null;
        }
        if (structural) evaluateInitial();
        else if (carry && rt.valLoss !== null && Number.isFinite(rt.valLoss)) {
          rt.best = { step: rt.step, val: rt.valLoss, params: carry.host }; // the carried weights
        }
      } catch (e) {
        if (params && rt.state === null) engine.dispose(params); // startTraining throws before consuming
        if (model && rt.model !== model) engine.disposeModel(model); // compiled but never used
        issue = e instanceof Error ? e.message : 'The model could not be compiled.';
      }
    }
    const canTrain = rt.state !== null;
    const keepRunning = !structural && wasRunning && canTrain; // a group / expand keeps running
    // One store update, so a reader never sees the new analysis with the old host.
    publish({
      analysis,
      trainIssue: issue,
      canTrain,
      running: false,
      diverged: false,
      probIssue: probSentence(),
    });
    if (keepRunning) startLoop();
  }

  /** Does this step add a log row: a Step press (not Run), the panel shown, a small model. */
  function logging(logRow: boolean): number | null {
    const { show, host, analysis } = get();
    if (!logRow || !show?.ui?.includes('panel-step-log') || !host) return null;
    const n = listParams(host, analysis.order).length;
    return n > 0 && n <= STEP_LOG_MAX_PARAMS ? n : null;
  }

  /**
   * Derivative of the full-batch loss per parameter entry at the current parameters, or null when
   * it is not what the step applies (not plain SGD, mini-batch, or dropout in the graph).
   */
  function slopesBefore(n: number): number[] | null {
    const { graph, dataset, host, analysis } = get();
    if (graph.train.optimizer !== 'sgd') return null; // momentum / Adam: the update is not lr x slope
    if (isMiniBatch(graph.train.batch, dataset.nTrain)) return null;
    if (graph.nodes.some((nd) => nd.type === 'dropout') || !host) return null;
    const out: number[] = [];
    for (const p of listParams(host, analysis.order).slice(0, n)) {
      const d = get().lossSlope(p.axis);
      if (d === null) return null;
      out.push(d);
    }
    return out;
  }

  /** First prediction pin at `params`: for the first training example, or the lone value when there is no input. */
  function predictionOf(params: Params): number | null {
    const { engine, model, trainBatch } = rt;
    if (!engine || !model || !trainBatch) return null;
    const d = get().dataInfo.columns.length;
    try {
      const x = trainBatch.x.slice(0, d);
      const v = engine.predict(model, params, x, 1)[
        plotIndex(get().graph, get().plotPrediction)
      ]?.[0];
      return v !== undefined && Number.isFinite(v) ? v : null;
    } catch {
      return null;
    }
  }

  function doStep(logRow = false): 'ok' | 'diverged' | 'range' {
    const { engine, model, state, batches } = rt;
    const { dataset, dataInfo } = get();
    if (!engine || !model || !state || !batches) return 'diverged';
    const idx = batches.next().value;
    const batch = gatherBatch(dataset.xTrain, dataset.yTrain, dataInfo.columns.length, idx);
    const log = logging(logRow);
    const slopes = log ? slopesBefore(log) : null; // before the step consumes the state
    const r = engine.step(model, state, batch);
    rt.state = r.state;
    if (r.diverged) {
      // the engine refuses a step whose probability prediction is outside (0, 1) and says so in `ranges`
      const sentence = r.ranges ? probabilitySentence(get().graph, r.ranges) : null;
      if (sentence === null) return 'diverged';
      set({ probIssue: sentence });
      return 'range';
    }
    if (get().probIssue !== null) set({ probIssue: null });
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
    if (log) {
      const values = listParams(engine.toHost(r.state.params), get().analysis.order).map(
        (p) => p.value,
      );
      rt.stepLog = [
        ...rt.stepLog,
        { step: rt.step, values, prediction: predictionOf(r.state.params), slopes, loss: r.loss },
      ].slice(-STEP_LOG_MAX_ROWS);
    }
    if (rt.history.length > MAX_HISTORY) {
      const last = rt.history.length - 1;
      rt.history = rt.history.filter((_, i) => i % 2 === 0 || i === last);
    }
    return 'ok';
  }

  /** Runs doStep and turns divergence or an engine error into store state. Returns whether to go on. */
  function guardedStep(logRow = false): boolean {
    try {
      const r = doStep(logRow);
      if (r === 'ok') return true;
      // A probability out of range stops the run like a divergence, but the weights are fine.
      set(r === 'range' ? { running: false } : { diverged: true, running: false });
      return false;
    } catch (e) {
      set({
        probIssue: null, // an engine error is never hidden behind a stale probability sentence
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
        probIssue: null,
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

  refreshExampleStatsRef = () => {
    const stats = computeExampleStats();
    if (stats !== get().exampleStats) set({ exampleStats: stats });
  };
  resyncRef = (carried) => resync(false, carried ?? null);

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
    probIssue: null,
    step: 0,
    trainLoss: null,
    valLoss: null,
    history: [],
    best: null,
    stepLog: [],
    lossMode: 'mean',
    setLossMode: (m) => set({ lossMode: m }),
    liveLoss: null,
    exampleStats: null,
    exampleLosses: exampleLossesOf,
    speed: DEFAULT_SPEED,
    host: null,
    currentHost: () => {
      const { engine, state } = rt;
      return engine && state ? engine.toHost(state.params) : null;
    },
    stepOnce: () => {
      const s = get();
      if (!s.canTrain || s.running || s.diverged) return;
      guardedStep(true);
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
    randomStart: () => {
      if (!get().analysis.trainable) return;
      rt.seed = makeRng(rt.seed)
        .fork('random-start')
        .int(2 ** 31);
      resync(true, null, true);
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
        rt.stepLog = []; // the rows no longer lead to these weights
        publish({ running: false, diverged: false, probIssue: probSentence() });
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
      publish({
        trainIssue: issue,
        canTrain: rt.state !== null,
        running: false,
        diverged: false,
        probIssue: probSentence(),
      });
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
    lossCurve: (axis, values) => {
      const { engine, model, state } = rt;
      if (!engine || !model || !state) return null;
      const { dataset } = get();
      try {
        const batch = { x: dataset.xTrain, y: dataset.yTrain, n: dataset.nTrain };
        return engine.lossCurve(model, state.params, batch, axis, values);
      } catch {
        return null;
      }
    },
    lossSlope: (axis) => {
      const { engine, model, state } = rt;
      if (!engine || !model || !state) return null;
      const { dataset } = get();
      try {
        const batch = { x: dataset.xTrain, y: dataset.yTrain, n: dataset.nTrain };
        return engine.lossSlope(model, state.params, batch, axis);
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
      rt.stepLog = []; // the rows no longer follow from the weights
      publish({ probIssue: probSentence() }); // a hand edit can take a probability out of (0, 1) and back
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
  // A mode switch or a new preset disclosure refreshes the per-example stats (they are only computed for the modes that need them).
  store.subscribe((s, prev) => {
    if (s.lossMode !== prev.lossMode || s.show !== prev.show) refreshExampleStatsRef?.();
  });
  store.subscribe(() => {
    const s = store.getState(); // the listener argument can be stale inside nested set() calls
    if (s.graph !== rt.lastGraph || s.dataset !== rt.lastDataset) {
      // Moving a piece changes only `positions`: nothing to recompile, the run goes on.
      if (rt.lastGraph && s.dataset === rt.lastDataset && onlyMoved(rt.lastGraph, s.graph)) {
        rt.lastGraph = s.graph;
        return;
      }
      resyncRef?.(rt.lastGraph ? scalarHostFor(rt.lastGraph, s.graph) : null);
    }
  });
  resyncRef?.();
}

let refreshExampleStatsRef: (() => void) | null = null;
let resyncRef: ((carried?: ScalarCarry | null) => void) | null = null;
