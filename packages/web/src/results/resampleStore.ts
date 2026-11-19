/**
 * State of the resampling overlay (tinker-iej.1): a small store of its own, not a slice. The run
 * happens in a module worker (resampleWorker.ts); this file posts the request, keeps the answer
 * and clears it when the graph or the data change. Only plain numbers and Float32Arrays live here.
 */
import { create } from 'zustand';
import { biasVariance, grid1d, grid2d, truthProbs } from '@tinker/core';
import { plotIndex } from '../state/plotPrediction.js';
import { useStore } from '../state/store.js';
import type { ResampleRequest, ResampleResponse } from './resampleRun.js';

/** Copies per overlay. */
export const COPIES = 20;
/** Steps used when the current run has none yet. */
export const FALLBACK_STEPS = 500;

export type OverlayResult = {
  /** 1 for a curve per copy, 2 for a surface per copy. */
  d: 1 | 2;
  gridN: number;
  /** Values per grid point in `curves` and `mean`: 1, or the number of classes (multi-class). */
  k: number;
  curves: Float32Array[];
  /** Pointwise average of `curves`. */
  mean: Float32Array;
  bias2: number;
  variance: number;
  /** Copies drawn, and copies that diverged and were left out. */
  used: number;
  diverged: number;
  steps: number;
  /** The run had 0 steps, so FALLBACK_STEPS was used. */
  fallbackSteps: boolean;
  /** The current graph trains with mini-batches; the copies are trained full batch. */
  fullBatchNote: boolean;
};

export type OverlayState = {
  status: 'idle' | 'busy' | 'done' | 'error';
  result: OverlayResult | null;
  message: string | null;
};

export const useOverlay = create<OverlayState>()(() => ({
  status: 'idle',
  result: null,
  message: null,
}));

let worker: Worker | null = null;
let token = 0;

function stopWorker(): void {
  worker?.terminate();
  worker = null;
}

/** Drops the result and any run in progress. */
export function clearOverlay(): void {
  token++;
  stopWorker();
  const s = useOverlay.getState();
  if (s.status !== 'idle') useOverlay.setState({ status: 'idle', result: null, message: null });
}

/** The overlay belongs to one graph and one dataset: any change to either clears it. */
useStore.subscribe((s, prev) => {
  if (s.graph !== prev.graph || s.dataset !== prev.dataset) clearOverlay();
});

/** Starts a run for the current graph, dataset and step count. No-op while one is running. */
export function runOverlay(): void {
  const st = useStore.getState();
  if (useOverlay.getState().status === 'busy' || !st.canTrain) return;
  if (typeof Worker === 'undefined') {
    useOverlay.setState({
      status: 'error',
      result: null,
      message: 'This browser cannot run the copies in the background.',
    });
    return;
  }
  const { graph, dataset } = st;
  const d = dataset.info.columns.length === 1 ? 1 : 2;
  const gridN = d === 1 ? 200 : 64;
  const fallbackSteps = st.step === 0;
  const steps = fallbackSteps ? FALLBACK_STEPS : st.step;
  const mine = ++token;
  stopWorker();
  useOverlay.setState({ status: 'busy', result: null, message: null });
  const w = new Worker(new URL('./resampleWorker.ts', import.meta.url), { type: 'module' });
  worker = w;
  const fail = (message: string) => {
    if (token !== mine) return;
    stopWorker();
    useOverlay.setState({ status: 'error', result: null, message });
  };
  w.onerror = () => fail('The copies could not be trained.');
  w.onmessage = (e: MessageEvent<ResampleResponse & { live?: number }>) => {
    if (token !== mine) return;
    const r = e.data;
    if (import.meta.env.DEV && r.live !== undefined) {
      const win = window as unknown as { __tinker?: Record<string, unknown> };
      win.__tinker = { ...win.__tinker, overlayLive: r.live };
    }
    if (!r.ok) return fail(r.message);
    // the plotted range: the grid spans the domain, so the plain mean over it is the average over x
    const grid = d === 1 ? grid1d(dataset.domain, gridN) : grid2d(dataset.domain, gridN);
    const multi = dataset.info.kind === 'multiclass';
    const k = multi ? dataset.info.numClasses : 1;
    const bv = biasVariance(r.curves, truthProbs(graph.data.set, grid.xRaw));
    if (bv.k === 0) return fail('Every copy diverged. Try a smaller learning rate.');
    useOverlay.setState({
      status: 'done',
      message: null,
      result: {
        d,
        gridN,
        k,
        curves: r.curves,
        mean: bv.mean,
        // summed over the k class probabilities of a point, averaged over the grid
        bias2: bv.bias2 * k,
        variance: bv.variance * k,
        used: bv.k,
        diverged: r.diverged,
        steps,
        fallbackSteps,
        fullBatchNote: graph.train.batch !== 'full',
      },
    });
  };
  const req: ResampleRequest = {
    graph,
    steps,
    k: COPIES,
    seed: graph.data.seed,
    gridN,
    prediction: plotIndex(graph, st.plotPrediction),
  };
  w.postMessage(req);
}
