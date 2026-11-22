import { useEffect, useMemo, useRef, useState } from 'react';
import { analyse, initParams, type Analysis, type HostParams } from '@tinker/core';
import { edgeIndexOf, neuronSelectionLive, pinLive } from '../canvas/pure.js';
import {
  PROB_BAR_EVERY_MS,
  PROB_BAR_MAX_ROWS,
  probBarDue,
  probBarTargets,
  probBars,
  type ProbBar,
} from '../canvas/probBarPure.js';
import { useStore, type Store } from './store.js';
import type { Selection } from './viewSlice.js';

/** Analysis of the current graph against the current dataset. Memoised on both. */
export function useAnalysis(): Analysis {
  const graph = useStore((s) => s.graph);
  const info = useStore((s) => s.dataInfo);
  return useMemo(() => analyse(graph, info), [graph, info]);
}

/** Host snapshot from the train slice, when it has one. */
export const selectTrainedHost = (s: Store): HostParams | undefined => s.host ?? undefined;

const selectNoHost = (): undefined => undefined;

/**
 * Weights at step 0: `initParams` for the current graph (CONTRACTS.md section 5), memoised on the
 * graph and analysis. Undefined when the graph cannot be initialised (the page never crashes).
 */
export function useInitialWeights(analysis: Analysis): HostParams | undefined {
  const graph = useStore((s) => s.graph);
  return useMemo(() => {
    try {
      return initParams(analysis, graph.train.seed);
    } catch {
      return undefined;
    }
  }, [graph, analysis]);
}

/**
 * The host weights to draw: the train slice's `host` when it has one, else the step-0 weights
 * (`useInitialWeights`). Arrays in it are immutable: read only. Entries can be missing (a Linear
 * off the loss path) or have another shape than the graph says (stale during an edit): callers
 * check lengths. With `live` false the hook does not follow `host`, so the caller does not
 * re-render on every training frame (the neuron view redraws its edges itself, see NeuronView).
 */
export function useWeights(analysis: Analysis, live = true): HostParams | undefined {
  const trained = useStore(live ? selectTrainedHost : selectNoHost);
  const initial = useInitialWeights(analysis);
  return trained ?? initial;
}

/**
 * Probability bars (tinker-6t7.17) for the Links that carry a ŷ pin, by node id: the value of the
 * current parameters, averaged over the training examples. Follows `host`, so it updates live; empty
 * until the engine has a model. Nothing engine-side is kept (`predictPins` copies out plain arrays).
 */
export function useProbBars(analysis: Analysis): Record<string, ProbBar> {
  const graph = useStore((s) => s.graph);
  const dataset = useStore((s) => s.dataset);
  const host = useStore((s) => s.host);
  const running = useStore((s) => s.running);
  const predictPins = useStore((s) => s.predictPins);
  const [, bump] = useState(0);
  const cache = useRef<{
    t: number;
    host: unknown;
    deps: unknown[];
    bars: Record<string, ProbBar>;
  } | null>(null);
  const deps = [graph, analysis, dataset, predictPins];
  const c = cache.current;
  const depsSame = c !== null && c.deps.every((d, i) => d === deps[i]);
  const d = dataset.info.columns.length;
  const now = performance.now();
  // while running, new parameters are read at most every PROB_BAR_EVERY_MS (a trailing timer catches up)
  if (!c || !depsSame || (c.host !== host && probBarDue(now, c.t, running, d === 0))) {
    let bars: Record<string, ProbBar> = {};
    if (probBarTargets(graph, analysis.outWidth).length > 0) {
      const n = d === 0 ? 1 : Math.min(dataset.nTrain, PROB_BAR_MAX_ROWS);
      const outs = n < 1 ? null : predictPins(dataset.xTrain.slice(0, n * d), n);
      if (outs) bars = probBars(graph, analysis.outWidth, outs, n);
    }
    cache.current = { t: now, host, deps, bars };
  }
  const pending = cache.current?.host !== host;
  useEffect(() => {
    if (!pending) return;
    const id = setTimeout(() => bump((k) => k + 1), PROB_BAR_EVERY_MS);
    return () => clearTimeout(id);
  });
  return cache.current?.bars ?? {};
}

/**
 * The selection if it still names something in the graph, else null. With the analysis, an edge is
 * also checked against the current widths (edges shift after edits).
 */
export function useSelection(analysis?: Analysis): Selection | null {
  const graph = useStore((s) => s.graph);
  const sel = useStore((s) => s.selection);
  if (!sel) return null;
  if (sel.kind === 'node') return graph.nodes.some((n) => n.id === sel.id) ? sel : null;
  if (sel.kind === 'wire') return edgeIndexOf(graph, sel.from, sel.to) >= 0 ? sel : null;
  if (sel.kind === 'pin') return pinLive(graph, sel.role, sel.node) ? sel : null;
  return neuronSelectionLive(sel, graph, analysis) ? sel : null;
}
