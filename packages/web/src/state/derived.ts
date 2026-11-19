import { useMemo } from 'react';
import { analyse, initParams, type Analysis, type HostParams } from '@tinker/core';
import { edgeIndexOf, neuronSelectionLive } from '../canvas/pure.js';
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
 * The selection if it still names something in the graph, else null. With the analysis, an edge is
 * also checked against the current widths (edges shift after edits).
 */
export function useSelection(analysis?: Analysis): Selection | null {
  const graph = useStore((s) => s.graph);
  const sel = useStore((s) => s.selection);
  if (!sel) return null;
  if (sel.kind === 'node') return graph.nodes.some((n) => n.id === sel.id) ? sel : null;
  if (sel.kind === 'wire') return edgeIndexOf(graph, sel.from, sel.to) >= 0 ? sel : null;
  return neuronSelectionLive(sel, graph, analysis) ? sel : null;
}
