import { useMemo } from 'react';
import { analyse, initParams, type Analysis, type HostParams } from '@tinker/core';
import { edgeIndexOf } from '../canvas/pure.js';
import { useStore, type Store } from './store.js';
import type { Selection } from './viewSlice.js';

/** Analysis of the current graph against the current dataset. Memoised on both. */
export function useAnalysis(): Analysis {
  const graph = useStore((s) => s.graph);
  const info = useStore((s) => s.dataInfo);
  return useMemo(() => analyse(graph, info), [graph, info]);
}

/** Host snapshot from the train slice, when it has one. */
export const selectTrainedHost = (s: Store): HostParams | undefined =>
  // The train slice has no fields yet; tinker-27m.2 adds `host`. Read it structurally until then.
  (s as { host?: HostParams }).host;

/**
 * The host weights to draw. The training slice (tinker-27m.2) will own `host`; until it has one
 * this falls back to `initParams` for the current graph (CONTRACTS.md section 5), which is what
 * the model looks like at step 0. Arrays in it are immutable: read only. Entries can be missing
 * (a Linear off the loss path) or have another shape than the graph says (stale during an edit):
 * callers check lengths.
 */
export function useWeights(analysis: Analysis): HostParams | undefined {
  const graph = useStore((s) => s.graph);
  const trained = useStore(selectTrainedHost);
  return useMemo(() => {
    if (trained) return trained;
    try {
      return initParams(graph, analysis, graph.train.seed);
    } catch {
      return undefined; // the page never crashes on an invalid graph
    }
  }, [trained, graph, analysis]);
}

/** The selection if it still names something in the graph, else null. */
export function useSelection(): Selection | null {
  const graph = useStore((s) => s.graph);
  const sel = useStore((s) => s.selection);
  if (!sel) return null;
  if (sel.kind === 'node') return graph.nodes.some((n) => n.id === sel.id) ? sel : null;
  return edgeIndexOf(graph, sel.from, sel.to) >= 0 ? sel : null;
}
