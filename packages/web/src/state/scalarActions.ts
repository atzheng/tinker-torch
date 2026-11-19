import {
  expandToScalars,
  groupIntoLinear,
  parsePort,
  type Analysis,
  type Graph,
  type NodeId,
  type ScalarEditResult,
  type HostParams,
  findLinearGroups,
} from '@tinker/core';
import { lockedIds } from './locks.js';
import { useStore } from './store.js';
import { recordScalarEdit } from './weightHistory.js';
import type { Selection } from './viewSlice.js';

/** How long the new pieces fade in (CONTRACTS.md section 10). */
export const FADE_MS = 300;

const cache = new WeakMap<Analysis, NodeId[][]>();
/** `findLinearGroups`, computed once per analysis (the canvas, the bar and the actions share it). */
export function linearGroups(graph: Graph, analysis: Analysis): NodeId[][] {
  let g = cache.get(analysis);
  if (!g) {
    g = findLinearGroups(graph, analysis);
    cache.set(analysis, g);
  }
  return g;
}

const DIVERGED = 'Reset first: the weights diverged.';
const NO_VALUES = 'The values of these pieces are not known yet, so they cannot be carried over.';

/** The group to act on: the one holding the selected piece, else the first; null when there is none. */
export function pickGroup(
  groups: readonly NodeId[][],
  selection: Selection | null,
): NodeId[] | null {
  if (selection?.kind === 'node') {
    const hit = groups.find((g) => g.includes(selection.id));
    if (hit) return hit;
  }
  return groups[0] ?? null;
}

/** The d to 1 Linear that Expand to scalars would act on: selected, fed by exactly one Input. */
export function expandTarget(graph: Graph, selection: Selection | null): NodeId | null {
  if (selection?.kind !== 'node') return null;
  return isExpandable(graph, selection.id) ? selection.id : null;
}

function isExpandable(graph: Graph, id: NodeId): boolean {
  const n = graph.nodes.find((x) => x.id === id);
  if (n?.type !== 'linear' || n.out !== 1) return false;
  const feeds = graph.edges.filter((e) => parsePort(e[1]).node === n.id);
  const from = feeds.length === 1 ? parsePort(feeds[0]?.[0] ?? '').node : '';
  return graph.nodes.find((x) => x.id === from)?.type === 'input';
}

/** Whether the equivalence caption applies: a group pattern, or any d to 1 Linear fed by an Input. */
export function hasScalarForm(graph: Graph, analysis: Analysis): boolean {
  return (
    linearGroups(graph, analysis).length > 0 || graph.nodes.some((n) => isExpandable(graph, n.id))
  );
}

/**
 * Apply a group / expand result as ONE undo step. The graph and the parameters move together: the
 * edit is recorded for the train slice (`recordScalarEdit`), which installs `r.host` under the new
 * graph and, on undo or redo, the matching side. History, step and loss carry on (the function is
 * the same). While Running the run keeps going: the values are identical, so loss continues; the
 * carried values are read from the engine at the moment of the edit (no steps are lost) and the
 * optimizer's own state (momentum, Adam moments) restarts. After a divergence the edit is refused.
 */
function run(make: (host: HostParams) => ScalarEditResult): boolean {
  const s = useStore.getState();
  if (s.diverged) {
    useStore.setState({ error: DIVERGED });
    return false;
  }
  const host = s.currentHost() ?? s.host;
  if (!host) {
    useStore.setState({ error: NO_VALUES });
    return false;
  }
  const r = make(host);
  if (!r.ok) {
    useStore.setState({ error: r.error });
    return false;
  }
  const before = new Set(s.graph.nodes.map((n) => n.id));
  recordScalarEdit(s.graph, r.graph, host, r.host, s.step);
  s.apply({ ok: true, graph: r.graph });
  s.select({ kind: 'node', id: r.select });
  const reduced =
    typeof window !== 'undefined' &&
    typeof window.matchMedia === 'function' &&
    window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  if (!reduced) {
    const fresh = new Set(r.graph.nodes.map((n) => n.id).filter((id) => !before.has(id)));
    s.setFresh(fresh);
    setTimeout(() => {
      if (useStore.getState().fresh === fresh) useStore.getState().setFresh(new Set());
    }, FADE_MS);
  }
  return true;
}

export function groupSelected(): boolean {
  const { graph, analysis, selection } = useStore.getState();
  const ids = pickGroup(linearGroups(graph, analysis), selection);
  if (!ids) return false;
  return run((host) => groupIntoLinear(graph, host, ids, { locked: lockedIds() }));
}

export function expandSelected(): boolean {
  const { graph, selection } = useStore.getState();
  const id = expandTarget(graph, selection);
  if (!id) return false;
  return run((host) => expandToScalars(graph, host, id, { locked: lockedIds() }));
}
