import {
  effectiveTab,
  parsePort,
  tabsSplit,
  viewOfTab,
  type Analysis,
  type Graph,
  type ViewTab,
} from '@tinker/core';
import { edgeIndexOf } from '../canvas/pure.js';
import type { Selection } from './viewSlice.js';

/**
 * Whether `selection` is about something the graph tab `tab` draws as its own piece or wire (a
 * stub of a piece on the other tab does not count). The Code tab draws no graph, so nothing is.
 */
export function selectionDrawnOn(
  selection: Selection,
  graph: Graph,
  analysis: Analysis,
  tab: ViewTab,
  ui: readonly string[] | undefined,
): boolean {
  const t = effectiveTab(tab, ui);
  if (t === 'code') return false;
  const { graph: view, ghosts } = viewOfTab(graph, analysis, t, !tabsSplit(ui));
  const own = (id: string) => view.nodes.some((n) => n.id === id) && ghosts[id] === undefined;
  if (selection.kind === 'wire') {
    const i = edgeIndexOf(graph, selection.from, selection.to);
    const e = view.edges[i];
    return (
      i >= 0 &&
      e !== undefined &&
      e[0] !== '' &&
      (own(parsePort(selection.from).node) || own(parsePort(selection.to).node))
    );
  }
  return own(selection.kind === 'node' ? selection.id : selection.node);
}
