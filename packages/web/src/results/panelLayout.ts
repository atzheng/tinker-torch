/** Pure helpers for the results panels (tinker-ong): order, drop position, resize. No DOM, no React. */

export const PANEL_IDS = ['fit', 'loss', 'surface'] as const;
export type PanelId = (typeof PANEL_IDS)[number];
export type PanelWeights = Record<PanelId, number>;

export const DEFAULT_ORDER: readonly PanelId[] = PANEL_IDS;
export const DEFAULT_WEIGHTS: PanelWeights = { fit: 1, loss: 1, surface: 1 };

/** A panel never shrinks below this share of the visible panels' total weight. */
export const MIN_SHARE = 0.15;

/**
 * Moves panel `id` so it ends at index `to` of the visible panels (counted without `id`, so
 * `to` is the number of other visible panels that stay above it). Hidden panels keep their place
 * relative to their neighbours: `id` goes right before the visible panel it now precedes, or right
 * after the last visible one.
 */
export function movePanel(
  order: readonly PanelId[],
  visible: ReadonlySet<PanelId>,
  id: PanelId,
  to: number,
): PanelId[] {
  const rest = order.filter((p) => p !== id);
  const vis = rest.filter((p) => visible.has(p));
  const k = Math.max(0, Math.min(vis.length, to));
  const anchor = vis[k];
  let at: number;
  if (anchor !== undefined) at = rest.indexOf(anchor);
  else {
    const last = vis[vis.length - 1];
    at = last === undefined ? rest.length : rest.indexOf(last) + 1;
  }
  return [...rest.slice(0, at), id, ...rest.slice(at)];
}

/** Where a dragged panel would land: how many of the other panels' midpoints (px) lie above `y`. */
export function dropIndex(otherMids: readonly number[], y: number): number {
  return otherMids.filter((m) => m < y).length;
}

/**
 * Drag the splitter between neighbours `a` (above) and `b` (below) by `dy` px of a column that is
 * `height` px tall and holds `visible` panels. Weights are shares of the column, so the shift is
 * `dy / height * total`; the pair's sum is kept and neither falls below `MIN_SHARE` of the total.
 */
export function resizePair(
  start: Readonly<PanelWeights>,
  visible: readonly PanelId[],
  a: PanelId,
  b: PanelId,
  dy: number,
  height: number,
): PanelWeights {
  const total = visible.reduce((s, p) => s + start[p], 0);
  const min = MIN_SHARE * total;
  const pair = start[a] + start[b];
  const want = start[a] + (dy / Math.max(1, height)) * total;
  const na = Math.min(Math.max(want, min), pair - min);
  return { ...start, [a]: na, [b]: pair - na };
}
