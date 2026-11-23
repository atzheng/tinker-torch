/** Pure helpers for the results panels (tinker-ong): order, drop position, resize. No DOM, no React. */
import type { Show, ViewTab } from '@tinker/core';

export const PANEL_IDS = [
  'fit',
  'loss',
  'surface',
  'histogram',
  'loss-1d',
  'step-log',
  'examples',
  'class-bars',
] as const;
export type PanelId = (typeof PANEL_IDS)[number];
export type PanelWeights = Record<PanelId, number>;

/** The tab each panel belongs to (CONTRACTS.md section 9, "Tabs"): the fit and the target histogram describe the model; the rest describe the loss. */
export const PANEL_TAB: Record<PanelId, 'model' | 'loss'> = {
  fit: 'model',
  histogram: 'model',
  loss: 'loss',
  surface: 'loss',
  'loss-1d': 'loss',
  examples: 'loss',
  'step-log': 'loss',
  'class-bars': 'model',
};

/** Whether a panel is on screen with `tab` showing; with one tab shown (`split` false), or on the full tab, every panel the preset lists is. */
export function panelOnTab(
  id: PanelId,
  tab: ViewTab,
  split: boolean,
  panelTabs?: Show['panelTabs'],
): boolean {
  if (!split || tab === 'full') return true;
  return (panelTabs?.[id] ?? [PANEL_TAB[id]]).includes(tab);
}

/**
 * Whether the preset's `ui` list (undefined: everything) puts panel `id` on screen. The Loss panel
 * is also on screen for `panel-score` alone: the score readout without the loss-over-steps chart
 * (tinker-bo1). Only `panel-loss` itself adds the chart; see `lossChartShown`.
 */
export function panelListed(ui: readonly string[] | undefined, id: PanelId): boolean {
  if (ui === undefined || ui.includes(`panel-${id}`)) return true;
  return id === 'loss' && ui.includes('panel-score');
}

/** Whether the Loss panel draws its chart (with its log toggle, Restore best and pins): `panel-loss` is listed or there is no list. */
export function lossChartShown(ui: readonly string[] | undefined): boolean {
  return ui === undefined || ui.includes('panel-loss');
}

export const DEFAULT_ORDER: readonly PanelId[] = PANEL_IDS;
export const DEFAULT_WEIGHTS: PanelWeights = {
  fit: 1,
  loss: 1,
  surface: 1,
  histogram: 1,
  'loss-1d': 1,
  examples: 1,
  'step-log': 1,
  'class-bars': 1,
};

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

/**
 * The panels lie in a column (landscape: stacked, split bars horizontal) or in a row (portrait:
 * side by side, split bars vertical). The reorder and resize maths above is one-dimensional: these
 * pick the coordinate along the panels' axis, so the same code serves both.
 */
export type Orientation = 'column' | 'row';

/** A rectangle's midpoint along the axis (y for a column, x for a row). */
export function midAlong(
  o: Orientation,
  r: { left: number; right: number; top: number; bottom: number },
): number {
  return o === 'row' ? (r.left + r.right) / 2 : (r.top + r.bottom) / 2;
}

/** A pointer position along the axis. */
export function posAlong(o: Orientation, p: { x: number; y: number }): number {
  return o === 'row' ? p.x : p.y;
}

/** A pointer movement along the axis. */
export function deltaAlong(o: Orientation, p: { dx: number; dy: number }): number {
  return o === 'row' ? p.dx : p.dy;
}

/** The container's extent along the axis (height for a column, width for a row). */
export function extentAlong(o: Orientation, r: { width: number; height: number }): number {
  return o === 'row' ? r.width : r.height;
}
