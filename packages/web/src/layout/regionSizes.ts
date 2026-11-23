/**
 * Pure rules for the resizable main regions (tinker-9qh): the width of the results column
 * (landscape), the height of the results band (portrait) and the height cap of the instructions
 * steps. Sizes are fractions of the window (or of the app grid) so a rotation or a window resize
 * keeps them sensible; `null` in the store means "the stylesheet default". No DOM, no React.
 */

export type RegionKey = 'resultsWidth' | 'resultsBand' | 'instructions';

export type RegionLimit = {
  /** The fraction of the extent the stylesheet uses when nothing was dragged. */
  def: number;
  min: number;
  max: number;
};

export const REGION_LIMITS: Record<RegionKey, RegionLimit> = {
  // Landscape: results share of the window width. At 1024 px that is 205 .. 512 px, which leaves
  // the palette (at least 140 px) and a canvas of about 350 px at the widest.
  resultsWidth: { def: 0.3, min: 0.2, max: 0.5 },
  // Portrait: results band share of the window height (the stylesheet clamps the default).
  resultsBand: { def: 0.32, min: 0.15, max: 0.5 },
  // The steps list of the instructions panel, share of the window height (28vh by default).
  instructions: { def: 0.28, min: 0.1, max: 0.6 },
};

/** Arrow key step and Shift+arrow step, as a fraction of the extent. */
export const KEY_STEP = 0.02;
export const KEY_STEP_BIG = 0.1;

export type RegionSizes = Record<RegionKey, number | null>;
export const DEFAULT_REGION_SIZES: RegionSizes = {
  resultsWidth: null,
  resultsBand: null,
  instructions: null,
};

export function clampSize(key: RegionKey, v: number): number {
  const { min, max } = REGION_LIMITS[key];
  return Math.min(max, Math.max(min, v));
}

/**
 * A drag of `deltaPx` along the axis in an extent of `extent` px. `sign` is +1 when moving the
 * pointer forward (right, down) grows the region and -1 when it shrinks it.
 */
export function dragSize(
  key: RegionKey,
  start: number,
  deltaPx: number,
  extent: number,
  sign: 1 | -1,
): number {
  return clampSize(key, start + (sign * deltaPx) / Math.max(1, extent));
}

/** One arrow key press: `dir` is the direction the boundary moves (-1 up or left, +1 down or right). */
export function stepSize(
  key: RegionKey,
  current: number,
  dir: -1 | 1,
  big: boolean,
  sign: 1 | -1,
): number {
  return clampSize(key, current + sign * dir * (big ? KEY_STEP_BIG : KEY_STEP));
}

/** Percent for `aria-valuenow` and friends. */
export function percent(v: number): number {
  return Math.round(v * 100);
}

/** `12.5%` style CSS length for a fraction. */
export function pctCss(v: number): string {
  return `${+(v * 100).toFixed(2)}%`;
}
