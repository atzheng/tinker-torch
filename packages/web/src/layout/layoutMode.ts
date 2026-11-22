/**
 * Which arrangement of the five regions fits the window (CONTRACTS.md section 10). Pure: the App
 * reads the window size and calls this on every resize, so rotating a tablet or resizing a window
 * switches the layout live. The layout is derived from the size, never stored.
 */
export type LayoutMode = 'landscape' | 'portrait' | 'notice';

/** Below this width or height (px) neither layout is usable: only the wider-screen notice shows. */
export const MIN_SIDE = 600;
/** The side-by-side layout (palette | canvas | results) needs at least this width (px). */
export const LANDSCAPE_MIN_WIDTH = 1024;

/** The stacked layout needs this height (px): below it the canvas would be under about 280 px tall. */
export const PORTRAIT_MIN_HEIGHT = 700;

export function layoutFor(width: number, height: number): LayoutMode {
  if (width < MIN_SIDE || height < MIN_SIDE) return 'notice';
  if (width >= LANDSCAPE_MIN_WIDTH && width >= height) return 'landscape';
  return height < PORTRAIT_MIN_HEIGHT ? 'notice' : 'portrait';
}
