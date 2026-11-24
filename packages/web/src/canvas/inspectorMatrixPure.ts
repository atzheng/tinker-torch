/** Pure helpers for the inspector's inline weight matrix (tinker-nj1). Unit tested in test/inspectorMatrixPure.test.ts. */

/**
 * Largest Linear whose matrix is drawn inside the piece inspector: at most 8 inputs (rows) and 4
 * output units (columns), 32 weights. The inspector is 248 px wide, which fits a 56 px label column
 * and four 36 px cells; more rows only make it taller (they stay readable), more columns would not
 * fit without scrolling. Bigger layers use the "Inspect weights" panel (tinker-fn5).
 */
export const INLINE_MAX_ROWS = 8;
export const INLINE_MAX_COLS = 4;

/** True when a `rows x cols` weight matrix is small enough to show inline. */
export function inlineFits(rows: number, cols: number): boolean {
  return (
    Number.isInteger(rows) &&
    Number.isInteger(cols) &&
    rows >= 1 &&
    cols >= 1 &&
    rows <= INLINE_MAX_ROWS &&
    cols <= INLINE_MAX_COLS
  );
}

/** Short column heading of output unit `j`. */
export const unitLabel = (j: number): string => `u${j}`;

/** The shape line of a Linear too large to draw: `8 inputs to 16 units`. */
export function shapeText(rows: number, cols: number): string {
  return `${rows} input${rows === 1 ? '' : 's'} to ${cols} unit${cols === 1 ? '' : 's'}`;
}
