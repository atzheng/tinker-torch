import { expect, test } from 'vitest';
import {
  INLINE_MAX_COLS,
  INLINE_MAX_ROWS,
  inlineFits,
  shapeText,
  unitLabel,
} from '../src/canvas/inspectorMatrixPure.js';

test('inlineFits: the limit is 8 rows by 4 columns', () => {
  expect(inlineFits(1, 1)).toBe(true);
  expect(inlineFits(INLINE_MAX_ROWS, INLINE_MAX_COLS)).toBe(true);
  expect(inlineFits(INLINE_MAX_ROWS + 1, 1)).toBe(false);
  expect(inlineFits(1, INLINE_MAX_COLS + 1)).toBe(false);
  expect(inlineFits(0, 3)).toBe(false);
  expect(inlineFits(2.5, 3)).toBe(false);
  expect(inlineFits(Number.NaN, 3)).toBe(false);
});

test('labels', () => {
  expect(unitLabel(3)).toBe('u3');
  expect(shapeText(8, 16)).toBe('8 inputs to 16 units');
  expect(shapeText(1, 1)).toBe('1 input to 1 unit');
});
