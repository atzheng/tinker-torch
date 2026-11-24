import { expect, test } from 'vitest';
import { placeIssues } from '../src/canvas/pure.js';

test('placeIssues: a global line about a piece on another tab names that tab', () => {
  const placed = placeIssues(
    [
      { at: { node: 'sub' }, message: 'needs two inputs' },
      { at: {}, message: 'no data' },
    ],
    new Set(),
    new Set(['lin']),
    undefined,
    undefined,
    (i) => (i.at.node === 'sub' ? 'Loss' : undefined),
  );
  expect(placed.global).toEqual(['Loss: needs two inputs', 'no data']);
});
