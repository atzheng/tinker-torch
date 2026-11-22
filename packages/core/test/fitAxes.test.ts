import { expect, test } from 'vitest';
import {
  axisCategoryCount,
  categoryJitter,
  categorySeries,
  continuousCols,
  fitView,
  grid1d,
  grid1dSlice,
  gridCategories,
} from '../src/index.js';

const num = { columns: ['x1'], columnKinds: ['numeric' as const] };
const cat1 = {
  columns: ['x1'],
  columnKinds: ['categorical' as const],
  categories: { x1: ['a', 'b', 'c'] },
};
const cat2 = {
  columns: ['x1', 'x2'],
  columnKinds: ['categorical' as const, 'categorical' as const],
  categories: { x1: ['a', 'b'], x2: ['u', 'v', 'w'] },
};
const mixed = {
  columns: ['x1', 'x2'],
  columnKinds: ['categorical' as const, 'numeric' as const],
  categories: { x1: ['a', 'b'] },
};

test('fitView picks the plot from the column kinds', () => {
  expect(fitView(num)).toEqual({ type: 'curve', col: 0 });
  expect(fitView({ columns: ['x1', 'x2'] })).toEqual({ type: 'surface' });
  expect(fitView(cat1)).toEqual({ type: 'categories', col: 0 });
  expect(fitView(cat2)).toEqual({ type: 'categories', col: 0 });
  expect(fitView(mixed)).toEqual({ type: 'curve', col: 1 });
  expect(continuousCols(mixed)).toEqual([1]);
});

test('grid1dSlice with d = 1 equals grid1d', () => {
  const a = grid1d([[0, 4]], 5, null);
  const b = grid1dSlice([[0, 4]], [0], 0, 5, null);
  expect(Array.from(b.xRaw)).toEqual(Array.from(a.xRaw));
});

test('gridCategories has one row per category and holds the other columns', () => {
  const st = { mean: [0, 10], std: [1, 2] };
  const g = gridCategories([0, 14], 0, 3, st);
  expect(g.n).toBe(3);
  expect(Array.from(g.xRaw)).toEqual([0, 14, 1, 14, 2, 14]);
  expect(Array.from(g.xStd)).toEqual([0, 2, 1, 2, 2, 2]);
});

test('categorySeries: regression, binary and multi-class values', () => {
  const reg = categorySeries(Float32Array.from([1, 2, 3]), 3, 'regression', 1);
  expect(Array.from(reg?.[0] ?? [])).toEqual([1, 2, 3]);
  const bin = categorySeries(Float32Array.from([0.9, 0.1, 0.2, 0.8]), 2, 'binary', 2);
  expect(Array.from(bin?.[0] ?? [])).toEqual([Math.fround(0.1), Math.fround(0.8)]);
  const mc = categorySeries(Float32Array.from([1, 2, 3, 4, 5, 6]), 2, 'multiclass', 3);
  expect(mc?.map((y) => Array.from(y))).toEqual([
    [1, 4],
    [2, 5],
    [3, 6],
  ]);
  expect(categorySeries(Float32Array.from([1, 2]), 3, 'multiclass', 3)).toBeNull();
});

test('axisCategoryCount and jitter bounds', () => {
  expect(axisCategoryCount(cat1, 0, 1)).toBe(3);
  expect(axisCategoryCount({ columns: ['x1'], columnKinds: ['categorical'] }, 0, 4)).toBe(5);
  for (let i = 0; i < 500; i++) expect(Math.abs(categoryJitter(i))).toBeLessThanOrEqual(0.3);
  expect(categoryJitter(7)).toBe(categoryJitter(7));
});
