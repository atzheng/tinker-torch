import { expect, test } from 'vitest';
import { parseFile } from '@tinker/core';
import mlp from '../src/presets/mlp.json';
import {
  cellText,
  gridOf,
  rowNames,
  statsOf,
  tintPct,
  windowOf,
} from '../src/canvas/weightsPure.js';

test('statsOf: min, max, mean, population std, ignores non-finite', () => {
  const s = statsOf([1, 3, NaN, 5]);
  expect(s).toMatchObject({ min: 1, max: 5, mean: 3, count: 3 });
  expect(s?.std).toBeCloseTo(Math.sqrt(8 / 3), 10);
  expect(statsOf([])).toBeNull();
});

test('windowOf pages by 32 and clamps', () => {
  expect(windowOf(10, 0)).toEqual({ from: 0, to: 10, pages: 1 });
  expect(windowOf(64, 1)).toEqual({ from: 32, to: 64, pages: 2 });
  expect(windowOf(70, 9)).toEqual({ from: 64, to: 70, pages: 3 });
});

test('tintPct and cellText', () => {
  expect(tintPct(0, 1)).toBe(4);
  expect(tintPct(-2, 1)).toBe(50);
  expect(tintPct(1, 0)).toBe(0);
  expect(cellText(-0.5)).toBe('−0.50');
  expect(cellText(12.34)).toBe('12.3');
});

test('gridOf refuses shapes that do not fit; rows are named after the Input columns', () => {
  const r = parseFile(mlp);
  if (!r.ok) throw new Error(r.error);
  const g = r.file.graph;
  const lin = g.nodes.find((n) => n.type === 'linear');
  if (!lin || lin.type !== 'linear') throw new Error('no linear');
  const host = { [lin.id]: { W: new Float32Array(2 * lin.out), b: new Float32Array(lin.out) } };
  expect(gridOf(host, lin, 2, undefined)?.rows).toBe(2);
  expect(gridOf(host, lin, 3, undefined)).toBeNull();
  expect(gridOf(host, lin, 2, [lin.out, 2])).toBeNull();
  expect(gridOf(undefined, lin, 2, undefined)).toBeNull();
  expect(rowNames(g, lin.id, 1, ['age', 'bp'])).toHaveLength(1);
});
