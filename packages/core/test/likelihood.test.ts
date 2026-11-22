import { expect, test } from 'vitest';
import { effectiveLossMode, exampleStats } from '../src/index.js';

test('exampleStats: total is the sum, likelihood the float32 product, 0 after 150 coin flips', () => {
  const ln2 = Math.fround(Math.LN2);
  const a = exampleStats(new Float32Array(10).fill(ln2));
  expect(a.total).toBeCloseTo(10 * Math.LN2, 5);
  expect(a.likelihood).toBeCloseTo(0.5 ** 10, 6);
  expect(exampleStats(new Float32Array(149).fill(ln2)).likelihood).toBeGreaterThan(0);
  expect(exampleStats(new Float32Array(150).fill(ln2)).likelihood).toBe(0);
  expect(exampleStats(new Float32Array(0))).toEqual({ n: 0, total: 0, likelihood: 1 });
});

test('effectiveLossMode: single shown mode is fixed; default mean, else first shown', () => {
  const all = ['likelihood', 'total', 'mean'] as const;
  const some = (ids: string[]) => (id: string) => ids.includes(id);
  expect(effectiveLossMode(all, () => true, 'mean')).toBe('mean');
  expect(effectiveLossMode(all, () => true, 'total')).toBe('total');
  expect(effectiveLossMode(all, some(['loss-mode-total']), 'mean')).toBe('total');
  expect(effectiveLossMode(all, some(['loss-mode-likelihood', 'loss-mode-total']), 'mean')).toBe(
    'likelihood',
  );
  expect(effectiveLossMode(all, some([]), 'mean')).toBeNull();
  expect(effectiveLossMode(['total', 'mean'], () => true, 'likelihood')).toBe('mean');
});
