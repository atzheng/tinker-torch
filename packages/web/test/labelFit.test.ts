import { describe, expect, it } from 'vitest';
import { fitCategoryLabels, truncateLabel } from '../src/plot/labelFit.js';
import { shouldSnapshotFit } from '../src/results/pinsModel.js';

const measure = (s: string) => s.length * 6;

describe('truncateLabel', () => {
  it('keeps a label that fits', () => expect(truncateLabel('abc', 30, measure)).toBe('abc'));
  it('cuts with an ellipsis within the width', () => {
    const t = truncateLabel('abcdefghij', 30, measure);
    expect(t).toBe('abcd…');
    expect(measure(t)).toBeLessThanOrEqual(30);
  });
  it('returns nothing when not even an ellipsis fits', () =>
    expect(truncateLabel('abc', 3, measure)).toBe(''));
});

describe('fitCategoryLabels', () => {
  it('leaves short names alone', () =>
    expect(fitCategoryLabels(['a', 'bb'], 40, measure)).toEqual(['a', 'bb']));
  it('truncates long names to the slot', () => {
    const out = fitCategoryLabels(['alpha-beta-gamma', 'x'], 50, measure);
    expect(measure(out[0] ?? '')).toBeLessThanOrEqual(44);
    expect(out[0]?.endsWith('…')).toBe(true);
    expect(out[1]).toBe('x');
  });
  it('thins every k-th label when slots are narrow, and shown ones fit', () => {
    const names = Array.from({ length: 12 }, (_, i) => `category-number-${i}`);
    const out = fitCategoryLabels(names, 20, measure);
    expect(out.filter((s) => s !== null).length).toBeLessThan(12);
    expect(out[0]).not.toBeNull();
    expect(out[1]).toBeNull();
    const idx = out.flatMap((s, i) => (s === null ? [] : [i]));
    const k = (idx[1] ?? 0) - (idx[0] ?? 0);
    expect(k).toBeGreaterThan(1);
    for (const i of idx) expect(measure(out[i] ?? '')).toBeLessThanOrEqual(k * 20 - 6);
  });
  it('handles an empty list', () => expect(fitCategoryLabels([], 20, measure)).toEqual([]));
});

describe('shouldSnapshotFit', () => {
  it('is true for one numeric column', () =>
    expect(shouldSnapshotFit({ columns: ['x'], columnKinds: ['numeric'] })).toBe(true));
  it('is false for one categorical column', () =>
    expect(
      shouldSnapshotFit({ columns: ['c'], columnKinds: ['categorical'], categories: { c: ['a'] } }),
    ).toBe(false));
  it('is false for two columns', () =>
    expect(shouldSnapshotFit({ columns: ['a', 'b'] })).toBe(false));
});
