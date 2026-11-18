import { expect, test } from 'vitest';
import {
  MIN_HALF,
  appendPath,
  axisLabels,
  contourLevels,
  contourSegments,
  heat,
  linspace,
  lossScale,
  nextRange,
  twoParams,
  type Pt,
} from '../src/plot/surface.js';

test('twoParams: exactly two parameters across nodes, else null', () => {
  const one = { l: { W: Float32Array.of(2), b: Float32Array.of(-1) } };
  const r = twoParams(one, ['l']);
  expect(r?.values).toEqual([2, -1]);
  expect(r?.axes.map((a) => [a.tensor, a.index])).toEqual([
    ['W', 0],
    ['b', 0],
  ]);
  expect(twoParams({ l: { W: Float32Array.of(1, 2, 3) } }, ['l'])).toBeNull();
  expect(twoParams({ l: { W: Float32Array.of(1) } }, ['l'])).toBeNull();
  const two = { a: { W: Float32Array.of(1) }, b: { W: Float32Array.of(4) } };
  const t = twoParams(two, ['a', 'b']);
  expect(t?.values).toEqual([1, 4]);
  if (t) expect(axisLabels(t.axes)).toEqual(['a weight', 'b weight']);
});

test('nextRange keeps the range while the path stays inside, then re-centres on the current point', () => {
  const r0 = nextRange(null, [[1, 2]]);
  expect(r0).toEqual({ a: [1 - MIN_HALF, 1 + MIN_HALF], b: [2 - MIN_HALF, 2 + MIN_HALF] });
  expect(
    nextRange(r0, [
      [1, 2],
      [2, 3],
    ]),
  ).toBe(r0);
  const r1 = nextRange(r0, [
    [1, 2],
    [10, 2],
  ]);
  expect(r1?.a).toEqual([10 - 11.25, 10 + 11.25]);
  expect(r1?.b).toEqual([2 - MIN_HALF, 2 + MIN_HALF]);
  expect(nextRange(null, [])).toBeNull();
});

test('appendPath halves past the cap and keeps the first and last points', () => {
  let p: Pt[] = [];
  for (let i = 0; i < 11; i++) p = appendPath(p, [i, i], 10);
  expect(p.length).toBeLessThanOrEqual(10);
  expect(p[p.length - 1]).toEqual([10, 10]);
  expect(p[0]).toEqual([0, 0]);
});

test('lossScale, contourLevels and heat', () => {
  expect(lossScale(Float32Array.of(Number.NaN))).toBeNull();
  const g = Float32Array.from({ length: 100 }, (_, i) => i);
  const s = lossScale(g);
  expect(s).toEqual({ min: 0, hi: 90 });
  const lv = contourLevels(s ?? { min: 0, hi: 1 }, 4);
  expect(lv).toHaveLength(4);
  expect(lv.every((v, i) => v > 0 && v < 90 && (i === 0 || v > (lv[i - 1] ?? 0)))).toBe(true);
  expect(heat(0, { min: 0, hi: 4 })).toBe(0);
  expect(heat(100, { min: 0, hi: 4 })).toBe(1);
  expect(heat(1, { min: 0, hi: 4 })).toBeCloseTo(0.5);
  expect(Number.isNaN(heat(Number.NaN, { min: 0, hi: 4 }))).toBe(true);
  expect(linspace(-1, 1, 3)).toEqual(Float32Array.of(-1, 0, 1));
});

test('contourSegments: a bowl gives a ring at the right radius; flat and NaN cells give nothing', () => {
  const n = 21;
  const g = new Float32Array(n * n);
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++) g[i * n + j] = (i - 10) ** 2 + (j - 10) ** 2;
  const seg = contourSegments(g, n, n, 25);
  expect(seg.length).toBeGreaterThan(0);
  for (let k = 0; k < seg.length; k += 2) {
    const r = Math.hypot((seg[k] ?? 0) - 10, (seg[k + 1] ?? 0) - 10);
    expect(Math.abs(r - 5)).toBeLessThan(0.35);
  }
  expect(contourSegments(new Float32Array(n * n), n, n, 1)).toEqual([]);
  expect(contourSegments(new Float32Array(n * n).fill(Number.NaN), n, n, 1)).toEqual([]);
  expect(contourSegments(Float32Array.of(2, 0, 0, 2), 2, 2, 1)).toHaveLength(8);
});
