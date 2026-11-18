import { expect, test } from 'vitest';
import {
  decimatePath,
  formatTick,
  linearScale,
  logScale,
  logTicks,
  niceTicks,
  paddedRange,
} from '../src/plot/axes.js';
import { lossRange } from '../src/plot/loss.js';

test('niceTicks gives 1-2-5 steps inside the range', () => {
  expect(niceTicks(0, 10, 5)).toEqual([0, 2, 4, 6, 8, 10]);
  expect(niceTicks(-3, 3, 6)).toEqual([-3, -2, -1, 0, 1, 2, 3]);
  expect(niceTicks(0, 0.3, 3)).toEqual([0, 0.1, 0.2, 0.3]);
  expect(niceTicks(2, 2)).toEqual([2]);
  expect(niceTicks(Number.NaN, 1)).toEqual([]);
});

test('logTicks: decades, with 2 and 5 on narrow ranges', () => {
  expect(logTicks(0.01, 100)).toEqual([0.01, 0.1, 1, 10, 100]);
  expect(logTicks(0.5, 8)).toEqual([0.5, 1, 2, 5]);
  expect(logTicks(0, 1)).toEqual([]);
});

test('scales map ends and midpoints, y can be flipped', () => {
  const s = linearScale(0, 10, 100, 200);
  expect([s.px(0), s.px(5), s.px(10)]).toEqual([100, 150, 200]);
  const y = linearScale(-1, 1, 300, 0);
  expect([y.px(-1), y.px(1)]).toEqual([300, 0]);
  const l = logScale(0.01, 100, 0, 400);
  expect(l.px(1)).toBeCloseTo(200);
  expect(l.px(0)).toBe(0); // non-positive values sit at the start
});

test('formatTick and paddedRange', () => {
  expect(formatTick(0)).toBe('0');
  expect(formatTick(0.5)).toBe('0.5');
  expect(formatTick(1e-5)).toBe('1e-5');
  expect(formatTick(3e6)).toBe('3e6');
  expect(formatTick(20000)).toBe('20000');
  expect(paddedRange(0, 10, 0.1)).toEqual([-1, 11]);
  expect(paddedRange(3, 3)).toEqual([2, 4]);
});

test('decimatePath keeps peaks and bounds the number of points by the pixel width', () => {
  const n = 10000;
  const ys = Array.from({ length: n }, (_, i) => Math.sin(i / 50) + (i === 5000 ? 5 : 0));
  const out: [number, number][] = [];
  decimatePath(
    n,
    (i) => i,
    (i) => ys[i] ?? 0,
    (v) => v / 10, // 1000 px
    (v) => -v * 10,
    (x, y) => out.push([x, y]),
  );
  expect(out.length).toBeLessThanOrEqual(4 * 1001);
  expect(out.length).toBeGreaterThan(1000);
  expect(Math.min(...out.map((p) => p[1]))).toBeLessThan(-44); // the spike (y = 4.5) survives
  for (let i = 1; i < out.length; i++) {
    expect(out[i]?.[0] ?? 0).toBeGreaterThanOrEqual((out[i - 1]?.[0] ?? 0) - 1);
  }
});

test('decimatePath skips non-finite values and handles empty input', () => {
  const out: number[] = [];
  decimatePath(
    0,
    () => 0,
    () => 0,
    (v) => v,
    (v) => v,
    (_x, y) => out.push(y),
  );
  expect(out).toEqual([]);
  const ys = [1, Number.NaN, 3];
  decimatePath(
    3,
    (i) => i * 10,
    (i) => ys[i] ?? 0,
    (v) => v,
    (v) => v,
    (_x, y) => out.push(y),
  );
  expect(out).toEqual([1, 3]);
});

test('lossRange: linear starts at zero, log ignores non-positive values, steps use `step`', () => {
  const h = [
    { step: 1, train: 4, val: null },
    { step: 40, train: 0.5, val: 2 },
    { step: 80, train: 0.25, val: 0 },
  ];
  expect(lossRange(h, false)).toEqual({ stepMax: 80, lo: 0, hi: 4 * 1.05 });
  const r = lossRange(h, true);
  expect(r.stepMax).toBe(80);
  expect(r.lo).toBeCloseTo(0.25 / 1.25);
  expect(r.hi).toBeCloseTo(5);
  expect(lossRange([], true).hi).toBeGreaterThan(lossRange([], true).lo);
  expect(lossRange(h, false, 9).hi).toBeCloseTo(9 * 1.05);
});
