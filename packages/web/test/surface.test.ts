import { expect, test } from 'vitest';
import { grid2d } from '@tinker/core';
import {
  column,
  contourSegments,
  fillGridImage,
  finiteRange,
  indexToDomain,
  makeRamp,
  parseColor,
  rampCss,
  unit,
} from '../src/plot/surface.js';

test('parseColor reads hex and rgb()', () => {
  expect(parseColor('#1f5fd1')).toEqual([0x1f, 0x5f, 0xd1]);
  expect(parseColor('#fff')).toEqual([255, 255, 255]);
  expect(parseColor('rgb(10, 20, 30)')).toEqual([10, 20, 30]);
  expect(parseColor('nonsense')).toEqual([128, 128, 128]);
});

test('ramp hits its stops and clamps t', () => {
  const lut = makeRamp(['#000000', '#ff0000', '#ffffff']);
  expect(rampCss(lut, 0)).toBe('rgb(0,0,0)');
  expect(rampCss(lut, 1)).toBe('rgb(255,255,255)');
  expect(rampCss(lut, -5)).toBe('rgb(0,0,0)');
  expect(rampCss(lut, 9)).toBe('rgb(255,255,255)');
  expect(rampCss(lut, 0.5)).toMatch(/^rgb\(255,[01],[01]\)$/); // 8-bit quantisation
  expect(unit(5, 0, 10)).toBe(0.5);
  expect(Number.isNaN(unit(Number.NaN, 0, 1))).toBe(true);
});

test('grid image: x1 increases left to right, x2 bottom to top (grid2d order iy*m+ix)', () => {
  const m = 4;
  const g = grid2d(
    [
      [0, 3],
      [0, 3],
    ],
    m,
  );
  // value = x1 on the grid: red channel should rise to the right, constant down a column
  const v1 = new Float32Array(m * m);
  const v2 = new Float32Array(m * m);
  for (let i = 0; i < m * m; i++) {
    v1[i] = g.xRaw[i * 2] ?? 0;
    v2[i] = g.xRaw[i * 2 + 1] ?? 0;
  }
  const lut = makeRamp(['#000000', '#ffffff']);
  const img = new Uint8ClampedArray(m * m * 4);
  const px = (x: number, y: number) => img[(y * m + x) * 4] ?? -1;
  fillGridImage(img, v1, m, lut, 0, 3);
  expect(px(0, 0)).toBe(0);
  expect(px(3, 0)).toBe(255);
  expect(px(0, 3)).toBe(0);
  expect(px(1, 2)).toBeLessThan(px(2, 2));
  fillGridImage(img, v2, m, lut, 0, 3);
  expect(px(0, 3)).toBe(0); // bottom row is the smallest x2
  expect(px(0, 0)).toBe(255); // top row is the largest x2
  expect(px(2, 0)).toBe(px(0, 0));
});

test('non-finite grid values stay transparent', () => {
  const img = new Uint8ClampedArray(4 * 4).fill(7);
  fillGridImage(img, [Number.NaN, 1, 1, 1], 2, makeRamp(['#000', '#fff']), 0, 1);
  expect(img[(1 * 2 + 0) * 4 + 3]).toBe(0); // grid (0,0) is the bottom-left pixel
});

test('contour of a linear ramp is a straight vertical line at the level', () => {
  const m = 5;
  const v = new Float32Array(m * m);
  for (let iy = 0; iy < m; iy++) for (let ix = 0; ix < m; ix++) v[iy * m + ix] = ix / (m - 1);
  const s = contourSegments(v, m, 0.5);
  expect(s.length).toBe((m - 1) * 4);
  for (let i = 0; i < s.length; i += 2) expect(s[i]).toBeCloseTo(2, 6);
});

test('contour of a step has no segments away from the step and handles saddles', () => {
  expect(contourSegments([0, 0, 0, 0], 2, 0.5)).toEqual([]);
  expect(contourSegments([1, 1, 1, 1], 2, 0.5)).toEqual([]);
  // saddle: bl and tr high -> two segments
  expect(contourSegments([1, 0, 0, 1], 2, 0.5).length).toBe(8);
  expect(contourSegments([Number.NaN, 0, 0, 1], 2, 0.5)).toEqual([]);
});

test('indexToDomain spans the ends; column and finiteRange', () => {
  expect(indexToDomain(0, 64, -2, 2)).toBe(-2);
  expect(indexToDomain(63, 64, -2, 2)).toBe(2);
  const out = new Float32Array(3);
  expect(Array.from(column(out, new Float32Array([1, 2, 3, 4, 5, 6]), 3, 1))).toEqual([2, 4, 6]);
  expect(Array.from(column(out, new Float32Array([1, 2, 3]), 3, 1))).toEqual([1, 2, 3]);
  expect(finiteRange([1, Number.NaN, 3], [-2])).toEqual([-2, 3]);
  expect(finiteRange([Number.NaN])).toBeNull();
});
