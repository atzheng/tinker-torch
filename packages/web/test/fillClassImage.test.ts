// @vitest-environment jsdom
import { expect, test } from 'vitest';
import { readColors } from '../src/plot/color.js';
import { fillClassImage } from '../src/results/FitPlot2D.js';

test('fillClassImage paints the argmax class colour, flipped so x2 grows upward', () => {
  const c = readColors(document.body); // jsdom has no theme: the fallbacks
  // 2 x 2 grid, 3 classes; grid row 0 (bottom) is class 0 then class 2, row 1 (top) class 1 then NaN
  const probs = Float32Array.from([
    0.8,
    0.1,
    0.1,
    0.1,
    0.1,
    0.8,
    0.1,
    0.8,
    0.1,
    Number.NaN,
    Number.NaN,
    Number.NaN,
  ]);
  const rgba = new Uint8ClampedArray(2 * 2 * 4);
  fillClassImage(rgba, probs, 2, 3, c);
  const px = (x: number, y: number) => Array.from(rgba.slice((y * 2 + x) * 4, (y * 2 + x) * 4 + 4));
  const blue = [0, 114, 178]; // fallback --class-0
  expect(px(0, 1).slice(0, 3)).toEqual(blue); // bottom-left: class 0
  expect(px(1, 1).slice(0, 3)).toEqual([0, 158, 115]); // bottom-right: class 2
  expect(px(0, 0).slice(0, 3)).toEqual([230, 159, 0]); // top-left: class 1
  expect(px(1, 0)[3]).toBe(0); // non-finite: transparent
});
