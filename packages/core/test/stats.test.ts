import { describe, expect, test } from 'vitest';
import { biasVariance, makeRng } from '../src/index.js';

describe('biasVariance', () => {
  test('hand-computed case', () => {
    // truth 0 everywhere; copies 1 and 3 at x0, 2 and 2 at x1
    // mean = [2, 2]; bias2 = 4; variance = (1 + 0) / 2 = 0.5
    const r = biasVariance([Float32Array.of(1, 2), Float32Array.of(3, 2)], Float32Array.of(0, 0));
    expect(Array.from(r.mean)).toEqual([2, 2]);
    expect(r.bias2).toBeCloseTo(4, 6);
    expect(r.variance).toBeCloseTo(0.5, 6);
    expect(r.k).toBe(2);
  });

  test('bias2 + variance equals the mean squared error to the truth', () => {
    const rng = makeRng(3);
    const m = 50;
    const truth = Float32Array.from({ length: m }, (_, j) => Math.sin(j / 7));
    const curves = Array.from({ length: 20 }, () =>
      Float32Array.from({ length: m }, (_, j) => Math.cos(j / 5) + rng.normal() * 0.3),
    );
    const r = biasVariance(curves, truth);
    let mse = 0;
    for (const c of curves) for (let j = 0; j < m; j++) mse += ((c[j] ?? 0) - (truth[j] ?? 0)) ** 2;
    mse /= curves.length * m;
    expect(r.bias2 + r.variance).toBeCloseTo(mse, 5);
    expect(r.mse).toBeCloseTo(mse, 5);
  });

  test('identical copies have no variance; copies centred on the truth have no bias', () => {
    const truth = Float32Array.of(0, 1, 2);
    const same = biasVariance([Float32Array.of(1, 1, 1), Float32Array.of(1, 1, 1)], truth);
    expect(same.variance).toBe(0);
    expect(same.bias2).toBeCloseTo(2 / 3, 6);
    const centred = biasVariance([Float32Array.of(-1, 0, 1), Float32Array.of(1, 2, 3)], truth);
    expect(centred.bias2).toBeCloseTo(0, 6);
    expect(centred.variance).toBeCloseTo(1, 6);
  });

  test('skips copies with non-finite values and returns NaN with none left', () => {
    const truth = Float32Array.of(0, 0);
    const r = biasVariance([Float32Array.of(1, 1), Float32Array.of(Number.NaN, 0)], truth);
    expect(r.k).toBe(1);
    expect(r.bias2).toBeCloseTo(1, 6);
    expect(biasVariance([], truth).bias2).toBeNaN();
  });
});
