import { describe, expect, test } from 'vitest';
import { makeRng } from '../src/rng.js';
import { generate } from '../src/data/sets.js';
import { SETS, sample, truth, type DataConfig } from '../src/index.js';
import {
  DELIVERY_DOMAIN,
  DELIVERY_ZONES,
  TIME_TRUE,
  generateDelivery,
} from '../src/data/delivery.js';

type Name = 'delivery_late' | 'delivery_time';
const cfg = (set: Name, over: Partial<DataConfig> = {}): DataConfig => ({
  set,
  n: 200,
  noise: 0.3,
  val: 0.25,
  seed: 3,
  standardize: false,
  ...over,
});
const truthAt = (set: Name, x: number[]): number => truth(set, Float32Array.from(x))[0] ?? NaN;

describe('delivery story', () => {
  test('specs: columns, kinds, targets', () => {
    expect(SETS.delivery_late.kind).toBe('binary');
    expect(SETS.delivery_time.kind).toBe('regression');
    expect(SETS.delivery_late.d).toBe(3);
    const ds = sample(cfg('delivery_late'));
    expect(ds.info.featureNames).toEqual(['distance (km)', 'rain (mm/h)', 'queue (orders)']);
    expect(ds.info.targetName).toBe('late');
    expect(ds.nTrain + ds.nVal).toBe(200);
    const small = sample(cfg('delivery_time', { n: 50 }));
    expect(small.nTrain + small.nVal).toBe(50);
  });

  test('same seed gives the same rows, different seeds different rows', () => {
    const a = sample(cfg('delivery_late'));
    const b = sample(cfg('delivery_late'));
    const c = sample(cfg('delivery_late', { seed: 4 }));
    expect(Array.from(b.xTrain)).toEqual(Array.from(a.xTrain));
    expect(Array.from(b.yTrain)).toEqual(Array.from(a.yTrain));
    expect(Array.from(c.xTrain)).not.toEqual(Array.from(a.xTrain));
  });

  test('both targets share the feature rows for a seed and n', () => {
    const a = sample(cfg('delivery_late'));
    const b = sample(cfg('delivery_time'));
    expect(Array.from(b.xTrain)).toEqual(Array.from(a.xTrain));
  });

  test('inputs stay in the domain', () => {
    const rows = generateDelivery('delivery_time', 2000, 0.3, makeRng(1));
    for (const r of rows) {
      r.x.forEach((v, c) => {
        const [lo, hi] = DELIVERY_DOMAIN[c] ?? [0, 0];
        expect(v).toBeGreaterThanOrEqual(lo);
        expect(v).toBeLessThanOrEqual(hi);
      });
      expect(r.zone).toBeGreaterThanOrEqual(0);
      expect(r.zone).toBeLessThan(DELIVERY_ZONES.length);
    }
  });

  test('zone shifts the outcome by its documented effect', () => {
    const rows = generateDelivery('delivery_time', 4000, 0, makeRng(5));
    TIME_TRUE.zone.forEach((e, z) => {
      const r = rows.filter((q) => q.zone === z);
      expect(r.length).toBeGreaterThan(500);
      const m = r.reduce((s, q) => s + (q.y - truthAt('delivery_time', q.x)), 0) / r.length;
      expect(m).toBeCloseTo(e, 4);
    });
  });

  test('late rate near the truth at large n', () => {
    const rows = generate('delivery_late', 40000, 0, makeRng(11));
    const p = rows.reduce((s, r) => s + r.y, 0) / rows.length;
    const pt = rows.reduce((s, r) => s + truthAt('delivery_late', r.x), 0) / rows.length;
    expect(Math.abs(p - pt)).toBeLessThan(0.01);
    expect(pt).toBeGreaterThan(0.3);
    expect(pt).toBeLessThan(0.5);
  });

  test('delivery time: residual spread is zone plus noise', () => {
    const rows = generate('delivery_time', 20000, 0.3, makeRng(2));
    const res = rows.map((r) => r.y - truthAt('delivery_time', r.x));
    const m = res.reduce((s, v) => s + v, 0) / res.length;
    const v = res.reduce((s, q) => s + (q - m) ** 2, 0) / res.length;
    // zone variance (mean of 4, 1, 1, 4 = 2.5) plus noise variance (3^2 = 9)
    expect(Math.abs(m)).toBeLessThan(0.1);
    expect(v).toBeGreaterThan(10.5);
    expect(v).toBeLessThan(12.5);
  });
});
