import { describe, expect, test } from 'vitest';
import { makeRng } from '../src/rng.js';
import { generate } from '../src/data/sets.js';
import { SETS, sample, truth, type DataConfig } from '../src/index.js';
import {
  DELIVERY_DOMAIN,
  DELIVERY_LATE_RATE,
  DELIVERY_ZONES,
  LATE_TRUE,
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
    expect(SETS.delivery_late.d).toBe(4);
    const ds = sample(cfg('delivery_late'));
    expect(ds.info.featureNames).toEqual([
      'distance (km)',
      'rain (mm/h)',
      'queue (orders)',
      'zone',
    ]);
    expect(ds.info.columnKinds).toEqual(['numeric', 'numeric', 'numeric', 'categorical']);
    expect(ds.info.categories).toEqual({ x4: ['centre', 'north', 'east', 'south'] });
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

  test('the first three columns do not depend on the zone column (its own random stream)', () => {
    // the same seed and n gave these feature rows before zone became a column
    const a = generateDelivery('delivery_late', 5, 0.3, makeRng(3));
    expect(a.map((r) => r.x.slice(0, 3))).toEqual(
      generateDelivery('delivery_time', 5, 0.9, makeRng(3)).map((r) => r.x.slice(0, 3)),
    );
  });

  test('truth uses the row zone', () => {
    const lo = truthAt('delivery_late', [5, 4, 6, 0]);
    const hi = truthAt('delivery_late', [5, 4, 6, 3]);
    expect(hi).toBeGreaterThan(lo);
    const t = (z: number) => truthAt('delivery_time', [5, 4, 6, z]);
    TIME_TRUE.zone.forEach((e, z) =>
      expect(t(z) - t(0)).toBeCloseTo(e - (TIME_TRUE.zone[0] ?? 0), 5),
    );
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
      expect(Number.isInteger(r.x[3])).toBe(true);
      expect(r.x[3]).toBeLessThan(DELIVERY_ZONES.length);
    }
  });

  test('zone shifts the outcome by its documented effect', () => {
    const rows = generateDelivery('delivery_time', 4000, 0, makeRng(5));
    TIME_TRUE.zone.forEach((e, z) => {
      const r = rows.filter((q) => q.x[3] === z);
      expect(r.length).toBeGreaterThan(500);
      // with no noise the minutes are the truth at the row's own zone, and the zone moves them by e
      const m = r.reduce((s, q) => s + (q.y - truthAt('delivery_time', q.x)), 0) / r.length;
      expect(m).toBeCloseTo(0, 4);
      const lin = (q: { x: number[] }) =>
        12 + 3 * (q.x[0] ?? 0) + 0.8 * (q.x[1] ?? 0) + (q.x[2] ?? 0);
      const base = r.reduce((s, q) => s + (q.y - lin(q)), 0);
      expect(base / r.length).toBeCloseTo(e, 4);
    });
  });

  test('late rate: the constant matches the integral and the sample at large n', () => {
    // Midpoint rule over distance and rain, every queue value and zone.
    const M = 100;
    let sum = 0;
    let cnt = 0;
    for (let i = 0; i < M; i++)
      for (let j = 0; j < M; j++)
        for (let q = 0; q <= 12; q++)
          for (const ze of LATE_TRUE.zone) {
            const z =
              LATE_TRUE.intercept +
              LATE_TRUE.distance * (0.5 + ((i + 0.5) / M) * 9.5) +
              LATE_TRUE.rain * (((j + 0.5) / M) * 10) +
              LATE_TRUE.queue * q +
              ze;
            sum += 1 / (1 + Math.exp(-z));
            cnt++;
          }
    expect(Math.abs(sum / cnt - DELIVERY_LATE_RATE)).toBeLessThan(0.001);
    const rows = generate('delivery_late', 40000, 0, makeRng(11));
    const p = rows.reduce((s, r) => s + r.y, 0) / rows.length;
    expect(Math.abs(p - DELIVERY_LATE_RATE)).toBeLessThan(0.02);
    const pt = rows.reduce((s, r) => s + truthAt('delivery_late', r.x), 0) / rows.length; // E over rows, zone included
    expect(Math.abs(pt - DELIVERY_LATE_RATE)).toBeLessThan(0.01);
  });

  test.each(['delivery_late', 'delivery_time'] as const)('%s: generic invariants', (set) => {
    for (const val of [0, 0.3, 0.5]) {
      const ds = sample(cfg(set, { n: 100, val }));
      expect(ds.nVal).toBe(Math.round(100 * val));
      expect(ds.nTrain + ds.nVal).toBe(100);
      expect(ds.xTrain.length).toBe(ds.nTrain * 4);
      expect(ds.xVal.length).toBe(ds.nVal * 4);
      for (const a of [ds.xTrain, ds.yTrain, ds.xVal, ds.yVal]) {
        expect(Array.from(a).every(Number.isFinite)).toBe(true);
      }
      expect(ds.info.columns).toEqual(['x1', 'x2', 'x3', 'x4']);
    }
    const ds = sample(cfg(set, { n: 300, val: 0.2, standardize: true }));
    // Standardise skips the categorical zone column: its values stay the category indices
    for (let i = 0; i < ds.nTrain; i++) expect(Number.isInteger(ds.xTrain[i * 4 + 3])).toBe(true);
    expect(ds.standardization?.mean[3]).toBe(0);
    expect(ds.standardization?.std[3]).toBe(1);
    for (let c = 0; c < 3; c++) {
      const col = Array.from({ length: ds.nTrain }, (_, i) => ds.xTrain[i * 4 + c] ?? 0);
      const m = col.reduce((s, v) => s + v, 0) / col.length;
      const sd = Math.sqrt(col.reduce((s, v) => s + (v - m) ** 2, 0) / col.length);
      expect(Math.abs(m)).toBeLessThan(1e-5);
      expect(sd).toBeCloseTo(1, 4);
    }
    if (set === 'delivery_late') {
      expect(ds.info.kind).toBe('binary');
      expect(ds.info.classNames).toEqual(['on time', 'late']);
      expect(Array.from(ds.yTrain).every((v) => v === 0 || v === 1)).toBe(true);
    } else {
      expect(ds.info.kind).toBe('regression');
      expect(ds.info.classNames).toBeUndefined();
    }
  });

  test('delivery time: residual spread is the noise alone', () => {
    const rows = generate('delivery_time', 20000, 0.3, makeRng(2));
    const res = rows.map((r) => r.y - truthAt('delivery_time', r.x));
    const m = res.reduce((s, v) => s + v, 0) / res.length;
    const v = res.reduce((s, q) => s + (q - m) ** 2, 0) / res.length;
    // the truth now includes the zone effect, so the residual is the noise alone (3^2 = 9)
    expect(Math.abs(m)).toBeLessThan(0.1);
    expect(v).toBeGreaterThan(8.3);
    expect(v).toBeLessThan(9.7);
  });
});
