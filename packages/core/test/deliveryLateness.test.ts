import { describe, expect, test } from 'vitest';
import { makeRng } from '../src/rng.js';
import { generate } from '../src/data/sets.js';
import { SETS, sample, truth, truthProbs, type DataConfig } from '../src/index.js';
import {
  LATENESS_CLASSES,
  LATENESS_CUTS,
  generateDelivery,
  latenessProbs,
} from '../src/data/delivery.js';

const cfg = (set: DataConfig['set'], over: Partial<DataConfig> = {}): DataConfig => ({
  set,
  n: 200,
  noise: 0.3,
  val: 0.25,
  seed: 3,
  standardize: false,
  ...over,
});

describe('delivery_lateness', () => {
  test('spec: 3 classes, same columns, noise applies', () => {
    expect(SETS.delivery_lateness.kind).toBe('multiclass');
    expect(SETS.delivery_lateness.numClasses).toBe(3);
    expect(SETS.delivery_lateness.noiseUnused).toBeUndefined();
    const ds = sample(cfg('delivery_lateness'));
    expect(ds.info.numClasses).toBe(3);
    expect(ds.info.targetName).toBe('lateness');
    expect(ds.info.classNames).toEqual([...LATENESS_CLASSES]);
    expect(ds.info.columnKinds).toEqual(['numeric', 'numeric', 'numeric', 'categorical']);
  });

  test('deterministic, and shares rows and noise with delivery_time', () => {
    const a = sample(cfg('delivery_lateness'));
    expect(Array.from(sample(cfg('delivery_lateness')).yTrain)).toEqual(Array.from(a.yTrain));
    expect(Array.from(sample(cfg('delivery_lateness', { seed: 4 })).yTrain)).not.toEqual(
      Array.from(a.yTrain),
    );
    const t = sample(cfg('delivery_time', { val: 0 }));
    const l = sample(cfg('delivery_lateness', { val: 0 }));
    expect(Array.from(l.xTrain)).toEqual(Array.from(t.xTrain));
    const [c0, c1] = LATENESS_CUTS;
    t.yTrain.forEach((m, i) => expect(l.yTrain[i]).toBe(m < c0 ? 0 : m < c1 ? 1 : 2));
  });

  test('the other delivery sets draw what they did before (zone/time unchanged)', () => {
    const t = generateDelivery('delivery_time', 3, 0, makeRng(3));
    const l = generateDelivery('delivery_late', 3, 0, makeRng(3));
    expect(t.map((r) => r.x)).toEqual(l.map((r) => r.x));
    const f = Float32Array.from(t.flatMap((r) => r.x));
    // at noise 0 the minutes are exactly the truth (the time draws are untouched)
    Array.from(truth('delivery_time', f)).forEach((v, k) => expect(t[k]?.y).toBeCloseTo(v, 4));
  });

  test('class proportions near 50 / 35 / 15 at noise 0.3; truth probabilities match', () => {
    const rows = generate('delivery_lateness', 40000, 0.3, makeRng(11));
    const freq = [0, 0, 0];
    const mean = [0, 0, 0];
    for (const r of rows) {
      freq[r.y] = (freq[r.y] ?? 0) + 1 / rows.length;
      latenessProbs(r.x, 0.3).forEach((p, c) => (mean[c] = (mean[c] ?? 0) + p / rows.length));
    }
    [0.5, 0.35, 0.15].forEach((w, c) => {
      expect(Math.abs((freq[c] ?? 0) - w)).toBeLessThan(0.02);
      expect(Math.abs((mean[c] ?? 0) - (freq[c] ?? 0))).toBeLessThan(0.01);
    });
  });

  test('truthProbs: rows sum to 1; one-hot at noise 0; soft otherwise', () => {
    const x = Float32Array.from([5, 4, 6, 0, 1, 0, 0, 2, 9, 9, 12, 3]);
    const t = truth('delivery_lateness', x);
    for (const nz of [0, 0.3, 1]) {
      const p = truthProbs('delivery_lateness', x, nz);
      for (let i = 0; i < 3; i++) {
        const r = [0, 1, 2].map((c) => p[i * 3 + c] ?? NaN);
        expect(r.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 5);
        if (nz === 0) expect(r[t[i] ?? 0]).toBe(1);
      }
    }
    const soft = truthProbs('delivery_lateness', Float32Array.from([5, 4, 6, 0]), 0.3);
    expect(Math.max(...soft)).toBeLessThan(1);
  });
});
