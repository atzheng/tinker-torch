import { describe, expect, test } from 'vitest';
import {
  DATASET_NAMES,
  SETS,
  batchSchedule,
  gatherBatch,
  grid1d,
  grid2d,
  sample,
  spread,
  stepsPerEpoch,
  truth,
  truthMargin,
  truthProbs,
  classCount,
  unstandardizeRows,
  type DataConfig,
  type DatasetName,
} from '../src/index.js';

const cfg = (set: DatasetName, over: Partial<DataConfig> = {}): DataConfig => ({
  set,
  n: 40,
  noise: 0.2,
  val: 0.3,
  seed: 7,
  standardize: false,
  ...over,
});

const SYNTH = DATASET_NAMES.filter((s) => !SETS[s].real);
// The delivery sets (d = 3, Bernoulli labels, own noise scale) have their own tests in delivery.test.ts.
const PLAIN = SYNTH.filter((s) => !s.startsWith('delivery_'));
const REG = PLAIN.filter((s) => SETS[s].kind === 'regression');
const CLS = PLAIN.filter((s) => SETS[s].kind === 'binary');
// delivery_lateness (d = 4, unbalanced) has its own tests in deliveryLateness.test.ts
const MULTI = SYNTH.filter((s) => SETS[s].kind === 'multiclass' && s !== 'delivery_lateness');

const mean = (a: ArrayLike<number>) => Array.from(a).reduce((s, v) => s + v, 0) / a.length;
const std = (a: ArrayLike<number>) => {
  const m = mean(a);
  return Math.sqrt(Array.from(a).reduce((s, v) => s + (v - m) ** 2, 0) / a.length);
};

describe('sample', () => {
  test.each(SYNTH)('%s is deterministic per seed and differs across seeds', (set) => {
    const a = sample(cfg(set));
    const b = sample(cfg(set));
    const c = sample(cfg(set, { seed: 8 }));
    expect(Array.from(a.xTrain)).toEqual(Array.from(b.xTrain));
    expect(Array.from(a.yTrain)).toEqual(Array.from(b.yTrain));
    expect(Array.from(a.xTrain)).not.toEqual(Array.from(c.xTrain));
  });

  test.each(SYNTH)('%s has the right sizes and shapes', (set) => {
    const d = SETS[set].d;
    for (const n of [10, 40, 500]) {
      for (const val of [0, 0.3, 0.5]) {
        const ds = sample(cfg(set, { n, val }));
        const nVal = Math.round(n * val);
        expect(ds.nVal).toBe(nVal);
        expect(ds.nTrain).toBe(n - nVal);
        expect(ds.xTrain.length).toBe(ds.nTrain * d);
        expect(ds.yTrain.length).toBe(ds.nTrain);
        expect(ds.xVal.length).toBe(nVal * d);
        expect(ds.yVal.length).toBe(nVal);
        expect(ds.info.columns).toEqual(Array.from({ length: d }, (_, c) => `x${c + 1}`));
      }
    }
  });

  test.each(REG)('%s: noise 0 gives y = f*(x) exactly', (set) => {
    const ds = sample(cfg(set, { noise: 0, n: 200 }));
    expect(Array.from(ds.yTrain)).toEqual(Array.from(truth(set, ds.xTrain)));
    expect(Array.from(ds.yVal)).toEqual(Array.from(truth(set, ds.xVal)));
  });

  test.each(REG)('%s: x lies in the domain and noise std is about noise * s', (set) => {
    const noise = 0.5;
    const ds = sample(cfg(set, { noise, n: 500, val: 0 }));
    const d = SETS[set].d;
    for (let i = 0; i < ds.xTrain.length; i++) {
      const [lo, hi] = ds.domain[i % d] ?? [0, 0];
      expect(ds.xTrain[i]).toBeGreaterThanOrEqual(lo);
      expect(ds.xTrain[i]).toBeLessThanOrEqual(hi);
    }
    const f = truth(set, ds.xTrain);
    const resid = Array.from(ds.yTrain, (y, i) => y - (f[i] ?? 0));
    const target = noise * spread(set);
    expect(std(resid)).toBeGreaterThan(0.85 * target);
    expect(std(resid)).toBeLessThan(1.15 * target);
  });

  test('spread constants are sensible', () => {
    expect(spread('line')).toBeCloseTo(0.8 * Math.sqrt(3), 2); // 0.8 * 6/sqrt(12)
    expect(spread('step')).toBeCloseTo(1, 2);
    expect(spread('plane')).toBeGreaterThan(0);
  });

  test.each(CLS)('%s: balanced; labels follow the rule at noise 0', (set) => {
    for (const val of [0, 0.3, 0.5]) {
      const ds = sample(cfg(set, { noise: 0, n: 100, val }));
      const ones = (y: Float32Array) => y.reduce((s, v) => s + v, 0);
      expect(Math.abs(ones(ds.yTrain) - ds.nTrain / 2)).toBeLessThanOrEqual(1);
      expect(Math.abs(ones(ds.yVal) - ds.nVal / 2)).toBeLessThanOrEqual(1);
      if (set === 'blobs') continue; // Gaussian overlap: the rule is probabilistic
      expect(Array.from(ds.yTrain)).toEqual(Array.from(truth(set, ds.xTrain)));
      expect(Array.from(ds.yVal)).toEqual(Array.from(truth(set, ds.xVal)));
    }
  });

  test('blobs: classes sit at (-1,-1) and (1,1); noise scales the spread', () => {
    const ds = sample(cfg('blobs', { noise: 0, n: 500, val: 0 }));
    const pick = (label: number, col: number) =>
      Array.from({ length: ds.nTrain }, (_, i) => i)
        .filter((i) => ds.yTrain[i] === label)
        .map((i) => ds.xTrain[i * 2 + col] ?? 0);
    expect(mean(pick(1, 0))).toBeGreaterThan(0.8);
    expect(mean(pick(0, 1))).toBeLessThan(-0.8);
    expect(std(pick(1, 0))).toBeCloseTo(0.3, 1);
    const noisy = sample(cfg('blobs', { noise: 1, n: 500, val: 0 }));
    const xs = Array.from({ length: noisy.nTrain }, (_, i) => noisy.xTrain[i * 2] ?? 0);
    expect(std(xs)).toBeGreaterThan(
      std(Array.from({ length: ds.nTrain }, (_, i) => ds.xTrain[i * 2] ?? 0)),
    );
  });

  test('circle: class 1 inside radius 1, class 0 in the ring 1.3 to 2', () => {
    const ds = sample(cfg('circle', { noise: 0, n: 200, val: 0 }));
    for (let i = 0; i < ds.nTrain; i++) {
      const r = Math.hypot(ds.xTrain[2 * i] ?? 0, ds.xTrain[2 * i + 1] ?? 0);
      if (ds.yTrain[i] === 1) expect(r).toBeLessThan(1.0001);
      else {
        expect(r).toBeGreaterThan(1.2999);
        expect(r).toBeLessThan(2.0001);
      }
    }
  });

  test('moons: lies on the two half circles at noise 0', () => {
    const ds = sample(cfg('moons', { noise: 0, n: 100, val: 0 }));
    for (let i = 0; i < ds.nTrain; i++) {
      const x = ds.xTrain[2 * i] ?? 0;
      const y = ds.xTrain[2 * i + 1] ?? 0;
      if (ds.yTrain[i] === 0) {
        expect(Math.hypot(x, y)).toBeCloseTo(1, 4);
        expect(y).toBeGreaterThanOrEqual(-1e-6);
      } else {
        expect(Math.hypot(x - 1, y - 0.5)).toBeCloseTo(1, 4);
        expect(y).toBeLessThanOrEqual(0.5 + 1e-6);
      }
    }
  });

  test('classification noise jitters positions', () => {
    const clean = sample(cfg('xor', { noise: 0, n: 200, val: 0 }));
    const noisy = sample(cfg('xor', { noise: 0.5, n: 200, val: 0 }));
    let wrong = 0;
    for (let i = 0; i < noisy.nTrain; i++) {
      const rule = (noisy.xTrain[2 * i] ?? 0) * (noisy.xTrain[2 * i + 1] ?? 0) > 0 ? 1 : 0;
      if (rule !== noisy.yTrain[i]) wrong++;
    }
    expect(wrong).toBeGreaterThan(0);
    expect(clean.nTrain).toBe(200);
  });

  test.each(SYNTH)('%s: standardised training columns have mean 0 and std 1', (set) => {
    const ds = sample(cfg(set, { standardize: true, n: 200 }));
    const d = SETS[set].d;
    expect(ds.standardization).not.toBeNull();
    for (let c = 0; c < d; c++) {
      if (ds.info.columnKinds?.[c] === 'categorical') continue; // category indices are not standardised
      const col = Array.from({ length: ds.nTrain }, (_, i) => ds.xTrain[i * d + c] ?? 0);
      expect(mean(col)).toBeCloseTo(0, 5);
      expect(std(col)).toBeCloseTo(1, 5);
    }
    // Val uses the training statistics, and unstandardising recovers raw x inside the domain.
    const raw = unstandardizeRows(ds.xVal, d, ds.standardization ?? { mean: [], std: [] });
    for (let i = 0; i < raw.length; i++) {
      const [lo, hi] = ds.domain[i % d] ?? [0, 0];
      expect(raw[i]).toBeGreaterThan(lo - 1e-3 - 1);
      expect(raw[i]).toBeLessThan(hi + 1e-3 + 1);
    }
  });

  test('standardize off leaves x raw and standardization null', () => {
    const ds = sample(cfg('line', { standardize: false }));
    expect(ds.standardization).toBeNull();
    expect(ds.domain).toEqual([[-3, 3]]);
  });

  test('truth is correct at known points', () => {
    expect(Array.from(truth('line', Float32Array.of(0, 1)))).toEqual([
      Math.fround(0.5),
      Math.fround(1.3),
    ]);
    expect(Array.from(truth('step', Float32Array.of(-0.1, 0.1)))).toEqual([-1, 1]);
    expect(truth('saddle', Float32Array.of(2, 0))[0]).toBeCloseTo(2);
    expect(truth('blobs', Float32Array.of(1, 1))[0]).toBeGreaterThan(0.99);
    expect(truth('blobs', Float32Array.of(0, 0))[0]).toBeCloseTo(0.5);
    expect(Array.from(truth('xor', Float32Array.of(1, 1, 1, -1)))).toEqual([1, 0]);
  });
});

describe('grids', () => {
  test('grid1d spans the domain, raw and standardised', () => {
    const g = grid1d([[-3, 3]], 200, { mean: [1], std: [2] });
    expect(g.n).toBe(200);
    expect(g.xRaw[0]).toBe(-3);
    expect(g.xRaw[199]).toBe(3);
    expect(g.xStd[0]).toBeCloseTo(-2);
    expect(g.xStd[199]).toBeCloseTo(1);
  });

  test('grid1d without standardisation: xStd equals xRaw', () => {
    const g = grid1d([[-3, 3]]);
    expect(g.n).toBe(200);
    expect(Array.from(g.xStd)).toEqual(Array.from(g.xRaw));
  });

  test('grid2d spans the domain with x1 fastest', () => {
    const g = grid2d(
      [
        [-1.5, 2.5],
        [-1, 1.5],
      ],
      64,
    );
    expect(g.n).toBe(64 * 64);
    expect(g.d).toBe(2);
    expect([g.xRaw[0], g.xRaw[1]]).toEqual([-1.5, -1]);
    expect(g.xRaw[2]).toBeGreaterThan(-1.5);
    expect(g.xRaw[3]).toBe(-1);
    const last = g.n - 1;
    expect([g.xRaw[2 * last], g.xRaw[2 * last + 1]]).toEqual([2.5, 1.5]);
    const s = grid2d(
      [
        [-2, 2],
        [-2, 2],
      ],
      64,
      { mean: [0, 0], std: [2, 2] },
    );
    expect(s.xStd[0]).toBeCloseTo(-1);
  });
});

describe('batchSchedule', () => {
  const take = (g: Generator<Int32Array>, k: number): Int32Array[] =>
    Array.from({ length: k }, () => g.next().value);

  test.each([
    [10, 3],
    [40, 8],
    [40, 1],
    [7, 100],
  ])('n=%i batch=%i covers every row once per epoch', (n, batch) => {
    const per = stepsPerEpoch(n, batch);
    const g = batchSchedule(n, batch, 3);
    for (let e = 0; e < 3; e++) {
      const seen = take(g, per).flatMap((b) => Array.from(b));
      expect(seen.length).toBe(n);
      expect(seen.sort((a, b) => a - b)).toEqual(Array.from({ length: n }, (_, i) => i));
    }
  });

  test('reproducible, seed-dependent, epochs differ', () => {
    const a = take(batchSchedule(20, 5, 1), 8).map((b) => Array.from(b));
    const b = take(batchSchedule(20, 5, 1), 8).map((b) => Array.from(b));
    const c = take(batchSchedule(20, 5, 2), 8).map((b) => Array.from(b));
    expect(a).toEqual(b);
    expect(a).not.toEqual(c);
    expect(a.slice(0, 4)).not.toEqual(a.slice(4, 8));
  });

  test('full batch yields all rows in order every step', () => {
    const g = batchSchedule(6, 'full', 1);
    for (let i = 0; i < 3; i++) expect(Array.from(g.next().value)).toEqual([0, 1, 2, 3, 4, 5]);
    expect(stepsPerEpoch(6, 'full')).toBe(1);
  });
});

describe('gatherBatch', () => {
  test('gathers the scheduled rows', () => {
    const ds = sample(cfg('plane', { n: 20, val: 0, standardize: true }));
    const idx = batchSchedule(ds.nTrain, 6, 5).next().value;
    const b = gatherBatch(ds.xTrain, ds.yTrain, 2, idx);
    expect(b.n).toBe(6);
    expect(b.x.length).toBe(12);
    idx.forEach((i, k) => {
      expect(b.y[k]).toBe(ds.yTrain[i]);
      expect(b.x[2 * k]).toBe(ds.xTrain[2 * i]);
      expect(b.x[2 * k + 1]).toBe(ds.xTrain[2 * i + 1]);
    });
  });
});

describe('multi-class sets', () => {
  test('there are at least two, each with k >= 3 classes', () => {
    expect(MULTI.length).toBeGreaterThanOrEqual(2);
    for (const set of MULTI) expect(classCount(set)).toBeGreaterThanOrEqual(3);
  });

  test.each(MULTI)(
    '%s: info carries names; labels cycle and are balanced; truth is a class',
    (set) => {
      const k = classCount(set);
      const ds = sample(cfg(set, { noise: 0, n: 90, val: 0.3 }));
      expect(ds.info).toMatchObject({ kind: 'multiclass', numClasses: k, targetName: 'y' });
      expect(ds.info.columns).toEqual(['x1', 'x2']);
      expect(ds.info.classNames).toEqual(Array.from({ length: k }, (_, i) => `class ${i}`));
      for (const ys of [ds.yTrain, ds.yVal]) {
        const counts = new Array<number>(k).fill(0);
        for (const y of ys) counts[y] = (counts[y] ?? 0) + 1;
        expect(Math.max(...counts) - Math.min(...counts)).toBeLessThanOrEqual(1);
      }
      expect(Array.from(ds.yTrain, (y) => Number.isInteger(y) && y >= 0 && y < k)).not.toContain(
        false,
      );
      const t = truth(set, ds.xTrain);
      expect(Array.from(t).every((c) => Number.isInteger(c) && c >= 0 && c < k)).toBe(true);
      const p = truthProbs(set, ds.xTrain);
      expect(p.length).toBe(ds.nTrain * k);
      expect(p.slice(0, k).reduce((a, b) => a + b, 0)).toBe(1);
    },
  );

  test('spiral3 at noise 0: every label equals the truth', () => {
    const ds = sample(cfg('spiral3', { noise: 0, n: 150, val: 0 }));
    expect(Array.from(ds.yTrain)).toEqual(Array.from(truth('spiral3', ds.xTrain)));
  });

  test('blobs3 at low noise: truth agrees with most labels', () => {
    const ds = sample(cfg('blobs3', { noise: 0, n: 150, val: 0 }));
    const t = truth('blobs3', ds.xTrain);
    const same = Array.from(ds.yTrain).filter((y, i) => y === t[i]).length;
    expect(same / ds.nTrain).toBeGreaterThan(0.9);
  });

  test('binary and regression sets get default names too', () => {
    expect(sample(cfg('xor')).info.classNames).toEqual(['class 0', 'class 1']);
    expect(sample(cfg('sine')).info.classNames).toBeUndefined();
    expect(sample(cfg('sine')).info.numClasses).toBe(1);
  });
});

describe('truthMargin', () => {
  test.each(['blobs', 'circle', 'xor', 'moons'] as const)(
    '%s: sign agrees with the truth class and the field is not binary',
    (set) => {
      const xs = new Float32Array(2 * 41 * 41);
      const dom = SETS[set].domain;
      let k = 0;
      for (let i = 0; i < 41; i++)
        for (let j = 0; j < 41; j++) {
          const [a0, a1] = dom[0] ?? [0, 1];
          const [b0, b1] = dom[1] ?? [0, 1];
          xs[k++] = a0 + ((a1 - a0) * (i + 0.37)) / 41;
          xs[k++] = b0 + ((b1 - b0) * (j + 0.21)) / 41;
        }
      const f = truth(set, xs);
      const m = truthMargin(set, xs);
      for (let i = 0; i < f.length; i++) {
        if (Math.abs(m[i] ?? 0) < 1e-9) continue;
        expect((m[i] ?? 0) > 0).toBe((f[i] ?? 0) >= 0.5);
      }
      expect(new Set(Array.from(m)).size).toBeGreaterThan(50);
    },
  );
  test('circle margin is the signed distance to the cut radius', () => {
    const m = truthMargin('circle', new Float32Array([0, 0, 1.15, 0, 0, 2]));
    expect(m[1]).toBeCloseTo(0, 6);
    expect(m[0]).toBeCloseTo(1.15, 6);
    expect(m[2]).toBeCloseTo(-0.85, 6);
  });
});

describe('trend_sine (clipped-normal x)', () => {
  const big = sample(cfg('trend_sine', { n: 20000, val: 0, noise: 0 }));
  test('x is clipped Normal(5, 2) on [0, 10] with point masses at the ends', () => {
    const x = Array.from(big.xTrain);
    expect(Math.min(...x)).toBe(0);
    expect(Math.max(...x)).toBe(10);
    const mean = x.reduce((a, b) => a + b, 0) / x.length;
    expect(Math.abs(mean - 5)).toBeLessThan(0.1);
    // P(x <= 0) = P(z <= -2.5) ~ 0.6%: some exactly at each end.
    expect(x.filter((v) => v === 0).length).toBeGreaterThan(20);
    expect(x.filter((v) => v === 10).length).toBeGreaterThan(20);
    // Not uniform: far more mass in [4, 6] than the uniform 20%.
    expect(x.filter((v) => v >= 4 && v <= 6).length / x.length).toBeGreaterThan(0.35);
  });
  test('truth is 0.5x + sin x, and noise 0 reproduces it', () => {
    const t = truth('trend_sine', Float32Array.of(0, 2, 10));
    expect(t[0]).toBeCloseTo(0, 6);
    expect(t[1]).toBeCloseTo(1 + Math.sin(2), 5);
    expect(t[2]).toBeCloseTo(5 + Math.sin(10), 5);
    const tr = truth('trend_sine', big.xTrain.subarray(0, 50));
    for (let i = 0; i < 50; i++) expect(big.yTrain[i]).toBeCloseTo(tr[i] ?? NaN, 5);
  });
  test('spread is that of f* over a uniform grid (noise 0.6466 gives sigma 1)', () => {
    expect(spread('trend_sine')).toBeCloseTo(1.5466, 3);
  });
});

describe('uniform sets are unchanged by xDist', () => {
  test('seeded sine sample matches the values recorded on origin/main', () => {
    const d = sample(cfg('sine', { n: 5, val: 0 }));
    expect(Array.from(d.xTrain)).toEqual([
      2.611025333404541, -1.2999413013458252, -0.8056244254112244, 1.2193810939788818,
      -2.5828166007995605,
    ]);
    expect(Array.from(d.yTrain)).toEqual([
      -0.5286279320716858, -1.039266586303711, -0.9604676961898804, 0.9178498387336731,
      0.5233895182609558,
    ]);
  });
});
