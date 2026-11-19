// trainMany (CONTRACTS.md section 7): each copy equals training that copy alone with `step`.
import { installProbe } from '../src/probe.js';
import { describe, expect, test, vi } from 'vitest';
import { loadFixture, toBatch } from './util.js';
import type { Batch, HostParams } from '@tinker/core';

installProbe();
// Logic tests, no timing assertions: a loaded machine (parallel agents, CI) must not fail them on vitest's 5 s default.
vi.setConfig({ testTimeout: 60_000 });
const { createEngine } = await import('../src/index.js');
const engine = await createEngine();

/** k datasets of n rows made from the fixture batch: the same x, different y (shifted and scaled). */
function datasets(fx: ReturnType<typeof loadFixture>['fx'], k: number, n = fx.batch.n): Batch[] {
  const base = toBatch(fx);
  const d = base.x.length / base.n;
  return Array.from({ length: k }, (_, i) => {
    const x = new Float32Array(n * d);
    const y = new Float32Array(n);
    for (let r = 0; r < n; r++) {
      const src = r % base.n;
      for (let c = 0; c < d; c++)
        x[r * d + c] = (base.x[src * d + c] ?? 0) + 0.05 * i * Math.sin(r + c);
      y[r] = (base.y[src] ?? 0) * (1 + 0.1 * i) + 0.02 * i * Math.cos(r);
    }
    return { x, y, n };
  });
}

function single(name: string, seeds: number[], data: Batch[], steps: number) {
  const { graph, data: info } = loadFixture(name);
  const model = engine.compile(graph, info);
  return seeds.map((s, i) => {
    let state = engine.startTraining(model, engine.init(model, s), s);
    for (let t = 0; t < steps; t++) {
      const r = engine.step(model, state, data[i] as Batch);
      expect(r.diverged).toBe(false);
      state = r.state;
    }
    const host = engine.toHost(state.params);
    engine.dispose(state);
    return host;
  });
}

function maxDiff(a: HostParams, b: HostParams): number {
  let worst = 0;
  for (const id of Object.keys(a)) {
    for (const t of ['W', 'b'] as const) {
      const u = a[id]?.[t];
      const v = b[id]?.[t];
      if (!u || !v) continue;
      expect(u.length).toBe(v.length);
      u.forEach((q, j) => (worst = Math.max(worst, Math.abs(q - (v[j] ?? NaN)))));
    }
  }
  return worst;
}

describe('trainMany equals training each copy alone', () => {
  for (const name of [
    'mlp_tanh_sgd_5steps',
    'mlp_tanh_momentum_5steps',
    'mlp_tanh_adam_5steps',
    'dropout_p0.5',
    'penalty_l2',
  ]) {
    test(name, () => {
      const { fx, graph, data: info } = loadFixture(name);
      const seeds = [3, 11, 42, 7];
      const data = datasets(fx, seeds.length);
      const steps = 25;
      const model = engine.compile(graph, info);
      const many = engine.trainMany(model, seeds, data, steps);
      const alone = single(name, seeds, data, steps);
      expect(many).toHaveLength(seeds.length);
      many.forEach((r, i) => {
        expect(r.diverged).toBe(false);
        expect(maxDiff(r.params, alone[i] ?? {})).toBeLessThan(1e-5);
      });
      // the copies really differ from each other
      const first = many[0]?.params;
      const second = many[1]?.params;
      expect(first && second ? maxDiff(first, second) : 0).toBeGreaterThan(1e-3);
    });
  }
});

describe('trainMany divergence and arguments', () => {
  test('one divergent copy is flagged, keeps finite params, and does not change the others', () => {
    const { fx, graph, data: info } = loadFixture('mlp_tanh_sgd_5steps');
    const model = engine.compile(graph, info);
    const seeds = [1, 2, 3];
    const good = datasets(fx, 3);
    const bad = good.map((b, i) => (i === 1 ? { ...b, y: Float32Array.from(b.y, () => NaN) } : b));
    const withBad = engine.trainMany(model, seeds, bad, 10);
    const clean = engine.trainMany(model, seeds, good, 10);
    expect(withBad.map((r) => r.diverged)).toEqual([false, true, false]);
    expect(clean.map((r) => r.diverged)).toEqual([false, false, false]);
    expect(maxDiff(withBad[0]?.params ?? {}, clean[0]?.params ?? {})).toBe(0);
    expect(maxDiff(withBad[2]?.params ?? {}, clean[2]?.params ?? {})).toBe(0);
    // frozen at its initialisation (the first step was already non-finite), and finite
    const init = engine.toHost(engine.init(model, 2));
    expect(maxDiff(withBad[1]?.params ?? {}, init)).toBe(0);
  });

  test('a copy that blows up mid-run keeps its last finite params; all diverged stops early', () => {
    const { fx, graph, data: info } = loadFixture('mlp_tanh_sgd_5steps');
    const wild = { ...graph, train: { ...graph.train, lr: 1e30 } };
    const model = engine.compile(wild, info);
    const out = engine.trainMany(model, [1, 2], datasets(fx, 2), 50);
    for (const r of out) {
      expect(r.diverged).toBe(true);
      for (const l of Object.values(r.params)) {
        expect(Array.from(l.W ?? []).every(Number.isFinite)).toBe(true);
      }
    }
  });

  test('bad arguments throw plain sentences', () => {
    const { fx, graph, data: info } = loadFixture('mlp_tanh_sgd_5steps');
    const model = engine.compile(graph, info);
    const d = datasets(fx, 2);
    expect(() => engine.trainMany(model, [], [], 5)).toThrow('at least one copy');
    expect(() => engine.trainMany(model, [1, 2], [d[0] as Batch], 5)).toThrow(
      'its own training set',
    );
    const short = { ...(d[1] as Batch), n: 3, x: new Float32Array(3), y: new Float32Array(3) };
    expect(() => engine.trainMany(model, [1, 2], [d[0] as Batch, short], 5)).toThrow('same size');
    expect(engine.trainMany(model, [1, 2], d, 0)).toHaveLength(2);
  });
});

describe('trainMany leaks nothing', () => {
  test('live arrays stay flat across repeated calls (also with dropout and a failing call)', () => {
    // the probe works: a live Params shows up in the count
    const probe = loadFixture('mlp_tanh_sgd_5steps');
    const held = engine.init(engine.compile(probe.graph, probe.data), 1);
    expect(engine.liveArrays()).toBeGreaterThan(0);
    engine.dispose(held);
    for (const name of ['mlp_tanh_adam_5steps', 'dropout_p0.5']) {
      const { fx, graph, data: info } = loadFixture(name);
      const model = engine.compile(graph, info);
      const data = datasets(fx, 5);
      engine.trainMany(model, [1, 2, 3, 4, 5], data, 5); // warm up
      const baseline = engine.liveArrays();
      for (let i = 0; i < 30; i++) engine.trainMany(model, [1, 2, 3, 4, 5], data, 5);
      expect(() => engine.trainMany(model, [1, 2], [data[0] as Batch], 5)).toThrow();
      expect(engine.liveArrays()).toBe(baseline);
    }
  });

  test('50 compiled models used once each leave no growth per model', () => {
    const { fx, graph, data: info } = loadFixture('mlp_tanh_sgd_5steps');
    const data = datasets(fx, 3);
    const run = () => engine.trainMany(engine.compile(graph, info), [1, 2, 3], data, 3);
    run();
    run();
    const baseline = engine.liveArrays();
    for (let i = 0; i < 50; i++) run();
    expect(engine.liveArrays()).toBe(baseline);
  });
});
