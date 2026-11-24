// 1-D loss curve (tinker-6t7.2): lossCurve equals evaluate with the value written; lossSlope equals
// the PyTorch gradient of the fixtures (the gradient Step uses, no dropout, full batch); no leaks.
import { readFileSync } from 'node:fs';
import { analyse, initParams, parseFile, sample } from '@tinker/core';
import { expect, test } from 'vitest';
import { installProbe } from '../src/probe.js';
import { fitFixtureTrees, fixtureNames, loadFixture, toBatch, toHostParams } from './util.js';
import type { ParamAxis } from '../src/types.js';

installProbe();
const { createEngine } = await import('../src/index.js');
const engine = await createEngine();

function intercept() {
  const url = new URL('../../web/src/presets/intercept-prob.json', import.meta.url);
  const file = parseFile(JSON.parse(readFileSync(url, 'utf8')));
  if (!file.ok) throw new Error(file.error);
  const g = file.file.graph;
  const ds = sample(g.data);
  const model = engine.compile(g, ds.info);
  const params = engine.fromHost(model, initParams(analyse(g, ds.info), 3));
  return { model, params, batch: { x: ds.xTrain, y: ds.yTrain, n: ds.nTrain } };
}

test('one-parameter intercept-prob: the curve equals evaluate at sampled values', () => {
  const { model, params, batch } = intercept();
  const axis: ParamAxis = { node: 'bias', tensor: 'b', index: 0 };
  const values = Float32Array.from([-3, -1, -0.2, 0, 0.4, 2.5]);
  const curve = engine.lossCurve(model, params, batch, axis, values);
  expect(curve).toHaveLength(values.length);
  values.forEach((v, i) => {
    const p = engine.write(engine.clone(params), 'bias', { b: Float32Array.of(v) });
    expect(curve[i]).toBeCloseTo(engine.evaluate(model, p, batch), 5);
    engine.dispose(p);
  });
  expect(engine.lossCurve(model, params, batch, axis, new Float32Array(0))).toHaveLength(0);
  expect(() => engine.lossCurve(model, params, batch, { ...axis, index: 4 }, values)).toThrow();
  expect(() => engine.lossCurve(model, params, batch, { ...axis, node: 'nope' }, values)).toThrow();
});

test('one-parameter intercept-prob: the slope is the finite difference of the curve', () => {
  const { model, params, batch } = intercept();
  const axis: ParamAxis = { node: 'bias', tensor: 'b', index: 0 };
  const b = engine.toHost(params)['bias']?.['b']?.[0] ?? 0;
  const h = 1e-2;
  const [lo, hi] = engine.lossCurve(model, params, batch, axis, Float32Array.of(b - h, b + h));
  const slope = engine.lossSlope(model, params, batch, axis);
  expect(slope).toBeCloseTo(((hi ?? 0) - (lo ?? 0)) / (2 * h), 2);
});

// the kink fixtures differ from torch by the owner's |w|'(0) = 1 / relu'(0) decision (reference.test.ts SKIPPED)
const KINKS = ['act_relu_zero_preact', 'penalty_l1_zero_weight'];
const NO_DROPOUT = fixtureNames().filter(
  (n) => !loadFixture(n).fx.expect.dropoutMask && !KINKS.includes(n),
);

test.each(NO_DROPOUT)('%s: lossSlope equals the PyTorch gradient of every entry', (name) => {
  const { fx, graph, data } = loadFixture(name);
  const batch = toBatch(fx);
  const model = engine.compile(graph, data, fitFixtureTrees(graph, data, batch));
  const params = engine.fromHost(model, toHostParams(fx.params));
  for (const [node, g] of Object.entries(fx.expect.grads)) {
    for (const tensor of ['W', 'b'] as const) {
      (g[tensor] ?? []).forEach((want, index) => {
        const got = engine.lossSlope(model, params, batch, { node, tensor, index });
        expect(Math.abs(got - want), `${node}.${tensor}[${index}] ${got} vs ${want}`).toBeLessThan(
          1e-4 + 1e-4 * Math.abs(want),
        );
      });
    }
  }
  engine.dispose(params);
  engine.disposeModel(model);
});

test('lossCurve and lossSlope leave the live buffer count flat (also when they throw)', () => {
  const { fx, graph, data } = loadFixture('two_param');
  const batch = toBatch(fx);
  const axis: ParamAxis = { node: 'lin1', tensor: 'W', index: 0 };
  const v = Float32Array.from([-1, 0, 1]);
  const cycle = () => {
    const model = engine.compile(graph, data);
    const params = engine.fromHost(model, toHostParams(fx.params));
    engine.lossCurve(model, params, batch, axis, v);
    engine.lossSlope(model, params, batch, axis);
    engine.dispose(params);
    engine.disposeModel(model);
  };
  for (let i = 0; i < 5; i++) cycle();
  const base = engine.liveArrays();
  for (let i = 0; i < 100; i++) cycle();
  expect(engine.liveArrays()).toBe(base);
  const model = engine.compile(graph, data);
  const params = engine.fromHost(model, toHostParams(fx.params));
  const before = engine.liveArrays();
  for (let i = 0; i < 10; i++) {
    expect(() => engine.lossCurve(model, params, batch, { ...axis, index: 9 }, v)).toThrow();
    expect(() => engine.lossSlope(model, params, batch, { ...axis, node: 'nope' })).toThrow();
  }
  expect(engine.liveArrays()).toBe(before);
  engine.dispose(params);
}, 120_000);

test('lossSlope on a Weights-tap-only layer leaves the live buffer count flat and matches the fixture', () => {
  const { fx, graph, data } = loadFixture('prim_weights_tap_only');
  const batch = toBatch(fx);
  const entries: ParamAxis[] = [];
  for (const [node, g] of Object.entries(fx.expect.grads))
    for (const tensor of ['W', 'b'] as const)
      (g[tensor] ?? []).forEach((_, index) => entries.push({ node, tensor, index }));
  const cycle = () => {
    const model = engine.compile(graph, data);
    const params = engine.fromHost(model, toHostParams(fx.params));
    for (const e of entries) {
      const got = engine.lossSlope(model, params, batch, e);
      const want = fx.expect.grads[e.node]?.[e.tensor]?.[e.index] ?? Number.NaN;
      expect(Math.abs(got - want)).toBeLessThan(1e-4 + 1e-4 * Math.abs(want));
      engine.lossCurve(model, params, batch, e, Float32Array.of(0, 1));
    }
    engine.dispose(params);
    engine.disposeModel(model);
  };
  for (let i = 0; i < 5; i++) cycle();
  const base = engine.liveArrays();
  for (let i = 0; i < 50; i++) cycle();
  expect(engine.liveArrays()).toBe(base);
}, 120_000);
