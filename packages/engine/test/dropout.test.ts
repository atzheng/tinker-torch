import type { Array as JArray } from '@jax-js/jax';
import { beforeAll, expect, test } from 'vitest';
import { createEngine } from '../src/index.js';
import type { Engine } from '../src/index.js';
import { loadFixture, toBatch, toHostParams, withLr } from './util.js';

let engine: Engine;
beforeAll(async () => {
  engine = await createEngine();
});

const keyOf = (k: unknown): number[] =>
  Array.from((k as JArray).ref.dataSync() as ArrayLike<number>);

test('dropout p = 0.5: step losses vary across steps (lr 0), predict and evaluate are deterministic', () => {
  const { fx, graph, data } = loadFixture('dropout_p0.5');
  const model = engine.compile(withLr(graph, 0), data);
  const batch = toBatch(fx);
  let state = engine.startTraining(model, engine.fromHost(model, toHostParams(fx.params)), 3);
  const losses: number[] = [];
  const keys: string[] = [JSON.stringify(keyOf(state.key))];
  for (let i = 0; i < 8; i++) {
    const r = engine.step(model, state, batch);
    expect(r.diverged).toBe(false);
    losses.push(r.loss);
    state = r.state;
    keys.push(JSON.stringify(keyOf(state.key)));
  }
  expect(new Set(losses).size).toBeGreaterThan(1);
  expect(new Set(keys).size).toBe(keys.length);
  const p1 = engine.predict(model, state.params, batch.x, batch.n);
  const p2 = engine.predict(model, state.params, batch.x, batch.n);
  expect([...p1]).toEqual([...p2]);
  expect(engine.evaluate(model, state.params, batch)).toBe(
    engine.evaluate(model, state.params, batch),
  );
  // the same seed gives the same sequence
  let s2 = engine.startTraining(model, engine.fromHost(model, toHostParams(fx.params)), 3);
  const again: number[] = [];
  for (let i = 0; i < 8; i++) {
    const r = engine.step(model, s2, batch);
    again.push(r.loss);
    s2 = r.state;
  }
  expect(again).toEqual(losses);
  engine.dispose(state);
  engine.dispose(s2);
});

test('a diverged step returns the state with the same key', () => {
  const { fx, graph, data } = loadFixture('dropout_p0.5');
  const model = engine.compile(graph, data);
  const batch = toBatch(fx);
  const bad = { ...batch, y: Float32Array.from(batch.y, () => NaN) };
  const state = engine.startTraining(model, engine.fromHost(model, toHostParams(fx.params)), 3);
  const before = keyOf(state.key);
  const r = engine.step(model, state, bad);
  expect(r.diverged).toBe(true);
  expect(keyOf(r.state.key)).toEqual(before);
  engine.dispose(r.state);
});
