// @vitest-environment jsdom
import { beforeAll, beforeEach, expect, test } from 'vitest';
import { mlpGraph as initialGraph } from './mlpGraph.js';
import { loadEngine } from '../src/engine.js';
import { useStore } from '../src/state/store.js';

const s = () => useStore.getState();

beforeAll(async () => {
  await loadEngine();
}, 60000);

beforeEach(() => {
  s().pause();
  s().loadGraph(initialGraph());
  s().reset();
});

/** Drive the model to a non-finite loss with an absurd learning rate. */
function diverge(): void {
  s().setLearningRate(1e30);
  for (let i = 0; i < 200 && !s().diverged; i++) s().stepOnce();
  expect(s().diverged).toBe(true);
}

function expectTrains(): void {
  expect(s().diverged).toBe(false);
  const before = s().step;
  s().stepOnce();
  expect(s().diverged).toBe(false);
  expect(s().step).toBe(before + 1);
  s().run();
  expect(s().diverged).toBe(false);
  expect(s().running).toBe(true);
  s().pause();
}

test('engine is available', () => {
  expect(s().canTrain).toBe(true);
});

test('lr change then resample then run clears diverged', () => {
  diverge();
  s().setLearningRate(0.3);
  expect(s().diverged).toBe(false);
  s().setData({ seed: s().graph.data.seed + 1 });
  expect(s().diverged).toBe(false);
  s().run();
  expect(s().diverged).toBe(false);
  expect(s().running).toBe(true);
  expect(s().step).toBe(0);
  s().stepOnce(); // no-op while running; pause first
  s().pause();
  s().stepOnce();
  expect(s().diverged).toBe(false);
  expect(s().step).toBe(1);
});

test('lowering lr clears diverged', () => {
  diverge();
  s().setLearningRate(0.1);
  expectTrains();
});

test('reset clears diverged', () => {
  diverge();
  s().reset();
  s().setLearningRate(0.1);
  expectTrains();
});

test('dataset change clears diverged', () => {
  diverge();
  s().setData({ seed: s().graph.data.seed + 1 });
  s().setLearningRate(0.1);
  expectTrains();
});

test('optimizer and batch change clear diverged', () => {
  diverge();
  s().setLearningRate(0.1);
  s().setOptimizer('adam');
  expectTrains();
  diverge();
  s().setLearningRate(0.01);
  s().setBatch(8);
  expectTrains();
});

test('restore best clears diverged', () => {
  s().stepOnce();
  diverge();
  s().setLearningRate(0.1);
  s().restoreBest();
  expectTrains();
});

test('loadHost clears diverged', () => {
  const saved = s().host;
  if (!saved) throw new Error('no host');
  diverge();
  s().loadGraph(initialGraph()); // a file load: new graph first, then its weights
  expect(s().loadHost(saved)).toBe(true);
  expectTrains();
});
