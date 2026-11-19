// @vitest-environment jsdom
import { beforeAll, beforeEach, expect, test } from 'vitest';
import { addPrediction, removePrediction } from '@tinker/core';
import { mlpGraph as initialGraph } from './mlpGraph.js';
import { loadEngine } from '../src/engine.js';
import { useStore } from '../src/state/store.js';

const s = () => useStore.getState();

beforeAll(async () => {
  await loadEngine();
}, 60000);

beforeEach(() => {
  s().pause();
});

test('frozen inert params are merged back into the live host and return bit for bit', () => {
  // A spare Linear on x, read by a second yhat pin: live while pinned, inert once the pin is gone.
  const g0 = initialGraph();
  const withSpare = {
    ...g0,
    nodes: [...g0.nodes, { id: 'spare', type: 'linear' as const, out: 1, bias: true }],
    edges: [...g0.edges, ['x', 'spare'] as (typeof g0.edges)[number]],
    predictions: [...g0.predictions, 'spare'] as typeof g0.predictions,
  };
  s().loadGraph(withSpare);
  s().reset();
  expect(s().canTrain).toBe(true);
  expect(s().analysis.paramNodes).toContain('spare');
  // The spare feeds no loss, so give it distinctive weights a fresh init could not reproduce.
  expect(
    s().loadHost({ spare: { W: new Float32Array([0.123456789]), b: new Float32Array([-0.75]) } }),
  ).toBe(true);
  s().stepOnce();
  const before = s().host?.spare;
  expect(before?.W).toBeDefined();
  const w0 = Array.from(before?.W ?? []);
  const b0 = Array.from(before?.b ?? []);

  const r = removePrediction(s().graph, 1);
  expect(r.ok).toBe(true);
  s().apply(r);
  expect(s().analysis.inert).toContain('spare');
  expect(s().analysis.paramNodes).not.toContain('spare');
  const lin1Before = Array.from(s().host?.lin1?.W ?? []);
  s().stepOnce(); // the engine holds no spare; the slice must keep its frozen weights
  s().stepOnce();
  // A later resync reads the live state out of the engine, which has no spare: the slice must
  // merge the frozen entry back instead of replacing the host with the live one.
  s().setLearningRate(0.05);
  s().stepOnce();
  expect(s().host?.spare).toBeUndefined(); // the published host is live owners only
  expect(Array.from(s().host?.lin1?.W ?? [])).not.toEqual(lin1Before); // training did move the live part

  const a = addPrediction(s().graph, 'spare');
  expect(a.ok).toBe(true);
  s().apply(a);
  expect(s().analysis.paramNodes).toContain('spare');
  expect(Array.from(s().host?.spare?.W ?? [])).toEqual(w0);
  expect(Array.from(s().host?.spare?.b ?? [])).toEqual(b0);
});
