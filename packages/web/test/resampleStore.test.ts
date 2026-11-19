import { beforeEach, expect, test } from 'vitest';
import { setLinearOut } from '@tinker/core';
import { mlpGraph as initialGraph } from './mlpGraph.js';
import { useStore } from '../src/state/store.js';
import { useOverlay, type OverlayResult } from '../src/results/resampleStore.js';

const s = () => useStore.getState();

const result: OverlayResult = {
  d: 1,
  gridN: 2,
  k: 1,
  curves: [Float32Array.of(0, 1)],
  mean: Float32Array.of(0, 1),
  bias2: 0.1,
  variance: 0.2,
  used: 1,
  diverged: 0,
  steps: 10,
  fallbackSteps: false,
  fullBatchNote: false,
};
const show = () => useOverlay.setState({ status: 'done', result, message: null });

beforeEach(() => {
  s().loadGraph(initialGraph());
  useOverlay.setState({ status: 'idle', result: null, message: null });
});

test('the overlay clears when the data change', () => {
  show();
  s().setData({ noise: 0.5 });
  expect(useOverlay.getState().status).toBe('idle');
  expect(useOverlay.getState().result).toBeNull();
});

test('the overlay clears when the graph changes', () => {
  show();
  s().apply(setLinearOut(s().graph, 'lin1', 5));
  expect(useOverlay.getState().result).toBeNull();
});

test('the overlay survives training steps (same graph, same data)', () => {
  show();
  s().stepOnce();
  expect(useOverlay.getState().status).toBe('done');
});
