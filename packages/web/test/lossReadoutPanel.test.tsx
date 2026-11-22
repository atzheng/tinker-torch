// @vitest-environment jsdom
import { beforeAll, beforeEach, expect, test, vi } from 'vitest';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { loadEngine } from '../src/engine.js';
import { mlpGraph } from './mlpGraph.js';
import { useStore } from '../src/state/store.js';
import { LossReadout } from '../src/results/LossReadout.js';
import { EM_DASH, formatLoss, lossReadout } from '../src/results/readoutModel.js';
import type { HistoryPoint } from '../src/state/trainSlice.js';

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;
vi.stubGlobal(
  'ResizeObserver',
  class {
    observe() {}
    disconnect() {}
  },
);
vi.stubGlobal('matchMedia', () => ({ addEventListener() {}, removeEventListener() {} }));

const h = (
  step: number,
  train: number,
  val: number | null,
  trainFull?: number | null,
): HistoryPoint => ({
  step,
  train,
  val,
  ...(trainFull !== undefined ? { trainFull } : {}),
});

test('formatLoss: 4 significant figures, exponent form outside [1e-3, 1e4)', () => {
  expect(formatLoss(0.693147)).toBe('0.6931');
  expect(formatLoss(12.3456)).toBe('12.35');
  expect(formatLoss(0.001)).toBe('0.001');
  expect(formatLoss(3.14159e-5)).toBe('3.142e-5');
  expect(formatLoss(12345.6)).toBe('1.235e+4');
  expect(formatLoss(9999)).toBe('9999');
  expect(formatLoss(0)).toBe('0');
  expect(formatLoss(null)).toBe(EM_DASH);
  expect(formatLoss(Number.NaN)).toBe(EM_DASH);
});

test('lossReadout: em dash with no history', () => {
  expect(lossReadout([], false, false)).toEqual({ train: EM_DASH, val: EM_DASH, diverged: false });
});

test('lossReadout: full batch follows train, val is the latest non-null', () => {
  const r = lossReadout([h(1, 0.5, 0.6), h(2, 0.25, null)], false, false);
  expect(r.train).toBe('0.25');
  expect(r.val).toBe('0.6');
});

test('lossReadout: mini-batch follows the latest non-null trainFull, not the batch loss', () => {
  const r = lossReadout([h(1, 0.9, 0.8, 0.7), h(2, 0.1, null, null)], true, false);
  expect(r.train).toBe('0.7');
  expect(r.val).toBe('0.8');
});

let root: Root;
let host: HTMLDivElement;
const q = (id: string) => host.querySelector(`[data-testid="${id}"]`);

beforeAll(async () => {
  await loadEngine();
}, 60000);
beforeEach(() => {
  useStore.getState().pause();
  useStore.getState().loadGraph(mlpGraph());
  useStore.getState().reset();
  host = document.createElement('div');
  document.body.appendChild(host);
  root = createRoot(host);
  act(() => root.render(<LossReadout />));
});

test('component: dash, history values, diverged', () => {
  act(() => useStore.setState({ liveLoss: null })); // the history fallback; live values: lossModes.test.tsx
  expect(q('loss-readout-train')?.textContent).toBe(EM_DASH);
  expect(q('loss-readout-val')?.textContent).toBe(EM_DASH);
  act(() => useStore.setState({ history: [h(1, 0.5, 0.6, 0.4)] }));
  expect(q('loss-readout-train')?.textContent).toBe('0.5');
  expect(q('loss-readout-val')?.textContent).toBe('0.6');
  act(() => useStore.setState({ diverged: true }));
  expect(q('loss-readout-train')?.textContent).toBe('diverged');
  act(() => useStore.setState({ diverged: false }));
  expect(q('loss-readout-train')?.textContent).toBe('0.5');
});

test('component: mini-batch follows trainFull', () => {
  const g = mlpGraph();
  useStore.getState().loadGraph({ ...g, train: { ...g.train, batch: 4 } });
  act(() => useStore.setState({ liveLoss: null }));
  act(() => useStore.setState({ history: [h(1, 0.9, 0.6, 0.4)] }));
  expect(q('loss-readout-train')?.textContent).toBe('0.4');
});
