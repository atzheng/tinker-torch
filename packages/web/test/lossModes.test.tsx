// @vitest-environment jsdom
// Score readout modes (tinker-6t7.5): likelihood / total / mean, live on hand edits, per preset.
import { act } from 'react';
import { createRoot } from 'react-dom/client';
import { beforeAll, beforeEach, expect, test, vi } from 'vitest';
import { exampleStats, type Graph } from '@tinker/core';
import { beginWeightEdit } from '../src/canvas/weightEdit.js';
import { loadEngine } from '../src/engine.js';
import { formatLoss } from '../src/results/readoutModel.js';
import { LossReadout } from '../src/results/LossReadout.js';
import { useStore } from '../src/state/store.js';
import { mlpGraph } from './mlpGraph.js';

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;
vi.stubGlobal('matchMedia', () => ({ addEventListener() {}, removeEventListener() {} }));
const s = () => useStore.getState();

/** A bare Bias p into log loss on a binary set: every example has probability p or 1-p. */
function biasGraph(n: number): Graph {
  const g = mlpGraph();
  return {
    ...g,
    nodes: [
      { id: 'b', type: 'bias' },
      { id: 'y', type: 'target' },
      { id: 'err', type: 'log_loss' },
      { id: 'loss', type: 'mean' },
    ],
    edges: [
      ['b', 'err.pred'],
      ['y', 'err.target'],
      ['err', 'loss'],
    ],
    predictions: ['b'],
    loss: 'loss',
    positions: {},
    data: { ...g.data, set: 'blobs', n, val: 0.25 },
    train: { ...g.train, optimizer: 'sgd', lr: 0.01, batch: 'full' },
  };
}

function setB(v: number): void {
  act(() => {
    const e = beginWeightEdit({ kind: 'bias', node: 'b' });
    e?.set(v);
    e?.commit();
  });
}

function mount() {
  const el = document.createElement('div');
  document.body.appendChild(el);
  const root = createRoot(el);
  act(() => root.render(<LossReadout />));
  const q = (id: string) => el.querySelector(`[data-testid="${id}"]`) as HTMLElement | null;
  return { q, root, click: (id: string) => act(() => q(id)?.click()) };
}

beforeAll(async () => {
  await loadEngine();
}, 60000);
beforeEach(() => {
  s().pause();
  useStore.setState({ lossMode: 'mean', show: null });
});

test('mean follows a hand edit; total = n x mean; likelihood = product of probabilities', () => {
  s().loadGraph(biasGraph(40));
  setB(0.5);
  const m = mount();
  const n = s().dataset.nTrain;
  expect(m.q('loss-readout-train')?.textContent).toBe('0.6931');
  expect(m.q('loss-readout-val')?.textContent).toBe('0.6931');
  setB(0.25); // live: no step taken
  const meanNow = s().liveLoss?.train ?? NaN;
  expect(m.q('loss-readout-train')?.textContent).not.toBe('0.6931');
  m.click('loss-mode-total');
  expect(Number(m.q('loss-readout-train')?.textContent)).toBeCloseTo(n * meanNow, 1);
  m.click('loss-mode-likelihood');
  const losses = s().exampleLosses('train');
  expect(losses?.length).toBe(n);
  const lik = exampleStats(losses as Float32Array).likelihood;
  expect(m.q('loss-readout-train')?.textContent).toBe(formatLoss(lik));
  setB(0.5);
  expect(m.q('loss-readout-train')?.textContent).toBe(formatLoss(0.5 ** n));
  act(() => m.root.unmount());
});

test('likelihood underflows to exactly 0 once the training set has 150 rows (n = 200, val 0.25)', () => {
  s().loadGraph(biasGraph(120)); // 90 training rows: 0.5^90 ~ 8e-28 > 0
  setB(0.5);
  const m = mount();
  m.click('loss-mode-likelihood');
  expect(m.q('loss-readout-train')?.textContent).toMatch(/e-28$/);
  act(() => s().loadGraph(biasGraph(200))); // 150 training rows: 2^-150 < 2^-149, the smallest float32
  setB(0.5);
  expect(s().dataset.nTrain).toBe(150);
  expect(m.q('loss-readout-train')?.textContent).toBe('0');
  expect(m.q('loss-readout-val')?.textContent).not.toBe('0'); // 50 rows
  act(() => m.root.unmount());
});

test('a preset fixes the mode by showing one id; no ui list shows the toggle', () => {
  s().loadGraph(biasGraph(40));
  setB(0.5);
  let m = mount();
  expect(m.q('loss-mode-mean')).not.toBeNull();
  act(() => m.root.unmount());
  useStore.setState({ show: { ui: ['loss-readout', 'loss-mode-total'] } });
  m = mount();
  expect(m.q('loss-mode-total')).toBeNull(); // single mode: no toggle
  const n = s().dataset.nTrain;
  expect(Number(m.q('loss-readout-train')?.textContent)).toBeCloseTo(n * Math.LN2, 1);
  act(() => m.root.unmount());
  useStore.setState({ show: { ui: ['loss-readout', 'loss-mode-likelihood', 'loss-mode-total'] } });
  m = mount();
  expect(m.q('loss-mode-mean')).toBeNull();
  expect(m.q('loss-mode-likelihood')?.getAttribute('aria-pressed')).toBe('true'); // mean hidden: first shown
  act(() => m.root.unmount());
});

test('squared error offers total and mean only', () => {
  s().loadGraph(mlpGraph());
  const m = mount();
  expect(m.q('loss-mode-likelihood')).toBeNull();
  expect(m.q('loss-mode-total')).not.toBeNull();
  expect(m.q('loss-mode-mean')).not.toBeNull();
  act(() => m.root.unmount());
});
