// @vitest-environment jsdom
// A probability used directly (an unwired Bias pinned as ŷ into a log loss, tinker-ddg): the run
// stops with one sentence when the prediction leaves (0, 1), and a hand edit shows and clears it.
import { act } from 'react';
import { createRoot } from 'react-dom/client';
import { beforeAll, beforeEach, expect, test, vi } from 'vitest';
import { beginWeightEdit } from '../src/canvas/weightEdit.js';
import { getEngine, loadEngine } from '../src/engine.js';
import { contourLevels, contourSegments, heat, lossScale } from '../src/plot/lossSurface.js';
import { probabilitySentence } from '../src/state/probRange.js';
import { useStore } from '../src/state/store.js';
import { LossReadout } from '../src/results/LossReadout.js';
import { mlpGraph } from './mlpGraph.js';
import type { Graph } from '@tinker/core';

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const s = () => useStore.getState();

/** Mounts the readout; `q` finds a test id in it. */
function mount() {
  const el = document.createElement('div');
  document.body.appendChild(el);
  const root = createRoot(el);
  act(() => root.render(<LossReadout />));
  return { el, q: (id: string) => el.querySelector(`[data-testid="${id}"]`), root };
}

function probGraph(lr: number): Graph {
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
    data: { ...g.data, set: 'blobs', n: 40, val: 0.25 },
    train: { ...g.train, optimizer: 'sgd', lr, batch: 'full' },
  };
}

beforeAll(async () => {
  await loadEngine();
}, 60000);

beforeEach(() => {
  s().pause();
});

function setB(v: number): void {
  const e = beginWeightEdit({ kind: 'bias', node: 'b' });
  expect(e).not.toBeNull();
  e?.set(v);
  e?.commit();
}

test('the graph is trainable and an unwired Bias can be set', () => {
  s().loadGraph(probGraph(0.1));
  expect(s().analysis.issues).toEqual([]);
  expect(s().canTrain).toBe(true);
});

test('sentence wording', () => {
  const g = probGraph(0.1);
  const r = (min: number, max: number) => [{ node: 'err', min, max }];
  expect(probabilitySentence(g, r(0.3, 0.3))).toBeNull();
  expect(probabilitySentence(g, r(1.27, 1.27))).toBe(
    'Log loss takes probabilities between 0 and 1, but the prediction reached 1.27.',
  );
  expect(probabilitySentence(g, r(-0.05, -0.05))).toContain('reached -0.05.');
  expect(probabilitySentence(g, r(0, 0))).toContain('reached 0.');
  expect(probabilitySentence(g, r(NaN, NaN))).toBe(
    'Log loss takes probabilities between 0 and 1, but the prediction is not a number.',
  );
  expect(probabilitySentence(g, r(0.2, Infinity))).toContain('became too large.');
  expect(probabilitySentence(g, r(0.2, 3e38))).toContain('became too large.');
  expect(probabilitySentence(g, r(-5e6, 0.5))).toContain('became too large.');
  expect(probabilitySentence(g, r(-Infinity, 0.5))).toContain('is not a number.');
  expect(probabilitySentence(g, r(0.2, 1296))).toContain('reached 1300.');
  expect(probabilitySentence(g, r(0.2, 5.0619))).toContain('reached 5.06.');
  expect(probabilitySentence(g, r(-1e-9, 0.5))).toContain('reached -0.000000001.');
  expect(probabilitySentence(g, r(0.2, 1))).toContain('reached 1.');
  for (const [lo, hi] of [
    [0.2, 1e38],
    [-1e-30, 0.5],
    [0.3, 123456],
    [-0.05, 0.5],
  ]) {
    expect(probabilitySentence(g, r(lo as number, hi as number))).not.toMatch(/e[+-]|Infinity|NaN/);
  }
});

test('training from p in range with a large lr stops on the step that sees the value, params where it left the range', () => {
  s().loadGraph(probGraph(5));
  setB(0.3);
  expect(s().probIssue).toBeNull();
  for (let i = 0; i < 50 && s().probIssue === null; i++) s().stepOnce();
  expect(s().probIssue).toMatch(
    /^Log loss takes probabilities between 0 and 1, but the prediction reached /,
  );
  expect(s().diverged).toBe(false);
  expect(s().running).toBe(false);
  // the update of the refused step is not applied: b is the value the sentence names
  const now = s().host?.b?.b?.[0] ?? NaN;
  expect(now >= 1 || now <= 0).toBe(true);
  expect(s().probIssue).toContain(`reached ${Number(now.toPrecision(3))}.`);
  // the readout shows the sentence, not "diverged"
  const m = mount();
  expect(m.q('loss-readout-prob')?.textContent).toBe(s().probIssue);
  act(() => m.root.unmount());
});

test('cross-entropy with p[class] above 1 (a finite loss) stops the run with the sentence', () => {
  const g = mlpGraph();
  s().loadGraph({
    ...g,
    nodes: [
      { id: 'in', type: 'input', columns: ['x1', 'x2'] },
      { id: 'lin', type: 'linear', out: 2, bias: true },
      { id: 'y', type: 'target' },
      { id: 'err', type: 'cross_entropy' },
      { id: 'loss', type: 'mean' },
    ],
    edges: [
      ['in', 'lin'],
      ['lin', 'err.pred'],
      ['y', 'err.target'],
      ['err', 'loss'],
    ],
    predictions: [],
    loss: 'loss',
    positions: {},
    data: { ...g.data, set: 'blobs', n: 40, val: 0.25 },
    train: { ...g.train, optimizer: 'sgd', lr: 0.1, batch: 'full' },
  });
  s().writeParams('lin', { W: new Float32Array(4), b: Float32Array.of(0.4, 1.5) });
  expect(s().probIssue).toContain('Cross-entropy takes probabilities between 0 and 1');
  const before = s().step;
  s().stepOnce();
  expect(s().step).toBe(before);
  expect(s().diverged).toBe(false);
  expect(s().probIssue).toContain('reached 1.5.');
});

test('a paused hand edit out of range shows the sentence and editing back clears it', () => {
  s().loadGraph(probGraph(0.1));
  setB(0.4);
  s().stepOnce();
  expect(s().probIssue).toBeNull();
  const m = mount();
  expect(m.el.textContent).not.toMatch(/reached/);
  act(() => setB(1.3));
  expect(s().probIssue).toBe(
    'Log loss takes probabilities between 0 and 1, but the prediction reached 1.3.',
  );
  expect(m.q('loss-readout-prob')?.textContent).toBe(s().probIssue);
  expect(s().diverged).toBe(false);
  act(() => setB(0.4));
  expect(s().probIssue).toBeNull();
  expect(m.q('loss-readout-prob')).toBeNull();
  expect(m.q('loss-readout-train')?.textContent).not.toBe('');
  act(() => m.root.unmount());
  const before = s().step;
  s().stepOnce();
  expect(s().step).toBe(before + 1);
});

test('SGD from p = 0.3 converges to the mean of the training targets', () => {
  s().loadGraph(probGraph(0.1));
  setB(0.3);
  const ds = s().dataset;
  const mean = ds.yTrain.reduce((a, v) => a + v, 0) / ds.nTrain;
  for (let i = 0; i < 400; i++) s().stepOnce();
  expect(s().probIssue).toBeNull();
  expect(s().host?.b?.b?.[0]).toBeCloseTo(mean, 3);
});

test('an engine error clears a stale probability sentence so the error shows', () => {
  s().loadGraph(probGraph(0.1));
  setB(1.3);
  expect(s().probIssue).not.toBeNull();
  // make the engine throw inside the step
  const engine = getEngine();
  if (!engine) throw new Error('no engine');
  const spy = vi.spyOn(engine, 'step').mockImplementation(() => {
    throw new Error('boom');
  });
  s().stepOnce();
  spy.mockRestore();
  expect(s().probIssue).toBeNull();
  expect(s().trainIssue).toBe('boom');
});

test('the loss surface of an unfused log loss has non-finite cells outside (0, 1) and the plot code copes', () => {
  const g = probGraph(0.1);
  s().loadGraph({
    ...g,
    nodes: [
      { id: 'c', type: 'column', name: 'x1' },
      { id: 'w', type: 'scalew' },
      { id: 'b', type: 'bias' },
      { id: 'y', type: 'target' },
      { id: 'err', type: 'log_loss' },
      { id: 'loss', type: 'mean' },
    ],
    edges: [
      ['c', 'w'],
      ['w', 'b'],
      ['b', 'err.pred'],
      ['y', 'err.target'],
      ['err', 'loss'],
    ],
  });
  expect(s().canTrain).toBe(true);
  s().writeParams('w', { W: Float32Array.of(0.1) });
  s().writeParams('b', { b: Float32Array.of(0.5) });
  const n = 21;
  const a = Float32Array.from({ length: n }, (_, i) => -1 + (2 * i) / (n - 1));
  const b = Float32Array.from({ length: n }, (_, i) => -0.5 + (2 * i) / (n - 1)); // b from -0.5 to 1.5
  const grid = s().lossSurface(
    [
      { node: 'w', tensor: 'W', index: 0 },
      { node: 'b', tensor: 'b', index: 0 },
    ],
    a,
    b,
  );
  expect(grid).not.toBeNull();
  const cells = Array.from(grid ?? []);
  expect(cells.some((v) => !Number.isFinite(v))).toBe(true); // outside (0, 1)
  expect(cells.some((v) => Number.isFinite(v))).toBe(true);
  const scale = lossScale(grid as Float32Array);
  expect(scale).not.toBeNull();
  if (!scale) return;
  // non-finite cells are blank (NaN heat), finite ones are in 0..1; contours never see a non-finite corner
  for (const v of cells) {
    const h = heat(v, scale);
    if (Number.isFinite(v)) expect(h >= 0 && h <= 1).toBe(true);
    else expect(Number.isNaN(h)).toBe(true);
  }
  for (const lv of contourLevels(scale)) {
    const seg = contourSegments(grid as Float32Array, n, n, lv);
    for (const q of seg) expect(Number.isFinite(q)).toBe(true);
  }
});
