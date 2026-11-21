// @vitest-environment jsdom
// A probability used directly (an unwired Bias pinned as ŷ into a log loss, tinker-ddg): the run
// stops with one sentence when the prediction leaves (0, 1), and a hand edit shows and clears it.
import { act } from 'react';
import { createRoot } from 'react-dom/client';
import { beforeAll, beforeEach, expect, test } from 'vitest';
import { beginWeightEdit } from '../src/canvas/weightEdit.js';
import { loadEngine } from '../src/engine.js';
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
  expect(probabilitySentence(g, r(NaN, NaN))).toBeNull();
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
