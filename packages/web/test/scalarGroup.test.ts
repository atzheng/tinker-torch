// @vitest-environment jsdom
/** Group into Linear / Expand to scalars in the store (tinker-9io.7): one undo step, params carried. */
import { beforeAll, beforeEach, expect, test } from 'vitest';
import { loadEngine } from '../src/engine.js';
import { parseGraph, type Edge, type Graph, type Node } from '@tinker/core';
import { expandSelected, groupSelected, pickGroup } from '../src/state/scalarActions.js';
import { useStore } from '../src/state/store.js';

const s = () => useStore.getState();

function scalarGraph(): Graph {
  const nodes: Node[] = [];
  const edges: Edge[] = [];
  ['x1', 'x2'].forEach((nm, i) => {
    nodes.push({ id: `c${i + 1}`, type: 'column', name: nm }, { id: `w${i + 1}`, type: 'scalew' });
    edges.push([`c${i + 1}`, `w${i + 1}`], [`w${i + 1}`, 'sum']);
  });
  nodes.push(
    { id: 'sum', type: 'sum' },
    { id: 'b', type: 'bias' },
    { id: 'y', type: 'target' },
    { id: 'diff', type: 'subtract' },
    { id: 'sq', type: 'square' },
    { id: 'm', type: 'mean' },
  );
  edges.push(['sum', 'b'], ['b', 'diff.a'], ['y', 'diff.b'], ['diff', 'sq'], ['sq', 'm']);
  const r = parseGraph({
    version: 2,
    nodes,
    edges,
    predictions: ['b'],
    loss: 'm',
    data: { set: 'plane', n: 60, noise: 0.1, val: 0.2, seed: 1, standardize: false },
    train: { optimizer: 'sgd', lr: 0.01, batch: 'full', seed: 1 },
  });
  if (!r.ok) throw new Error(r.error);
  return r.graph;
}

/** Every parameter of every layer, flattened to a sorted list of "node.key=values". */
function flat(): string[] {
  const h = s().host ?? {};
  const out: string[] = [];
  for (const [id, layer] of Object.entries(h)) {
    for (const [k, v] of Object.entries(layer)) out.push(`${id}.${k}=${Array.from(v ?? [])}`);
  }
  return out.sort();
}

/** The predictions of the model on the training inputs, through the store's own predictor. */
function fit(): number[] {
  return Array.from(s().predictGrid(Float32Array.of(0.1, 0.2, 0.5, 0.4, 0.9, 0.7), 3) ?? []);
}

beforeAll(async () => {
  await loadEngine();
}, 60000);

beforeEach(() => {
  s().pause();
  s().loadGraph(scalarGraph());
  s().setFresh(new Set());
});

test('group then expand: loss and fit are unchanged; each is one undo step', () => {
  for (let i = 0; i < 5; i++) s().stepOnce();
  const loss = s().trainLoss;
  const val = s().valLoss;
  const steps = s().step;
  const params0 = flat();
  expect(steps).toBe(5);

  expect(groupSelected()).toBe(true);
  expect(s().error).toBeNull();
  expect(s().graph.nodes.some((n) => n.type === 'linear')).toBe(true);
  expect(s().graph.nodes.some((n) => n.type === 'scalew')).toBe(false);
  expect(s().step).toBe(steps); // the run carries on
  expect(s().trainLoss).toBe(loss);
  expect(s().valLoss).toBe(val);
  const linId = s().graph.nodes.find((n) => n.type === 'linear')?.id ?? '';
  expect(s().selection).toEqual({ kind: 'node', id: linId });
  const grouped = flat();
  expect(grouped).toHaveLength(2); // W and b of the Linear
  const fitGrouped = fit();

  expect(expandSelected()).toBe(true);
  expect(s().graph.nodes.some((n) => n.type === 'scalew')).toBe(true);
  expect(s().trainLoss).toBe(loss);
  expect(s().step).toBe(steps);
  expect(fit()).toEqual(fitGrouped);

  // The next step from the expanded graph continues the same run.
  s().stepOnce();
  expect(s().step).toBe(steps + 1);

  // Undo the step-free edits: one undo each, graph and params together.
  s().undo(); // expand
  expect(s().graph.nodes.some((n) => n.type === 'linear')).toBe(true);
  expect(flat()).toEqual(grouped);
  s().undo(); // group
  expect(s().graph.nodes.some((n) => n.type === 'scalew')).toBe(true);
  expect(flat()).toEqual(params0);
  expect(s().canUndo).toBe(false);
  s().redo();
  expect(flat()).toEqual(grouped);
  s().redo();
  expect(s().graph.nodes.some((n) => n.type === 'scalew')).toBe(true);
});

test('grouping while running keeps the run going', () => {
  s().run();
  expect(s().running).toBe(true);
  expect(groupSelected()).toBe(true);
  expect(s().running).toBe(true);
  s().pause();
});

test('a refused edit is one sentence in the toast slot and changes nothing', () => {
  const g = s().graph;
  s().setPreset({ title: 't', prompt: '', locked: ['sum'] });
  expect(groupSelected()).toBe(false);
  expect(s().error).toMatch(/locked|cannot|not/i);
  expect(s().graph).toBe(g);
  s().setPreset(null);
});

test('pickGroup prefers the group holding the selection', () => {
  const groups = [
    ['a', 'b'],
    ['c', 'd'],
  ];
  expect(pickGroup(groups, { kind: 'node', id: 'd' })).toEqual(['c', 'd']);
  expect(pickGroup(groups, null)).toEqual(['a', 'b']);
  expect(pickGroup([], null)).toBeNull();
});

test('undo and redo after more training return step, losses and history with the weights', () => {
  for (let i = 0; i < 3; i++) s().stepOnce();
  const edit = { step: s().step, hist: s().history.length };
  groupSelected();
  for (let i = 0; i < 4; i++) s().stepOnce();
  expect(s().step).toBe(edit.step + 4);
  const trained = s().trainLoss;
  s().undo();
  expect(s().step).toBe(edit.step);
  expect(s().history.length).toBe(edit.hist);
  expect(s().history.at(-1)?.step).toBe(edit.step);
  expect(s().trainLoss).not.toBe(trained);
  const lossBack = s().trainLoss;
  expect(Number.isFinite(lossBack)).toBe(true);
  s().redo();
  expect(s().step).toBe(edit.step);
  expect(s().trainLoss).toBe(lossBack); // same weights, same loss
});

test('undo back to an earlier step drops the log rows after it', () => {
  useStore.setState({ show: { ui: ['panel-step-log'] } });
  for (let i = 0; i < 3; i++) s().stepOnce();
  groupSelected();
  for (let i = 0; i < 4; i++) s().stepOnce();
  expect(s().stepLog.length).toBeGreaterThan(0);
  s().undo();
  expect(s().stepLog.every((r) => r.step <= s().step)).toBe(true);
});

test('the carried weights are the exact current ones (no lost steps)', () => {
  for (let i = 0; i < 3; i++) s().stepOnce();
  const exact = s().currentHost();
  expect(exact).not.toBeNull();
  groupSelected();
  const lin = s().graph.nodes.find((n) => n.type === 'linear')?.id ?? '';
  expect(Array.from(s().host?.[lin]?.['b'] ?? [])).toEqual(Array.from(exact?.['b']?.['b'] ?? []));
});

test('group is refused after a divergence', () => {
  s().setLearningRate(1e30);
  for (let i = 0; i < 200 && !s().diverged; i++) s().stepOnce();
  expect(s().diverged).toBe(true);
  const g = s().graph;
  expect(groupSelected()).toBe(false);
  expect(s().graph).toBe(g);
  expect(s().error).toMatch(/Reset first/);
});
