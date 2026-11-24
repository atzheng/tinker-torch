// @vitest-environment jsdom
// The Tree piece in the store (tinker-6h1.4): fitted on the host before analysis, refits on a data or
// tree-setting change and re-initialises the weights, never on a move.
import { beforeAll, beforeEach, expect, test } from 'vitest';
import { moveNodes, parseGraph, setSetting, type Graph } from '@tinker/core';
import { loadEngine } from '../src/engine.js';
import { useStore } from '../src/state/store.js';
import { treeFitsOf } from '../src/state/treeFits.js';

const s = () => useStore.getState();

function treeGraph(depth = 3, standardize = true): Graph {
  const r = parseGraph({
    version: 2,
    nodes: [
      { id: 'x', type: 'input', columns: ['x1'] },
      {
        id: 'tree',
        type: 'tree',
        criterion: 'squared_error',
        depth,
        minSplit: 2,
        minLeaf: 1,
        minDecrease: 0,
      },
      { id: 'lin1', type: 'linear', out: 1, bias: true },
      { id: 'y', type: 'target' },
      { id: 'diff', type: 'subtract' },
      { id: 'err', type: 'square' },
      { id: 'loss', type: 'mean' },
    ],
    edges: [
      ['x', 'tree'],
      ['tree', 'lin1'],
      ['lin1', 'diff.a'],
      ['y', 'diff.b'],
      ['diff', 'err'],
      ['err', 'loss'],
    ],
    predictions: ['lin1'],
    loss: 'loss',
    data: { set: 'trend_sine', n: 43, noise: 0.6466, val: 0.3, seed: 3, standardize },
    train: { optimizer: 'adam', lr: 0.05, batch: 'full', seed: 1 },
  });
  if (!r.ok) throw new Error(r.error);
  return r.graph;
}

beforeAll(async () => {
  await loadEngine();
}, 60000);

beforeEach(() => {
  s().pause();
  s().loadGraph(treeGraph());
});

const leaves = () => s().analysis.inWidth['lin1'];

test('the Linear after a Tree reads its in-width from the fitted leaf count', () => {
  expect(s().canTrain).toBe(true);
  const fit = treeFitsOf(s().graph, s().dataset)['tree'];
  expect(fit?.leaves).toBeGreaterThan(1);
  expect(leaves()).toBe(fit?.leaves);
  expect(s().analysis.outWidth['tree']).toBe(fit?.leaves);
  expect(s().host?.lin1?.W?.length).toBe(fit?.leaves);
});

test('a depth change refits and resizes; the same leaf count still resets the weights', () => {
  const before = leaves();
  s().apply(setSetting(s().graph, 'tree', 'depth', 6));
  expect(leaves()).toBeGreaterThan(before ?? 0);
  expect(s().host?.lin1?.W?.length).toBe(leaves());
  // train a little, then a setting change that leaves the fit identical: weights start again
  s().reset();
  const start = Array.from(s().host?.lin1?.W ?? []);
  for (let i = 0; i < 5; i++) s().stepOnce();
  expect(Array.from(s().host?.lin1?.W ?? [])).not.toEqual(start);
  s().apply(setSetting(s().graph, 'tree', 'minDecrease', 1e-12));
  expect(Array.from(s().host?.lin1?.W ?? [])).toEqual(start);
});

test('undo of a tree setting refits', () => {
  const before = leaves();
  s().apply(setSetting(s().graph, 'tree', 'depth', 6));
  expect(leaves()).not.toBe(before);
  s().undo();
  expect(leaves()).toBe(before);
  expect(s().host?.lin1?.W?.length).toBe(before);
});

test('a data change refits (Resample, n); a move does not', () => {
  const a = treeFitsOf(s().graph, s().dataset);
  s().resample();
  const b = treeFitsOf(s().graph, s().dataset);
  expect(b).not.toBe(a);
  expect(s().host?.lin1?.W?.length).toBe(leaves());
  s().setData({ n: 80 });
  expect(treeFitsOf(s().graph, s().dataset)).not.toBe(b);
  expect(s().host?.lin1?.W?.length).toBe(leaves());
  const c = treeFitsOf(s().graph, s().dataset);
  const moved = moveNodes(s().graph, { tree: [2, 2] });
  s().apply(moved);
  expect(treeFitsOf(s().graph, s().dataset)).toBe(c);
});

test('a Tree not wired from an Input explains itself and disables training', () => {
  const g = s().graph;
  s().apply({ ok: true, graph: { ...g, edges: g.edges.slice(1) } });
  expect(s().canTrain).toBe(false);
  expect(s().trainIssue).toBeTruthy();
});

test('a depth-8 tree on trend_sine is driven to near zero training loss with Adam', () => {
  s().apply(setSetting(s().graph, 'tree', 'depth', 8));
  s().reset();
  const lr = 0.1;
  s().setLearningRate(lr);
  for (let i = 0; i < 400; i++) s().stepOnce();
  const { trainLoss, valLoss } = s();
  expect(trainLoss).not.toBeNull();
  expect(trainLoss ?? 1).toBeLessThan(0.02);
  expect(valLoss ?? 0).toBeGreaterThan(trainLoss ?? 1);
});

test('a locked Tree refuses setting edits unless its field id is listed in show.ui', async () => {
  const { editSettingOpen } = await import('../src/canvas/drawings/fields.js');
  useStore.setState({
    preset: { title: 't', locked: ['tree'] } as never,
    show: { ui: ['tree-min-leaf'] },
  });
  editSettingOpen('tree', 'depth', 5, 'tree-depth');
  expect(s().graph.nodes.find((n) => n.id === 'tree')).toMatchObject({ depth: 3 });
  editSettingOpen('tree', 'minLeaf', 4, 'tree-min-leaf');
  expect(s().graph.nodes.find((n) => n.id === 'tree')).toMatchObject({ minLeaf: 4 });
  useStore.setState({ preset: null, show: null });
});
