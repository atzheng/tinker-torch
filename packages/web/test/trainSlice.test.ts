import { beforeEach, expect, test } from 'vitest';
import { setLinearOut } from '@tinker/core';
import { initialGraph } from '../src/state/graphSlice.js';
import { isMiniBatch } from '../src/state/trainSlice.js';
import { useStore } from '../src/state/store.js';

const s = () => useStore.getState();

beforeEach(() => {
  s().loadGraph(initialGraph());
});

test('host and graph change in the same store update', () => {
  const seen: { out: number; wLen: number | null }[] = [];
  const unsub = useStore.subscribe(() => {
    const st = s(); // current state, as React reads it (the listener argument can be stale)
    const lin = st.graph.nodes.find((n) => n.id === 'lin1');
    if (lin?.type !== 'linear' || !st.host) return;
    seen.push({ out: lin.out, wLen: st.host.lin1?.W?.length ?? null });
  });
  const r = setLinearOut(s().graph, 'lin1', 6);
  s().apply(r);
  unsub();
  expect(seen.length).toBeGreaterThan(0);
  for (const x of seen) expect(x.wLen).toBe(x.out); // in = 1
});

test('optimizer and batch are graph.train changes with one undo step each', () => {
  s().setOptimizer('adam');
  s().setBatch(7);
  expect(s().graph.train).toMatchObject({ optimizer: 'adam', batch: 7 });
  s().undo();
  expect(s().graph.train).toMatchObject({ optimizer: 'adam', batch: 'full' });
  s().undo();
  expect(s().graph.train.optimizer).toBe('sgd');
  s().redo();
  expect(s().graph.train.optimizer).toBe('adam');
});

test('a batch size at or above nTrain counts as full batch', () => {
  expect(isMiniBatch('full', 28)).toBe(false);
  expect(isMiniBatch(28, 28)).toBe(false);
  expect(isMiniBatch(16, 28)).toBe(true);
  s().setBatch(64);
  s().setData({ n: 100, val: 0.3 }); // nTrain 70
  expect(isMiniBatch(s().graph.train.batch, s().dataset.nTrain)).toBe(true);
  s().setData({ n: 40, val: 0.3 }); // nTrain 28
  expect(isMiniBatch(s().graph.train.batch, s().dataset.nTrain)).toBe(false);
});

test('an invalid graph clears host and disables training', () => {
  const g = s().graph;
  s().apply({ ok: true, graph: { ...g, edges: g.edges.slice(1) } });
  expect(s().host).toBeNull();
  expect(s().canTrain).toBe(false);
  expect(s().trainIssue).toBeTruthy();
});
