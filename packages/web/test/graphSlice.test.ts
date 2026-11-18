import { beforeEach, expect, test } from 'vitest';
import type { Graph } from '@tinker/core';
import { initialGraph } from '../src/state/graphSlice.js';
import { useStore } from '../src/state/store.js';

function withLr(g: Graph, lr: number): Graph {
  return { ...g, train: { ...g.train, lr } };
}

const s = () => useStore.getState();

beforeEach(() => {
  s().loadGraph(initialGraph());
});

test('initial graph is the SPEC example', () => {
  expect(s().graph.nodes.map((n) => n.id)).toEqual([
    'x',
    'lin1',
    'act1',
    'lin2',
    'yhat',
    'y',
    'loss',
  ]);
  expect(s().canUndo).toBe(false);
  expect(s().canRedo).toBe(false);
});

test('apply pushes history; undo and redo walk it', () => {
  const g0 = s().graph;
  const g1 = withLr(g0, 0.1);
  const g2 = withLr(g0, 0.2);
  expect(s().apply({ ok: true, graph: g1 }).ok).toBe(true);
  s().apply({ ok: true, graph: g2 });
  expect(s().graph).toBe(g2);
  expect(s().canUndo).toBe(true);

  s().undo();
  expect(s().graph).toBe(g1);
  expect(s().canRedo).toBe(true);
  s().undo();
  expect(s().graph).toBe(g0);
  expect(s().canUndo).toBe(false);
  s().undo(); // no-op at the start
  expect(s().graph).toBe(g0);

  s().redo();
  s().redo();
  expect(s().graph).toBe(g2);
  expect(s().canRedo).toBe(false);
  s().redo(); // no-op at the end
  expect(s().graph).toBe(g2);
});

test('a new edit after undo clears redo', () => {
  const g0 = s().graph;
  s().apply({ ok: true, graph: withLr(g0, 0.1) });
  s().undo();
  expect(s().canRedo).toBe(true);
  s().apply({ ok: true, graph: withLr(g0, 0.5) });
  expect(s().canRedo).toBe(false);
  s().redo();
  expect(s().graph.train.lr).toBe(0.5);
});

test('an error result leaves graph and history alone and sets the message', () => {
  const g0 = s().graph;
  const r = s().apply({ ok: false, error: 'This piece is locked.' });
  expect(r.ok).toBe(false);
  expect(s().graph).toBe(g0);
  expect(s().canUndo).toBe(false);
  expect(s().error).toBe('This piece is locked.');
  s().apply({ ok: true, graph: withLr(g0, 0.1) });
  expect(s().error).toBeNull();
});

test('apply returns the result so callers can read select', () => {
  const r = s().apply({ ok: true, graph: withLr(s().graph, 0.1), select: 'lin1' });
  expect(r.ok && r.select).toBe('lin1');
});

test('loadGraph replaces the graph and clears history', () => {
  s().apply({ ok: true, graph: withLr(s().graph, 0.1) });
  s().loadGraph(initialGraph());
  expect(s().canUndo).toBe(false);
  expect(s().canRedo).toBe(false);
});
