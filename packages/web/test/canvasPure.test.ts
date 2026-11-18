import { expect, test } from 'vitest';
import { analyse, sample } from '@tinker/core';
import { initialGraph } from '../src/state/graphSlice.js';
import {
  edgeIndexOf,
  errorNodes,
  fitTransform,
  heatMap,
  hitPad,
  placeInspector,
  placeIssues,
  resolvedEdges,
  wireOfNode,
} from '../src/canvas/pure.js';

test('fitTransform centres and never exceeds maxScale', () => {
  const t = fitTransform([400, 200], { w: 1000, h: 800 }, 20, 1.4);
  expect(t.scale).toBe(1.4);
  expect(t.tx).toBeCloseTo((1000 - 400 * 1.4) / 2);
  const small = fitTransform([2000, 200], { w: 1000, h: 800 }, 20);
  expect(small.scale).toBeCloseTo(960 / 2000);
});

test('hitPad grows small shapes to 32 screen px and leaves big ones alone', () => {
  expect(hitPad(10, 0.5)).toBe((64 - 10) / 2); // 32 px at scale 0.5 is 64 units
  expect(hitPad(112, 1)).toBe(0);
  expect(hitPad(20, 1) * 2 + 20).toBe(32);
});

test('placeIssues: wire issues on the wire, unresolved edges fall back to node then global', () => {
  const placed = placeIssues(
    [
      { at: { edge: 1, node: 'add' }, message: 'mismatch' },
      { at: { edge: 7 }, message: 'bad edge, no wire' },
      { at: { edge: 8, node: 'lin' }, message: 'bad edge at node' },
      { at: { node: 'lin' }, message: 'node only' },
      { at: {}, message: 'global' },
      { at: { node: 'ghost' }, message: 'unknown node' },
    ],
    new Set([1]),
    new Set(['add', 'lin']),
  );
  expect([...placed.byEdge]).toEqual([[1, ['mismatch']]]);
  expect(placed.byNode.get('lin')).toEqual(['bad edge at node', 'node only']);
  expect(placed.global).toEqual(['bad edge, no wire', 'global', 'unknown node']);
});

test('placeIssues puts a node issue on its wire when wireOfNode names one', () => {
  const placed = placeIssues(
    [{ at: { node: 'yhat' }, message: 'width' }],
    new Set([4]),
    new Set(['yhat']),
    () => 4,
  );
  expect([...placed.byEdge]).toEqual([[4, ['width']]]);
  expect(placed.byNode.size).toBe(0);
});

test('wireOfNode: an Output points at its wire into the loss, other pieces at none', () => {
  const g = initialGraph();
  expect(wireOfNode(g, 'yhat')).toBe(4);
  expect(wireOfNode(g, 'lin1')).toBeUndefined();
  expect(wireOfNode(g, 'ghost')).toBeUndefined();
});

test('placeInspector goes below when there is room, above otherwise, and stays in view', () => {
  const view = { w: 700, h: 600 };
  const below = placeInspector({ cx: 100, top: 100, bottom: 130 }, view, 248);
  expect(below).toEqual({ left: 4, top: 146 });
  const above = placeInspector({ cx: 690, top: 450, bottom: 480 }, view, 248);
  expect(above).toEqual({ left: 448, bottom: 600 - 434 });
});

test('errorNodes lists nodes named by any issue', () => {
  expect(
    errorNodes([
      { at: { edge: 1, node: 'a' }, message: 'm' },
      { at: {}, message: 'g' },
    ]),
  ).toEqual(new Set(['a']));
});

test('resolvedEdges skips edges that do not resolve', () => {
  const g = initialGraph();
  const bad = { ...g, edges: [...g.edges, ['x', 'ghost'] as [string, string]] };
  expect(resolvedEdges(bad).has(g.edges.length)).toBe(false);
  expect(resolvedEdges(bad).size).toBe(g.edges.length);
  void analyse(bad, sample(g.data).info);
});

test('edgeIndexOf follows a wire by its endpoints after indices shift', () => {
  const g = initialGraph();
  const shifted = { ...g, edges: [['x', 'lin1'] as [string, string], ...g.edges] };
  expect(edgeIndexOf(g, 'lin1', 'act1')).toBe(1);
  expect(edgeIndexOf(shifted, 'lin1', 'act1')).toBe(2);
  expect(edgeIndexOf(g, 'nope', 'act1')).toBe(-1);
});

test('heatMap averages blocks, caps the size, and does not touch W', () => {
  const W = Float32Array.from({ length: 4 * 20 }, (_, k) => (k % 2 ? -1 : 1) * (k / 80));
  const copy = Float32Array.from(W);
  const h = heatMap(W, 4, 20, 10);
  expect(h).not.toBeNull();
  expect(h?.rows).toBe(4);
  expect(h?.cols).toBe(10);
  expect(h?.values).toHaveLength(40);
  expect(Array.from(W)).toEqual(Array.from(copy));
  expect(heatMap(W, 3, 20)).toBeNull(); // shape does not match length
  const one = heatMap(Float32Array.of(2, -4), 1, 2);
  expect(one?.values).toEqual([2, -4]);
  expect(one?.maxAbs).toBe(4);
});
