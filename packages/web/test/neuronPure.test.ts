import { expect, test } from 'vitest';
import {
  analyse,
  layoutNeurons,
  LAYOUT,
  parseGraph,
  sample,
  type Graph,
  type HostParams,
} from '@tinker/core';
import { mlpGraph as initialGraph } from './mlpGraph.js';
import {
  distToSegment,
  edgeStyle,
  nearestEdge,
  neuronHitShapes,
  pickNeuron,
  trimSegment,
  weightAt,
} from '../src/canvas/neuronPure.js';
import { fitTransform } from '../src/canvas/pure.js';

function graphWith(hidden: number[]): Graph {
  const nodes: unknown[] = [{ id: 'x', type: 'input', columns: ['x1'] }];
  const edges: string[][] = [];
  let prev = 'x';
  hidden.forEach((h, k) => {
    nodes.push({ id: `l${k}`, type: 'linear', out: h, bias: true });
    nodes.push({ id: `a${k}`, type: 'activation', fn: 'tanh' });
    edges.push([prev, `l${k}`], [`l${k}`, `a${k}`]);
    prev = `a${k}`;
  });
  nodes.push({ id: 'last', type: 'linear', out: 1, bias: true });
  nodes.push({ id: 'yhat', type: 'output', link: 'identity' });
  nodes.push({ id: 'y', type: 'target' });
  nodes.push({ id: 'loss', type: 'loss', fn: 'squared_error' });
  edges.push([prev, 'last'], ['last', 'yhat'], ['yhat', 'loss.pred'], ['y', 'loss.target']);
  const parsed = parseGraph({
    version: 1,
    nodes,
    edges,
    data: { set: 'sine', n: 40, noise: 0.2, val: 0.3, seed: 7 },
    train: { optimizer: 'sgd', lr: 0.03, batch: 'full', seed: 11 },
  });
  if (!parsed.ok) throw new Error(parsed.error);
  return parsed.graph;
}

const info = sample(initialGraph().data).info;

test('edgeStyle: sign picks the colour, size picks the thickness, bad values are safe', () => {
  expect(edgeStyle(0.5).sign).toBe('pos');
  expect(edgeStyle(-0.5).sign).toBe('neg');
  expect(edgeStyle(-1).width).toBeGreaterThan(edgeStyle(-0.2).width);
  expect(edgeStyle(100).width).toBe(edgeStyle(1.5).width);
  expect(Number.isFinite(edgeStyle(NaN).width)).toBe(true);
});

test('every edge maps to one entry of a hand-built HostParams (W row-major [in, out])', () => {
  const g = graphWith([4]);
  const a = analyse(g, info);
  const layout = layoutNeurons(g, a);
  // W[i][j] = 10 * i + j, so a wrong index shows up as a wrong number.
  const host: HostParams = {
    l0: { W: Float32Array.from([0, 1, 2, 3]), b: new Float32Array(4) },
    last: { W: Float32Array.from([0, 10, 20, 30]), b: new Float32Array(1) },
  };
  const l0 = layout.edges.filter((e) => e.node === 'l0');
  expect(l0).toHaveLength(4);
  for (const [k, e] of l0.entries()) {
    expect(e.i * 4 + e.j).toBe(k); // ordered i-major, then j
    expect(weightAt(host, 'l0', e.i, e.j, 1, 4)).toBe(e.i * 4 + e.j);
  }
  const last = layout.edges.filter((e) => e.node === 'last');
  expect(last).toHaveLength(4);
  for (const e of last) expect(weightAt(host, 'last', e.i, e.j, 4, 1)).toBe(10 * e.i);
});

test('weightAt: missing entries and wrong sizes give undefined, never a throw', () => {
  const host: HostParams = { l: { W: new Float32Array(5) } };
  expect(weightAt(undefined, 'l', 0, 0, 1, 5)).toBeUndefined();
  expect(weightAt(host, 'nope', 0, 0, 1, 5)).toBeUndefined();
  expect(weightAt(host, 'l', 0, 0, 2, 3)).toBeUndefined(); // stale shape
  expect(weightAt(host, 'l', 0, 5, 1, 5)).toBeUndefined();
});

test('trimSegment shortens both ends and leaves short segments alone', () => {
  expect(trimSegment([0, 0], [10, 0], 2, 3)).toEqual([2, 0, 7, 0]);
  expect(trimSegment([0, 0], [4, 0], 2, 3)).toEqual([0, 0, 4, 0]);
});

test('distToSegment and nearestEdge pick the closest line, ties to the earlier edge', () => {
  expect(distToSegment(5, 3, [0, 0], [10, 0])).toBe(3);
  expect(distToSegment(-4, 3, [0, 0], [10, 0])).toBe(5);
  const edges = [
    { node: 'n', i: 0, j: 0, from: [0, 0] as [number, number], to: [100, 0] as [number, number] },
    { node: 'n', i: 0, j: 1, from: [0, 10] as [number, number], to: [100, 10] as [number, number] },
  ];
  expect(nearestEdge(edges, 50, 7, 20)).toBe(1);
  expect(nearestEdge(edges, 50, 3, 20)).toBe(0);
  expect(nearestEdge(edges, 50, 5, 20)).toBe(0);
  expect(nearestEdge(edges, 50, 60, 20)).toBeUndefined();
});

test('at the smallest scale an 8-8-8 graph still has non-overlapping unit cells and 32 px pieces', () => {
  const g = graphWith([8, 8]);
  const a = analyse(g, info);
  const layout = layoutNeurons(g, a);
  const t = fitTransform(layout.size, { w: 1024, h: 600 });
  expect(t.scale).toBeLessThan(0.7);
  const shapes = neuronHitShapes(layout, t.scale, LAYOUT.unitPitch, LAYOUT.unitW);
  const units = shapes.filter((s) => s.kind === 'unit');
  expect(units).toHaveLength(layout.units.length);
  for (const u of units) expect(u.w * t.scale).toBeGreaterThanOrEqual(32 - 1e-9);
  for (const s of shapes.filter((x) => x.kind === 'node')) {
    expect(s.w * t.scale).toBeGreaterThanOrEqual(32 - 1e-9);
    expect(s.h * t.scale).toBeGreaterThanOrEqual(32 - 1e-9);
  }
  // cells of one column tile: neighbours touch, never overlap
  for (const u of units) {
    for (const v of units) {
      if (u === v || u.kind !== 'unit' || v.kind !== 'unit' || u.node !== v.node) continue;
      if (u.index + 1 === v.index) expect(u.y + u.h).toBeLessThanOrEqual(v.y + 1e-9);
    }
  }
});

test('pickNeuron: a unit wins over an edge, an edge within reach is picked, empty space is nothing', () => {
  const g = graphWith([4]);
  const layout = layoutNeurons(g, analyse(g, info));
  const t = fitTransform(layout.size, { w: 1024, h: 600 });
  const shapes = neuronHitShapes(layout, t.scale, LAYOUT.unitPitch, LAYOUT.unitW);
  const u = layout.units.find((x) => x.node === 'l0' && x.index === 2);
  expect(u).toBeDefined();
  if (!u) return;
  expect(pickNeuron(layout, shapes, t.scale, 32, u.x, u.y)).toEqual({
    kind: 'unit',
    node: 'l0',
    index: 2,
  });
  const e = layout.edges.find((x) => x.node === 'last' && x.i === 3);
  expect(e).toBeDefined();
  if (!e) return;
  const mid: [number, number] = [(e.from[0] + e.to[0]) / 2, (e.from[1] + e.to[1]) / 2];
  const got = pickNeuron(layout, shapes, t.scale, 32, mid[0], mid[1]);
  expect(got?.kind).toBe('edge');
  expect(pickNeuron(layout, shapes, t.scale, 32, -500, -500)).toBeUndefined();
});

test('layouts: 1-4-1 has 8 edges, a 16-wide layer has none (collapsed)', () => {
  const small = initialGraph();
  const ls = layoutNeurons(small, analyse(small, info));
  expect(ls.edges).toHaveLength(8);
  const wide = graphWith([16]);
  const lw = layoutNeurons(wide, analyse(wide, info));
  expect(lw.edges).toHaveLength(0);
  expect(lw.collapsed).toEqual(['l0', 'last']);
});
