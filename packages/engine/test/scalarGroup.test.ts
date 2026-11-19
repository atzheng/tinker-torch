// Group into Linear / Expand to scalars (core/scalar.ts): the grouped, expanded and re-grouped
// graphs, carrying the host params across, give the same predictions and loss.
import { installProbe } from '../src/probe.js';
import { expect, test, vi } from 'vitest';
import {
  analyse,
  expandToScalars,
  findLinearGroups,
  groupIntoLinear,
  parseGraph,
} from '@tinker/core';
import type { DataInfo, Edge, Graph, HostParams, Node } from '@tinker/core';
import { compare, predict0 } from './util.js';

installProbe();
vi.setConfig({ testTimeout: 120_000 });
const { createEngine } = await import('../src/index.js');
const engine = await createEngine();

const data: DataInfo = { columns: ['x1', 'x2'], kind: 'regression', numClasses: 1 };

/** Columns wired in non-dataset order (w1 reads x2) so the Linear's W is a permutation. */
function scalarGraph(): Graph {
  const nodes: Node[] = [
    { id: 'c1', type: 'column', name: 'x2' },
    { id: 'c2', type: 'column', name: 'x1' },
    { id: 'w1', type: 'scalew' },
    { id: 'w2', type: 'scalew' },
    { id: 'sum', type: 'sum' },
    { id: 'b', type: 'bias' },
    { id: 'y', type: 'target' },
    { id: 'diff', type: 'subtract' },
    { id: 'sq', type: 'square' },
    { id: 'm', type: 'mean' },
  ];
  const edges: Edge[] = [
    ['c1', 'w1'],
    ['c2', 'w2'],
    ['w1', 'sum'],
    ['w2', 'sum'],
    ['sum', 'b'],
    ['b', 'diff.a'],
    ['y', 'diff.b'],
    ['diff', 'sq'],
    ['sq', 'm'],
  ];
  const r = parseGraph({
    version: 2,
    nodes,
    edges,
    predictions: ['b'],
    loss: 'm',
    data: { set: 'plane', n: 100, noise: 0.1, val: 0.2, seed: 1, standardize: false },
    train: { optimizer: 'adam', lr: 0.05, batch: 'full', seed: 1 },
  });
  if (!r.ok) throw new Error(r.error);
  return r.graph;
}

const host = (): HostParams => {
  const h = Object.create(null) as HostParams;
  h['w1'] = { W: Float32Array.of(0.7) }; // on x2
  h['w2'] = { W: Float32Array.of(-1.3) }; // on x1
  h['b'] = { b: Float32Array.of(0.25) };
  return h;
};

function batch(n: number) {
  const x = new Float32Array(n * 2);
  const y = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    x[2 * i] = Math.sin(i + 1);
    x[2 * i + 1] = Math.cos(2 * i + 1);
    y[i] = 0.5 * (x[2 * i] ?? 0) - (x[2 * i + 1] ?? 0) + 0.1;
  }
  return { x, y, n };
}

function run(g: Graph, h: HostParams) {
  const b = batch(16);
  const model = engine.compile(g, data);
  const params = engine.fromHost(model, h);
  const pred = predict0(engine, model, params, b.x, b.n);
  const loss = engine.evaluate(model, params, b);
  engine.dispose(params);
  return { pred, loss };
}

test('group, expand, group keep predictions and loss, and the params bitwise', () => {
  const g0 = scalarGraph();
  const h0 = host();
  expect(analyse(g0, data).issues).toEqual([]);
  const base = run(g0, h0);

  const found = findLinearGroups(g0, analyse(g0, data));
  expect(found).toHaveLength(1);
  const a = groupIntoLinear(g0, h0, found[0] ?? []);
  if (!a.ok) throw new Error(a.error);
  expect(analyse(a.graph, data).issues).toEqual([]);
  const lin = run(a.graph, a.host);
  expect(compare(lin.pred, base.pred, 1e-6).ok).toBe(true);
  expect(Math.abs(lin.loss - base.loss)).toBeLessThan(1e-6);
  // W is in dataset order: [x1 weight, x2 weight]
  expect(Array.from(a.host['linear1']?.['W'] ?? [])).toEqual([Math.fround(-1.3), Math.fround(0.7)]);

  const e = expandToScalars(a.graph, a.host, 'linear1');
  if (!e.ok) throw new Error(e.error);
  expect(analyse(e.graph, data).issues).toEqual([]);
  const exp = run(e.graph, e.host);
  expect(compare(exp.pred, base.pred, 1e-6).ok).toBe(true);
  expect(Math.abs(exp.loss - base.loss)).toBeLessThan(1e-6);

  const again = findLinearGroups(e.graph, analyse(e.graph, data));
  const b = groupIntoLinear(e.graph, e.host, again[0] ?? []);
  if (!b.ok) throw new Error(b.error);
  expect(Array.from(b.host['linear1']?.['W'] ?? [])).toEqual(
    Array.from(a.host['linear1']?.['W'] ?? []),
  );
  expect(Array.from(b.host['linear1']?.['b'] ?? [])).toEqual(
    Array.from(a.host['linear1']?.['b'] ?? []),
  );
  const again2 = run(b.graph, b.host);
  expect(Math.abs(again2.loss - base.loss)).toBeLessThan(1e-6);
});
