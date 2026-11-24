/** Fan-out (tinker-0sq): one output port feeding any number of wires, through analyse, edit and layout. */
import { describe, expect, test } from 'vitest';
import {
  analyse,
  connect,
  deleteEdge,
  deleteNode,
  insertOnWire,
  layoutLayers,
  layoutNeurons,
  moveNodes,
  parseGraph,
  rewireEdge,
  resolveEdge,
  type DataInfo,
  type Graph,
} from '../src/index.js';

const info: DataInfo = { columns: ['x1', 'x2'], kind: 'regression', numClasses: 1 };

/** x -> h (linear 1) -> {a1, a2, a3} (tanh) -> s (sum); h also feeds the prediction. */
function fan(n = 3): Graph {
  const acts = Array.from({ length: n }, (_, i) => `a${i + 1}`);
  const r = parseGraph({
    version: 2,
    nodes: [
      { id: 'x', type: 'input', columns: ['x1', 'x2'] },
      { id: 'h', type: 'linear', out: 1, bias: true },
      ...acts.map((id) => ({ id, type: 'activation', fn: 'tanh' })),
      { id: 's', type: 'sum' },
    ],
    edges: [['x', 'h'], ...acts.map((a) => ['h', a]), ...acts.map((a) => [a, 's'])],
    predictions: ['s'],
    loss: null,
    data: { set: 'sine', n: 40, noise: 0.2, val: 0.3, seed: 7 },
    train: { optimizer: 'sgd', lr: 0.03, batch: 'full', seed: 11 },
  });
  if (!r.ok) throw new Error(r.error);
  return r.graph;
}

const pillH = 24;
const labelW = 36;

function labelsApart(l: ReturnType<typeof layoutLayers>, edges: number[]) {
  const at = edges.map((e) => l.wires.find((w) => w.edge === e)?.labelAt as [number, number]);
  for (let i = 0; i < at.length; i++) {
    for (let j = i + 1; j < at.length; j++) {
      const a = at[i] as [number, number];
      const b = at[j] as [number, number];
      expect(Math.abs(a[0] - b[0]) >= labelW || Math.abs(a[1] - b[1]) >= pillH).toBe(true);
    }
  }
}

describe('analyse and edit', () => {
  test('a node feeding several consumers is valid and every wire resolves', () => {
    const g = fan(4);
    expect(analyse(g, info).issues.filter((i) => !i.message.startsWith('No loss'))).toEqual([]);
    expect(g.edges.filter((e) => resolveEdge(g.nodes, e).ok)).toHaveLength(g.edges.length);
  });

  test('connect from an output that already has wires adds one and keeps the others', () => {
    const g = fan(3);
    const added = parseGraph({
      ...g,
      nodes: [...g.nodes, { id: 'a4', type: 'activation', fn: 'relu' }],
    });
    if (!added.ok) throw new Error(added.error);
    const r = connect(added.graph, 'h', 'a4');
    if (!r.ok) throw new Error(r.error);
    expect(r.graph.edges.filter((e) => e[0] === 'h')).toHaveLength(4);
    expect(r.graph.edges).toHaveLength(added.graph.edges.length + 1);
    expect(connect(r.graph, 'h', 'a4').ok).toBe(false);
  });

  test('rewire, insert and delete act on one wire of a fanned-out output', () => {
    const g = fan(3);
    const i2 = g.edges.findIndex((e) => e[0] === 'h' && e[1] === 'a2');
    const ins = insertOnWire(g, i2, { type: 'activation', fn: 'relu' });
    if (!ins.ok) throw new Error(ins.error);
    expect(ins.graph.edges.filter((e) => e[0] === 'h')).toHaveLength(3);
    expect(ins.graph.edges.some((e) => e[0] === 'h' && e[1] === 'a2')).toBe(false);
    expect(ins.graph.edges.some((e) => e[0] === 'h' && e[1] === 'a1')).toBe(true);
    expect(ins.graph.edges.some((e) => e[0] === 'h' && e[1] === 'a3')).toBe(true);

    const del = deleteEdge(g, i2);
    if (!del.ok) throw new Error(del.error);
    expect(del.graph.edges.filter((e) => e[0] === 'h').map((e) => e[1])).toEqual(['a1', 'a3']);

    // rewire the destination end of one wire: the others stay
    const withB = parseGraph({ ...g, nodes: [...g.nodes, { id: 'b', type: 'bias' }] });
    if (!withB.ok) throw new Error(withB.error);
    const rw = rewireEdge(withB.graph, i2, 'to', 'b');
    if (!rw.ok) throw new Error(rw.error);
    expect(
      rw.graph.edges
        .filter((e) => e[0] === 'h')
        .map((e) => e[1])
        .sort(),
    ).toEqual(['a1', 'a3', 'b']);
  });

  test('deleting one consumer of a fanned-out output leaves the other wires', () => {
    const g = fan(3);
    const r = deleteNode(g, 'a2');
    if (!r.ok) throw new Error(r.error);
    expect(r.graph.edges.filter((e) => e[0] === 'h').map((e) => e[1])).toEqual(['a1', 'a3', 's']);
  });
});

describe('layout', () => {
  test('wires leaving one port: labels do not overlap and no segment enters a box', () => {
    for (const n of [3, 5]) {
      const g = fan(n);
      const a = analyse(g, info);
      const l = layoutLayers(g, a);
      const out = g.edges.flatMap((e, i) => (e[0] === 'h' ? [i] : []));
      expect(out).toHaveLength(n);
      labelsApart(l, out);
      for (const w of l.wires) {
        for (let k = 0; k + 1 < w.points.length; k++) {
          const p = w.points[k] as [number, number];
          const q = w.points[k + 1] as [number, number];
          for (const [id, b] of Object.entries(l.nodes)) {
            const hit =
              Math.max(p[0], q[0]) > b.x &&
              Math.min(p[0], q[0]) < b.x + b.w &&
              Math.max(p[1], q[1]) > b.y &&
              Math.min(p[1], q[1]) < b.y + b.h;
            // a wire may touch only its own endpoints' box edges, never the interior
            expect(hit, `wire ${w.edge} crosses ${id}`).toBe(false);
          }
        }
      }
    }
  });

  test('with manual positions the labels still do not overlap', () => {
    const g = fan(3);
    const moved = moveNodes(g, { a1: [300, 200], a2: [300, 20] });
    if (!moved.ok) throw new Error(moved.error);
    const l = layoutLayers(moved.graph, analyse(moved.graph, info));
    labelsApart(
      l,
      g.edges.flatMap((e, i) => (e[0] === 'h' ? [i] : [])),
    );
  });

  test('wires out of one output into Weight boxes do not overlap', () => {
    const r = parseGraph({
      version: 2,
      nodes: [
        { id: 'x1', type: 'column', name: 'x1' },
        { id: 'w1', type: 'scalew' },
        { id: 'w2', type: 'scalew' },
        { id: 'sum', type: 'sum' },
        { id: 'p1', type: 'activation', fn: 'tanh' },
        { id: 'p2', type: 'activation', fn: 'relu' },
      ],
      edges: [
        ['x1', 'w1'],
        ['x1', 'w2'],
        ['w1', 'sum'],
        ['w2', 'sum'],
        ['x1', 'p1'],
        ['x1', 'p2'],
      ],
      predictions: ['sum'],
      loss: null,
      data: { set: 'sine', n: 40, noise: 0.2, val: 0.3, seed: 7 },
      train: { optimizer: 'sgd', lr: 0.03, batch: 'full', seed: 11 },
    });
    if (!r.ok) throw new Error(r.error);
    const l = layoutLayers(r.graph, analyse(r.graph, info));
    expect(Object.keys(l.nodes)).toContain('w1');
    // labels are about 24 by 17
    const items = l.wires.map((w) => ({
      at: w.labelAt,
      hw: 12,
      hh: 9,
    }));
    for (let i = 0; i < items.length; i++) {
      for (let j = i + 1; j < items.length; j++) {
        const a = items[i];
        const b = items[j];
        if (!a || !b) continue;
        const apart =
          Math.abs(a.at[0] - b.at[0]) >= a.hw + b.hw || Math.abs(a.at[1] - b.at[1]) >= a.hh + b.hh;
        expect(apart, `labels ${i} and ${j}`).toBe(true);
      }
    }
  });

  test('neuron view keeps one wire per fanned-out edge, labels apart', () => {
    const g = fan(3);
    const n = layoutNeurons(g, analyse(g, info));
    const out = g.edges.flatMap((e, i) => (e[0] === 'h' ? [i] : []));
    expect(n.wires.filter((w) => out.includes(w.edge))).toHaveLength(3);
    labelsApart({ ...n, nodes: n.boxes } as ReturnType<typeof layoutLayers>, out);
  });
});
