import { describe, expect, test } from 'vitest';
import {
  analyse,
  layoutLayers,
  layoutNeurons,
  type DataInfo,
  type Edge,
  type Graph,
  type Node,
} from '../src/index.js';

const info: DataInfo = { columns: ['x1'], kind: 'regression', numClasses: 2 };

function g(nodes: Node[], edges: Edge[]): Graph {
  return {
    version: 1,
    nodes,
    edges,
    data: { set: 'line', n: 100, noise: 0.1, val: 0.2, seed: 1, standardize: false },
    train: { optimizer: 'adam', lr: 0.01, batch: 'full', seed: 1 },
  };
}
const lin = (id: string, out: number): Node => ({ id, type: 'linear', out, bias: true });
const act = (id: string): Node => ({ id, type: 'activation', fn: 'relu' });
const tail: Node[] = [
  { id: 'yhat', type: 'output', link: 'identity' },
  { id: 'y', type: 'target' },
  { id: 'loss', type: 'loss', fn: 'squared_error' },
];
const tailEdges: Edge[] = [
  ['yhat', 'loss.pred'],
  ['y', 'loss.target'],
];

function mlp(hidden: number): Graph {
  return g(
    [
      { id: 'x', type: 'input', columns: ['x1'] },
      lin('lin1', hidden),
      act('act1'),
      lin('lin2', 1),
      ...tail,
    ],
    [['x', 'lin1'], ['lin1', 'act1'], ['act1', 'lin2'], ['lin2', 'yhat'], ...tailEdges],
  );
}
function skip(): Graph {
  return g(
    [
      { id: 'x', type: 'input', columns: ['x1'] },
      lin('lin1', 4),
      act('act1'),
      lin('lin2', 4),
      { id: 'add', type: 'add' },
      lin('lin3', 1),
      ...tail,
    ],
    [
      ['x', 'lin1'],
      ['lin1', 'act1'],
      ['act1', 'lin2'],
      ['lin2', 'add.a'],
      ['lin1', 'add.b'],
      ['add', 'lin3'],
      ['lin3', 'yhat'],
      ...tailEdges,
    ],
  );
}

type Seg = [[number, number], [number, number]];
function interiorHit(s: Seg, b: { x: number; y: number; w: number; h: number }): boolean {
  const [[ax, ay], [bx, by]] = s;
  const x0 = Math.min(ax, bx);
  const x1 = Math.max(ax, bx);
  const y0 = Math.min(ay, by);
  const y1 = Math.max(ay, by);
  return x1 > b.x && x0 < b.x + b.w && y1 > b.y && y0 < b.y + b.h;
}

describe('layoutLayers', () => {
  test('SPEC example: columns x, lin1, act1, lin2, yhat, loss; y beside yhat', () => {
    const gr = mlp(4);
    const L = layoutLayers(gr, analyse(gr, info));
    const cols = ['x', 'lin1', 'act1', 'lin2', 'yhat', 'loss'].map((id) => L.nodes[id]?.col);
    expect(cols).toEqual([0, 1, 2, 3, 4, 5]);
    expect(L.nodes['y']?.col).toBe(4);
    const yhat = L.nodes['yhat'];
    const y = L.nodes['y'];
    expect(y && yhat && y.y).toBeGreaterThan((yhat?.y ?? 0) + (yhat?.h ?? 0) - 1);
    expect(y?.x).toBe(yhat?.x);
    expect(L.wires.map((w) => w.edge)).toEqual([0, 1, 2, 3, 4, 5]);
  });

  test('x positions increase with column and nothing is outside size', () => {
    const gr = mlp(4);
    const L = layoutLayers(gr, analyse(gr, info));
    for (const b of Object.values(L.nodes)) {
      expect(b.x).toBeGreaterThanOrEqual(0);
      expect(b.x + b.w).toBeLessThanOrEqual(L.size[0]);
      expect(b.y + b.h).toBeLessThanOrEqual(L.size[1]);
    }
  });

  test('a skip wire routes around node boxes', () => {
    const gr = skip();
    const L = layoutLayers(gr, analyse(gr, info));
    const edge = gr.edges.findIndex((e) => e[1] === 'add.b');
    const wire = L.wires.find((w) => w.edge === edge);
    expect(wire).toBeDefined();
    const pts = wire?.points ?? [];
    expect(pts.length).toBeGreaterThan(4);
    const minTop = Math.min(...Object.values(L.nodes).map((b) => b.y));
    expect(Math.min(...pts.map((p) => p[1]))).toBeLessThan(minTop);
    for (const w of L.wires) {
      for (let k = 0; k + 1 < w.points.length; k++) {
        const a = w.points[k];
        const b = w.points[k + 1];
        if (!a || !b) continue;
        for (const box of Object.values(L.nodes)) expect(interiorHit([a, b], box)).toBe(false);
      }
    }
    // the two inputs of add enter at different heights
    const a = L.wires.find((w) => gr.edges[w.edge]?.[1] === 'add.a')?.points.at(-1);
    expect(a?.[1]).not.toBe(pts.at(-1)?.[1]);
  });

  test('identical across calls and does not mutate inputs', () => {
    const gr = skip();
    const an = analyse(gr, info);
    const snap = JSON.stringify(gr);
    expect(layoutLayers(gr, an)).toEqual(layoutLayers(gr, an));
    expect(layoutNeurons(gr, an)).toEqual(layoutNeurons(gr, an));
    expect(JSON.stringify(gr)).toBe(snap);
  });

  test('invalid graphs lay out every node without throwing', () => {
    const gr = g(
      [
        { id: 'x', type: 'input', columns: ['x1'] },
        lin('a', 3),
        lin('b', 3),
        lin('c', 2),
        { id: 'add', type: 'add' },
        { id: 'loose', type: 'dropout', p: 0.1 },
        { id: 'constructor', type: 'activation', fn: 'tanh' },
        ...tail,
        { id: 'y2', type: 'target' },
        { id: 'loss2', type: 'loss', fn: 'huber' },
      ],
      [
        ['x', 'a'],
        ['a', 'b'],
        ['b', 'a'],
        ['b', 'b'],
        ['x', 'c'],
        ['a', 'c'],
        ['c', 'add.a'],
        ['ghost', 'add.b'],
        ['add', 'yhat'],
        ['yhat', 'loss.pred'],
        ['y', 'loss.target'],
        ['y2', 'loss2.target'],
      ],
    );
    const an = analyse(gr, info);
    const L = layoutLayers(gr, an);
    for (const n of gr.nodes) {
      const b = L.nodes[n.id];
      expect(b).toBeDefined();
      expect(Number.isFinite(b?.x) && Number.isFinite(b?.y)).toBe(true);
    }
    for (const w of L.wires) for (const p of w.points) expect(p.every(Number.isFinite)).toBe(true);
    expect(L.wires.some((w) => w.edge === 7)).toBe(false); // edge from an unknown node is skipped
    const N = layoutNeurons(gr, an);
    expect(Object.keys(N.boxes).length).toBe(gr.nodes.length);
    expect(layoutLayers(g([], []), an).size[0]).toBeGreaterThan(0);
    expect(layoutNeurons(g([], []), an).units).toEqual([]);
  });

  test('a node without wires goes in the last column; a lone input stays first', () => {
    const gr = mlp(4);
    gr.nodes.push({ id: 'drop', type: 'dropout', p: 0.5 });
    const L = layoutLayers(gr, analyse(gr, info));
    expect(L.nodes['drop']?.col).toBe(5);
    expect(L.nodes['x']?.col).toBe(0);
  });
});

describe('review fixes', () => {
  test('an unwired piece listed before the loss does not move the loss or target', () => {
    const base = mlp(4);
    const before = layoutLayers(base, analyse(base, info));
    const gr = mlp(4);
    gr.nodes.splice(1, 0, { id: 'drop', type: 'dropout', p: 0.5 });
    const after = layoutLayers(gr, analyse(gr, info));
    for (const id of ['x', 'yhat', 'y', 'loss']) expect(after.nodes[id]).toEqual(before.nodes[id]);
    expect(after.nodes['drop']?.y).toBeGreaterThan(before.nodes['x']?.y ?? 0);
  });

  test('label anchors: longest horizontal segment midpoint', () => {
    const gr = skip();
    const L = layoutLayers(gr, analyse(gr, info));
    for (const w of L.wires) {
      const [lx, ly] = w.labelAt;
      const onSeg = w.points.some((p, k) => {
        const q = w.points[k + 1];
        return (
          q &&
          p[1] === q[1] &&
          ly === p[1] &&
          lx >= Math.min(p[0], q[0]) &&
          lx <= Math.max(p[0], q[0])
        );
      });
      expect(onSeg).toBe(true);
    }
    const w0 = L.wires[0];
    const p = w0?.points ?? [];
    const l1 = Math.abs((p[1]?.[0] ?? 0) - (p[0]?.[0] ?? 0));
    const l2 = Math.abs((p[3]?.[0] ?? 0) - (p[2]?.[0] ?? 0));
    const [m, n] = l1 >= l2 ? [0, 1] : [2, 3];
    expect(w0?.labelAt).toEqual([((p[m]?.[0] ?? 0) + (p[n]?.[0] ?? 0)) / 2, p[m]?.[1]]);
    const sk = L.wires.find((w) => gr.edges[w.edge]?.[1] === 'add.b');
    const sp = sk?.points ?? [];
    expect(sk?.labelAt).toEqual([((sp[2]?.[0] ?? 0) + (sp[3]?.[0] ?? 0)) / 2, sp[2]?.[1]]);
  });

  test('neuron edges are i-major then j', () => {
    const gr = mlp(4);
    const es = layoutNeurons(gr, analyse(gr, info)).edges.filter((e) => e.node === 'lin2');
    expect(es.map((e) => e.i * 1 + e.j)).toEqual([0, 1, 2, 3]);
    const e1 = layoutNeurons(gr, analyse(gr, info)).edges.filter((e) => e.node === 'lin1');
    expect(e1.map((e) => [e.i, e.j])).toEqual([
      [0, 0],
      [0, 1],
      [0, 2],
      [0, 3],
    ]);
  });
});

describe('layoutNeurons', () => {
  test('1-4-1 has 4+4 weight edges and units at distinct positions', () => {
    const gr = mlp(4);
    const N = layoutNeurons(gr, analyse(gr, info));
    expect(N.edges.filter((e) => e.node === 'lin1').length).toBe(4);
    expect(N.edges.filter((e) => e.node === 'lin2').length).toBe(4);
    expect(N.collapsed).toEqual([]);
    const keys = new Set(N.units.map((u) => `${u.x},${u.y}`));
    expect(keys.size).toBe(N.units.length);
    const kinds = (id: string) => [
      ...new Set(N.units.filter((u) => u.node === id).map((u) => u.kind)),
    ];
    expect(kinds('x')).toEqual(['in']);
    expect(kinds('lin1')).toEqual(['sum']);
    expect(kinds('act1')).toEqual(['act']);
    expect(kinds('y')).toEqual(['target']);
    expect(N.units.filter((u) => u.node === 'act1').length).toBe(4);
    // lin2's edges start at act1's circles
    const act1 = N.units.filter((u) => u.node === 'act1');
    for (const e of N.edges.filter((e) => e.node === 'lin2')) {
      const u = act1[e.i];
      expect(e.from).toEqual([u?.x, u?.y]);
    }
    for (const e of N.edges) expect(e.from[0]).toBeLessThan(e.to[0]);
  });

  test('a 1-16-1 Linear stays collapsed, its neighbour that is too wide on one side too', () => {
    const gr = mlp(16);
    const N = layoutNeurons(gr, analyse(gr, info));
    expect(N.collapsed).toEqual(['lin1', 'lin2']);
    expect(N.edges).toEqual([]);
    expect(N.units.some((u) => u.node === 'lin1' || u.node === 'act1')).toBe(false);
    expect(N.boxes['lin1']?.w).toBeGreaterThan(36);
  });

  test('9 units is collapsed, 8 expands', () => {
    expect(layoutNeurons(mlp(8), analyse(mlp(8), info)).edges.length).toBe(16);
    expect(layoutNeurons(mlp(9), analyse(mlp(9), info)).collapsed).toContain('lin1');
  });

  test('expanded Linear after a collapsed upstream spreads edges along its box', () => {
    const gr = g(
      [
        { id: 'x', type: 'input', columns: ['x1'] },
        { id: 'd', type: 'dropout', p: 0.1 },
        lin('l', 2),
        ...tail,
      ],
      [['x', 'd'], ['d', 'l'], ['l', 'yhat'], ...tailEdges],
    );
    const N = layoutNeurons(gr, analyse(gr, info));
    const es = N.edges.filter((e) => e.node === 'l');
    expect(es.length).toBe(2);
    const d = N.boxes['d'];
    expect(es[0]?.from[0]).toBe((d?.x ?? 0) + (d?.w ?? 0));
  });

  test('skip graph neuron wires avoid boxes too', () => {
    const gr = skip();
    const N = layoutNeurons(gr, analyse(gr, info));
    for (const w of N.wires) {
      for (let k = 0; k + 1 < w.points.length; k++) {
        const a = w.points[k];
        const b = w.points[k + 1];
        if (!a || !b) continue;
        for (const box of Object.values(N.boxes)) expect(interiorHit([a, b], box)).toBe(false);
      }
    }
  });
});
