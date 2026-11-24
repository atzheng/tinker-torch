import { describe, expect, test } from 'vitest';
import {
  analyse,
  layoutLayers,
  LAYOUT,
  paramWireCurve,
  wireGaps,
  type DataInfo,
  type Graph,
} from '../src/index.js';

const info: DataInfo = { columns: ['x1'], kind: 'regression', numClasses: 2 };
const lin = (id: string, out: number) => ({ id, type: 'linear' as const, out, bias: true });
const base: Omit<Graph, 'nodes' | 'edges'> = {
  version: 2,
  predictions: [],
  loss: null,
  data: { set: 'line' as const, n: 100, noise: 0.1, val: 0.2, seed: 1, standardize: false },
  train: { optimizer: 'adam', lr: 0.01, batch: 'full', seed: 1 },
};
// The synthetic Sum graph of tinker-6wk: l3's first run leaves at the height where l1's skip wire ends.
const sumGraph: Graph = {
  ...base,
  nodes: [
    { id: 'x', type: 'input', columns: ['x1'] },
    lin('l1', 3),
    { id: 'a1', type: 'activation', fn: 'relu' },
    lin('l2', 3),
    { id: 'a2', type: 'activation', fn: 'relu' },
    lin('l3', 3),
    { id: 's', type: 'sum' },
  ],
  edges: [
    ['x', 'l1'],
    ['l1', 'a1'],
    ['a1', 'l2'],
    ['l2', 'a2'],
    ['a2', 'l3'],
    ['x', 's.in'],
    ['l1', 's.in'],
    ['l3', 's.in'],
  ],
};

type H = { from: string; y: number; lo: number; hi: number };
function horizontals(g: Graph): H[] {
  const L = layoutLayers(g, analyse(g, info));
  const out: H[] = [];
  for (const w of L.wires)
    for (let i = 0; i + 1 < w.points.length; i++) {
      const p = w.points[i] as [number, number];
      const q = w.points[i + 1] as [number, number];
      if (p[1] === q[1] && p[0] !== q[0])
        out.push({
          from: g.edges[w.edge]?.[0] ?? '',
          y: p[1],
          lo: Math.min(p[0], q[0]),
          hi: Math.max(p[0], q[0]),
        });
    }
  return out;
}

/** Smallest collinear distance between horizontals of different sources (negative: they overlap). */
function minBreak(hs: H[]): number {
  let m = Infinity;
  for (let i = 0; i < hs.length; i++)
    for (let j = i + 1; j < hs.length; j++) {
      const a = hs[i] as H;
      const b = hs[j] as H;
      if (a.from === b.from || Math.abs(a.y - b.y) > 1e-6) continue;
      m = Math.min(m, Math.max(a.lo, b.lo) - Math.min(a.hi, b.hi));
    }
  return m;
}

describe('horizontal runs of different wires are separated (tinker-6wk)', () => {
  test('Sum graph: a first run and another wire last run at one height break by breakLanes steps', () => {
    expect(minBreak(horizontals(sumGraph))).toBeGreaterThanOrEqual(
      LAYOUT.breakLanes * LAYOUT.laneStep,
    );
  });

  test('wires keep their ends', () => {
    const L = layoutLayers(sumGraph, analyse(sumGraph, info));
    for (const w of L.wires) {
      const b = L.nodes[sumGraph.edges[w.edge]?.[1].split('.')[0] ?? ''];
      expect(w.points[w.points.length - 1]?.[0]).toBe(b?.x);
    }
  });
});

describe('param wires over/under (tinker-6wk)', () => {
  const lb = { x: 100, y: 50, w: 80, h: 40 };
  const wb = { x: 300, y: 200, w: 60, h: 30 };
  test('paramWireCurve flattens the cubic from the Linear bottom to the Weights left side', () => {
    const c = paramWireCurve(lb, wb);
    expect(c.points[0]).toEqual([140, 90]);
    expect(c.points[c.points.length - 1]).toEqual([300, 215]);
    expect(c.d.startsWith('M140 90C140 126 260 215 300 215')).toBe(true);
  });
  test('a data wire crossing a param wire gets a gap; the param wire, ordered last, none', () => {
    const c = paramWireCurve(lb, wb);
    const p0 = c.points[12] as [number, number];
    const p1 = c.points[13] as [number, number];
    const mid: [number, number] = [(p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2];
    const cross: [number, number][] = [
      [mid[0] - 60, mid[1] - 20],
      [mid[0] + 60, mid[1] + 20],
    ];
    const g = wireGaps([cross, c.points]);
    expect(g[0]?.length).toBe(1);
    expect(g[1]).toEqual([]);
  });
});
