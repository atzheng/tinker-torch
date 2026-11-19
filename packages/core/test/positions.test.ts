import { describe, expect, test } from 'vitest';
import {
  analyse,
  clearPositions,
  layoutLayers,
  layoutNeurons,
  moveNodes,
  parseFileText,
  parseGraph,
  serializeFile,
  serializeGraph,
  LAYOUT,
  type DataInfo,
  type Graph,
} from '../src/index.js';

const info: DataInfo = { columns: ['x1'], kind: 'regression', numClasses: 2 };

function chain(): Graph {
  const r = parseGraph({
    version: 2,
    nodes: [
      { id: 'x', type: 'input', columns: ['x1'] },
      { id: 'lin1', type: 'linear', out: 4, bias: true },
      { id: 'act1', type: 'activation', fn: 'relu' },
      { id: 'lin2', type: 'linear', out: 1, bias: true },
    ],
    edges: [
      ['x', 'lin1'],
      ['lin1', 'act1'],
      ['act1', 'lin2'],
    ],
    predictions: ['lin2'],
    loss: null,
    data: { set: 'sine', n: 40, noise: 0.2, val: 0.3, seed: 7 },
    train: { optimizer: 'sgd', lr: 0.03, batch: 'full', seed: 11 },
  });
  if (!r.ok) throw new Error(r.error);
  return r.graph;
}
const lay = (g: Graph) => layoutLayers(g, analyse(g, info));
const boxes = (g: Graph) => Object.values(lay(g).nodes);
const apart = (
  a: { x: number; y: number; w: number; h: number },
  b: { x: number; y: number; w: number; h: number },
) => a.x + a.w <= b.x || b.x + b.w <= a.x || a.y + a.h <= b.y || b.y + b.h <= a.y;
const mv = (g: Graph, moves: Record<string, [number, number]>): Graph => {
  const m = moveNodes(g, moves);
  if (!m.ok) throw new Error(m.error);
  return m.graph;
};

describe('positions in the graph', () => {
  test('absent by default, round-trips through serialise and parse', () => {
    const g = chain();
    expect(g.positions).toBeUndefined();
    expect(serializeGraph(g)).not.toContain('positions');
    const text = serializeGraph(mv(g, { lin1: [200, 90] }));
    expect(text).toContain('"positions"');
    const back = parseGraph(JSON.parse(text));
    expect(back.ok && back.graph.positions).toEqual({ lin1: [200, 90] });
  });

  test('entries for missing nodes are dropped; bad shapes are refused', () => {
    const g = chain();
    const ok = parseGraph({ ...g, positions: { lin1: [1, 2], ghost: [3, 4] } });
    expect(ok.ok && ok.graph.positions).toEqual({ lin1: [1, 2] });
    expect(parseGraph({ ...g, positions: { lin1: [1] } }).ok).toBe(false);
    expect(parseGraph({ ...g, positions: { lin1: ['a', 'b'] } }).ok).toBe(false);
  });

  test('survive file save and load; a removed node drops its position on re-parse', () => {
    const m = mv(chain(), { act1: [300, 50], lin2: [60, 50] });
    const f = parseFileText(serializeFile(m));
    expect(f.ok && f.file.graph.positions).toEqual({ act1: [300, 50], lin2: [60, 50] });
    const cut = parseGraph({
      ...m,
      nodes: m.nodes.filter((n) => n.id !== 'act1'),
      edges: [],
      predictions: [],
    });
    expect(cut.ok && cut.graph.positions).toEqual({ lin2: [60, 50] });
  });
});

describe('moveNodes and clearPositions', () => {
  test('one edit, pure, keeps nodes and edges, clamps at the margin', () => {
    const g = chain();
    const m = mv(g, { lin1: [-5, 40], lin2: [100, 100] });
    expect(g.positions).toBeUndefined();
    expect(m.nodes).toBe(g.nodes);
    expect(m.edges).toBe(g.edges);
    expect(m.positions).toEqual({ lin1: [LAYOUT.pad, 40], lin2: [100, 100] });
    expect(mv(m, { lin1: [70, 80] }).positions).toEqual({ lin1: [70, 80], lin2: [100, 100] });
  });

  test('refuses an unknown piece and non-finite numbers', () => {
    const g = chain();
    expect(moveNodes(g, { nope: [1, 1] }).ok).toBe(false);
    expect(moveNodes(g, { lin1: [NaN, 1] }).ok).toBe(false);
    expect(moveNodes(g, { lin1: [Infinity, 1] }).ok).toBe(false);
  });

  test('clearPositions removes some or all, and refuses when nothing was moved', () => {
    const g = chain();
    expect(clearPositions(g).ok).toBe(false);
    const m = mv(g, { lin1: [31, 31], lin2: [32, 32] });
    const one = clearPositions(m, ['lin1']);
    expect(one.ok && one.graph.positions).toEqual({ lin2: [32, 32] });
    const all = clearPositions(m);
    expect(all.ok && 'positions' in all.graph).toBe(false);
  });
});

describe('layout with manual positions', () => {
  test('no positions left: identical to the automatic layout', () => {
    const g = chain();
    const cleared = clearPositions(mv(g, { lin1: [1, 1] }));
    if (!cleared.ok) throw new Error(cleared.error);
    expect(lay(cleared.graph)).toEqual(lay(g));
  });

  test('a placed piece goes exactly there; others keep their auto place', () => {
    const g = chain();
    const base = lay(g);
    const l = lay(mv(g, { act1: [400, 200] }));
    expect(l.nodes['act1']).toMatchObject({ x: 400, y: 200 });
    expect(l.nodes['lin1']).toEqual(base.nodes['lin1']);
    expect(l.nodes['x']).toEqual(base.nodes['x']);
    expect(l.size[0]).toBeGreaterThanOrEqual(400 + LAYOUT.boxW + LAYOUT.pad);
    expect(l.size[1]).toBeGreaterThanOrEqual(200 + LAYOUT.boxH + LAYOUT.pad);
  });

  test('wires follow the moved piece, end at its ports, and stay orthogonal', () => {
    const l = lay(mv(chain(), { act1: [400, 200] }));
    const a = l.nodes['act1'];
    if (!a) throw new Error('no box');
    expect(l.wires.find((w) => w.edge === 1)?.points.at(-1)).toEqual([a.x, a.y + a.h / 2]);
    expect(l.wires.find((w) => w.edge === 2)?.points[0]).toEqual([a.x + a.w, a.y + a.h / 2]);
    for (const w of l.wires) {
      for (let k = 0; k + 1 < w.points.length; k++) {
        const p = w.points[k];
        const q = w.points[k + 1];
        if (p && q) expect(p[0] === q[0] || p[1] === q[1]).toBe(true);
      }
    }
  });

  test('a backwards wire goes round its pieces', () => {
    const g = chain();
    const lin1 = lay(g).nodes['lin1'];
    if (!lin1) throw new Error('no box');
    const l = lay(mv(g, { act1: [lin1.x - 20, lin1.y + 150] }));
    expect(l.wires.find((x) => x.edge === 1)?.points.length).toBeGreaterThan(4);
  });

  test('an automatic piece under a placed one moves down, clear of it; deterministic', () => {
    const g = chain();
    const lin1 = lay(g).nodes['lin1'];
    if (!lin1) throw new Error('no box');
    const m = mv(g, { lin2: [lin1.x + 5, lin1.y + 5] });
    const l = lay(m);
    const a = l.nodes['lin1'];
    const b = l.nodes['lin2'];
    if (!a || !b) throw new Error('no box');
    expect(a.y).toBeGreaterThan(lin1.y);
    expect(apart(a, b)).toBe(true);
    expect(lay(m)).toEqual(l);
  });

  test('automatic pieces never overlap placed ones; the neuron layout ignores positions', () => {
    const g = chain();
    const m = mv(g, { act1: [30, 30], lin2: [600, 600] });
    const rs = boxes(m);
    for (const [i, a] of rs.entries())
      for (const b of rs.slice(i + 1)) expect(apart(a, b)).toBe(true);
    expect(layoutNeurons(m, analyse(m, info)).boxes).toEqual(
      layoutNeurons(g, analyse(g, info)).boxes,
    );
  });
});

describe('moveNodes limit', () => {
  test('a far move is clamped to a bound that grows with the piece count', () => {
    const g = chain();
    const m = mv(g, { lin1: [1e9, 1e9] });
    const [x, y] = m.positions?.['lin1'] ?? [0, 0];
    const spare = g.nodes.length + 2;
    expect(x).toBe(LAYOUT.pad + spare * (LAYOUT.boxW + LAYOUT.layerGap));
    expect(y).toBe(LAYOUT.pad + spare * (LAYOUT.boxH + LAYOUT.rowGap));
  });
});
