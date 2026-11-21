import { expect, test } from 'vitest';
import { analyse, layoutLayers, parseGraph, sample, type Graph } from '@tinker/core';
import { mlpGraph as initialGraph } from './mlpGraph.js';
import {
  edgeIndexOf,
  errorNodes,
  fitTransform,
  heatMap,
  hitPad,
  placeInspector,
  placeIssues,
  resolvedEdges,
  hitAt,
  layerHitShapes,
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

const info = () => sample(initialGraph().data).info;

/** Where `placeIssues` puts the issues of `g`, the way the canvas calls it. */
function placed(g: Graph) {
  const a = analyse(g, info());
  const p = placeIssues(a.issues, resolvedEdges(g), new Set(g.nodes.map((n) => n.id)));
  return { a, p };
}
const edit = (g: Graph, f: (g: Graph) => Partial<Graph>): Graph => ({ ...g, ...f(g) });

test('a ŷ of the wrong width: the wire into b and the pinned piece each get their sentence', () => {
  const g = edit(initialGraph(), (g) => ({
    nodes: g.nodes.map((n) => (n.id === 'lin2' && n.type === 'linear' ? { ...n, out: 2 } : n)),
  }));
  const { p } = placed(g);
  expect([...p.byEdge.keys()]).toEqual([4]); // y -> diff.b
  expect(p.byEdge.get(4)?.[0]).toContain('same width');
  expect(p.byNode.get('lin2')?.[0]).toContain('width 1');
});

test('log loss without a sigmoid: no wire sentence; dataset sentence stays at the node', () => {
  const g = edit(initialGraph(), (g) => ({
    nodes: [
      ...g.nodes.filter((n) => n.id !== 'diff' && n.id !== 'err'),
      { id: 'err', type: 'log_loss' },
    ],
    edges: [
      ['x', 'lin1'],
      ['lin1', 'act1'],
      ['act1', 'lin2'],
      ['lin2', 'err.pred'],
      ['y', 'err.target'],
      ['err', 'loss'],
    ],
  }));
  const { p } = placed(g);
  expect(p.byEdge.size).toBe(0);
  expect(p.byNode.get('err')?.[0]).toContain('two classes');
});

test('"needs an input" stays at the piece, not on a wire', () => {
  const g = edit(initialGraph(), (g) => ({ edges: g.edges.filter((e) => e[1] !== 'diff.a') }));
  const { a, p } = placed(g);
  const msg = a.issues.find((i) => i.at.node === 'diff')?.message;
  expect(msg).toBe('Subtract needs both of its inputs connected.');
  expect(p.byNode.get('diff')).toContain(msg);
  expect(p.byEdge.size).toBe(0);
});

test('a cycle stays at its node', () => {
  const g = edit(initialGraph(), (g) => ({
    edges: [...g.edges.filter((e) => e[0] !== 'x'), ['act1', 'lin1']],
  }));
  const { a, p } = placed(g);
  const cyc = a.issues.find((i) => i.at.node !== undefined && i.at.edge === undefined);
  expect(cyc).toBeDefined();
  const at = cyc?.at.node ?? '';
  expect(p.byNode.get(at)).toContain(cyc?.message);
});

test('a duplicate id stays at the node', () => {
  const g0 = initialGraph();
  const act = g0.nodes.find((n) => n.id === 'act1');
  if (!act) throw new Error('no act1');
  const g = edit(g0, (g) => ({ nodes: [...g.nodes, act] }));
  const { a, p } = placed(g);
  const dup = a.issues.find((i) => i.message.startsWith('Two nodes'));
  expect(dup).toBeDefined();
  expect(p.byNode.get('act1')).toContain(dup?.message);
  expect(p.byEdge.size).toBe(0);
});

test('every wire label and every piece owns a 32x32 px region at the 1024x768 scale (0.475)', () => {
  const scale = 0.475;
  const graphs = [initialGraph(), skipGraph()];
  for (const g of graphs) {
    const a = analyse(g, info());
    const layout = layoutLayers(g, a);
    const shapes = layerHitShapes(layout, scale, a.edgeWidth);
    expect(shapes.filter((s) => s.kind === 'wire')).toHaveLength(layout.wires.length);
    for (const s of shapes) {
      // Is there a 32x32 px square, inside the shape, in which every sampled point resolves to it?
      const u = 32 / scale;
      let found = false;
      for (let x = s.x; x + u <= s.x + s.w + 1e-9 && !found; x += 2 / scale) {
        for (let y = s.y; y + u <= s.y + s.h + 1e-9 && !found; y += 2 / scale) {
          let all = true;
          for (let i = 0; i <= 8 && all; i++) {
            for (let j = 0; j <= 8 && all; j++) {
              const top = hitAt(
                shapes,
                x + 0.01 + ((u - 0.02) * i) / 8,
                y + 0.01 + ((u - 0.02) * j) / 8,
              );
              all = top === s;
            }
          }
          found = all;
        }
      }
      expect(found, JSON.stringify(s)).toBe(true);
    }
  }
});

function skipGraph(): Graph {
  const g = initialGraph();
  const parsed = parseGraph({
    ...g,
    nodes: [
      ...g.nodes.filter((n) => n.id !== 'lin2'),
      { id: 'add1', type: 'add' },
      { id: 'lin2', type: 'linear', out: 4, bias: true },
      { id: 'lin3', type: 'linear', out: 1, bias: true },
    ],
    edges: [
      ['x', 'lin1'],
      ['lin1', 'act1'],
      ['act1', 'lin2'],
      ['lin2', 'add1.a'],
      ['lin1', 'add1.b'],
      ['add1', 'lin3'],
      ['lin3', 'diff.a'],
      ['y', 'diff.b'],
      ['diff', 'err'],
      ['err', 'loss'],
    ],
    predictions: ['lin3'],
  });
  if (!parsed.ok) throw new Error(parsed.error);
  return parsed.graph;
}

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
