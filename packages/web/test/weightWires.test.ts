// @vitest-environment jsdom
import { beforeAll, beforeEach, expect, test } from 'vitest';
import {
  LAYOUT,
  PALETTE,
  analyse,
  layoutLayers,
  parseGraph,
  type Graph,
  type PaletteEntry,
} from '@tinker/core';
import { editForDrop, inputEndAt, placeable, resolveDrop } from '../src/canvas/dnd.js';
import {
  fmtWeight,
  HIT_MIN_PX,
  layerHitShapes,
  pillSize,
  pillTexts,
  weightTex,
} from '../src/canvas/pure.js';
import { beginWeightEdit, WEIGHT_LIMIT } from '../src/canvas/weightEdit.js';
import { loadEngine } from '../src/engine.js';
import { useStore } from '../src/state/store.js';

const s = () => useStore.getState();

function scalar(): Graph {
  const r = parseGraph({
    version: 2,
    nodes: [
      { id: 'x1', type: 'column', name: 'x1' },
      { id: 'x2', type: 'column', name: 'x2' },
      { id: 'w1', type: 'scalew' },
      { id: 'w2', type: 'scalew' },
      { id: 'sum1', type: 'sum' },
      { id: 'bias1', type: 'bias' },
      { id: 'target', type: 'target' },
      { id: 'diff', type: 'subtract' },
      { id: 'err', type: 'square' },
      { id: 'loss', type: 'mean' },
    ],
    edges: [
      ['x1', 'w1'],
      ['x2', 'w2'],
      ['w1', 'sum1'],
      ['w2', 'sum1'],
      ['sum1', 'bias1'],
      ['bias1', 'diff.a'],
      ['target', 'diff.b'],
      ['diff', 'err'],
      ['err', 'loss'],
    ],
    predictions: ['bias1'],
    loss: 'loss',
    data: { set: 'plane', n: 40, noise: 0.1, val: 0.3, seed: 3 },
    train: { optimizer: 'sgd', lr: 0.05, batch: 'full', seed: 5 },
  });
  if (!r.ok) throw new Error(r.error);
  return r.graph;
}
const info = { columns: ['x1', 'x2'], kind: 'regression', numClasses: 2 } as never;
const entry = (key: string): PaletteEntry => {
  const e = PALETTE.find((x) => x.key === key);
  if (!e) throw new Error(key);
  return e;
};

beforeAll(async () => {
  await loadEngine();
}, 60000);
beforeEach(() => {
  s().pause();
  s().loadGraph(scalar());
});

test('a Weight is set by hand: one undo step, clamped, refused while running', () => {
  expect(s().canTrain).toBe(true);
  const before = s().host?.['w1']?.['W']?.[0];
  expect(before).toBeDefined();
  const past = s().past.length;
  const edit = beginWeightEdit({ kind: 'scalew', node: 'w1' });
  expect(edit?.start).toBe(before);
  edit?.set(0.25);
  edit?.set(0.75);
  expect(s().past.length).toBe(past);
  expect(s().host?.['w1']?.['W']?.[0]).toBeCloseTo(0.75, 6);
  edit?.commit();
  expect(s().past.length).toBe(past + 1);
  s().undo();
  expect(s().host?.['w1']?.['W']?.[0]).toBe(before);
  s().redo();
  expect(s().host?.['w1']?.['W']?.[0]).toBeCloseTo(0.75, 6);
  const big = beginWeightEdit({ kind: 'scalew', node: 'w2' });
  big?.set(1e6);
  expect(s().host?.['w2']?.['W']?.[0]).toBe(WEIGHT_LIMIT);
  big?.cancel();
  useStore.setState({ running: true });
  expect(beginWeightEdit({ kind: 'scalew', node: 'w1' })).toBeNull();
  useStore.setState({ running: false });
});

test('a Bias is set by hand like a weight', () => {
  const edit = beginWeightEdit({ kind: 'bias', node: 'bias1' });
  expect(edit).not.toBeNull();
  edit?.set(-2);
  edit?.commit();
  expect(s().host?.['bias1']?.['b']?.[0]).toBeCloseTo(-2, 6);
  s().undo();
  expect(s().host?.['bias1']?.['b']?.[0]).not.toBeCloseTo(-2, 3);
  // a target of the wrong kind has nothing to set
  expect(beginWeightEdit({ kind: 'scalew', node: 'bias1' })).toBeNull();
});

test('a weight pill is at least 32 px each way and sized by a fixed text', () => {
  const g = scalar();
  const layout = layoutLayers(g, analyse(g, info));
  expect(layout.nodes['w1']).toBeUndefined();
  const pills = pillTexts(layout);
  expect([...pills.values()]).toEqual(['w1 = -0.00', 'w2 = -0.00']);
  for (const scale of [0.5, 1, 1.4]) {
    const shapes = layerHitShapes(layout, scale, [], new Set(), pills);
    for (const w of layout.wires.filter((x) => x.weightNode)) {
      const hit = shapes.find((x) => x.kind === 'wire' && x.edge === w.edge);
      expect(hit).toBeDefined();
      expect((hit?.w ?? 0) * scale).toBeGreaterThanOrEqual(HIT_MIN_PX - 1e-6);
      expect((hit?.h ?? 0) * scale).toBeGreaterThanOrEqual(HIT_MIN_PX - 1e-6);
    }
  }
  expect(pillSize('w1 = -0.00', 1).lw).toBeGreaterThan(pillSize('w1', 1).lw);
  expect(weightTex('w1')).toBe('w_{1}');
  expect(weightTex('w_a')).toBe('\\mathrm{w\\_a}');
  expect(fmtWeight(0.4249)).toBe('0.42');
  expect(fmtWeight(-12.34)).toBe('-12.3');
  expect(fmtWeight(100)).toBe('100');
});

test('every piece and group dropped on empty canvas is placed there', () => {
  const g = scalar();
  const a = analyse(g, info);
  const layout = layoutLayers(g, a);
  const empty = { x: layout.size[0] + 200, y: layout.size[1] + 200 };
  const drop = (key: string) =>
    resolveDrop({ kind: 'palette', entry: entry(key) }, g, layout, a.edgeWidth, 0.6, empty);
  for (const key of PALETTE.map((e) => e.key)) {
    expect(placeable(entry(key))).toBe(true);
    const t = drop(key);
    expect(t?.kind).toBe('free');
    const at = t?.kind === 'free' ? t.at : undefined;
    expect(at?.[0]).toBe(Math.round((empty.x - LAYOUT.boxW / 2) / 8) * 8);
    expect(at?.[1]).toBe(Math.round((empty.y - LAYOUT.boxH / 2) / 8) * 8);
  }
});

test('dropping a Column on empty canvas is one edit: a new unconnected node at the drop spot', () => {
  const g = scalar();
  const r = editForDrop(
    g,
    { kind: 'palette', entry: entry('column') },
    { kind: 'free', at: [480, 320] },
  );
  if (!r?.ok) throw new Error('no edit');
  expect(r.graph.nodes).toHaveLength(g.nodes.length + 1);
  expect(r.graph.edges).toEqual(g.edges);
  expect(r.select).toBe('x3');
  expect(r.graph.positions?.['x3']).toEqual([480, 320]);
  // left of or above the margin: pieces with a position move by the difference, automatic ones do not
  const placed = editForDrop(
    g,
    { kind: 'palette', entry: entry('column') },
    { kind: 'free', at: [200, 200] },
  );
  if (!placed?.ok) throw new Error('no edit');
  const layout = layoutLayers(placed.graph, analyse(placed.graph, info));
  const left = editForDrop(
    placed.graph,
    { kind: 'palette', entry: entry('sum') },
    { kind: 'free', at: [-100, 8] },
    { boxes: layout.nodes },
  );
  if (!left?.ok) throw new Error('no edit');
  expect(Object.keys(left.graph.positions ?? {}).sort()).toEqual(['sum2', 'x3']);
  // (LAYOUT.pad, LAYOUT.pad) is where the shifted pieces now sit, so the new piece is nudged one box right
  expect(left.graph.positions?.['sum2']).toEqual([LAYOUT.pad + LAYOUT.boxW + 8, LAYOUT.pad]);
  expect(left.graph.positions?.['x3']).toEqual([324, 216]);
});

test('a wire dragged to the Sum enters at its own slot of the port, ordered by source height', () => {
  const g = scalar();
  const layout = layoutLayers(g, analyse(g, info));
  const sum = layout.nodes['sum1'];
  if (!sum) throw new Error('no sum');
  const centre = layout.nodes['sum1'];
  expect(centre).toBeDefined();
  const first = layout.wires.find((w) => w.weightNode === 'w1')?.points.at(-1);
  const second = layout.wires.find((w) => w.weightNode === 'w2')?.points.at(-1);
  // existing ends are the layout's; a third wire from above x1 would take the top slot of three
  const above = inputEndAt(
    g,
    {
      ...layout,
      nodes: { ...layout.nodes, tmp: { x: 0, y: -500, w: 10, h: 10, col: 0 } },
    },
    'sum1',
    'in',
    'tmp',
  );
  expect(above?.[0]).toBe(sum.x);
  expect(above?.[1]).toBeLessThan(Math.min(first?.[1] ?? 0, second?.[1] ?? 0));
  // a plain port gives its centre, whatever is there
  const bias = layout.nodes['bias1'];
  const at = inputEndAt(g, layout, 'bias1', 'in', 'sum1');
  expect(at?.[1]).toBeCloseTo((bias?.y ?? 0) + (bias?.h ?? 0) / 2, 6);
});
