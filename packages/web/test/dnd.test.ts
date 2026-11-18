import { expect, test } from 'vitest';
import { PALETTE, analyse, layoutLayers, type Graph, type PaletteEntry } from '@tinker/core';
import { editForDrop, resolveDrop, type DragPayload } from '../src/canvas/dnd.js';
import { initialGraph } from '../src/state/graphSlice.js';
import { PORT_INSET, hitAt, layerHitShapes } from '../src/canvas/pure.js';

const SCALE = 0.475;
const info = () => ({ columns: ['x1'], numClasses: 1, n: 10 }) as never;

function setup(g: Graph = initialGraph()) {
  const a = analyse(g, info());
  const layout = layoutLayers(g, a);
  return { g, a, layout };
}
const entry = (key: string): PaletteEntry => {
  const e = PALETTE.find((x) => x.key === key);
  if (!e) throw new Error(key);
  return e;
};
const palette = (key: string): DragPayload => ({ kind: 'palette', entry: entry(key) });

function centreOf(layout: ReturnType<typeof setup>['layout'], id: string) {
  const b = layout.nodes[id];
  if (!b) throw new Error(id);
  return { x: b.x + b.w / 2, y: b.y + b.h / 2 };
}

test('a palette activation over a piece of the same family targets it, over another family does not', () => {
  const { g, a, layout } = setup();
  const over = (id: string, key: string) =>
    resolveDrop(palette(key), g, layout, a.edgeWidth, SCALE, centreOf(layout, id));
  expect(over('act1', 'activation')).toEqual({ kind: 'node', id: 'act1' });
  expect(over('lin1', 'activation')).toBeNull();
  expect(over('loss', 'huber')).toEqual({ kind: 'node', id: 'loss' });
  expect(over('loss', 'weight_penalty')).toEqual({ kind: 'node', id: 'loss' });
  expect(over('act1', 'weight_penalty')).toBeNull();
});

test('a palette piece over a wire label targets that wire; a loss or input does not insert', () => {
  const { g, a, layout } = setup();
  const w = layout.wires.find((x) => g.edges[x.edge]?.[0] === 'lin1');
  if (!w) throw new Error('no wire');
  const at = { x: w.labelAt[0], y: w.labelAt[1] };
  expect(resolveDrop(palette('activation'), g, layout, a.edgeWidth, SCALE, at)).toEqual({
    kind: 'wire',
    edge: w.edge,
  });
  expect(resolveDrop(palette('huber'), g, layout, a.edgeWidth, SCALE, at)).toBeNull();
  expect(resolveDrop(palette('input'), g, layout, a.edgeWidth, SCALE, at)).toBeNull();
  expect(resolveDrop(palette('weight_penalty'), g, layout, a.edgeWidth, SCALE, at)).toBeNull();
});

test('dropping on empty canvas cancels', () => {
  const { g, a, layout } = setup();
  expect(
    resolveDrop(palette('activation'), g, layout, a.edgeWidth, SCALE, { x: -500, y: -500 }),
  ).toBeNull();
});

test('every wire label resolves to its own wire, including the two that meet at the loss', () => {
  const { g, a, layout } = setup();
  for (const w of layout.wires) {
    const t = resolveDrop(palette('dropout'), g, layout, a.edgeWidth, SCALE, {
      x: w.labelAt[0],
      y: w.labelAt[1],
    });
    expect(t, `edge ${w.edge}`).toEqual({ kind: 'wire', edge: w.edge });
  }
});

test('editForDrop: insert, swap, penalty, connect, and a refused cycle', () => {
  const g = initialGraph();
  const ins = editForDrop(g, palette('dropout'), { kind: 'wire', edge: 1 });
  expect(ins?.ok).toBe(true);
  if (ins?.ok) {
    expect(ins.graph.nodes).toHaveLength(g.nodes.length + 1);
    expect(ins.select).toBe('dropout1');
  }
  const swap = editForDrop(g, palette('activation'), { kind: 'node', id: 'act1' });
  expect(swap?.ok).toBe(true);
  const pen = editForDrop(g, palette('weight_penalty'), { kind: 'node', id: 'loss' });
  expect(pen?.ok).toBe(true);
  if (pen?.ok) {
    const loss = pen.graph.nodes.find((n) => n.id === 'loss');
    expect(loss?.type === 'loss' && loss.penalty?.lambda).toBe(0.01);
  }
  const cycle = editForDrop(
    g,
    { kind: 'wire', from: 'lin2' },
    { kind: 'port', node: 'lin1', port: 'lin1' },
  );
  expect(cycle?.ok).toBe(false);
  const skip = editForDrop(
    g,
    { kind: 'wire', from: 'x' },
    { kind: 'port', node: 'lin2', port: 'lin2' },
  );
  expect(skip?.ok).toBe(true);
});

test('dragging a wire from an output over a piece targets its nearest input port; not its own piece', () => {
  const { g, a, layout } = setup();
  const t = resolveDrop(
    { kind: 'wire', from: 'x' },
    g,
    layout,
    a.edgeWidth,
    SCALE,
    centreOf(layout, 'loss'),
  );
  expect(t?.kind).toBe('port');
  const self = resolveDrop(
    { kind: 'wire', from: 'x' },
    g,
    layout,
    a.edgeWidth,
    SCALE,
    centreOf(layout, 'x'),
  );
  expect(self).toBeNull();
});

test('output port handles are 32 px squares that win presses at the right edge of their piece', () => {
  const { g, a, layout } = setup();
  const outputs = new Set(g.nodes.filter((n) => n.type !== 'loss').map((n) => n.id));
  const shapes = layerHitShapes(layout, SCALE, a.edgeWidth, outputs);
  for (const id of outputs) {
    const b = layout.nodes[id];
    if (!b) throw new Error(id);
    const top = hitAt(shapes, b.x + b.w - PORT_INSET, b.y + b.h / 2);
    expect(top?.kind === 'port' ? top.node : top, id).toBe(id);
  }
});
