import { expect, test } from 'vitest';
import {
  LAYOUT,
  PALETTE,
  analyse,
  layoutLayers,
  type Graph,
  type PaletteEntry,
} from '@tinker/core';
import {
  dnd,
  dropCandidates,
  editForDrop,
  resolveDrop,
  type DragPayload,
} from '../src/canvas/dnd.js';
import { mlpGraph as initialGraph } from './mlpGraph.js';
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
  expect(over('err', 'huber')).toEqual({ kind: 'node', id: 'err' });
  expect(over('err', 'activation')).toBeNull();
  // a ready-made group is never swapped onto a piece
  expect(over('loss', 'loss_huber')).toBeNull();
  expect(over('act1', 'weight_penalty')).toBeNull();
});

test('a palette piece over a wire label targets that wire; a piece with no input does not insert', () => {
  const { g, a, layout } = setup();
  const w = layout.wires.find((x) => g.edges[x.edge]?.[0] === 'lin1');
  if (!w) throw new Error('no wire');
  const at = { x: w.labelAt[0], y: w.labelAt[1] };
  expect(resolveDrop(palette('activation'), g, layout, a.edgeWidth, SCALE, at)).toEqual({
    kind: 'wire',
    edge: w.edge,
  });
  expect(resolveDrop(palette('huber'), g, layout, a.edgeWidth, SCALE, at)).toEqual({
    kind: 'wire',
    edge: w.edge,
  });
  expect(resolveDrop(palette('input'), g, layout, a.edgeWidth, SCALE, at)).toBeNull();
  expect(resolveDrop(palette('weights'), g, layout, a.edgeWidth, SCALE, at)).toBeNull();
  expect(resolveDrop(palette('loss_huber'), g, layout, a.edgeWidth, SCALE, at)).toEqual({
    kind: 'wire',
    edge: w.edge,
  });
});

test('every palette entry dropped on empty canvas goes in free-standing at the drop point', () => {
  const { g, a, layout } = setup();
  const at = { x: -500, y: -500 };
  for (const entry of PALETTE) {
    const t = resolveDrop({ kind: 'palette', entry }, g, layout, a.edgeWidth, SCALE, at);
    expect(t, entry.key).toEqual({ kind: 'free', at: [-552, -528] });
  }
});

test('editForDrop puts every piece and group on empty canvas, positioned, as one edit', () => {
  const g = initialGraph();
  for (const entry of PALETTE) {
    const r = editForDrop(g, { kind: 'palette', entry }, { kind: 'free', at: [480, 320] });
    if (!r?.ok) {
      // the only refusals: a Weights tap needs a Linear; the penalty needs a Linear
      expect(['weights', 'weight_penalty'], entry.key).toContain(entry.key);
      continue;
    }
    expect(r.graph.nodes.length, entry.key).toBeGreaterThan(g.nodes.length);
    const added = r.graph.nodes.filter((n) => !g.nodes.some((m) => m.id === n.id));
    for (const n of added) expect(r.graph.positions?.[n.id], `${entry.key} ${n.id}`).toBeDefined();
  }
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

test('editForDrop: insert, swap, ready-made group, connect, and a refused cycle', () => {
  const g = initialGraph();
  const ins = editForDrop(g, palette('dropout'), { kind: 'wire', edge: 1 });
  expect(ins?.ok).toBe(true);
  if (ins?.ok) {
    expect(ins.graph.nodes).toHaveLength(g.nodes.length + 1);
    expect(ins.select).toBe('dropout1');
  }
  const swap = editForDrop(g, palette('activation'), { kind: 'node', id: 'act1' });
  expect(swap?.ok).toBe(true);
  const pen = editForDrop(g, palette('weight_penalty'), { kind: 'free' });
  expect(pen?.ok).toBe(true);
  if (pen?.ok) {
    // one edit: the group's nodes arrive together and L moves to the new total
    expect(pen.graph.nodes.map((n) => n.id)).toContain('total');
    expect(pen.graph.loss).toBe('total');
    expect(pen.select).toBe('total');
  }
  const onWire = editForDrop(g, palette('loss_huber'), { kind: 'wire', edge: 1 });
  expect(onWire?.ok).toBe(true);
  if (onWire?.ok) expect(onWire.graph.edges).toContainEqual(['lin1', 'diff_2.a']);
  expect(editForDrop(g, palette('loss_huber'), { kind: 'node', id: 'err' })).toBeNull();
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
  const outputs = new Set(g.nodes.map((n) => n.id));
  const shapes = layerHitShapes(layout, SCALE, a.edgeWidth, outputs);
  for (const id of outputs) {
    const b = layout.nodes[id];
    if (!b) throw new Error(id);
    const top = hitAt(shapes, b.x + b.w - PORT_INSET, b.y + b.h / 2);
    expect(top?.kind === 'port' ? top.node : top, id).toBe(id);
  }
});

test('rewire: a wire end dropped on another input port moves the wire; the old port and its source do not', () => {
  const { g, a, layout } = setup();
  const payload: DragPayload = { kind: 'rewire', edge: 1, from: 'lin1', to: 'act1' }; // lin1 -> act1
  const over = (id: string) =>
    resolveDrop(payload, g, layout, a.edgeWidth, SCALE, centreOf(layout, id));
  expect(over('lin2')).toEqual({ kind: 'port', node: 'lin2', port: 'lin2' });
  expect(over('act1')).toBeNull(); // the port it already ends at
  expect(over('lin1')).toBeNull(); // its own source
  const edit = editForDrop(g, payload, { kind: 'port', node: 'lin2', port: 'lin2' });
  expect(edit?.ok).toBe(true);
  if (edit?.ok) {
    expect(edit.graph.edges).toContainEqual(['lin1', 'lin2']);
    expect(edit.graph.edges).not.toContainEqual(['lin1', 'act1']);
  }
});

test('rewire onto a locked piece is refused; a cycle is refused', () => {
  const { g } = setup();
  const payload: DragPayload = { kind: 'rewire', edge: 1, from: 'lin1', to: 'act1' };
  const target = { kind: 'port', node: 'lin2', port: 'lin2' } as const;
  expect(editForDrop(g, payload, target, { locked: ['lin2'] })?.ok).toBe(false);
  const loop = editForDrop(
    g,
    { kind: 'rewire', edge: 2, from: 'act1', to: 'lin2' },
    { kind: 'port', node: 'lin1', port: 'lin1' },
  );
  expect(loop?.ok).toBe(false);
});

test('tap-to-add: arm toggles, and a tap applies the entry once then disarms', () => {
  const g = initialGraph();
  const committed: unknown[] = [];
  const off = dnd.setHandler({
    resolve: (_p, x) => (x > 0 ? { kind: 'wire', edge: 1 } : null),
    commit: (p, t) => committed.push([p.kind, t]),
  });
  dnd.arm(entry('activation'));
  expect(dnd.armed()?.key).toBe('activation');
  dnd.arm(entry('activation')); // the same entry again cancels
  expect(dnd.armed()).toBeNull();
  dnd.arm(entry('activation'));
  expect(dnd.tapAt(-1, 0)).toBe(false); // nothing there: cancel
  expect(dnd.armed()).toBeNull();
  expect(committed).toHaveLength(0);
  dnd.arm(entry('activation'));
  expect(dnd.tapAt(5, 0)).toBe(true);
  expect(committed).toHaveLength(1);
  expect(dnd.armed()).toBeNull();
  expect(g.edges.length).toBeGreaterThan(1);
  off();
});

test('dropCandidates: wires for insertable pieces and groups, same-family pieces', () => {
  const g = initialGraph();
  const act = dropCandidates(entry('activation'), g);
  expect(act.edges.size).toBe(g.edges.length);
  expect([...act.nodes]).toEqual(['act1']);
  const pen = dropCandidates(entry('weight_penalty'), g);
  expect(pen.edges.size).toBe(g.edges.length);
  expect(pen.nodes.size).toBe(0);
  expect(dropCandidates(entry('input'), g).edges.size).toBe(0);
});

test('a piece dropped on an occupied spot is nudged to the nearest free grid slot', async () => {
  const { freeSlot } = await import('../src/canvas/dnd.js');
  expect(freeSlot([200, 100], [])).toEqual([200, 100]);
  const others: [number, number][] = [[200, 100]];
  const s = freeSlot([200, 100], others);
  expect(s).not.toEqual([200, 100]);
  expect(s[0] % 8).toBe(0);
  expect(s[1] % 8).toBe(0);
  const overlap = Math.abs(s[0] - 200) < LAYOUT.boxW + 8 && Math.abs(s[1] - 100) < LAYOUT.boxH + 8;
  expect(overlap).toBe(false);
  // Several pieces in a row: still free of all of them.
  const row: [number, number][] = [
    [200, 100],
    [320, 100],
    [80, 100],
  ];
  const r = freeSlot([200, 100], row);
  for (const [ox, oy] of row) {
    expect(Math.abs(r[0] - ox) >= LAYOUT.boxW + 8 || Math.abs(r[1] - oy) >= LAYOUT.boxH + 8).toBe(
      true,
    );
  }
  // Via editForDrop: a Column dropped on top of an existing piece does not share its position.
  const g = initialGraph();
  const a = analyse(g, { columns: ['x1'], kind: 'regression', numClasses: 1 });
  const layout = layoutLayers(g, a);
  const b = layout.nodes['lin1'];
  if (!b) throw new Error('no box');
  const r2 = editForDrop(
    g,
    { kind: 'palette', entry: PALETTE.find((e) => e.key === 'column') as PaletteEntry },
    { kind: 'free', at: [b.x, b.y] },
    { boxes: layout.nodes },
  );
  if (!r2?.ok) throw new Error('no edit');
  const id = r2.select ?? '';
  const p = r2.graph.positions?.[id];
  expect(p).toBeDefined();
  expect(p?.[0] !== b.x || p?.[1] !== b.y).toBe(true);
});
