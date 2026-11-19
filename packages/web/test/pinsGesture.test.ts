import { expect, test } from 'vitest';
import { addPrediction, analyse, layoutLayers, type Graph } from '@tinker/core';
import { editForDrop, pinCanLandOn, resolveDrop, type DragPayload } from '../src/canvas/dnd.js';
import { mlpGraph as initialGraph } from './mlpGraph.js';
import {
  HIT_MIN_PX,
  PIN,
  hitAt,
  layerHitShapes,
  pinCentre,
  pinHitRect,
  pinLive,
  pinsOf,
  placeIssues,
  errorNodes,
} from '../src/canvas/pure.js';
import { plotIndex, predictionNodes } from '../src/state/plotPrediction.js';

const SCALE = 0.733;
const info = () => ({ columns: ['x1'], numClasses: 1, n: 10 }) as never;

function setup(g: Graph = initialGraph()) {
  const a = analyse(g, info());
  return { g, a, layout: layoutLayers(g, a) };
}
function centreOf(layout: ReturnType<typeof setup>['layout'], id: string) {
  const b = layout.nodes[id];
  if (!b) throw new Error(id);
  return { x: b.x + b.w / 2, y: b.y + b.h / 2 };
}
const pin = (role: 'pred' | 'loss', node: string): DragPayload => ({ kind: 'pin', role, node });

test('pinsOf lists the ŷ pins in order, then L; pinLive checks one', () => {
  const g = initialGraph();
  expect(pinsOf(g)).toEqual([
    { role: 'pred', node: 'lin2' },
    { role: 'loss', node: 'loss' },
  ]);
  expect(pinLive(g, 'pred', 'lin2')).toBe(true);
  expect(pinLive(g, 'loss', 'lin2')).toBe(false);
});

test('a dragged pin lands on another piece (body or port), never on itself or a taken ŷ', () => {
  const { g, a, layout } = setup();
  const at = (id: string, p: DragPayload) =>
    resolveDrop(p, g, layout, a.edgeWidth, SCALE, centreOf(layout, id));
  expect(at('act1', pin('pred', 'lin2'))).toEqual({ kind: 'node', id: 'act1' });
  expect(at('lin2', pin('pred', 'lin2'))).toBeNull();
  expect(at('err', pin('loss', 'loss'))).toEqual({ kind: 'node', id: 'err' });
  expect(at('loss', pin('loss', 'loss'))).toBeNull();
  // a second ŷ cannot land on a node that already is one; L can land on a ŷ node
  const two = addPrediction(g, 'act1');
  if (!two.ok) throw new Error(two.error);
  const s2 = setup(two.graph);
  expect(
    resolveDrop(
      pin('pred', 'lin2'),
      s2.g,
      s2.layout,
      s2.a.edgeWidth,
      SCALE,
      centreOf(s2.layout, 'act1'),
    ),
  ).toBeNull();
  expect(pinCanLandOn(s2.g, { role: 'loss', node: 'loss' }, 'act1')).toBe(true);
  // empty canvas is no target
  expect(
    resolveDrop(pin('pred', 'lin2'), g, layout, a.edgeWidth, SCALE, { x: -50, y: -50 }),
  ).toBeNull();
});

test('dropping a pin is one pin edit; locked pieces refuse', () => {
  const { g } = setup();
  const moved = editForDrop(g, pin('pred', 'lin2'), { kind: 'node', id: 'act1' });
  expect(moved?.ok && moved.graph.predictions).toEqual(['act1']);
  const loss = editForDrop(g, pin('loss', 'loss'), { kind: 'node', id: 'err' });
  expect(loss?.ok && loss.graph.loss).toBe('err');
  const locked = editForDrop(
    g,
    pin('pred', 'lin2'),
    { kind: 'node', id: 'act1' },
    { locked: ['lin2'] },
  );
  expect(locked).toEqual({ ok: false, error: 'This piece is locked.' });
  expect(editForDrop(g, pin('pred', 'lin2'), { kind: 'free' })).toBeNull();
});

test('a pin press rectangle stays clear of its piece and of the output-port band, and is touch sized', () => {
  const { layout } = setup();
  const box = layout.nodes['lin2'];
  if (!box) throw new Error('no box');
  for (const scale of [0.5, 0.733, 1, 1.4]) {
    for (const role of ['pred', 'loss'] as const) {
      const r = pinHitRect(box, role, scale);
      expect(r.x).toBeGreaterThan(box.x + box.w); // outside the piece
      const mid = box.y + box.h / 2;
      const band = HIT_MIN_PX / 2 / scale;
      if (role === 'pred') expect(r.y + r.h).toBeLessThanOrEqual(mid - band + 1e-9);
      else expect(r.y).toBeGreaterThanOrEqual(mid + band - 1e-9);
    }
  }
  // 32 px each way from the floor scale up
  for (const scale of [0.733, 1, 1.4]) {
    const r = pinHitRect(box, 'pred', scale);
    expect(r.w * scale).toBeGreaterThanOrEqual(HIT_MIN_PX - 1);
    expect(r.h * scale).toBeGreaterThanOrEqual(HIT_MIN_PX - 1);
  }
  const [cx, cy] = pinCentre(box, 'pred');
  expect(cx - (box.x + box.w)).toBe(PIN.dx);
  expect(cy).toBe(box.y + box.h / 2 - PIN.dy);
});

test('a pin hit rectangle never shadows the connect handle or a piece', () => {
  const { g, layout } = setup();
  const outputs = new Set(g.nodes.map((n) => n.id));
  const shapes = layerHitShapes(layout, 0.733, [], outputs);
  const box = layout.nodes['lin2'];
  if (!box) throw new Error('no box');
  for (const role of ['pred', 'loss'] as const) {
    const r = pinHitRect(box, role, 0.733);
    const corners = [
      [r.x, r.y],
      [r.x + r.w, r.y],
      [r.x, r.y + r.h],
      [r.x + r.w, r.y + r.h],
    ] as const;
    for (const [x, y] of corners) {
      const hit = hitAt(shapes, x, y);
      expect(hit?.kind === 'port' && hit.node === 'lin2').toBe(false);
    }
  }
});

test('pin issues are placed at the pin when asked, at the node otherwise', () => {
  const g = initialGraph();
  const two = addPrediction(g, 'act1');
  if (!two.ok) throw new Error(two.error);
  const a = analyse(two.graph, info());
  const ids = new Set(two.graph.nodes.map((n) => n.id));
  const nodeOf = (p: { role: string; index?: number }) =>
    p.role === 'loss' ? 'loss' : predictionNodes(two.graph)[p.index ?? 0];
  const at = placeIssues(a.issues, new Set(), ids, undefined, nodeOf);
  expect([...at.byPin.keys()]).toEqual(['pred:act1']);
  expect(at.byNode.size).toBe(0);
  expect([...errorNodes(a.issues, true)]).toEqual([]);
  const old = placeIssues(a.issues, new Set(), ids);
  expect(old.byPin.size).toBe(0);
  expect([...old.byNode.keys()]).toEqual(['act1']);
  expect([...errorNodes(a.issues)]).toEqual(['act1']);
});

test('the plot shows the picked ŷ node, else the first', () => {
  const g = { predictions: ['lin2', 'act1'] };
  expect(plotIndex(g, null)).toBe(0);
  expect(plotIndex(g, 'act1')).toBe(1);
  expect(plotIndex(g, 'gone')).toBe(0);
  expect(plotIndex({ predictions: [] }, 'act1')).toBe(0);
});
