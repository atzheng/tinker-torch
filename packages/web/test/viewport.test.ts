import { expect, test } from 'vitest';
import { analyse, layoutNeurons, LAYOUT, parseGraph, type Graph } from '@tinker/core';
import { useStore } from '../src/state/store.js';
import { LAYER_MIN_SCALE, layerBaseView } from '../src/canvas/viewport.js';

test('layerBaseView: fit when names are legible, else the floor, overflow left-aligned', () => {
  const view = { w: 800, h: 500 };
  const small = { ...fitTransform([400, 200], { w: 800, h: 456 }), ty: 100 };
  expect(layerBaseView(small, [400, 200], view, 44).scale).toBe(small.scale);
  const wide = fitTransform([2400, 200], { w: 800, h: 456 });
  expect(wide.scale).toBeLessThan(LAYER_MIN_SCALE);
  const t = layerBaseView(wide, [2400, 200], view, 44);
  expect(t.scale).toBe(LAYER_MIN_SCALE);
  expect(t.tx).toBe(24);
  expect(t.ty).toBeGreaterThan(44);
  expect(LAYER_MIN_SCALE * 15).toBeCloseTo(11, 6);
});
import { fitTransform } from '../src/canvas/pure.js';
import { neuronHitShapes, pickNeuron } from '../src/canvas/neuronPure.js';
import {
  clampView,
  neuronBaseView,
  pinchView,
  touchScaleFloor,
  zoomAbout,
} from '../src/canvas/viewport.js';

/** An MLP with `cols` input columns and the given hidden widths, one output. */
function mlp(cols: number, hidden: number[]): Graph {
  const nodes: unknown[] = [
    { id: 'x', type: 'input', columns: Array.from({ length: cols }, (_, i) => `x${i + 1}`) },
  ];
  const edges: unknown[] = [];
  let prev = 'x';
  [...hidden, 1].forEach((out, i) => {
    nodes.push({ id: `lin${i}`, type: 'linear', out, bias: true });
    edges.push([prev, `lin${i}`]);
    prev = `lin${i}`;
    if (i < hidden.length) {
      nodes.push({ id: `act${i}`, type: 'activation', fn: 'relu' });
      edges.push([prev, `act${i}`]);
      prev = `act${i}`;
    }
  });
  nodes.push({ id: 'yhat', type: 'output', link: 'identity' }, { id: 'y', type: 'target' });
  nodes.push({ id: 'loss', type: 'loss', fn: 'squared_error', penalty: { kind: 'l2', lambda: 0 } });
  edges.push([prev, 'yhat'], ['yhat', 'loss.pred'], ['y', 'loss.target']);
  const r = parseGraph({
    version: 1,
    nodes,
    edges,
    data: { set: 'sine', n: 40, noise: 0.2, val: 0.3, seed: 7 },
    train: { optimizer: 'sgd', lr: 0.03, batch: 'full', seed: 11 },
  });
  if (!r.ok) throw new Error(r.error);
  return r.graph;
}

const HEADER = 44;
// Canvas region sizes: the viewport minus the palette column and top/train bars (about 0.7 of it).
const VIEWS = [
  { name: '1024x768', w: 1024 - 250, h: 768 - 200 },
  { name: '1180x820', w: 1180 - 250, h: 820 - 200 },
];

test.each(
  VIEWS.flatMap((v) =>
    [
      ['1-8-8-1', 1, [8, 8]],
      ['2-8-8-8-1', 2, [8, 8, 8]],
    ].map(([n, c, h]) => [v, n, c, h] as const),
  ),
)('touch: every unit cell is at least 32 px square-ish at %o %s', (v, _name, cols, hidden) => {
  const g = mlp(cols as number, hidden as number[]);
  const layout = layoutNeurons(g, analyse(g, useStore.getState().dataInfo));
  const size: [number, number] = [layout.size[0], layout.size[1] + 16];
  const f = fitTransform(size, { w: v.w, h: v.h - HEADER });
  const fit = { ...f, ty: f.ty + HEADER };
  const t = neuronBaseView(fit, size, v, HEADER, true, LAYOUT.unitPitch);
  expect(t.scale).toBeGreaterThanOrEqual(touchScaleFloor(LAYOUT.unitPitch));
  const shapes = neuronHitShapes(layout, t.scale, LAYOUT.unitPitch, LAYOUT.unitW);
  const units = shapes.filter((s) => s.kind === 'unit');
  expect(units.length).toBeGreaterThan(20);
  for (const u of units) {
    expect(u.w * t.scale).toBeGreaterThanOrEqual(32 - 1e-6);
    expect(u.h * t.scale).toBeGreaterThanOrEqual(32 - 1e-6);
  }
  // The centre of every unit picks that unit.
  for (const u of layout.units) {
    const hit = pickNeuron(layout, shapes, t.scale, 32, u.x, u.y);
    expect(hit).toMatchObject({ kind: 'unit', node: u.node, index: u.index });
  }
  // Without the floor (a laptop fit) the same cells are too small: the floor is what fixes it.
  expect(fit.scale * LAYOUT.unitPitch).toBeLessThan(32);
});

test('the base view is the plain fit for a fine pointer or when the fit is already large', () => {
  const fit = { scale: 0.3, tx: 5, ty: 60 };
  expect(neuronBaseView(fit, [1000, 400], { w: 700, h: 500 }, HEADER, false, 44)).toEqual(fit);
  const big = { scale: 1, tx: 5, ty: 60 };
  expect(neuronBaseView(big, [500, 300], { w: 700, h: 500 }, HEADER, true, 44)).toEqual(big);
});

test('zoomAbout keeps the point under the cursor fixed and respects limits', () => {
  const t = { scale: 1, tx: 10, ty: 20 };
  const z = zoomAbout(t, 2, 110, 120, 0.5, 3);
  expect(z.scale).toBe(2);
  // abstract point under (110,120) before: (100,100); after: (110-tx)/2
  expect((110 - z.tx) / z.scale).toBeCloseTo(100);
  expect((120 - z.ty) / z.scale).toBeCloseTo(100);
  expect(zoomAbout(t, 10, 0, 0, 0.5, 3).scale).toBe(3);
  expect(zoomAbout(t, 0.01, 0, 0, 0.5, 3).scale).toBe(0.5);
});

test('pinchView zooms by the finger distance and pans with the midpoint', () => {
  const t = { scale: 1, tx: 0, ty: 0 };
  const z = pinchView(t, [100, 100], [200, 100], [100, 100], [300, 100], 0.1, 5);
  expect(z.scale).toBeCloseTo(2);
  expect((200 - z.tx) / z.scale).toBeCloseTo(150); // the point under the old midpoint follows it
  const pan = pinchView(t, [100, 100], [200, 100], [130, 90], [230, 90], 0.1, 5);
  expect(pan.scale).toBeCloseTo(1);
  expect(pan.tx).toBeCloseTo(30);
  expect(pan.ty).toBeCloseTo(-10);
});

test('clampView centres a small drawing and keeps a large one reachable', () => {
  const view = { w: 700, h: 500 };
  const small = clampView({ scale: 1, tx: 999, ty: 999 }, [300, 100], view, HEADER);
  expect(small.tx).toBeCloseTo(200);
  expect(small.ty).toBeCloseTo(HEADER + (456 - 100) / 2);
  const wide = (tx: number) => clampView({ scale: 1, tx, ty: 100 }, [1500, 100], view, HEADER).tx;
  expect(wide(500)).toBe(24);
  expect(wide(-5000)).toBe(700 - 24 - 1500);
  expect(wide(-100)).toBe(-100);
});
