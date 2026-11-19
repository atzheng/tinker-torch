// @vitest-environment jsdom
import { afterEach, beforeEach, expect, test } from 'vitest';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { analyse, layoutNeurons, LAYOUT, sample } from '@tinker/core';
import { Canvas } from '../src/canvas/Canvas.js';
import { initialGraph } from '../src/state/graphSlice.js';
import { editSetting } from '../src/canvas/drawings/fields.js';
import { useStore } from '../src/state/store.js';
import { fitName, MIN_TEXT_PX } from '../src/canvas/pure.js';

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

let host: HTMLDivElement;
let root: Root;
beforeEach(() => {
  useStore.getState().loadGraph(initialGraph());
  host = document.createElement('div');
  document.body.appendChild(host);
  root = createRoot(host);
  act(() => root.render(<Canvas />));
});
afterEach(() => {
  act(() => root.unmount());
  host.remove();
});

const nameOf = (id: string) =>
  host.querySelector(`[data-testid="node-${id}"] .tk-name`)?.textContent ?? null;

test('layer view: each node shows its type, the activation its function', () => {
  expect(nameOf('x')).toBe('Input');
  expect(nameOf('lin1')).toBe('Linear');
  expect(nameOf('act1')).toBe('ReLU');
  expect(nameOf('lin2')).toBe('Linear');
  expect(nameOf('y')).toBe('Target');
  expect(nameOf('diff')).toBe('Subtract');
  expect(nameOf('err')).toBe('Square');
  expect(nameOf('loss')).toBe('Mean');
});

test('fitName keeps names at least MIN_TEXT_PX on screen where the width allows, and never exceeds the width', () => {
  expect(fitName(1, 'Linear', 88, 15)).toEqual({ text: 'Linear', size: 15 });
  expect(fitName(undefined, 'Linear', 88, 15).size).toBe(15);
  const f = fitName(0.45, 'Linear', 88, 15);
  expect(f.size * 0.45).toBeGreaterThanOrEqual(MIN_TEXT_PX - 1e-9);
  const long = fitName(1, 'Absolute error', 88, 15);
  expect(long.size).toBeLessThan(15);
  expect(long.size).toBeGreaterThanOrEqual(10);
  // Too narrow even at 10 units: cut with an ellipsis, and the estimate stays within the width.
  for (const w of [30, 36, 50]) {
    for (const t of ['Input', 'Target', 'Sigmoid', 'Absolute error']) {
      const r = fitName(0.4, t, w, 13, 16);
      expect(r.text.length * r.size * 0.52, `${t} in ${w}`).toBeLessThanOrEqual(w + 1e-9);
      expect(r.text.length).toBeGreaterThanOrEqual(2);
    }
  }
  expect(fitName(1, 'Sigmoid', 36, 13).text.endsWith('\u2026')).toBe(true);
});

test('Tanh has no glyph that repeats its name; ReLU keeps max(0,x)', () => {
  // The glyph is KaTeX; its TeX source is the span's aria-label.
  const g = (id: string) =>
    host.querySelector(`[data-testid="node-${id}"] .tk-fo .tk-tex`)?.getAttribute('aria-label') ??
    undefined;
  expect(g('act1')).toBe('\\max(0,x)');
  act(() => {
    editSetting('act1', 'fn', 'tanh');
  });
  expect(nameOf('act1')).toBe('Tanh');
  expect(g('act1')).toBeUndefined();
});

type Rect = { x: number; y: number; w: number; h: number };

/** Whether the segment p-q crosses the axis-aligned rectangle (slab clipping). */
function crosses(p: readonly [number, number], q: readonly [number, number], r: Rect): boolean {
  let t0 = 0;
  let t1 = 1;
  for (const k of [0, 1] as const) {
    const d = q[k] - p[k];
    const lo = k === 0 ? r.x : r.y;
    const hi = lo + (k === 0 ? r.w : r.h);
    if (Math.abs(d) < 1e-12) {
      if (p[k] < lo || p[k] > hi) return false;
    } else {
      const a = (lo - p[k]) / d;
      const b = (hi - p[k]) / d;
      t0 = Math.max(t0, Math.min(a, b));
      t1 = Math.min(t1, Math.max(a, b));
      if (t0 > t1) return false;
    }
  }
  return true;
}

test('neuron view: the name strip above each unit column crosses no weight edge, wire, unit or box', () => {
  const g = initialGraph();
  const layout = layoutNeurons(g, analyse(g, sample(g.data).info));
  const half = 18; // the name is centred on the column and at most one column (36 units) wide
  let strips = 0;
  for (const [id, b] of Object.entries(layout.boxes)) {
    if (!layout.units.some((u) => u.node === id)) continue;
    strips++;
    const r: Rect = { x: b.x + b.w / 2 - half, y: b.y, w: 2 * half, h: LAYOUT.unitLabelH };
    for (const e of layout.edges) expect(crosses(e.from, e.to, r), `edge near ${id}`).toBe(false);
    for (const w of layout.wires) {
      for (let i = 1; i < w.points.length; i++) {
        const a = w.points[i - 1];
        const c = w.points[i];
        if (a && c) expect(crosses(a, c, r), `wire ${w.edge} near ${id}`).toBe(false);
      }
    }
    for (const u of layout.units) {
      const inside =
        u.x + 12 > r.x && u.x - 12 < r.x + r.w && u.y + 12 > r.y && u.y - 12 < r.y + r.h;
      expect(inside, `unit ${u.node}:${u.index} under the ${id} name`).toBe(false);
    }
    for (const [other, ob] of Object.entries(layout.boxes)) {
      if (other === id) continue;
      const hit = ob.x < r.x + r.w && ob.x + ob.w > r.x && ob.y < r.y + r.h && ob.y + ob.h > r.y;
      expect(hit, `box ${other} under the ${id} name`).toBe(false);
    }
  }
  expect(strips).toBeGreaterThanOrEqual(4);
});
