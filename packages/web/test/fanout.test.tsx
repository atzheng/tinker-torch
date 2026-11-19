// @vitest-environment jsdom
// Fan-out (tinker-0sq): a connect drag from an output that already has wires adds a wire.
import { afterEach, beforeEach, expect, test } from 'vitest';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { analyse, layoutLayers, parseGraph } from '@tinker/core';
import { Canvas } from '../src/canvas/Canvas.js';
import { fitTransform } from '../src/canvas/pure.js';
import { layerBaseView } from '../src/canvas/viewport.js';
import { useStore } from '../src/state/store.js';

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const s = () => useStore.getState();
let host: HTMLDivElement;
let root: Root;

function startGraph() {
  const r = parseGraph({
    version: 2,
    nodes: [
      { id: 'x', type: 'input', columns: ['x1', 'x2'] },
      { id: 'h', type: 'linear', out: 1, bias: true },
      { id: 'a0', type: 'activation', fn: 'tanh' },
      { id: 'a1', type: 'activation', fn: 'tanh' },
      { id: 'a2', type: 'activation', fn: 'relu' },
      { id: 'a3', type: 'activation', fn: 'sigmoid' },
    ],
    edges: [
      ['x', 'h'],
      ['h', 'a0'],
    ],
    predictions: ['a0'],
    loss: null,
    data: { set: 'sine', n: 40, noise: 0.2, val: 0.3, seed: 7 },
    train: { optimizer: 'sgd', lr: 0.03, batch: 'full', seed: 11 },
  });
  if (!r.ok) throw new Error(r.error);
  return r.graph;
}

beforeEach(() => {
  s().loadGraph(startGraph());
  s().select(null);
  host = document.createElement('div');
  document.body.appendChild(host);
  root = createRoot(host);
  act(() => root.render(<Canvas />));
});
afterEach(() => {
  act(() => root.unmount());
  host.remove();
});

const must = (id: string): Element => {
  const el = host.querySelector(`[data-testid="${id}"]`);
  if (!el) throw new Error(`missing ${id}`);
  return el;
};

function ptr(el: Element, type: string, x: number, y: number, pointerType: string) {
  const e = new Event(type, { bubbles: true, cancelable: true });
  Object.assign(e, { pointerId: 1, clientX: x, clientY: y, button: 0, pointerType });
  act(() => void el.dispatchEvent(e));
}

function screenOf(fn: (l: ReturnType<typeof layoutLayers>) => [number, number]): [number, number] {
  const g = s().graph;
  const layout = layoutLayers(g, analyse(g, s().dataInfo));
  const view = { w: 800, h: 500 };
  const f = fitTransform(layout.size, { w: view.w, h: view.h - 44 });
  const t = layerBaseView({ ...f, ty: f.ty + 44 }, layout.size, view, 44);
  const [x, y] = fn(layout);
  return [x * t.scale + t.tx, y * t.scale + t.ty];
}

for (const pointerType of ['mouse', 'touch']) {
  test(`${pointerType}: three drags from an output that already has a wire add three wires`, () => {
    for (const id of ['a1', 'a2', 'a3']) {
      const before = s().graph.edges.length;
      const [fx, fy] = screenOf((l) => {
        const b = l.nodes['h'];
        return b ? [b.x + b.w - 14, b.y + b.h / 2] : [0, 0];
      });
      const [tx, ty] = screenOf((l) => {
        const b = l.nodes[id];
        return b ? [b.x + b.w / 2, b.y + b.h / 2] : [0, 0];
      });
      const port = must('port-out-h');
      ptr(port, 'pointerdown', fx, fy, pointerType);
      ptr(port, 'pointermove', fx + 20, fy + 20, pointerType);
      ptr(port, 'pointermove', tx, ty, pointerType);
      // nothing is dimmed as "replaced": the target port is free, and the output's wires are kept
      expect(host.querySelector('[data-replace="true"]')).toBeNull();
      ptr(port, 'pointerup', tx, ty, pointerType);
      expect(s().graph.edges).toHaveLength(before + 1);
      expect(s().graph.edges).toContainEqual(['h', id]);
    }
    const fromH = s().graph.edges.filter((e) => e[0] === 'h');
    expect(fromH.map((e) => e[1])).toEqual(['a0', 'a1', 'a2', 'a3']);
    expect(s().graph.edges).toContainEqual(['x', 'h']);
  });
}
