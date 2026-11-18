// @vitest-environment jsdom
import { afterEach, beforeEach, expect, test } from 'vitest';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { analyse, layoutLayers } from '@tinker/core';
import { Canvas } from '../src/canvas/Canvas.js';
import { fitTransform } from '../src/canvas/pure.js';
import { Palette } from '../src/palette/Palette.js';
import { initialGraph } from '../src/state/graphSlice.js';
import { useStore } from '../src/state/store.js';
import { UndoRedo } from '../src/topbar/UndoRedo.js';

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const s = () => useStore.getState();
let host: HTMLDivElement;
let root: Root;

beforeEach(() => {
  s().loadGraph(initialGraph());
  s().select(null);
  host = document.createElement('div');
  document.body.appendChild(host);
  root = createRoot(host);
  act(() =>
    root.render(
      <>
        <UndoRedo />
        <Palette />
        <Canvas />
      </>,
    ),
  );
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

function ptr(el: Element, type: string, x: number, y: number) {
  const e = new Event(type, { bubbles: true, cancelable: true });
  Object.assign(e, { pointerId: 1, clientX: x, clientY: y, button: 0, pointerType: 'touch' });
  act(() => void el.dispatchEvent(e));
}

/** Where (client px) a layout point lands; the canvas is 800x500 at the origin in jsdom. */
function screenOf(fn: (l: ReturnType<typeof layoutLayers>) => [number, number]): [number, number] {
  const g = s().graph;
  const layout = layoutLayers(g, analyse(g, s().dataInfo));
  const t = fitTransform(layout.size, { w: 800, h: 500 - 44 }); // 44: header strip above the drawing
  const [x, y] = fn(layout);
  return [x * t.scale + t.tx, y * t.scale + t.ty + 44];
}

function dragEntry(key: string, to: [number, number]) {
  const el = must(`palette-${key}`);
  ptr(el, 'pointerdown', 20, 20);
  ptr(el, 'pointermove', 60, 60);
  ptr(el, 'pointermove', to[0], to[1]);
  return () => ptr(el, 'pointerup', to[0], to[1]);
}

test('drag a piece onto a wire inserts it, onto a piece swaps it; undo restores', () => {
  const original = s().graph;
  const [lx, ly] = screenOf((l) => {
    const w = l.wires.find((x) => x.edge === 1);
    return w ? w.labelAt : [0, 0];
  });
  const release = dragEntry('dropout', [lx, ly]);
  expect(must('palette-ghost').getAttribute('data-target')).toBe('wire');
  expect(host.querySelector('[data-testid="wire-1"]')?.getAttribute('data-drop')).toBe('true');
  release();
  expect(s().graph.nodes.some((n) => n.id === 'dropout1')).toBe(true);
  expect(s().selection).toEqual({ kind: 'node', id: 'dropout1' });

  const [ax, ay] = screenOf((l) => {
    const b = l.nodes['act1'];
    return b ? [b.x + b.w / 2, b.y + b.h / 2] : [0, 0];
  });
  dragEntry('activation', [ax, ay])();
  expect(s().graph.nodes.find((n) => n.id === 'act1')).toMatchObject({ type: 'activation' });

  act(() => must('btn-undo').dispatchEvent(new MouseEvent('click', { bubbles: true })));
  act(() => must('btn-undo').dispatchEvent(new MouseEvent('click', { bubbles: true })));
  expect(s().graph).toEqual(original);
});

test('dropping on empty canvas does nothing', () => {
  const before = s().graph;
  dragEntry('dropout', [790, 490])();
  expect(s().graph).toBe(before);
  expect(host.querySelector('[data-testid="palette-ghost"]')).toBeNull();
});

test('Delete removes the selected piece, Cmd+Z restores, and a cycle shows the toast', () => {
  const original = s().graph;
  act(() => s().selectNode('act1'));
  act(() => void window.dispatchEvent(new KeyboardEvent('keydown', { key: 'Delete' })));
  expect(s().graph.nodes.some((n) => n.id === 'act1')).toBe(false);
  expect(s().graph.edges).toContainEqual(['lin1', 'lin2']);
  act(() => void window.dispatchEvent(new KeyboardEvent('keydown', { key: 'z', metaKey: true })));
  expect(s().graph).toEqual(original);
  act(
    () =>
      void window.dispatchEvent(
        new KeyboardEvent('keydown', { key: 'z', metaKey: true, shiftKey: true }),
      ),
  );
  expect(s().graph.nodes.some((n) => n.id === 'act1')).toBe(false);
  act(() => s().undo());

  // Wire from lin2's output back to lin1's input: a cycle.
  const [fx, fy] = screenOf((l) => {
    const b = l.nodes['lin2'];
    return b ? [b.x + b.w - 28, b.y + b.h / 2] : [0, 0];
  });
  const [tx, ty] = screenOf((l) => {
    const b = l.nodes['lin1'];
    return b ? [b.x + b.w / 2, b.y + b.h / 2] : [0, 0];
  });
  const port = must('port-out-lin2');
  ptr(port, 'pointerdown', fx, fy);
  ptr(port, 'pointermove', fx + 20, fy + 20);
  ptr(port, 'pointermove', tx, ty);
  expect(host.querySelector('[data-testid="wire-preview"]')).not.toBeNull();
  ptr(port, 'pointerup', tx, ty);
  expect(s().graph).toBe(original);
  expect(must('toast').textContent?.length).toBeGreaterThan(5);
});

test('edits during a run: a drop and a delete do not throw and pause the run; a refused drop toasts', () => {
  act(() => useStore.setState({ running: true }));
  const [lx, ly] = screenOf((l) => {
    const w = l.wires.find((x) => x.edge === 1);
    return w ? w.labelAt : [0, 0];
  });
  expect(() => dragEntry('dropout', [lx, ly])()).not.toThrow();
  expect(s().graph.nodes.some((n) => n.id === 'dropout1')).toBe(true);
  expect(s().running).toBe(false);

  act(() => useStore.setState({ running: true }));
  act(() => s().selectNode('dropout1'));
  expect(() =>
    act(() => void window.dispatchEvent(new KeyboardEvent('keydown', { key: 'Backspace' }))),
  ).not.toThrow();
  expect(s().graph.nodes.some((n) => n.id === 'dropout1')).toBe(false);
  expect(s().running).toBe(false);

  // Target onto the Output: same family, but the edit refuses it.
  const before = s().graph;
  const [yx, yy] = screenOf((l) => {
    const b = l.nodes['yhat'];
    return b ? [b.x + b.w / 2, b.y + b.h / 2] : [0, 0];
  });
  dragEntry('target', [yx, yy])();
  expect(s().graph).toBe(before);
  expect(must('toast').textContent?.length).toBeGreaterThan(5);
});
