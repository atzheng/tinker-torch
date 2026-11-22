// @vitest-environment jsdom
import { afterEach, beforeAll, beforeEach, expect, test } from 'vitest';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { PALETTE, analyse, connect, layoutLayers, setLoss } from '@tinker/core';
import { Canvas } from '../src/canvas/Canvas.js';
import { fitTransform } from '../src/canvas/pure.js';
import { layerBaseView } from '../src/canvas/viewport.js';
import { dnd } from '../src/canvas/dnd.js';
import { Palette } from '../src/palette/Palette.js';
import { mlpGraph as initialGraph } from './mlpGraph.js';
import { useStore } from '../src/state/store.js';
import { UndoRedo } from '../src/topbar/UndoRedo.js';

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const s = () => useStore.getState();
let host: HTMLDivElement;
let root: Root;

// The tests have no timers and no waitFor: every gesture is dispatched synchronously inside act().
// The only time dependence is the 5 s default timeout against a cold first run (module JIT, first
// layout and render), which starved CPUs stretch from ~0.4 s to seconds. The warm-up runs the
// work once in a hook (10 s budget) so no test pays for it.
beforeAll(() => {
  const warm = document.createElement('div');
  document.body.appendChild(warm);
  const r = createRoot(warm);
  s().loadGraph(initialGraph());
  act(() =>
    r.render(
      <>
        <Palette />
        <Canvas />
      </>,
    ),
  );
  act(() => r.unmount());
  warm.remove();
});

beforeEach(() => {
  // jsdom lays nothing out: the canvas region is 800 x 500 at the origin (the size `screenOf` assumes).
  Element.prototype.getBoundingClientRect = () =>
    ({
      left: 0,
      top: 0,
      right: 800,
      bottom: 500,
      x: 0,
      y: 0,
      width: 800,
      height: 500,
      toJSON: () => ({}),
    }) as DOMRect;
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
  // `dnd` is module state: a test that fails after arming an entry must not leak it into the next
  // one (the mouse and touch cases of an entry are neighbours, and arming twice disarms).
  act(() => dnd.arm(null));
  dnd.cancel();
  act(() => root.unmount());
  host.remove();
});

const must = (id: string): Element => {
  const el = host.querySelector(`[data-testid="${id}"]`);
  if (!el) throw new Error(`missing ${id}`);
  return el;
};

function ptr(el: Element, type: string, x: number, y: number, pointerType = 'touch') {
  const e = new Event(type, { bubbles: true, cancelable: true });
  Object.assign(e, { pointerId: 1, clientX: x, clientY: y, button: 0, pointerType });
  act(() => void el.dispatchEvent(e));
}

/** Where (client px) a layout point lands; the canvas is 800x500 at the origin in jsdom. */
function screenOf(fn: (l: ReturnType<typeof layoutLayers>) => [number, number]): [number, number] {
  const g = s().graph;
  const layout = layoutLayers(g, analyse(g, s().dataInfo));
  const view = { w: 800, h: 500 };
  const f = fitTransform(layout.size, { w: view.w, h: view.h - 44 }); // 44: header strip above the drawing
  const t = layerBaseView({ ...f, ty: f.ty + 44 }, layout.size, view, 44);
  const [x, y] = fn(layout);
  return [x * t.scale + t.tx, y * t.scale + t.ty];
}

function dragEntry(key: string, to: [number, number], kind = 'touch') {
  const el = must(`palette-${key}`);
  ptr(el, 'pointerdown', 20, 20, kind);
  ptr(el, 'pointermove', 60, 60, kind);
  ptr(el, 'pointermove', to[0], to[1], kind);
  return () => ptr(el, 'pointerup', to[0], to[1], kind);
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

const EMPTY: [number, number] = [780, 470];
const newNodes = (before: { nodes: { id: string }[] }) =>
  s().graph.nodes.filter((n) => !before.nodes.some((m) => m.id === n.id));

// One test per entry and pointer kind, so each stays far under the timeout under load.
const ENTRY_KINDS = PALETTE.flatMap((entry) =>
  ['mouse', 'touch'].map((kind) => [entry, kind] as const),
);

test.each(ENTRY_KINDS)(
  'palette entry %o dragged onto empty canvas (%s) is placed there, inert, in one undo step',
  (entry, kind) => {
    {
      const before = s().graph;
      dragEntry(entry.key, EMPTY, kind)();
      const added = newNodes(before);
      expect(added.length, entry.key).toBeGreaterThan(0);
      for (const n of added)
        expect(s().graph.positions?.[n.id], `${entry.key} ${n.id}`).toBeDefined();
      expect(s().analysis.trainable, entry.key).toBe(true);
      act(() => s().undo());
      expect(s().graph, entry.key).toEqual(before);
      act(() => s().redo());
      expect(newNodes(before).length, entry.key).toBe(added.length);
      act(() => s().undo());
    }
  },
);

test.each(ENTRY_KINDS)(
  'tap-to-add: arm palette entry %o (%s), tap empty canvas, it lands there',
  (entry, kind) => {
    {
      const before = s().graph;
      const el = must(`palette-${entry.key}`);
      ptr(el, 'pointerdown', 20, 20, kind);
      ptr(el, 'pointerup', 20, 20, kind);
      expect(must('armed-hint').textContent, entry.key).toMatch(/empty canvas/);
      const svg = must('canvas-svg');
      ptr(svg, 'pointerdown', EMPTY[0], EMPTY[1], kind);
      ptr(svg, 'pointerup', EMPTY[0], EMPTY[1], kind);
      const added = newNodes(before);
      expect(added.length, entry.key).toBeGreaterThan(0);
      for (const n of added)
        expect(s().graph.positions?.[n.id], `${entry.key} ${n.id}`).toBeDefined();
      act(() => s().undo());
    }
  },
);

test('an unwired piece shows the inert state and training stays possible; wiring it in makes it live', () => {
  dragEntry('square', EMPTY)();
  const id = newNodes(initialGraph())[0]?.id ?? '';
  expect(must(`node-${id}`).getAttribute('data-inert')).toBe('true');
  expect(must(`inert-${id}`).textContent).toBe('not connected');
  expect(s().analysis.trainable).toBe(true);
  act(() => void s().apply(connect(s().graph, 'err', id)));
  act(() => void s().apply(setLoss(s().graph, id)));
  expect(must(`node-${id}`).getAttribute('data-inert')).toBe('false');
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

  // A Scale onto the Subtract: same family, but its second wire would be lost, so the edit refuses it.
  const before = s().graph;
  const [yx, yy] = screenOf((l) => {
    const b = l.nodes['diff'];
    return b ? [b.x + b.w / 2, b.y + b.h / 2] : [0, 0];
  });
  dragEntry('scale', [yx, yy])();
  expect(s().graph).toBe(before);
  expect(must('toast').textContent?.length).toBeGreaterThan(5);
});
