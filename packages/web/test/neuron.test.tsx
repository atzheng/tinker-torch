// @vitest-environment jsdom
import { afterEach, beforeEach, expect, test } from 'vitest';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { analyse, layoutNeurons, setLinearOut, type HostParams } from '@tinker/core';
import { Canvas } from '../src/canvas/Canvas.js';
import { fitTransform } from '../src/canvas/pure.js';
import { DataControls } from '../src/topbar/DataControls.js';
import { initialGraph } from '../src/state/graphSlice.js';
import { useStore } from '../src/state/store.js';

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const s = () => useStore.getState();
let el: HTMLDivElement;
let root: Root;

beforeEach(() => {
  s().loadGraph(initialGraph());
  s().select(null);
  s().setGrain('layer');
  useStore.setState({ host: null });
  el = document.createElement('div');
  document.body.appendChild(el);
  root = createRoot(el);
  act(() =>
    root.render(
      <>
        <DataControls />
        <Canvas />
      </>,
    ),
  );
});
afterEach(() => {
  act(() => root.unmount());
  el.remove();
});

const q = (id: string) => el.querySelector(`[data-testid="${id}"]`);
const must = (id: string): Element => {
  const e = q(id);
  if (!e) throw new Error(`missing ${id}`);
  return e;
};

function fire(target: Element, type: string, x: number, y: number, pointerType: string) {
  const e = new Event(type, { bubbles: true, cancelable: true });
  Object.assign(e, { pointerId: 1, clientX: x, clientY: y, button: 0, pointerType });
  target.dispatchEvent(e);
}
function tapAt(target: Element, x: number, y: number) {
  act(() => {
    fire(target, 'pointerdown', x, y, 'touch');
    fire(target, 'pointerup', x, y, 'touch');
  });
}
/** Client position of an abstract point: jsdom has no layout, so the canvas uses 800 x 500. */
function screenOf(x: number, y: number): [number, number] {
  const g = s().graph;
  const layout = layoutNeurons(g, analyse(g, s().dataInfo));
  const t = fitTransform(layout.size, { w: 800, h: 500 });
  return [x * t.scale + t.tx, y * t.scale + t.ty];
}
const neuron = () => layoutNeurons(s().graph, analyse(s().graph, s().dataInfo));

// lin1 is 1x4, lin2 is 4x1 in the initial graph.
const HAND: HostParams = {
  lin1: { W: Float32Array.from([0.5, -0.4, 0.3, -1.2]), b: new Float32Array(4) },
  lin2: { W: Float32Array.from([1, -1, 0.2, -0.2]), b: new Float32Array(1) },
};

const toggle = () => act(() => void (must('grain-toggle') as HTMLButtonElement).click());

test('the grain toggle is enabled and switches drawings; the stored graph never changes', () => {
  const before = s().graph;
  const pastBefore = s().past.length;
  const btn = must('grain-toggle') as HTMLButtonElement;
  expect(btn.disabled).toBe(false);
  expect(q('layer-view')).not.toBeNull();
  toggle();
  expect(s().grain).toBe('neuron');
  expect(q('neuron-view')).not.toBeNull();
  expect(q('layer-view')).toBeNull();
  expect(s().graph).toBe(before);
  expect(s().past.length).toBe(pastBefore);
  toggle();
  expect(s().grain).toBe('layer');
  expect(q('layer-view')).not.toBeNull();
  expect(s().graph).toBe(before);
  expect(s().canUndo).toBe(false);
});

test('1-4-1: 8 weight edges whose colour follows the sign of the host weights', () => {
  toggle();
  expect(el.querySelectorAll('.tk-edge')).toHaveLength(8);
  act(() => useStore.setState({ host: HAND }));
  const sign = (id: string) => must(id).querySelector('.tk-edge-line')?.getAttribute('data-sign');
  const width = (id: string) =>
    Number(must(id).querySelector('.tk-edge-line')?.getAttribute('stroke-width'));
  expect(sign('edge-lin1-0-0')).toBe('pos');
  expect(sign('edge-lin1-0-1')).toBe('neg');
  expect(sign('edge-lin1-0-3')).toBe('neg');
  expect(sign('edge-lin2-0-0')).toBe('pos');
  expect(sign('edge-lin2-1-0')).toBe('neg');
  expect(width('edge-lin1-0-3')).toBeGreaterThan(width('edge-lin1-0-2')); // |-1.2| > |0.3|
  // a new host with flipped signs repaints without a re-layout
  const flipped: HostParams = {
    lin1: { W: HAND.lin1?.W.map((v) => -v) ?? new Float32Array(4), b: new Float32Array(4) },
    lin2: { W: HAND.lin2?.W.map((v) => -v) ?? new Float32Array(4), b: new Float32Array(1) },
  };
  const node = must('edge-lin1-0-0');
  act(() => useStore.setState({ host: flipped }));
  expect(must('edge-lin1-0-0')).toBe(node); // same element: no remount
  expect(sign('edge-lin1-0-0')).toBe('neg');
  expect(sign('edge-lin1-0-1')).toBe('pos');
  expect(Array.from(HAND.lin1?.W ?? [])).toEqual(Array.from(Float32Array.of(0.5, -0.4, 0.3, -1.2))); // never written into
});

test('before any training the edges show the step-0 weights from initParams', () => {
  toggle();
  const signs = [...el.querySelectorAll('.tk-edge-line')].map((l) => l.getAttribute('data-sign'));
  expect(signs).toHaveLength(8);
  for (const sg of signs) expect(['pos', 'neg']).toContain(sg);
});

test('a stale host entry of the wrong size is drawn neutral, not wrong', () => {
  toggle();
  act(() => useStore.setState({ host: { lin1: { W: new Float32Array(3) } } }));
  expect(must('edge-lin1-0-0').querySelector('.tk-edge-line')?.getAttribute('data-sign')).toBe(
    'none',
  );
  expect(must('edge-lin2-0-0').querySelector('.tk-edge-line')?.getAttribute('data-sign')).toBe(
    'none',
  );
});

test('a 16-wide Linear stays collapsed and says how many edges it hides', () => {
  act(() => void s().apply(setLinearOut(s().graph, 'lin1', 16)));
  toggle();
  expect(el.querySelectorAll('.tk-edge')).toHaveLength(0);
  expect(must('edges-lin1').textContent).toBe('16 edges');
  expect(must('edges-lin2').textContent).toBe('16 edges');
});

test('hover with a mouse shows the weight value, and it follows the host', () => {
  toggle();
  act(() => useStore.setState({ host: HAND }));
  const e = neuron().edges.find((x) => x.node === 'lin1' && x.j === 1);
  expect(e).toBeDefined();
  if (!e) return;
  const [x, y] = screenOf((e.from[0] + e.to[0]) / 2, (e.from[1] + e.to[1]) / 2);
  const target = must('edge-lin1-0-1');
  act(() => fire(target, 'pointermove', x, y, 'mouse'));
  expect(must('edge-tip').textContent).toBe('-0.400');
  act(() =>
    useStore.setState({
      host: {
        ...HAND,
        lin1: { W: Float32Array.from([0.5, 0.25, 0.3, 0]), b: new Float32Array(4) },
      },
    }),
  );
  expect(must('edge-tip').textContent).toBe('0.250');
  act(() => fire(target, 'pointerout', x, y, 'mouse'));
  expect(q('edge-tip')).toBeNull();
});

test('touch does not open the hover tip; a tap selects the edge and the inspector shows the value', () => {
  toggle();
  act(() => useStore.setState({ host: HAND }));
  const e = neuron().edges.find((x) => x.node === 'lin2' && x.i === 1);
  expect(e).toBeDefined();
  if (!e) return;
  const [x, y] = screenOf((e.from[0] + e.to[0]) / 2, (e.from[1] + e.to[1]) / 2);
  const target = must('edge-lin2-1-0');
  act(() => fire(target, 'pointermove', x, y, 'touch'));
  expect(q('edge-tip')).toBeNull();
  tapAt(target, x, y);
  expect(s().selection).toEqual({ kind: 'edge', node: 'lin2', i: 1, j: 0 });
  expect(must('inspector-weight').textContent).toBe('Value -1.0000');
  expect(q('edge-selected')).not.toBeNull();
  expect(q('inspector-delete')).toBeNull(); // an edge is not deleted
});

test('where fat edge hit strokes overlap, the line nearest to the press is selected', () => {
  toggle();
  // lin1 edges fan out from one input to four units; press on lin1 j=3 but through the hit
  // stroke of lin1 j=2's element (the DOM target is whichever is on top).
  const e3 = neuron().edges.find((x) => x.node === 'lin1' && x.j === 3);
  expect(e3).toBeDefined();
  if (!e3) return;
  const [x, y] = screenOf(e3.from[0] * 0.3 + e3.to[0] * 0.7, e3.from[1] * 0.3 + e3.to[1] * 0.7);
  tapAt(must('edge-lin1-0-2'), x, y);
  expect(s().selection).toEqual({ kind: 'edge', node: 'lin1', i: 0, j: 3 });
});

test('tapping a unit selects it and shows its bias; the background clears', () => {
  toggle();
  act(() =>
    useStore.setState({
      host: {
        ...HAND,
        lin1: {
          W: HAND.lin1?.W ?? new Float32Array(4),
          b: Float32Array.from([0.1, 0.2, 0.3, 0.4]),
        },
      },
    }),
  );
  tapAt(must('unit-lin1-2'), 10, 10);
  expect(s().selection).toEqual({ kind: 'unit', node: 'lin1', index: 2 });
  expect(must('inspector-bias').textContent).toBe('Bias 0.3000');
  expect(must('unit-lin1-2').getAttribute('data-selected')).toBe('true');
  act(() => fire(must('canvas-svg'), 'pointerdown', 5, 5, 'touch'));
  expect(s().selection).toBeNull();
});

test('leaving neuron view clears an edge or unit selection but keeps a piece selected', () => {
  toggle();
  act(() => s().select({ kind: 'unit', node: 'lin1', index: 0 }));
  toggle();
  expect(s().selection).toBeNull();
  toggle();
  act(() => s().selectNode('lin1'));
  toggle();
  expect(s().selection).toEqual({ kind: 'node', id: 'lin1' });
});

test('a selected edge that stops existing after an edit is not shown', () => {
  toggle();
  act(() => s().select({ kind: 'edge', node: 'lin2', i: 3, j: 0 }));
  expect(q('inspector')).not.toBeNull();
  act(() => void s().apply(setLinearOut(s().graph, 'lin1', 2))); // lin2 is now 2x1
  expect(q('inspector')).toBeNull();
});

test('Delete with an edge or unit selected does nothing to the graph', () => {
  toggle();
  const before = s().graph;
  act(() => s().select({ kind: 'unit', node: 'lin1', index: 1 }));
  act(() => void window.dispatchEvent(new KeyboardEvent('keydown', { key: 'Delete' })));
  expect(s().graph).toBe(before);
});
