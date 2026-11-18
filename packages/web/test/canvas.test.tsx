// @vitest-environment jsdom
import { afterEach, beforeEach, expect, test } from 'vitest';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { analyse, parseGraph, setLinearOut, type Graph, type HostParams } from '@tinker/core';
import { Canvas } from '../src/canvas/Canvas.js';
import { initialGraph } from '../src/state/graphSlice.js';
import { useStore } from '../src/state/store.js';

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
  act(() => root.render(<Canvas />));
});
afterEach(() => {
  act(() => root.unmount());
  host.remove();
});

const q = (id: string) => host.querySelector(`[data-testid="${id}"]`);
const must = (id: string): Element => {
  const el = q(id);
  if (!el) throw new Error(`missing ${id}`);
  return el;
};

/** A press and release on an element: what a click or a touch tap sends. */
function tap(el: Element) {
  const ev = (type: string) => {
    const e = new Event(type, { bubbles: true, cancelable: true });
    Object.assign(e, { pointerId: 1, clientX: 10, clientY: 10, button: 0, pointerType: 'touch' });
    return e;
  };
  act(() => {
    el.dispatchEvent(ev('pointerdown'));
    el.dispatchEvent(ev('pointerup'));
  });
}

function setValue(el: Element, value: string, proto: object, event: string) {
  const setter = Object.getOwnPropertyDescriptor(proto, 'value')?.set;
  if (!setter) throw new Error('no value setter');
  act(() => {
    setter.call(el, value);
    el.dispatchEvent(new Event(event, { bubbles: true }));
  });
}

const fillOpacities = (id: string) =>
  Array.from(must(id).querySelectorAll('rect[fill]')).map((r) => r.getAttribute('fill-opacity'));

test('the SPEC example draws every piece and width labels 1, 4, 4, 1', () => {
  for (const id of ['x', 'lin1', 'act1', 'lin2', 'yhat', 'y', 'loss']) {
    expect(q(`node-${id}`)).not.toBeNull();
  }
  const label = (i: number) => must(`wire-label-${i}`).textContent;
  expect([0, 1, 2, 3].map(label)).toEqual(['1', '4', '4', '1']);
  expect(host.querySelectorAll('[data-error="true"]')).toHaveLength(0);
  expect(must('node-lin1').textContent).toContain('1 to 4');
  expect(must('node-lin1').textContent).toContain('8 params');
  expect(q('heat-lin1')).not.toBeNull();
});

test('lin2 out = 2 turns the wire red with the sentence; undo clears it', () => {
  act(() => void s().apply(setLinearOut(s().graph, 'lin2', 2)));
  const a = analyse(s().graph, s().dataInfo);
  const issue = a.issues[0];
  expect(issue?.message).toContain('width 1');
  const edge = s().graph.edges.findIndex((e) => e[0] === 'yhat'); // the wire into the loss
  expect(must(`wire-${edge}`).getAttribute('data-error')).toBe('true');
  expect(must(`issue-wire-${edge}`).textContent).toBe(issue?.message);
  act(() => s().undo());
  expect(host.querySelectorAll('[data-error="true"]')).toHaveLength(0);
  expect(q(`issue-wire-${edge}`)).toBeNull();
});

test('tapping a piece selects it and opens the inspector with its settings', () => {
  tap(must('node-lin1'));
  expect(s().selection).toEqual({ kind: 'node', id: 'lin1' });
  expect(must('node-lin1').getAttribute('data-selected')).toBe('true');
  expect(must('inspector-params').textContent).toBe('1 x 4 + 4 = 8 parameters');
  tap(must('canvas-svg'));
  expect(s().selection).toBeNull();
  expect(q('inspector')).toBeNull();
});

test('the inspector edits width and activation; undo restores both', () => {
  tap(must('node-lin1'));
  setValue(must('field-out'), '6', HTMLInputElement.prototype, 'input');
  const lin1 = () => s().graph.nodes.find((n) => n.id === 'lin1');
  expect(lin1()).toMatchObject({ out: 6 });
  expect(must('wire-label-1').textContent).toBe('6');
  expect(must('inspector-params').textContent).toBe('1 x 6 + 6 = 12 parameters');

  tap(must('node-act1'));
  setValue(must('field-fn'), 'tanh', HTMLSelectElement.prototype, 'change');
  expect(s().graph.nodes.find((n) => n.id === 'act1')).toMatchObject({ fn: 'tanh' });
  expect(must('node-act1').textContent).toBe('tanh');

  act(() => s().undo());
  expect(s().graph.nodes.find((n) => n.id === 'act1')).toMatchObject({ fn: 'relu' });
  act(() => s().undo());
  expect(lin1()).toMatchObject({ out: 4 });
  expect(must('wire-label-1').textContent).toBe('4');
});

test('an out-of-range width is refused and the graph is unchanged', () => {
  tap(must('node-lin1'));
  setValue(must('field-out'), '99', HTMLInputElement.prototype, 'input');
  expect(s().graph.nodes.find((n) => n.id === 'lin1')).toMatchObject({ out: 4 });
  expect(s().error).not.toBeNull();
});

test('tapping a wire selects it by its endpoints and shows its width', () => {
  tap(must('wire-1'));
  expect(s().selection).toEqual({ kind: 'wire', from: 'lin1', to: 'act1' });
  expect(must('inspector-width').textContent).toBe('Width 4');
  // An edit that shifts edge indices keeps the selection on the same wire.
  const g = s().graph;
  act(() => void s().apply({ ok: true, graph: { ...g, edges: [...g.edges].reverse() } }));
  expect(must('inspector-width').textContent).toBe('Width 4');
});

test('selection that names something deleted counts as none', () => {
  tap(must('node-act1'));
  expect(q('inspector')).not.toBeNull();
  const g = s().graph;
  act(
    () =>
      void s().apply({ ok: true, graph: { ...g, nodes: g.nodes.filter((n) => n.id !== 'act1') } }),
  );
  expect(q('inspector')).toBeNull();
});

test('a Linear with no HostParams entry (off the loss path) draws without a heat map', () => {
  const g = initialGraph();
  const parsed = parseGraph({
    ...g,
    nodes: [...g.nodes, { id: 'side', type: 'linear', out: 3, bias: true }],
    edges: [...g.edges, ['x', 'side']],
  });
  if (!parsed.ok) throw new Error(parsed.error);
  act(() => s().loadGraph(parsed.graph));
  expect(q('node-side')).not.toBeNull();
  expect(q('heat-side')).toBeNull();
  expect(q('heat-lin1')).not.toBeNull();
});

test('edges that do not resolve show their issue at a node or globally, not on a wire', () => {
  const g: Graph = { ...initialGraph(), edges: [...initialGraph().edges, ['x', 'ghost']] };
  act(() => s().loadGraph(g));
  const bad = g.edges.length - 1;
  expect(q(`wire-${bad}`)).toBeNull();
  const a = analyse(s().graph, s().dataInfo);
  expect(a.issues.length).toBeGreaterThan(0);
  const shown = host.querySelectorAll('.tk-callout, .tk-global-issues');
  expect(shown.length).toBeGreaterThan(0);
});

test('the heat map follows the weights from the train slice when it has them', () => {
  const before = fillOpacities('heat-lin1');
  const W = new Float32Array([5, 0, 0, 0]);
  const trained: HostParams = { lin1: { W, b: new Float32Array(4) } };
  act(() => useStore.setState({ host: trained } as never));
  const after = fillOpacities('heat-lin1');
  expect(after).not.toEqual(before);
  expect(after[0]).toBe('1');
  expect(Array.from(W)).toEqual([5, 0, 0, 0]);
  act(() => useStore.setState({ host: undefined } as never));
});

test('a stale weight array of the wrong size is ignored, not drawn', () => {
  act(() => useStore.setState({ host: { lin1: { W: new Float32Array(3) } } } as never));
  expect(q('heat-lin1')).toBeNull();
  act(() => useStore.setState({ host: undefined } as never));
});
