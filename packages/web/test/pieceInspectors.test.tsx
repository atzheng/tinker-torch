// @vitest-environment jsdom
import { afterEach, beforeEach, expect, test } from 'vitest';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { analyse, parseGraph, partitionNodes, type Graph } from '@tinker/core';
import { legibleAt } from '../src/canvas/pure.js';
import { Canvas } from '../src/canvas/Canvas.js';
import { mlpGraph as initialGraph } from './mlpGraph.js';
import { useStore } from '../src/state/store.js';

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const s = () => useStore.getState();
let host: HTMLDivElement;
let root: Root;

function everyPieceGraph(): Graph {
  const g = initialGraph();
  const parsed = parseGraph({
    ...g,
    nodes: [
      { id: 'x', type: 'input', columns: ['x1'] },
      { id: 'feat', type: 'feature', degree: 2, sin: true, cos: false, products: false },
      { id: 'lin1', type: 'linear', out: 3, bias: true },
      { id: 'drop', type: 'dropout', p: 0.2 },
      { id: 'lin2', type: 'linear', out: 3, bias: true },
      { id: 'add', type: 'add' },
      { id: 'lin3', type: 'linear', out: 1, bias: true },
      { id: 'yhat', type: 'link', fn: 'sigmoid' },
      { id: 'y', type: 'target' },
      { id: 'diff', type: 'subtract' },
      { id: 'hub', type: 'huber', delta: 1 },
      { id: 'mean', type: 'mean' },
      { id: 'wl1', type: 'weights', of: 'lin1' },
      { id: 'n1', type: 'norm', kind: 'l2' },
      { id: 'pen', type: 'scale', c: 0.01 },
      { id: 'total', type: 'add' },
    ],
    edges: [
      ['x', 'feat'],
      ['feat', 'lin1'],
      ['lin1', 'drop'],
      ['drop', 'lin2'],
      ['lin2', 'add.a'],
      ['lin1', 'add.b'],
      ['add', 'lin3'],
      ['lin3', 'yhat'],
      ['yhat', 'diff.a'],
      ['y', 'diff.b'],
      ['diff', 'hub'],
      ['hub', 'mean'],
      ['wl1', 'n1'],
      ['n1', 'pen'],
      ['mean', 'total.a'],
      ['pen', 'total.b'],
    ],
    predictions: ['yhat'],
    loss: 'total',
  });
  if (!parsed.ok) throw new Error(parsed.error);
  return parsed.graph;
}

beforeEach(() => {
  s().loadGraph(everyPieceGraph());
  s().setViewTab('model');
  s().select(null);
  host = document.createElement('div');
  document.body.appendChild(host);
  root = createRoot(host);
  act(() => root.render(<Canvas />));
});
afterEach(() => {
  act(() => root.unmount());
  host.remove();
  act(() => useStore.setState({ preset: null }));
});

const q = (id: string) => host.querySelector(`[data-testid="${id}"]`);
const must = (id: string): Element => {
  const el = q(id);
  if (!el) throw new Error(`missing ${id}`);
  return el;
};
const node = (id: string) => s().graph.nodes.find((n) => n.id === id);
// Each piece is on the tab it belongs to (CONTRACTS.md section 9, "Tabs"): show it, then select it.
const tabOf = (id: string) =>
  partitionNodes(s().graph, analyse(s().graph, s().dataInfo))[id] ?? 'model';
const pick = (id: string) =>
  act(() => {
    s().setViewTab(tabOf(id));
    s().select({ kind: 'node', id });
  });

function setValue(el: Element, value: string, proto: object, event: string) {
  const setter = Object.getOwnPropertyDescriptor(proto, 'value')?.set;
  if (!setter) throw new Error('no value setter');
  act(() => {
    setter.call(el, value);
    el.dispatchEvent(new Event(event, { bubbles: true }));
  });
}
function typeAndBlur(el: Element, value: string) {
  setValue(el, value, HTMLInputElement.prototype, 'input');
  act(() => void el.dispatchEvent(new Event('focusout', { bubbles: true })));
}
const choose = (id: string, v: string) =>
  setValue(must(id), v, HTMLSelectElement.prototype, 'change');
const check = (id: string) => act(() => void (must(id) as HTMLInputElement).click());
const steps = () => s().past.length;

test('feature inspector: degree, sin, cos, products; width shown; one undo step each', () => {
  pick('feat');
  const out = () => must('feature-width').textContent;
  expect(out()).toBe('Produces 3 columns'); // x, x^2, sin
  const before = steps();
  typeAndBlur(must('field-degree'), '3');
  expect(node('feat')).toMatchObject({ degree: 3 });
  expect(steps()).toBe(before + 1);
  expect(out()).toBe('Produces 4 columns');
  check('field-cos');
  expect(node('feat')).toMatchObject({ cos: true });
  expect(out()).toBe('Produces 5 columns');
  check('field-sin');
  expect(node('feat')).toMatchObject({ sin: false });
  check('field-products');
  expect(node('feat')).toMatchObject({ products: true });
  expect(steps()).toBe(before + 4);
  act(() => s().undo());
  expect(node('feat')).toMatchObject({ products: false });
  // The main label (typeset by KaTeX; its TeX is the aria-label) stays at any scale.
  // The tag (`\cos`) is drawn too when the scale allows (layer view starts at 0.73), so look at all.
  const labels = [...must('node-feat').querySelectorAll('.tk-tex')].map((e) =>
    e.getAttribute('aria-label'),
  );
  expect(labels).toContain('x, \\dots, x^{3}');
});

test('dropout inspector: p commits once; out of range is refused', () => {
  pick('drop');
  const before = steps();
  typeAndBlur(must('field-p'), '0.5');
  expect(node('drop')).toMatchObject({ p: 0.5 });
  expect(steps()).toBe(before + 1);
  expect(must('node-drop').textContent).toContain('0.5');
  typeAndBlur(must('field-p'), '1');
  expect(node('drop')).toMatchObject({ p: 0.5 });
  expect(steps()).toBe(before + 1);
  act(() => s().undo());
  expect(node('drop')).toMatchObject({ p: 0.2 });
});

test('add inspector: both input widths, and the mismatch sentence', () => {
  pick('add');
  expect(must('add-widths').textContent).toBe('Input a: 3. Input b: 3.');
  act(() => {
    const r = s().graph;
    s().loadGraph({
      ...r,
      nodes: r.nodes.map((n) => (n.id === 'lin1' ? { ...n, out: 5 } : n)),
    });
  });
  pick('add');
  expect(must('add-widths').textContent).toBe('Input a: 3. Input b: 5.');
  expect(must('inspector-issues').textContent).toContain('same width');
});

test('huber inspector: delta commits once and shows in the piece; zero is refused', () => {
  pick('hub');
  const before = steps();
  typeAndBlur(must('field-delta'), '2');
  expect(node('hub')).toMatchObject({ delta: 2 });
  expect(steps()).toBe(before + 1);
  typeAndBlur(must('field-delta'), '0');
  expect(node('hub')).toMatchObject({ delta: 2 });
  expect(steps()).toBe(before + 1);
  expect(s().error).toContain('above 0');
  act(() => s().undo());
  expect(node('hub')).toMatchObject({ delta: 1 });
});

test('scale inspector: c commits once, any finite number', () => {
  pick('pen');
  const before = steps();
  typeAndBlur(must('field-c'), '-0.5');
  expect(node('pen')).toMatchObject({ c: -0.5 });
  expect(steps()).toBe(before + 1);
  act(() => s().undo());
  expect(node('pen')).toMatchObject({ c: 0.01 });
});

test('norm inspector: kind select is one undo step; weights says which Linear it reads', () => {
  pick('n1');
  const before = steps();
  choose('field-norm', 'l1');
  expect(node('n1')).toMatchObject({ kind: 'l1' });
  expect(steps()).toBe(before + 1);
  expect(must('node-n1').textContent).toContain('L1 norm');
  act(() => s().undo());
  expect(node('n1')).toMatchObject({ kind: 'l2' });
  pick('wl1');
  expect(must('weights-of').textContent).toContain('lin1');
});

test('link inspector: link select is one undo step', () => {
  pick('yhat');
  const before = steps();
  choose('field-link', 'softmax');
  expect(node('yhat')).toMatchObject({ fn: 'softmax' });
  expect(steps()).toBe(before + 1);
  act(() => s().undo());
  expect(node('yhat')).toMatchObject({ fn: 'sigmoid' });
});

test('the L node carries its role; the node pinned as ŷ carries none', () => {
  act(() => s().setViewTab('loss'));
  expect(must('node-yhat').getAttribute('data-role')).toBeNull();
  expect(must('node-total').getAttribute('data-role')).toBe('loss');
  expect(must('node-mean').getAttribute('data-role')).toBeNull();
});

test('locked pieces refuse every inspector edit', () => {
  act(() => useStore.setState({ preset: { title: 'T', locked: ['feat', 'drop', 'hub', 'yhat'] } }));
  const before = steps();
  pick('feat');
  typeAndBlur(must('field-degree'), '4');
  check('field-sin');
  pick('drop');
  typeAndBlur(must('field-p'), '0.4');
  pick('hub');
  typeAndBlur(must('field-delta'), '3');
  pick('yhat');
  choose('field-link', 'softmax');
  expect(steps()).toBe(before);
  expect(s().error).toBe('This piece is locked.');
  expect(node('feat')).toMatchObject({ degree: 2, sin: true });
  expect(node('yhat')).toMatchObject({ fn: 'sigmoid' });
});

test('secondary text is hidden when it would be under 8 px on screen', () => {
  expect(legibleAt(undefined, 9)).toBe(true);
  expect(legibleAt(1, 9)).toBe(true);
  expect(legibleAt(0.3, 12.5)).toBe(false);
  expect(legibleAt(0.64, 12.5)).toBe(true);
  expect(legibleAt(0.88, 9)).toBe(false);
  expect(legibleAt(0.89, 9)).toBe(true);
});
