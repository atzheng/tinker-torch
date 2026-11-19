// @vitest-environment jsdom
import { afterEach, beforeEach, expect, test } from 'vitest';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { parseGraph, type Graph } from '@tinker/core';
import { legibleAt } from '../src/canvas/pure.js';
import { Canvas } from '../src/canvas/Canvas.js';
import { initialGraph } from '../src/state/graphSlice.js';
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
      { id: 'yhat', type: 'output', link: 'identity' },
      { id: 'y', type: 'target' },
      { id: 'loss', type: 'loss', fn: 'huber', delta: 1, penalty: { kind: 'l2', lambda: 0.01 } },
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
      ['yhat', 'loss.pred'],
      ['y', 'loss.target'],
    ],
  });
  if (!parsed.ok) throw new Error(parsed.error);
  return parsed.graph;
}

beforeEach(() => {
  s().loadGraph(everyPieceGraph());
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
const pick = (id: string) => act(() => s().select({ kind: 'node', id }));

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
  expect(must('node-feat').textContent).toContain('x to x^3'); // main label stays at any scale
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

test('loss inspector: fn, huber delta, penalty kind, lambda, remove; tag drawn', () => {
  pick('loss');
  const before = steps();
  typeAndBlur(must('field-delta'), '2');
  expect(node('loss')).toMatchObject({ delta: 2 });
  typeAndBlur(must('field-penalty-lambda'), '0.5');
  expect(node('loss')).toMatchObject({ penalty: { kind: 'l2', lambda: 0.5 } });
  choose('field-penalty-kind', 'l1');
  expect(node('loss')).toMatchObject({ penalty: { kind: 'l1', lambda: 0.5 } });
  expect(steps()).toBe(before + 3);
  choose('field-penalty-kind', 'none');
  expect(node('loss')).not.toHaveProperty('penalty.kind');
  expect(q('penalty-tag-loss')).toBeNull();
  expect(q('field-penalty-lambda')).toBeNull();
  choose('field-loss-fn', 'squared_error');
  expect(node('loss')).toMatchObject({ fn: 'squared_error' });
  expect(q('field-delta')).toBeNull();
  act(() => s().undo());
  expect(node('loss')).toMatchObject({ fn: 'huber' });
});

test('output inspector: link select is one undo step', () => {
  pick('yhat');
  const before = steps();
  choose('field-link', 'sigmoid');
  expect(node('yhat')).toMatchObject({ link: 'sigmoid' });
  expect(steps()).toBe(before + 1);
  act(() => s().undo());
  expect(node('yhat')).toMatchObject({ link: 'identity' });
});

test('locked pieces refuse every inspector edit', () => {
  act(() =>
    useStore.setState({ preset: { title: 'T', locked: ['feat', 'drop', 'loss', 'yhat'] } }),
  );
  const before = steps();
  pick('feat');
  typeAndBlur(must('field-degree'), '4');
  check('field-sin');
  pick('drop');
  typeAndBlur(must('field-p'), '0.4');
  pick('loss');
  typeAndBlur(must('field-delta'), '3');
  choose('field-penalty-kind', 'none');
  pick('yhat');
  choose('field-link', 'sigmoid');
  expect(steps()).toBe(before);
  expect(s().error).toBe('This piece is locked.');
  expect(node('feat')).toMatchObject({ degree: 2, sin: true });
  expect(node('yhat')).toMatchObject({ link: 'identity' });
});

test('lambda above 100 is refused with the schema sentence', () => {
  pick('loss');
  const before = steps();
  typeAndBlur(must('field-penalty-lambda'), '101');
  expect(node('loss')).toMatchObject({ penalty: { lambda: 0.01 } });
  expect(steps()).toBe(before);
  expect(s().error).toContain('0 to 100');
});

test('secondary text is hidden when it would be under 8 px on screen', () => {
  expect(legibleAt(undefined, 9)).toBe(true);
  expect(legibleAt(1, 9)).toBe(true);
  expect(legibleAt(0.3, 12.5)).toBe(false);
  expect(legibleAt(0.64, 12.5)).toBe(true);
  expect(legibleAt(0.88, 9)).toBe(false);
  expect(legibleAt(0.89, 9)).toBe(true);
});
