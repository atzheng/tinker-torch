// @vitest-environment jsdom
import { afterEach, beforeEach, expect, test } from 'vitest';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { Canvas } from '../src/canvas/Canvas.js';
import { dnd } from '../src/canvas/dnd.js';
import { Palette } from '../src/palette/Palette.js';
import { cardOwner, placeCard } from '../src/palette/PieceCard.js';
import { useStore } from '../src/state/store.js';
import { mlpGraph } from './mlpGraph.js';

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const s = () => useStore.getState();
let host: HTMLDivElement;
let root: Root;

beforeEach(() => {
  s().loadGraph(mlpGraph());
  s().setViewTab('model');
  s().select(null);
  host = document.createElement('div');
  document.body.appendChild(host);
  root = createRoot(host);
  act(() =>
    root.render(
      <>
        <Palette />
        <Canvas />
      </>,
    ),
  );
});
afterEach(() => {
  act(() => cardOwner.close());
  act(() => root.unmount());
  host.remove();
  act(() => useStore.setState({ show: null }));
});

const q = (id: string) => document.querySelector(`[data-testid="${id}"]`);
const must = (id: string): HTMLElement => {
  const el = q(id);
  if (!el) throw new Error(`missing ${id}`);
  return el as HTMLElement;
};
const click = (el: Element) => act(() => void (el as HTMLElement).click());
function ptr(el: Element, type: string, x = 5, y = 5, pointerType = 'mouse') {
  const e = new Event(type, { bubbles: true, cancelable: true });
  Object.assign(e, { pointerId: 1, clientX: x, clientY: y, button: 0, pointerType });
  act(() => void el.dispatchEvent(e));
}
const key = (k: string, el: Element = document.body) =>
  act(() => void el.dispatchEvent(new KeyboardEvent('keydown', { key: k, bubbles: true })));
const cards = () => document.querySelectorAll('[data-testid="piece-card"]').length;
const texOf = (id: string) => must(id).querySelector('.tk-tex')?.getAttribute('aria-label');

test('palette entries have no native title; the summary is the accessible description', () => {
  const entry = must('palette-linear');
  expect(entry.getAttribute('title')).toBeNull();
  expect(entry.getAttribute('aria-description')).toMatch(/\.$/);
  expect(must('piece-doc-linear').getAttribute('aria-label')).toBe('About Linear');
});

test('the palette "?" opens the symbolic card and never arms or drags', () => {
  const b = must('piece-doc-linear');
  ptr(b, 'pointerdown', 5, 5, 'touch');
  ptr(b, 'pointermove', 60, 60, 'touch');
  ptr(b, 'pointerup', 60, 60, 'touch');
  click(b);
  expect(dnd.armed()).toBeNull();
  expect(dnd.get()).toBeNull();
  expect(cards()).toBe(1);
  expect(must('piece-card').getAttribute('data-key')).toBe('linear');
  expect(texOf('piece-card-params')).toContain('m'); // symbolic: d \cdot m + m
  expect(must('piece-card-torch').textContent).toContain('nn.Linear');
  click(b); // the "?" again closes it
  expect(cards()).toBe(0);
});

test('Escape closes the card and returns focus to its "?"', () => {
  const b = must('piece-doc-linear');
  click(b);
  expect(cards()).toBe(1);
  key('Escape');
  expect(cards()).toBe(0);
  expect(document.activeElement).toBe(b);
});

test('the close button and an outside pointerdown close it; a press inside does not', () => {
  click(must('piece-doc-linear'));
  ptr(must('piece-card'), 'pointerdown');
  expect(cards()).toBe(1);
  click(must('piece-card-close'));
  expect(cards()).toBe(0);
  click(must('piece-doc-linear'));
  ptr(must('canvas'), 'pointerdown');
  expect(cards()).toBe(0);
});

test('only one card is open at a time', () => {
  click(must('piece-doc-linear'));
  click(must('piece-doc-input'));
  expect(cards()).toBe(1);
  expect(must('piece-card').getAttribute('data-key')).toBe('input');
  act(() => s().select({ kind: 'node', id: 'lin1' }));
  click(must('piece-doc-inspector'));
  expect(cards()).toBe(1);
  expect(must('piece-card').getAttribute('data-placement')).toBe('inspector');
});

test('the inspector card is live: a width edit changes the parameter count; selection change closes it', () => {
  act(() => s().select({ kind: 'node', id: 'lin1' }));
  click(must('piece-doc-inspector'));
  expect(texOf('piece-card-params')).toBe('1 \\cdot 4 + 4 = 8');
  const input = must('field-out') as HTMLInputElement;
  const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')?.set;
  act(() => {
    setter?.call(input, '6');
    input.dispatchEvent(new Event('input', { bubbles: true }));
  });
  act(() => void input.dispatchEvent(new Event('focusout', { bubbles: true })));
  expect(texOf('piece-card-params')).toBe('1 \\cdot 6 + 6 = 12');
  expect(must('piece-card-torch').textContent).toContain('6');
  act(() => s().select({ kind: 'node', id: 'act1' }));
  expect(cards()).toBe(0);
});

test('a wire inspector has no "?"', () => {
  act(() => s().select({ kind: 'wire', from: 'x', to: 'lin1' }));
  expect(q('piece-doc-inspector')).toBeNull();
});

test('show.docs: false removes both "?" buttons', () => {
  act(() => s().select({ kind: 'node', id: 'lin1' }));
  expect(q('piece-doc-inspector')).not.toBeNull();
  act(() => useStore.setState({ show: { docs: false } }));
  expect(q('piece-doc-inspector')).toBeNull();
  expect(q('piece-doc-linear')).toBeNull();
  expect(q('palette-linear')).not.toBeNull();
  act(() => useStore.setState({ show: { docs: true } }));
  expect(q('piece-doc-linear')).not.toBeNull();
});

test('placeCard keeps the card inside the box, beside the side, never wider than 320', () => {
  const box = { left: 0, top: 0, right: 800, bottom: 500 };
  const right = placeCard(box, { left: 100, right: 348, bottom: 300 }, 200, 600);
  expect(right.left).toBe(356);
  expect(right.width).toBe(320);
  expect(right.top).toBeGreaterThanOrEqual(4);
  expect(right.maxHeight).toBeLessThanOrEqual(492);
  const left = placeCard(box, { left: 500, right: 748, bottom: 300 }, 200, 300);
  expect(left.left).toBe(500 - 8 - 320);
  expect(left.top + 300).toBeLessThanOrEqual(496);
  const tight = placeCard(box, { left: 100, right: 560, bottom: 300 }, 0, 100);
  expect(tight.left).toBe(568); // beside it, narrower than 320
  expect(tight.width).toBe(228);
  const narrow = placeCard({ ...box, right: 200 }, { left: 0, right: 10, bottom: 10 }, 0, 100);
  expect(narrow.left + 192).toBeLessThanOrEqual(196);
});
