// @vitest-environment jsdom
import { afterEach, beforeEach, expect, test } from 'vitest';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { setLinearOut } from '@tinker/core';
import { Canvas } from '../src/canvas/Canvas.js';
import { mlpGraph as initialGraph } from './mlpGraph.js';
import { useStore } from '../src/state/store.js';
import { wheelZoomFactor } from '../src/canvas/viewport.js';

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const s = () => useStore.getState();
let el: HTMLDivElement;
let root: Root;

beforeEach(() => {
  s().loadGraph(initialGraph());
  s().select(null);
  s().setGrain('neuron');
  useStore.setState({ host: null });
  el = document.createElement('div');
  document.body.appendChild(el);
  root = createRoot(el);
  act(() => root.render(<Canvas />));
});
afterEach(() => {
  act(() => root.unmount());
  el.remove();
});

const q = (id: string) => el.querySelector(`[data-testid="${id}"]`);
function wheel(target: Element, init: WheelEventInit): WheelEvent {
  const ev = new WheelEvent('wheel', { bubbles: true, cancelable: true, ...init });
  act(() => void target.dispatchEvent(ev));
  return ev;
}
const canvas = () => {
  const c = q('canvas');
  if (!c) throw new Error('no canvas');
  return c;
};

test('wheelZoomFactor caps one notch at a quarter step and keeps small deltas smooth', () => {
  expect(wheelZoomFactor(-100)).toBeCloseTo(1.25);
  expect(wheelZoomFactor(100)).toBeCloseTo(0.8);
  expect(wheelZoomFactor(-3, 0)).toBeCloseTo(Math.exp(0.03));
  expect(wheelZoomFactor(-3, 1)).toBeLessThanOrEqual(1.25); // line mode is converted, then capped
  expect(wheelZoomFactor(0)).toBe(1);
});

test('a wheel that changes nothing does not show Reset view; a zoom does', () => {
  wheel(canvas(), { deltaY: 40 }); // plain wheel on a drawing that fits: clamped back to centre
  expect(q('reset-view')).toBeNull();
  wheel(canvas(), { deltaY: -100, ctrlKey: true });
  expect(q('reset-view')).not.toBeNull();
});

test('a wheel over the inspector is left alone', () => {
  const inner = document.createElement('div');
  inner.className = 'tk-inspector-pos';
  canvas().appendChild(inner);
  const ev = wheel(inner, { deltaY: -100, ctrlKey: true });
  expect(ev.defaultPrevented).toBe(false);
  expect(q('reset-view')).toBeNull();
  inner.remove();
});

test('loading a different graph resets the view; an edit keeps it', () => {
  wheel(canvas(), { deltaY: -100, ctrlKey: true });
  expect(q('reset-view')).not.toBeNull();
  act(() => void s().apply(setLinearOut(s().graph, 'lin1', 5)));
  expect(q('reset-view')).not.toBeNull();
  act(() => s().undo());
  expect(q('reset-view')).not.toBeNull();
  act(() => s().loadGraph(initialGraph()));
  expect(q('reset-view')).toBeNull();
});
