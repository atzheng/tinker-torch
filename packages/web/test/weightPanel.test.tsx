// @vitest-environment jsdom
import { afterEach, beforeAll, beforeEach, expect, test } from 'vitest';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { WeightInspector, weightStats } from '../src/canvas/WeightInspector.js';
import { loadEngine } from '../src/engine.js';
import { mlpGraph } from './mlpGraph.js';
import { useStore } from '../src/state/store.js';

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;
const s = () => useStore.getState();
let el: HTMLDivElement;
let root: Root;

beforeAll(async () => {
  await loadEngine();
}, 60000);
beforeEach(() => {
  s().pause();
  s().loadGraph(mlpGraph());
  act(() => s().openWeights('lin1'));
  el = document.createElement('div');
  document.body.appendChild(el);
  root = createRoot(el);
  act(() => root.render(<WeightInspector />));
});
afterEach(() => {
  act(() => root.unmount());
  el.remove();
  s().openWeights(null);
  s().setPreset(null);
});
const q = (id: string) => el.querySelector(`[data-testid="${id}"]`) as HTMLElement | null;
function type(input: HTMLInputElement, value: string) {
  const set = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')?.set;
  act(() => {
    set?.call(input, value);
    input.dispatchEvent(new Event('input', { bubbles: true }));
  });
}
function fire(target: Element, type: string, x: number, y: number) {
  const e = new Event(type, { bubbles: true, cancelable: true });
  Object.assign(e, { pointerId: 1, clientX: x, clientY: y, button: 0, pointerType: 'touch' });
  target.dispatchEvent(e);
}
const tap = (node: HTMLElement) =>
  act(() => {
    fire(node, 'pointerdown', 5, 5);
    fire(node, 'pointerup', 5, 5);
  });

test('cells hold exactly the live params of the layer', () => {
  const W = s().host?.lin1?.W;
  const lin = s().graph.nodes.find((n) => n.id === 'lin1');
  if (!W || !lin || lin.type !== 'linear') throw new Error('no params');
  const cells = el.querySelectorAll('[data-testid^="wcell-"]');
  const inW = s().analysis.inWidth['lin1'] ?? 0;
  expect(cells.length).toBe(W.length + (lin.bias ? lin.out : 0));
  for (let i = 0; i < inW; i++) {
    for (let j = 0; j < lin.out; j++) {
      expect(Number(q(`wcell-${i}-${j}`)?.dataset['value'])).toBe(W[i * lin.out + j]);
    }
  }
  expect(q('weights-stats-W')).not.toBeNull();
});

test('typing a value is one undo step, and undo restores it', () => {
  const before = s().host?.lin1?.W?.[1];
  const past = s().past.length;
  tap(q('wcell-0-1') as HTMLElement);
  const field = q('weights-value-input') as HTMLInputElement;
  field.focus();
  type(field, '0.25');
  act(() => field.blur());
  expect(s().past.length).toBe(past + 1);
  expect(s().host?.lin1?.W?.[1]).toBeCloseTo(0.25, 6);
  expect(q('wcell-0-1')?.dataset['value']).toBe(String(Math.fround(0.25)));
  act(() => s().undo());
  expect(s().host?.lin1?.W?.[1]).toBe(before);
});

test('a locked piece makes cells diagnostics', () => {
  expect(q('wcell-0-0')?.dataset['editable']).toBe('true');
  act(() => s().setPreset({ title: 't', prompt: '', locked: ['lin1'] }));
  expect(q('wcell-0-0')?.dataset['editable']).toBe('false');
  expect(q('weights-value-input')).toBeNull();
});

test('the panel closes when its piece does not exist', () => {
  expect(q('weights-panel')).not.toBeNull();
  act(() => s().openWeights('nope'));
  expect(q('weights-panel')).toBeNull();
  expect(s().weightsOf).toBeNull();
});

test('a paused step shows at once', () => {
  const r0 = weightStats.renders;
  act(() => s().stepOnce());
  expect(weightStats.renders).toBeGreaterThan(r0);
});

test('a drag on a cell is one undo step; a running run refuses it', () => {
  const cell = q('wcell-0-0') as HTMLElement;
  const start = s().host?.lin1?.W?.[0] ?? 0;
  const past = s().past.length;
  act(() => {
    fire(cell, 'pointerdown', 5, 100);
    fire(cell, 'pointermove', 5, 50);
    fire(cell, 'pointerup', 5, 50);
  });
  expect(s().past.length).toBe(past + 1);
  expect(s().host?.lin1?.W?.[0]).toBeCloseTo(start + 0.5, 5);
  useStore.setState({ running: true });
  act(() => {
    fire(cell, 'pointerdown', 5, 100);
    fire(cell, 'pointermove', 5, 50);
    fire(cell, 'pointerup', 5, 50);
  });
  expect(s().past.length).toBe(past + 1);
  useStore.setState({ running: false });
});
