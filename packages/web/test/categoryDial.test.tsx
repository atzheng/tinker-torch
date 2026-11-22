// @vitest-environment jsdom
import { afterEach, beforeAll, beforeEach, expect, test } from 'vitest';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { parseGraph, type Graph } from '@tinker/core';
import { Canvas } from '../src/canvas/Canvas.js';
import { loadEngine } from '../src/engine.js';
import { useStore } from '../src/state/store.js';

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const s = () => useStore.getState();
let el: HTMLDivElement;
let root: Root;

beforeAll(async () => {
  await loadEngine();
}, 60000);

function build(nodes: object[], edges: [string, string][]): Graph {
  const r = parseGraph({
    version: 2,
    nodes: [
      { id: 'z', type: 'input', columns: ['x4'] },
      ...nodes,
      { id: 'p', type: 'link', fn: 'sigmoid' },
      { id: 'y', type: 'target' },
      { id: 'err', type: 'log_loss' },
      { id: 'loss', type: 'mean' },
    ],
    edges: [...edges, ['p', 'err.pred'], ['y', 'err.target'], ['err', 'loss']],
    predictions: ['p'],
    loss: 'loss',
    data: { set: 'delivery_late', n: 100, noise: 0.3, val: 0.3, seed: 2, standardize: false },
    train: { optimizer: 'sgd', lr: 0.1, batch: 'full', seed: 1 },
  });
  if (!r.ok) throw new Error(r.error);
  return r.graph;
}
const lookupGraph = () =>
  build(
    [{ id: 'look', type: 'lookup', width: 1 }],
    [
      ['z', 'look'],
      ['look', 'p'],
    ],
  );
const onehotGraph = () =>
  build(
    [
      { id: 'oh', type: 'onehot' },
      { id: 'lin', type: 'linear', out: 1, bias: true },
    ],
    [
      ['z', 'oh'],
      ['oh', 'lin'],
      ['lin', 'p'],
    ],
  );

function mount(g: Graph) {
  s().pause();
  s().loadGraph(g);
  s().select(null);
  s().setGrain('neuron');
  el = document.createElement('div');
  document.body.appendChild(el);
  root = createRoot(el);
  act(() => root.render(<Canvas />));
}
afterEach(() => {
  act(() => root.unmount());
  el.remove();
});
beforeEach(() => {
  useStore.setState({ host: null });
});

const must = (id: string): Element => {
  const e = el.querySelector(`[data-testid="${id}"]`);
  if (!e) throw new Error(`missing ${id}`);
  return e;
};
function fire(target: Element, type: string, x: number, y: number) {
  const e = new Event(type, { bubbles: true, cancelable: true });
  Object.assign(e, { pointerId: 1, clientX: x, clientY: y, button: 0, pointerType: 'touch' });
  target.dispatchEvent(e);
}
/** A vertical touch drag of `dy` px from the middle of the element. */
function drag(target: Element, dy: number) {
  act(() => {
    fire(target, 'pointerdown', 300, 200);
    fire(target, 'pointermove', 300, 200 + dy / 2);
    fire(target, 'pointermove', 300, 200 + dy);
    fire(target, 'pointerup', 300, 200 + dy);
  });
}

test('a width-1 Lookup draws one labelled dial per category; a drag moves only that row', () => {
  mount(lookupGraph());
  const names = s().dataInfo.categories?.['x4'] ?? [];
  expect(names).toHaveLength(4);
  names.forEach((name, k) => {
    must(`unit-look-${k}`);
    expect(must(`dial-label-look-${k}`).textContent).toContain(name.slice(0, 5));
  });
  const before = Array.from(s().host?.look?.W ?? []);
  expect(before).toHaveLength(4);
  const past = s().past.length;
  drag(must('unit-look-2'), -30); // up 30 px raises by 0.3
  const after = Array.from(s().host?.look?.W ?? []);
  expect(after[2]).toBeCloseTo((before[2] ?? 0) + 0.3, 5);
  for (const k of [0, 1, 3]) expect(after[k]).toBe(before[k]);
  expect(s().past.length).toBe(past + 1);
  act(() => s().undo());
  expect(Array.from(s().host?.look?.W ?? [])).toEqual(before);
});

test('One-hot into Linear(out 1): the category dial sets that row of the Linear weight', () => {
  mount(onehotGraph());
  for (let k = 0; k < 4; k++) must(`dial-label-oh-${k}`);
  const before = Array.from(s().host?.lin?.W ?? []);
  expect(before).toHaveLength(4);
  const bias = Array.from(s().host?.lin?.b ?? []);
  drag(must('unit-oh-1'), 20);
  const after = Array.from(s().host?.lin?.W ?? []);
  expect(after[1]).toBeCloseTo((before[1] ?? 0) - 0.2, 5);
  for (const k of [0, 2, 3]) expect(after[k]).toBe(before[k]);
  expect(Array.from(s().host?.lin?.b ?? [])).toEqual(bias);
});
