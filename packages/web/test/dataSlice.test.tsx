// @vitest-environment jsdom
import { beforeEach, expect, test } from 'vitest';
import { act, type ReactElement } from 'react';
import { createRoot } from 'react-dom/client';
import { analyse, sample } from '@tinker/core';
import { initialGraph } from '../src/state/graphSlice.js';
import { useStore } from '../src/state/store.js';
import { DataControls } from '../src/topbar/DataControls.js';
import { InputFields } from '../src/canvas/drawings/InputFields.js';

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const s = () => useStore.getState();
const input = () => {
  const n = s().graph.nodes.find((x) => x.type === 'input');
  if (!n || n.type !== 'input') throw new Error('no input');
  return n;
};

beforeEach(() => {
  s().loadGraph(initialGraph());
});

test('initial dataset matches graph.data', () => {
  expect(s().dataset.nTrain + s().dataset.nVal).toBe(40);
  expect(s().dataInfo.columns).toEqual(['x1']);
});

test('setData updates the dataset and clamps to schema ranges', () => {
  s().setData({ n: 100, noise: 0.5, val: 0.2 });
  expect(s().dataset.nVal).toBe(20);
  s().setData({ n: 5, noise: 3, val: 0.9 });
  expect(s().graph.data).toMatchObject({ n: 10, noise: 1, val: 0.5 });
});

test('same seed gives the same dataset, resample changes it reproducibly', () => {
  s().setData({ seed: 3 });
  const a = Array.from(s().dataset.yTrain);
  s().setData({ seed: 4 });
  s().setData({ seed: 3 });
  expect(Array.from(s().dataset.yTrain)).toEqual(a);
  s().resample();
  const seed1 = s().graph.data.seed;
  expect(seed1).not.toBe(3);
  s().setData({ seed: 3 });
  s().resample();
  expect(s().graph.data.seed).toBe(seed1);
  expect(Array.from(s().dataset.yTrain)).toEqual(Array.from(sample(s().graph.data).yTrain));
});

test('switching to a 2-D set gives the Input two columns; widths follow', () => {
  s().setData({ set: 'xor' });
  expect(input().columns).toEqual(['x1', 'x2']);
  expect(s().dataInfo.kind).toBe('binary');
  const r = analyse(s().graph, s().dataInfo);
  expect(r.outWidth['x']).toBe(2);
  s().setData({ set: 'sine' });
  expect(input().columns).toEqual(['x1']);
});

test('changing set within the same columns keeps the column choice; undo recomputes the dataset', () => {
  s().setData({ set: 'plane' });
  s().setInputColumns('x', ['x2']);
  s().setData({ set: 'bowl' });
  expect(input().columns).toEqual(['x2']);
  s().loadGraph(initialGraph());
  expect(s().dataInfo.columns).toEqual(['x1']);
});

test('standardise toggles stored standardisation', () => {
  expect(s().dataset.standardization).toBeNull();
  s().setData({ standardize: true });
  expect(s().dataset.standardization).not.toBeNull();
});

test('setInputColumns keeps dataset order and refuses an empty list', () => {
  s().setData({ set: 'plane' });
  s().setInputColumns('x', ['x2', 'x1']);
  expect(input().columns).toEqual(['x1', 'x2']);
  s().setInputColumns('x', []);
  expect(input().columns).toEqual(['x1', 'x2']);
});

function mount(el: ReactElement) {
  const host = document.createElement('div');
  document.body.appendChild(host);
  const root = createRoot(host);
  act(() => root.render(el));
  return host;
}

test('DataControls: picking a set and Resample go through the store', () => {
  const host = mount(<DataControls />);
  const sel = host.querySelector<HTMLSelectElement>('[data-testid="data-set"]');
  expect(sel?.querySelectorAll('option').length).toBe(12);
  act(() => {
    if (!sel) return;
    sel.value = 'circle';
    sel.dispatchEvent(new Event('change', { bubbles: true }));
  });
  expect(s().graph.data.set).toBe('circle');
  const before = s().graph.data.seed;
  act(() => host.querySelector<HTMLButtonElement>('[data-testid="data-resample"]')?.click());
  expect(s().graph.data.seed).not.toBe(before);
  expect(host.querySelector<HTMLButtonElement>('[data-testid="grain-toggle"]')?.disabled).toBe(
    true,
  );
});

test('InputFields toggles a column off and on', () => {
  s().setData({ set: 'plane' });
  const host = mount(<InputFields node={input()} />);
  const box = host.querySelector<HTMLInputElement>('[data-testid="input-col-x2"]');
  act(() => box?.click());
  expect(input().columns).toEqual(['x1']);
});
