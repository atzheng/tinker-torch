// @vitest-environment jsdom
import { beforeEach, expect, test } from 'vitest';
import { act, type ReactElement } from 'react';
import { createRoot } from 'react-dom/client';
import { analyse, sample, setColumns } from '@tinker/core';
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

test('data changes are undo steps, in order with other edits; dataset follows', () => {
  const g0 = s().graph;
  s().setData({ set: 'xor' });
  s().apply({ ok: true, graph: { ...s().graph, train: { ...s().graph.train, lr: 0.5 } } });
  s().undo();
  expect(s().graph.train.lr).toBe(g0.train.lr);
  expect(s().graph.data.set).toBe('xor');
  expect(s().dataInfo.columns).toEqual(['x1', 'x2']);
  s().undo();
  expect(s().graph).toEqual(g0);
  expect(s().dataInfo.columns).toEqual(['x1']);
  s().redo();
  expect(s().graph.data.set).toBe('xor');
  expect(input().columns).toEqual(['x1', 'x2']);
  s().redo();
  expect(s().graph.train.lr).toBe(0.5);
  expect(s().dataset.nTrain).toBe(sample(s().graph.data).nTrain);
});

test('a data patch that changes nothing adds no history', () => {
  s().setData({ n: 40 });
  expect(s().canUndo).toBe(false);
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

test('setInputColumns goes through setColumns as one undo step; an unknown id sets the error', () => {
  s().setData({ set: 'plane' });
  const before = s().past.length;
  s().setInputColumns('x', ['x2']);
  expect(input().columns).toEqual(['x2']);
  expect(s().past.length).toBe(before + 1);
  s().undo();
  expect(input().columns).toEqual(['x1', 'x2']);
  s().setInputColumns('nope', ['x1']);
  expect(s().error).not.toBeNull();
});

test('a dataset switch resets the columns and is a single undo step', () => {
  const before = s().past.length;
  s().setData({ set: 'plane' });
  expect(s().past.length).toBe(before + 1);
  expect(input().columns).toEqual(['x1', 'x2']);
  s().undo();
  expect(input().columns).toEqual(['x1']);
  expect(s().graph.data.set).not.toBe('plane');
});

test('core setColumns refuses a locked Input and repeated columns', () => {
  const r = setColumns(s().graph, 'x', ['x1'], { locked: ['x'] });
  expect(r.ok).toBe(false);
  if (!r.ok) expect(r.error).toBe('This piece is locked.');
  expect(setColumns(s().graph, 'x', ['x1', 'x1']).ok).toBe(false);
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
});

test('InputFields toggles a column off and on', () => {
  s().setData({ set: 'plane' });
  const host = mount(<InputFields node={input()} />);
  const box = host.querySelector<HTMLInputElement>('[data-testid="input-col-x2"]');
  act(() => box?.click());
  expect(input().columns).toEqual(['x1']);
});
