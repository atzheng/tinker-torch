// @vitest-environment jsdom
import { afterEach, beforeEach, expect, test } from 'vitest';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import type { Graph } from '@tinker/core';
import { Canvas } from '../src/canvas/Canvas.js';
import { useStore } from '../src/state/store.js';

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const s = () => useStore.getState();
let host: HTMLDivElement;
let root: Root;

// Three skip-ish wires into one Sum: their drops share a gap and cross other wires' runs.
const graph: Graph = {
  version: 2,
  nodes: [
    { id: 'x', type: 'input', columns: ['x1'] },
    { id: 'l1', type: 'linear', out: 3, bias: true },
    { id: 'a1', type: 'activation', fn: 'relu' },
    { id: 'l2', type: 'linear', out: 3, bias: true },
    { id: 'a2', type: 'activation', fn: 'relu' },
    { id: 'l3', type: 'linear', out: 3, bias: true },
    { id: 's', type: 'sum' },
  ],
  edges: [
    ['x', 'l1'],
    ['l1', 'a1'],
    ['a1', 'l2'],
    ['l2', 'a2'],
    ['a2', 'l3'],
    ['x', 's.in'],
    ['l1', 's.in'],
    ['l3', 's.in'],
  ],
  predictions: [],
  loss: null,
  data: { set: 'line', n: 100, noise: 0.1, val: 0.2, seed: 1, standardize: false },
  train: { optimizer: 'adam', lr: 0.01, batch: 'full', seed: 1 },
};

beforeEach(() => {
  s().loadGraph(graph);
  s().setViewTab('full');
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

const wireIds = () =>
  [...host.querySelectorAll('g[data-testid^="wire-"]')]
    .map((g) => g.getAttribute('data-testid') ?? '')
    .filter((id) => /^wire-\d+$/.test(id));

test('crossing gaps reach the visible line as stroke-dasharray', () => {
  const dashed = [...host.querySelectorAll('[data-testid^="wire-"] .tk-wire-line')].filter((p) =>
    p.hasAttribute('stroke-dasharray'),
  );
  expect(dashed.length).toBeGreaterThan(0);
  for (const p of dashed) expect(p.closest('g')?.getAttribute('data-gapped')).toBe('true');
});

test('the selected wire is drawn last (on top)', () => {
  expect(wireIds().at(-1)).toBe('wire-7');
  act(() => s().select({ kind: 'wire', from: 'x', to: 's.in' }));
  const ids = wireIds();
  expect(ids).toHaveLength(8);
  expect(ids.at(-1)).toBe('wire-5');
});
