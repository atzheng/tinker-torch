// @vitest-environment jsdom
import { afterEach, beforeEach, expect, test } from 'vitest';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { NODE_TYPES, type Graph, type Node } from '@tinker/core';
import { Canvas } from '../src/canvas/Canvas.js';
import { shapeOfNode } from '../src/canvas/drawings/NodeShell.js';
import { mlpGraph } from './mlpGraph.js';
import { useStore } from '../src/state/store.js';

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const VALUES = ['input', 'target', 'column', 'weights', 'bias'];
test('shapeOfNode: every piece type', () => {
  for (const type of NODE_TYPES) {
    const node = { id: 'n', type } as unknown as Node;
    const want = VALUES.includes(type) ? 'value' : 'op';
    expect(shapeOfNode(node, { edges: [] }), type).toBe(want);
  }
});

test('shapeOfNode: a Bias with an input is an operation', () => {
  const bias = { id: 'b', type: 'bias' } as Node;
  expect(shapeOfNode(bias, { edges: [] })).toBe('value');
  expect(shapeOfNode(bias, { edges: [['x', 'b.in']] })).toBe('op');
  expect(shapeOfNode(bias, { edges: [['b', 'x.in']] })).toBe('value');
});

let host: HTMLDivElement;
let root: Root;
beforeEach(() => {
  host = document.createElement('div');
  document.body.appendChild(host);
  root = createRoot(host);
  useStore.getState().setViewTab('model');
  useStore.getState().select(null);
});
afterEach(() => {
  act(() => root.unmount());
  host.remove();
});
const shapeOf = (id: string) =>
  host.querySelector(`[data-testid="node-${id}"]`)?.getAttribute('data-shape');
const bodyRx = (id: string) =>
  host.querySelector(`[data-testid="node-${id}"] .tk-body`)?.getAttribute('rx');

test('Canvas: Input is a value, Linear an operation, Bias switches when wired', () => {
  const g = mlpGraph();
  const first = g.nodes.find((n) => n.type === 'input')!;
  const lin = g.nodes.find((n) => n.type === 'linear')!;
  const bare: Graph = { ...g, nodes: [...g.nodes, { id: 'bz', type: 'bias' }] };
  act(() => useStore.getState().loadGraph(bare));
  act(() => root.render(<Canvas />));
  expect(shapeOf(first.id)).toBe('value');
  expect(shapeOf(lin.id)).toBe('op');
  expect(shapeOf('bz')).toBe('value');
  expect(Number(bodyRx(first.id))).toBeGreaterThan(Number(bodyRx(lin.id)));
  const wired: Graph = { ...bare, edges: [...bare.edges, [first.id, 'bz.in']] };
  act(() => useStore.getState().loadGraph(wired));
  expect(shapeOf('bz')).toBe('op');
});
