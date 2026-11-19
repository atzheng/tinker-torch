// @vitest-environment jsdom
import { afterEach, beforeEach, expect, test } from 'vitest';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { Canvas } from '../src/canvas/Canvas.js';
import { mlpGraph as initialGraph } from './mlpGraph.js';
import { useStore } from '../src/state/store.js';

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

let host: HTMLDivElement;
let root: Root;
beforeEach(() => {
  useStore.getState().loadGraph(initialGraph());
  host = document.createElement('div');
  document.body.appendChild(host);
  root = createRoot(host);
  act(() => root.render(<Canvas />));
});
afterEach(() => {
  act(() => root.unmount());
  host.remove();
});

test('every piece is drawn by the one node shell: one body, same radius, ports on the edges', () => {
  const nodes = [...host.querySelectorAll('[data-testid^="node-"]')];
  expect(nodes.length).toBe(8);
  for (const n of nodes) {
    const bodies = n.querySelectorAll('.tk-body');
    expect(bodies, n.getAttribute('data-testid') ?? '').toHaveLength(1);
    expect(bodies[0]?.getAttribute('rx')).toBe('8');
    expect(n.querySelector('.tk-name')?.textContent).toBeTruthy();
    // No per-type outline: the only circles in a node are its ports and a role badge.
    expect(n.querySelectorAll('circle.tk-shape')).toHaveLength(0);
    expect(n.querySelectorAll('.tk-port-static').length).toBeGreaterThan(0);
  }
});

test('only inputs, the node pinned as ŷ and the node pinned as L carry a role', () => {
  const role = (id: string) =>
    host.querySelector(`[data-testid="node-${id}"]`)?.getAttribute('data-role') ?? null;
  expect(['x', 'y', 'lin2', 'loss'].map(role)).toEqual(['input', 'input', 'pred', 'loss']);
  expect(['lin1', 'act1', 'diff', 'err'].map(role)).toEqual([null, null, null, null]);
  // the pins are drawn as pins at the pinned node's output, not as pieces of their own
  expect(host.querySelector('[data-testid="pin-pred-lin2"]')).not.toBeNull();
  expect(host.querySelector('[data-testid="pin-loss-loss"]')).not.toBeNull();
  expect(
    host.querySelectorAll('[data-testid^="pin-pred-"], [data-testid^="pin-loss-"]'),
  ).toHaveLength(2);
});

test('a node pinned as both ŷ and L shows both badges', () => {
  act(() => root.unmount());
  useStore.getState().loadGraph({ ...initialGraph(), predictions: ['loss'] });
  root = createRoot(host);
  act(() => root.render(<Canvas />));
  expect(host.querySelector('[data-testid="pin-pred-loss"]')).not.toBeNull();
  expect(host.querySelector('[data-testid="pin-loss-loss"]')).not.toBeNull();
});
