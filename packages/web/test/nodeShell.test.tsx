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
  useStore.getState().setViewTab('model');
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
  expect(nodes.length).toBe(5); // the model tab: four model pieces and the stub of the Subtract
  act(() => useStore.getState().setViewTab('loss'));
  const lossNodes = [...host.querySelectorAll('[data-testid^="node-"]')];
  expect(lossNodes.length).toBe(5); // the stub of the prediction and the four loss pieces
  for (const n of [...nodes, ...lossNodes].filter((x) => x.getAttribute('data-stub') !== 'true')) {
    const bodies = n.querySelectorAll('.tk-body');
    expect(bodies, n.getAttribute('data-testid') ?? '').toHaveLength(1);
    // the radius is one of two: an operation's 8, or a value's pill (tinker-paq)
    expect(bodies[0]?.getAttribute('rx')).toBe(
      n.getAttribute('data-shape') === 'value' ? '28' : '8',
    );
    expect(n.querySelector('.tk-name')?.textContent).toBeTruthy();
    // No per-type outline: the only circles in a node are its ports and a role badge.
    expect(n.querySelectorAll('circle.tk-shape')).toHaveLength(0);
    expect(n.querySelectorAll('.tk-port-static').length).toBeGreaterThan(0);
  }
});

test('inputs, parameter pieces and the node pinned as L carry a role; the node pinned as ŷ gets none', () => {
  const role = (id: string) =>
    host.querySelector(`[data-testid="node-${id}"]`)?.getAttribute('data-role') ?? null;
  expect(['x', 'lin2'].map(role)).toEqual(['input', 'param']);
  expect(['lin1'].map(role)).toEqual(['param']);
  expect(['act1'].map(role)).toEqual([null]);
  // the pins are drawn as pins at the pinned node's output, not as pieces of their own
  expect(host.querySelector('[data-testid="pin-pred-lin2"]')).not.toBeNull();
  expect(host.querySelectorAll('[data-testid^="pin-pred-"]')).toHaveLength(1);
  act(() => useStore.getState().setViewTab('loss'));
  expect(['y', 'loss'].map(role)).toEqual(['input', 'loss']);
  expect(['diff', 'err'].map(role)).toEqual([null, null]);
  expect(host.querySelector('[data-testid="pin-loss-loss"]')).not.toBeNull();
  // the prediction pin shows on the stub of its piece too
  expect(host.querySelector('[data-testid="pin-pred-lin2"]')).not.toBeNull();
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
