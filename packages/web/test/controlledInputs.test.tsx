// @vitest-environment jsdom
import { afterEach, expect, test, vi } from 'vitest';
import { act } from 'react';
import { createRoot } from 'react-dom/client';
import type { Graph } from '@tinker/core';
import { App } from '../src/App.js';
import { mlpGraph as initialGraph } from './mlpGraph.js';
import { useStore } from '../src/state/store.js';
import { PRESET_NAMES, bundledPreset } from '../src/presets/index.js';

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

class FakeResizeObserver {
  observe() {}
  unobserve() {}
  disconnect() {}
}
(globalThis as { ResizeObserver?: unknown }).ResizeObserver = FakeResizeObserver;

window.matchMedia = ((q: string) => ({
  matches: false,
  media: q,
  addEventListener() {},
  removeEventListener() {},
})) as unknown as typeof window.matchMedia;
HTMLCanvasElement.prototype.getContext = (() =>
  null) as unknown as typeof HTMLCanvasElement.prototype.getContext;

(globalThis as { ImageData?: unknown }).ImageData = class {
  constructor(
    public width: number,
    public height: number,
  ) {}
};

afterEach(() => vi.restoreAllMocks());

/** A graph as a hand-built caller might pass it: optional fields left out. */
function sparseGraph(): Graph {
  const g = initialGraph();
  const data: Record<string, unknown> = { ...g.data };
  delete data['standardize'];
  return { ...g, data: data as Graph['data'] };
}

test('loading graphs never switches an input between controlled and uncontrolled', () => {
  const errors = vi.spyOn(console, 'error').mockImplementation(() => {});
  const host = document.createElement('div');
  document.body.appendChild(host);
  const root = createRoot(host);
  act(() => useStore.getState().loadGraph(initialGraph()));
  act(() => root.render(<App />));
  const graphs: Graph[] = [sparseGraph()];
  for (const name of PRESET_NAMES) {
    const f = bundledPreset(name);
    if (f) graphs.push(f.graph);
  }
  graphs.push(initialGraph());
  for (const g of graphs) act(() => useStore.getState().loadGraph(g));
  act(() => root.unmount());
  const msgs = errors.mock.calls.map((c) => c.join(' '));
  expect(msgs.filter((m) => /controlled|uncontrolled/.test(m))).toEqual([]);
});
