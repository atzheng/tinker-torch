// @vitest-environment jsdom
import { afterEach, beforeEach, expect, test, vi } from 'vitest';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { FitPlot } from '../src/results/FitPlot.js';
import { initialGraph } from '../src/state/graphSlice.js';
import { useStore } from '../src/state/store.js';

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

// jsdom has no ResizeObserver or matchMedia; the plot's canvas hook needs both.
class RO {
  observe() {}
  disconnect() {}
}
vi.stubGlobal('ResizeObserver', RO);
vi.stubGlobal('matchMedia', () => ({ addEventListener() {}, removeEventListener() {} }));

let host: HTMLDivElement;
let root: Root;

beforeEach(() => {
  host = document.createElement('div');
  document.body.appendChild(host);
  root = createRoot(host);
});
afterEach(() => {
  act(() => root.unmount());
  host.remove();
});

test('the fit plot says so when no prediction is pinned, and not otherwise', () => {
  useStore.getState().loadGraph(initialGraph());
  act(() => root.render(<FitPlot />));
  expect(host.querySelector('[data-testid="no-prediction"]')).toBeNull();
  act(() => useStore.getState().loadGraph({ ...initialGraph(), predictions: [] }));
  expect(host.querySelector('[data-testid="no-prediction"]')?.textContent).toBe(
    'No prediction selected.',
  );
});
