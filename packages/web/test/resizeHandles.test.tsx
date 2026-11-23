// @vitest-environment jsdom
import { afterEach, beforeEach, expect, test } from 'vitest';
import { act } from 'react';
import { createRoot } from 'react-dom/client';
import { App, gridFor } from '../src/App.js';
import { mlpGraph } from './mlpGraph.js';
import { useStore } from '../src/state/store.js';
import { loadText, saveText } from '../src/topbar/FileMenu.js';
import { DEFAULT_WEIGHTS } from '../src/results/panelLayout.js';
import { DEFAULT_REGION_SIZES, REGION_LIMITS } from '../src/layout/regionSizes.js';

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;
(globalThis as { ResizeObserver?: unknown }).ResizeObserver = class {
  observe() {}
  unobserve() {}
  disconnect() {}
};
window.matchMedia = ((q: string) => ({
  matches: false,
  media: q,
  addEventListener() {},
  removeEventListener() {},
})) as unknown as typeof window.matchMedia;
HTMLCanvasElement.prototype.getContext = (() =>
  null) as unknown as typeof HTMLCanvasElement.prototype.getContext;
const captured = new Set<number>();
HTMLElement.prototype.setPointerCapture = (id: number) => void captured.add(id);
HTMLElement.prototype.hasPointerCapture = (id: number) => captured.has(id);
HTMLElement.prototype.releasePointerCapture = (id: number) => void captured.delete(id);

const s = () => useStore.getState();

beforeEach(() => {
  s().loadGraph(mlpGraph());
  s().setRegionSize('resultsWidth', null);
  s().setRegionSize('resultsBand', null);
  s().setRegionSize('instructions', null);
  s().setPanelWeights({ ...DEFAULT_WEIGHTS });
});
afterEach(() => {
  window.innerWidth = 1024;
  window.innerHeight = 768;
});

function ptr(type: string, x: number, y: number): Event {
  const ev = new Event(type, { bubbles: true, cancelable: true });
  Object.assign(ev, { pointerId: 1, clientX: x, clientY: y, pointerType: 'touch', button: 0 });
  return ev;
}

function drag(el: Element, dx: number, dy: number) {
  act(() => void el.dispatchEvent(ptr('pointerdown', 100, 100)));
  act(() => void el.dispatchEvent(ptr('pointermove', 100 + dx, 100 + dy)));
  act(() => void el.dispatchEvent(ptr('pointerup', 100 + dx, 100 + dy)));
}

function key(el: Element, k: string, shiftKey = false) {
  act(
    () => void el.dispatchEvent(new KeyboardEvent('keydown', { key: k, shiftKey, bubbles: true })),
  );
}

function mount() {
  const host = document.createElement('div');
  document.body.appendChild(host);
  const root = createRoot(host);
  act(() => root.render(<App />));
  const q = (id: string) => host.querySelector(`[data-testid="${id}"]`) as HTMLElement;
  return {
    host,
    q,
    done: () => {
      act(() => root.unmount());
      host.remove();
    },
  };
}

test('panel splitter: focusable separator with values; drag and arrow keys resize the pair', () => {
  const { host, q, done } = mount();
  act(() => s().setViewTab('full'));
  const col = q('results-panels');
  col.getBoundingClientRect = () => ({ width: 300, height: 400 }) as DOMRect;
  const h = q('splitter-fit-loss');
  expect(h.getAttribute('role')).toBe('separator');
  expect(h.getAttribute('aria-orientation')).toBe('horizontal');
  expect(h.tabIndex).toBe(0);
  expect(h.getAttribute('aria-valuenow')).toBe('50');
  expect(Number(h.getAttribute('aria-valuemin'))).toBeLessThan(50);
  expect(Number(h.getAttribute('aria-valuemax'))).toBeGreaterThan(50);
  // two visible panels in this graph's model tab: fit and loss
  const total = s().panelWeights.fit + s().panelWeights.loss;
  drag(h, 0, 100); // a quarter of the 400 px column
  expect(s().panelWeights.fit).toBeGreaterThan(1);
  expect(s().panelWeights.fit + s().panelWeights.loss).toBeCloseTo(total);
  expect(h.getAttribute('aria-valuenow')).not.toBe('50');
  const before = s().panelWeights.fit;
  key(h, 'ArrowUp');
  expect(s().panelWeights.fit).toBeLessThan(before);
  key(h, 'ArrowDown', true);
  expect(s().panelWeights.fit).toBeGreaterThan(before);
  // a press on the handle does not reorder panels
  expect(s().panelOrder.slice(0, 2)).toEqual(['fit', 'loss']);
  expect(host.scrollWidth).toBeLessThanOrEqual(host.clientWidth);
  done();
});

test('results region handle (landscape): drag left widens, keys step, limits hold', () => {
  const { host, q, done } = mount();
  const h = q('resize-results');
  expect(h.getAttribute('aria-orientation')).toBe('vertical');
  expect(h.getAttribute('aria-label')).toBe('Resize the results column');
  expect(h.getAttribute('aria-valuenow')).toBe('30');
  drag(h, -102.4, 0); // 10 % of the 1024 px window
  expect(s().regionSizes.resultsWidth).toBeCloseTo(0.4, 1);
  const grid = host.querySelector('[data-layout]') as HTMLElement;
  expect(grid.style.gridTemplateColumns).toContain('40%');
  key(h, 'ArrowRight'); // right shrinks the column
  expect(s().regionSizes.resultsWidth).toBeCloseTo(0.38, 1);
  drag(h, -5000, 0);
  expect(s().regionSizes.resultsWidth).toBe(REGION_LIMITS.resultsWidth.max);
  drag(h, 5000, 0);
  expect(s().regionSizes.resultsWidth).toBe(REGION_LIMITS.resultsWidth.min);
  done();
});

test('results region handle (portrait): drag up grows the band', () => {
  window.innerWidth = 768;
  window.innerHeight = 1024;
  const { host, q, done } = mount();
  const h = q('resize-results');
  expect(h.getAttribute('aria-orientation')).toBe('horizontal');
  expect(h.getAttribute('aria-label')).toBe('Resize the results band');
  drag(h, 0, -102.4); // 10 % of the 1024 px window
  expect(s().regionSizes.resultsBand).toBeCloseTo(0.42, 1);
  expect(s().regionSizes.resultsWidth).toBeNull();
  const grid = host.querySelector('[data-layout]') as HTMLElement;
  expect(grid.style.gridTemplateRows).toContain('42');
  key(h, 'ArrowDown', true);
  expect(s().regionSizes.resultsBand).toBeCloseTo(0.32, 1);
  done();
});

test('instructions handle: drag down grows the steps cap; arrow keys step', () => {
  const g = JSON.parse(saveText(false)) as Record<string, unknown>;
  loadText(JSON.stringify({ ...g, title: 'T', instructions: ['First step.', 'Second step.'] }));
  const { q, done } = mount();
  const h = q('resize-instructions');
  expect(h.getAttribute('aria-orientation')).toBe('horizontal');
  drag(h, 0, 76.8); // 10 % of the 768 px window
  expect(s().regionSizes.instructions).toBeCloseTo(0.38, 1);
  expect(q('instructions-steps').style.height).toMatch(/^3[78]/);
  key(h, 'ArrowUp');
  expect(s().regionSizes.instructions).toBeCloseTo(0.36, 1);
  done();
});

test('gridFor: a dragged size changes only its own track', () => {
  const all = { top: true, palette: true, results: true, train: true };
  expect(gridFor(all, 'landscape', DEFAULT_REGION_SIZES)).toBeUndefined();
  expect(gridFor(all, 'portrait', DEFAULT_REGION_SIZES)).toBeUndefined();
  // a landscape size does not restyle the portrait layout
  expect(gridFor(all, 'portrait', { ...DEFAULT_REGION_SIZES, resultsWidth: 0.4 })).toBeUndefined();
  const l = gridFor(all, 'landscape', { ...DEFAULT_REGION_SIZES, resultsWidth: 0.4 });
  expect(l?.gridTemplateColumns).toBe('minmax(140px, 15%) minmax(0, 1fr) minmax(0, 40%)');
  const p = gridFor(all, 'portrait', { ...DEFAULT_REGION_SIZES, resultsBand: 0.4 });
  expect(p?.gridTemplateRows).toContain('clamp(120px, 40%, 50%)');
});
