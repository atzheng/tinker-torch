// @vitest-environment jsdom
import { beforeEach, expect, test } from 'vitest';
import { act } from 'react';
import { createRoot } from 'react-dom/client';
import { App } from '../src/App.js';
import { mlpGraph } from './mlpGraph.js';
import { useStore } from '../src/state/store.js';
import { loadText, saveText } from '../src/topbar/FileMenu.js';
import scalar from '../src/presets/scalar.json';

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

const s = () => useStore.getState();

beforeEach(() => {
  s().loadGraph(mlpGraph());
});

function mount() {
  const host = document.createElement('div');
  document.body.appendChild(host);
  const root = createRoot(host);
  act(() => root.render(<App />));
  const q = (sel: string) => host.querySelector(sel) as HTMLInputElement | null;
  return { host, q, done: () => act(() => root.unmount()) };
}

function presetText(extra: Record<string, unknown>): string {
  const g = JSON.parse(saveText(false)) as Record<string, unknown>;
  delete g['options'];
  delete g['show'];
  // the loss panels (log scale, truth toggle) are on the loss tab
  return JSON.stringify({ ...g, title: 'T', show: { tab: 'loss' }, ...extra });
}

test('log scale and truth start on from a preset, off without options', () => {
  const { q, done } = mount();
  act(() => void loadText(presetText({})));
  expect(q('[data-testid="loss-log-toggle"]')?.checked).toBe(false);
  act(
    () => void loadText(presetText({ options: { 'loss-log-toggle': true, 'truth-toggle': true } })),
  );
  expect(q('[data-testid="loss-log-toggle"]')?.checked).toBe(true);
  act(() => s().setViewTab('model')); // the truth toggle is on the fit plot
  expect(q('[data-testid="truth-toggle"]')?.checked).toBe(true);
  done();
});

test('the student can change a shown toggle; it survives a remount; a new load resets it', () => {
  const { q, done } = mount();
  act(() => void loadText(presetText({ options: { 'loss-log-toggle': true } })));
  act(() => q('[data-testid="loss-log-toggle"]')?.click());
  expect(q('[data-testid="loss-log-toggle"]')?.checked).toBe(false);
  done();
  const m2 = mount();
  expect(m2.q('[data-testid="loss-log-toggle"]')?.checked).toBe(false);
  act(() => void loadText(presetText({ options: { 'loss-log-toggle': true } })));
  expect(m2.q('[data-testid="loss-log-toggle"]')?.checked).toBe(true);
  m2.done();
});

test('undo of a load returns to the previous preset values', () => {
  const { q, done } = mount();
  act(() => void loadText(presetText({ options: { 'loss-log-toggle': true } })));
  act(() => void loadText(presetText({})));
  expect(q('[data-testid="loss-log-toggle"]')?.checked).toBe(false);
  act(() => s().undo());
  expect(q('[data-testid="loss-log-toggle"]')?.checked).toBe(true);
  done();
});

test('squares, tangent and explored-only start on from a preset', () => {
  const { q, done } = mount();
  const text = JSON.stringify({
    ...scalar,
    show: {
      ui: [
        'panel-histogram',
        'panel-loss-1d',
        'histogram-squares',
        'loss-1d-tangent',
        'loss-1d-explored',
        'loss-1d-axis',
      ],
    },
    options: { 'histogram-squares': true, 'loss-1d-tangent': true, 'loss-1d-explored': true },
  });
  act(() => void loadText(text));
  const found = ['histogram-squares', 'loss-1d-tangent', 'loss-1d-explored']
    .map((id) => q(`[data-testid="${id}"]`))
    .filter((x) => x !== null);
  // A panel renders only where the model allows; every control that is on screen starts checked.
  expect(found.length).toBeGreaterThan(0);
  for (const el of found) expect(el?.checked).toBe(true);
  done();
});

test('a hidden toggle is not rendered and keeps the preset value', () => {
  act(
    () =>
      void loadText(presetText({ options: { 'truth-toggle': true }, show: { ui: ['panel-fit'] } })),
  );
  const { q, done } = mount();
  expect(q('[data-testid="truth-toggle"]')).toBeNull();
  expect(s().preset?.options?.['truth-toggle']).toBe(true);
  done();
});

test('loss-mode sets the readout mode on load; Save writes options back', () => {
  const opts = { 'loss-mode': 'total', 'histogram-squares': true };
  act(() => void loadText(presetText({ options: opts })));
  expect(s().lossMode).toBe('total');
  expect(JSON.parse(saveText(false)).options).toEqual(opts);
  act(() => void loadText(presetText({})));
  expect(JSON.parse(saveText(false)).options).toBeUndefined();
});

test('an invalid option is refused with a sentence and the graph is left alone', () => {
  const before = s().graph;
  const r = loadText(presetText({ options: { 'truth-toggle': 'yes' } }));
  expect(r.ok).toBe(false);
  expect(r.message).toMatch(/must be true or false/);
  expect(s().graph).toBe(before);
});

test('validation points toggle: on by default (plain file and preset without options), off from options, student can change it', () => {
  const { q, done } = mount();
  act(() => s().setViewTab('model'));
  expect(q('[data-testid="val-points-toggle"]')?.checked).toBe(true);
  act(() => void loadText(presetText({ show: { tab: 'model' } })));
  expect(q('[data-testid="val-points-toggle"]')?.checked).toBe(true);
  act(
    () =>
      void loadText(
        presetText({ show: { tab: 'model' }, options: { 'val-points-toggle': false } }),
      ),
  );
  expect(q('[data-testid="val-points-toggle"]')?.checked).toBe(false);
  act(() => q('[data-testid="val-points-toggle"]')?.click());
  expect(q('[data-testid="val-points-toggle"]')?.checked).toBe(true);
  // hidden by show.ui: fixed to the preset value (default on, or the option)
  act(
    () =>
      void loadText(
        presetText({ options: { 'val-points-toggle': false }, show: { ui: ['panel-fit'] } }),
      ),
  );
  expect(q('[data-testid="val-points-toggle"]')).toBeNull();
  expect(s().preset?.options?.['val-points-toggle']).toBe(false);
  done();
});
