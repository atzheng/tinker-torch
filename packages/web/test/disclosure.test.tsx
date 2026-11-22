// @vitest-environment jsdom
import { beforeEach, expect, test } from 'vitest';
import { act } from 'react';
import { createRoot } from 'react-dom/client';
import { PALETTE_GROUP_IDS, setLinearOut } from '@tinker/core';
import { App, gridFor } from '../src/App.js';
import { GROUPS } from '../src/palette/Palette.js';
import { mlpGraph } from './mlpGraph.js';
import { useStore } from '../src/state/store.js';
import { loadText, saveText } from '../src/topbar/FileMenu.js';

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

function presetText(show: unknown): string {
  const g = JSON.parse(saveText(false)) as Record<string, unknown>;
  return JSON.stringify({ ...g, title: 'T', show });
}

function mount() {
  const host = document.createElement('div');
  document.body.appendChild(host);
  const root = createRoot(host);
  act(() => root.render(<App />));
  const q = (sel: string) => host.querySelector(sel);
  return { host, q, done: () => act(() => root.unmount()) };
}

test('the palette groups are the ids core validates', () => {
  expect(GROUPS.map((g) => g.id)).toEqual([...PALETTE_GROUP_IDS]);
});

test('a preset without show looks as today: every region and control', () => {
  const { host, q, done } = mount();
  for (const r of ['top-bar', 'palette', 'canvas', 'results', 'training-bar']) {
    expect(q(`[data-region="${r}"]`)).not.toBeNull();
  }
  for (const id of [
    'btn-step',
    'btn-run',
    'optimizer',
    'btn-undo',
    'data-resample',
    'palette-sigmoid',
  ]) {
    expect(host.querySelector(`[data-testid="${id}"]`)).not.toBeNull();
  }
  expect(q('[class*="app"]')?.getAttribute('style')).toBeNull();
  done();
});

test('stage 0: empty palette and no training-bar controls remove those regions', () => {
  expect(loadText(presetText({ palette: [], ui: ['data-set', 'data-n', 'panel-fit'] })).ok).toBe(
    true,
  );
  const { host, q, done } = mount();
  expect(q('[data-region="palette"]')).toBeNull();
  expect(q('[data-region="training-bar"]')).toBeNull();
  expect(q('[data-region="results"]')).not.toBeNull();
  expect(host.querySelector('[data-testid="data-set"]')).not.toBeNull();
  expect(host.querySelector('[data-testid="data-resample"]')).toBeNull();
  expect(host.querySelector('[data-testid="btn-undo"]')).toBeNull();
  expect(host.querySelector('[data-testid="presets-button"]')).toBeNull();
  expect(host.querySelector('[data-testid="grain-toggle"]')).toBeNull();
  expect(host.querySelector('[data-testid="equation-toggle"]')).toBeNull();
  expect(host.querySelector('[data-testid="loss-curves"]')).toBeNull();
  expect(host.querySelector('[data-testid="fit-plot"]')).not.toBeNull();
  done();
});

test('stage 2: only the listed palette entries and controls', () => {
  const ui = ['btn-step', 'btn-run', 'btn-reset', 'lr-slider', 'panel-loss', 'btn-undo'];
  loadText(presetText({ palette: ['sigmoid', 'loss_log_loss'], ui, tab: 'loss' }));
  const { host, q, done } = mount();
  const entries = () =>
    [...host.querySelectorAll('.tk-palette-entry')].map((e) => e.getAttribute('data-testid'));
  // the preset opens on the loss tab, which offers the ready-made loss; the model tab the model piece
  expect(entries()).toEqual(['palette-loss_log_loss']);
  act(() => s().setViewTab('model'));
  expect(entries()).toEqual(['palette-sigmoid']);
  act(() => s().setViewTab('loss'));
  for (const id of ui.filter((x) => x !== 'panel-loss')) {
    expect(host.querySelector(`[data-testid="${id}"]`)).not.toBeNull();
  }
  for (const id of ['btn-restore-best', 'optimizer', 'batch', 'speed', 'param-count', 'btn-redo']) {
    expect(host.querySelector(`[data-testid="${id}"]`)).toBeNull();
  }
  expect(q('[data-region="results"]')).not.toBeNull();
  expect(host.querySelector('[data-testid="fit-plot"]')).toBeNull();
  done();
});

test('hiding never changes behaviour; keyboard shortcuts of hidden controls are off', () => {
  loadText(presetText({ ui: ['btn-undo', 'btn-step'] }));
  expect(s().graph.train.optimizer).toBe(mlpGraph().train.optimizer);
  s().apply(setLinearOut(s().graph, 'lin1', 6)); // an edit after the load, so undo keeps the preset
  const mounted = mount();
  const before = s().graph;
  const key = (init: KeyboardEventInit) =>
    act(() => void window.dispatchEvent(new KeyboardEvent('keydown', { ...init })));
  // The undo shortcut is live (btn-undo shown); redo is hidden so Ctrl+Shift+Z does nothing.
  key({ key: 'z', ctrlKey: true });
  expect(s().canRedo).toBe(true);
  const after = s().graph;
  key({ key: 'z', ctrlKey: true, shiftKey: true });
  expect(s().graph).toBe(after);
  expect(after).not.toBe(before);
  mounted.done();
});

test('disclosure travels with the preset in the undo history, and Save writes it back', () => {
  const show = { palette: ['sigmoid'], ui: ['btn-undo'] };
  loadText(presetText(show));
  expect(s().show).toEqual(show);
  expect(JSON.parse(saveText(false)).show).toEqual(show);
  s().undo(); // back to the plain graph: its show is restored too
  expect(s().show).toBeNull();
  s().redo();
  expect(s().show).toEqual(show);
  s().loadGraph(mlpGraph()); // a new graph shows everything again
  expect(s().show).toBeNull();
});

test('a results list naming only the surface panel on a model without one removes the region', () => {
  expect(s().host).not.toBeNull();
  loadText(presetText({ ui: ['panel-surface', 'btn-undo'] }));
  const { q, done } = mount();
  expect(q('[data-region="results"]')).toBeNull();
  done();
});

test('ids for the panel controls hide them one by one', () => {
  loadText(presetText({ ui: ['panel-loss', 'loss-log-toggle', 'pins'], tab: 'loss' }));
  const { host, done } = mount();
  expect(host.querySelector('[data-testid="loss-log-toggle"]')).not.toBeNull();
  expect(host.querySelector('[data-testid="pins"]')).not.toBeNull();
  expect(host.querySelector('[data-testid="loss-restore-best"]')).toBeNull();
  expect(host.querySelector('[data-testid="loss-readout"]')).toBeNull();
  expect(host.querySelector('[data-testid="scalar-bar"]')).toBeNull();
  done();
});

test('a bad id leaves the current graph and show alone', () => {
  const r = loadText(presetText({ ui: ['btn-stepp'] }));
  expect(r.ok).toBe(false);
  expect(r.message).toContain('"btn-stepp"');
  expect(s().show).toBeNull();
});

test('gridFor: the remaining regions fill the grid, no gaps', () => {
  expect(gridFor({ top: true, palette: true, results: true, train: true })).toBeUndefined();
  const g = gridFor({ top: true, palette: false, results: true, train: false });
  expect(g?.gridTemplateColumns).toBe('minmax(0, 1fr) minmax(0, 30%)');
  expect(g?.gridTemplateRows).toBe('48px minmax(0, 1fr)');
  expect(g?.gridTemplateAreas).toBe("'top top' 'canvas results'");
  expect(gridFor({ top: false, palette: false, results: false, train: false })).toMatchObject({
    gridTemplateColumns: 'minmax(0, 1fr)',
    gridTemplateAreas: "'canvas'",
  });
});
