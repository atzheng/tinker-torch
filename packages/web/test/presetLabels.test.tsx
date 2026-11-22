// @vitest-environment jsdom
import { beforeEach, expect, test } from 'vitest';
import { act } from 'react';
import { createRoot } from 'react-dom/client';
import { App } from '../src/App.js';
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

function mount() {
  const host = document.createElement('div');
  document.body.appendChild(host);
  const root = createRoot(host);
  act(() => root.render(<App />));
  const q = (sel: string) => host.querySelector(sel);
  return { host, q, done: () => act(() => root.unmount()) };
}

const LABELS = {
  'btn-run': 'Keep stepping',
  'lr-slider': 'Step size',
  optimizer: 'How to move',
  'panel-fit': 'The picture',
  'data-set': 'Which data',
};

function presetText(labels: unknown): string {
  const g = JSON.parse(saveText(false)) as Record<string, unknown>;
  return JSON.stringify({ ...g, title: 'T', labels });
}

test('labels rename controls and aria-labels; others are unchanged', () => {
  const { host, q, done } = mount();
  act(() => void loadText(presetText(LABELS)));
  expect(q('[data-testid="btn-run"]')?.textContent).toBe('Keep stepping');
  expect(q('[data-testid="btn-run"]')?.getAttribute('aria-label')).toBe('Keep stepping');
  expect(q('[data-testid="lr-slider"]')?.getAttribute('aria-label')).toBe('Step size');
  expect(q('[data-testid="optimizer"]')?.getAttribute('aria-label')).toBe('How to move');
  expect(host.textContent).toContain('The picture');
  expect(host.textContent).toContain('Which data');
  expect(q('[data-testid="btn-step"]')?.textContent).toBe('Step');
  expect(q('[data-testid="btn-reset"]')?.textContent).toBe('Reset');
  done();
});

test('labels travel with the preset: Save writes them, undo restores the previous ones', () => {
  const { q, done } = mount();
  act(() => void loadText(presetText({ 'btn-step': 'Nudge' })));
  expect(JSON.parse(saveText(false)).labels).toEqual({ 'btn-step': 'Nudge' });
  act(() => void loadText(presetText({ 'btn-step': 'Move once' })));
  expect(q('[data-testid="btn-step"]')?.textContent).toBe('Move once');
  act(() => s().undo());
  expect(q('[data-testid="btn-step"]')?.textContent).toBe('Nudge');
  act(() => s().undo());
  expect(q('[data-testid="btn-step"]')?.textContent).toBe('Step');
  done();
});

test('the loss readout gets a title only when the preset gives one', () => {
  const { q, done } = mount();
  expect(q('[data-testid="loss-readout-title"]')).toBeNull();
  act(() => void loadText(presetText({ 'loss-readout': 'Average surprise' })));
  act(() => s().setViewTab('loss')); // the loss readout is on the loss tab
  expect(q('[data-testid="loss-readout-title"]')?.textContent).toBe('Average surprise');
  done();
});
