// @vitest-environment jsdom
import { beforeAll, expect, test } from 'vitest';
import { act } from 'react';
import { createRoot } from 'react-dom/client';
import { analyse, insertOnWire, sample, setColumns } from '@tinker/core';
import { App } from '../src/App.js';
import { loadEngine } from '../src/engine.js';
import { bundledPreset } from '../src/presets/index.js';
import { useStore } from '../src/state/store.js';
import { loadBundled } from '../src/topbar/PresetMenu.js';

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

(globalThis as { ImageData?: unknown }).ImageData = class {
  data: Uint8ClampedArray;
  constructor(w: number, h: number) {
    this.data = new Uint8ClampedArray(w * h * 4);
  }
};

const s = () => useStore.getState();
const NAMES = [0, 1, 2, 3, 4, 5].map((i) => `lecture1-stage${i}`);

beforeAll(async () => {
  await loadEngine();
}, 60000);

function mount() {
  const host = document.createElement('div');
  document.body.appendChild(host);
  const root = createRoot(host);
  act(() => root.render(<App />));
  const has = (id: string) => host.querySelector(`[data-testid="${id}"]`) !== null;
  const region = (r: string) => host.querySelector(`[data-region="${r}"]`) !== null;
  const entries = () =>
    [...host.querySelectorAll('.tk-palette-entry')].map((e) => e.getAttribute('data-testid'));
  return {
    host,
    has,
    region,
    entries,
    done: () => {
      act(() => root.unmount());
      host.remove();
    },
  };
}

function load(name: string) {
  expect(loadBundled(name), name).toBe(true);
  return s();
}

function train(steps: number): void {
  s().pause();
  for (let i = 0; i < steps; i++) s().stepOnce();
}

const bias = () => Array.from(s().host?.bias?.b ?? []);

test('every lecture 1 stage is a titled preset with a prompt, steps, a valid graph and a hand-set start', () => {
  for (const name of NAMES) {
    const f = bundledPreset(name);
    expect(f, name).not.toBeNull();
    if (!f || !('title' in f)) throw new Error(name);
    expect(f.prompt, name).toBeTruthy();
    expect((f.instructions ?? []).length, name).toBeGreaterThan(0);
    const a = analyse(f.graph, sample(f.graph.data).info);
    expect(a.issues, name).toEqual([]);
    expect(a.trainable, name).toBe(true);
  }
});

test('stage 0: the loss tab only, the score and data, no palette and no training bar', () => {
  load(NAMES[0] as string);
  const m = mount();
  expect(m.region('palette')).toBe(false);
  expect(m.region('training-bar')).toBe(false);
  expect(m.region('results')).toBe(true);
  expect(s().show?.tab).toBe('loss');
  expect(m.has('tab-bar')).toBe(false); // one tab: no tab bar, the whole graph is drawn
  for (const id of ['loss-readout', 'loss-mode-likelihood', 'loss-1d']) {
    // the 1-D panel has no test id of its own; it is found by its data-panel attribute below
    if (id !== 'loss-1d') expect(m.has(id), id).toBe(true);
  }
  expect(m.host.querySelector('[data-panel="loss-1d"]')).not.toBeNull();
  expect(m.host.querySelector('[data-panel="examples"]')).not.toBeNull();
  for (const id of ['btn-step', 'btn-run', 'btn-reset', 'btn-random-start', 'lr-slider']) {
    expect(m.has(id), id).toBe(false);
  }
  expect(s().lossMode).toBe('likelihood');
  m.done();
});

test('stage 1: Step, Reset and step size, no Run; the first step from 0.5 lands on 0.4', () => {
  load(NAMES[1] as string);
  const m = mount();
  expect(m.region('training-bar')).toBe(true);
  for (const id of ['btn-step', 'btn-reset', 'lr-slider']) expect(m.has(id), id).toBe(true);
  for (const id of ['btn-run', 'btn-random-start', 'optimizer']) expect(m.has(id), id).toBe(false);
  expect(m.region('palette')).toBe(false);
  expect(m.host.querySelector('[data-panel="step-log"]')).not.toBeNull();
  m.done();
  // hand calculation: 10 of the 30 deliveries are late, so the mean log loss has slope
  // -(1/3)/p + (2/3)/(1-p) = 2/3 at p = 0.5; with step size 0.15 one step is 0.5 - 0.15 * 2/3 = 0.4
  const y = s().dataset.yTrain;
  expect(y.length).toBe(30);
  expect(y.reduce((a, b) => a + b, 0)).toBe(10);
  expect(bias()[0]).toBeCloseTo(0.5, 6);
  expect(s().graph.train.lr).toBeCloseTo(0.15, 6);
  train(1);
  expect(bias()[0]).toBeCloseTo(0.4, 4);
  const row = s().stepLog.at(-1);
  expect(row?.loss).toBeCloseTo(-(Math.log(0.5) * 1), 4); // log loss at p = 0.5 is log 2
  s().reset();
  expect(bias()[0]).toBeCloseTo(0.5, 6);
  train(200);
  expect(s().diverged).toBe(false);
  expect(bias()[0]).toBeCloseTo(1 / 3, 3);
});

test('stage 2: a palette of Sigmoid and log loss, Run; adding the Sigmoid trains to the proportion', async () => {
  load(NAMES[2] as string);
  const m = mount();
  expect(m.has('btn-run')).toBe(true);
  expect(m.entries()).toEqual(['palette-sigmoid']); // the model tab offers the model piece
  act(() => s().setViewTab('loss'));
  expect(m.entries()).toEqual(['palette-log_loss']);
  for (const id of ['panel-fit', 'optimizer', 'batch']) expect(m.has(id), id).toBe(false);
  expect(m.host.querySelector('[data-panel="histogram"]')).toBeNull();
  m.done();
  // without the Sigmoid the number leaves 0 to 1 within a few steps and the run stops with a sentence
  train(10);
  expect(s().probIssue).toMatch(/between 0 and 1/);
  s().reset();
  const g = s().graph;
  const r = insertOnWire(g, 0, { type: 'link', fn: 'sigmoid' }, { locked: s().preset?.locked });
  expect(r.ok ? '' : r.error).toBe('');
  if (!r.ok) return;
  act(() => s().apply(r));
  await new Promise((res) => setTimeout(res, 100));
  expect(s().canTrain).toBe(true);
  train(300);
  expect(s().diverged).toBe(false);
  expect(1 / (1 + Math.exp(-(bias()[0] as number)))).toBeCloseTo(1 / 3, 3);
});

test('stage 3: squared error, the histogram view and its squares option', () => {
  load(NAMES[3] as string);
  const m = mount();
  expect(m.has('btn-run')).toBe(true);
  expect(m.host.querySelector('[data-panel="histogram"]')).not.toBeNull();
  expect(m.has('histogram-squares')).toBe(true);
  expect(m.entries().sort()).toEqual(['palette-sigmoid']);
  act(() => s().setViewTab('loss'));
  expect(m.entries().sort()).toEqual(['palette-log_loss', 'palette-loss_log_loss', 'palette-loss_squared_error']);
  m.done();
  train(400);
  expect(s().diverged).toBe(false);
  expect(1 / (1 + Math.exp(-(bias()[0] as number)))).toBeCloseTo(1 / 3, 2);
});

test('stage 4: Input, Weight, Bias, categories, fit plot and the 2-parameter surface', () => {
  load(NAMES[4] as string);
  const m = mount();
  expect(m.region('palette')).toBe(true);
  expect(m.entries().sort()).toEqual(
    ['palette-bias', 'palette-input', 'palette-column', 'palette-lookup', 'palette-onehot', 'palette-scalew', 'palette-sigmoid', 'palette-sum'].sort(),
  );
  expect(m.has('grain-toggle')).toBe(true);
  expect(m.has('btn-random-start')).toBe(true);
  expect(m.has('data-standardize')).toBe(false);
  expect(m.host.querySelector('[data-panel="fit"]')).not.toBeNull();
  act(() => s().setViewTab('loss'));
  expect(m.host.querySelector('[data-panel="surface"]')).not.toBeNull();
  m.done();
  expect(s().graph.data.standardize).toBe(true);
  train(300);
  expect(s().diverged).toBe(false);
  const w = s().host?.w?.W?.[0] as number;
  expect(w).toBeGreaterThan(0.2); // longer distance, more likely late
});

test('stage 5: Standardize, several inputs, Linear and Softmax, trains without diverging', () => {
  load(NAMES[5] as string);
  const m = mount();
  expect(m.has('data-standardize')).toBe(true);
  expect(m.entries()).toContain('palette-linear');
  expect(m.entries()).toContain('palette-softmax');
  m.done();
  const g = s().graph;
  const next = setColumns(g, 'x', ['x1', 'x2', 'x3'], { locked: s().preset?.locked });
  expect(next.ok).toBe(true);
  if (!next.ok) return;
  act(() => s().apply(next));
  // raw units (as loaded): small steps still do not blow up
  train(1);
  const start = s().liveLoss?.train ?? NaN;
  train(300);
  expect(s().diverged).toBe(false);
  act(() => s().setData({ standardize: true }));
  expect(s().canTrain).toBe(true);
  train(1);
  const start2 = s().liveLoss?.train ?? NaN;
  train(300);
  expect(s().diverged).toBe(false);
  expect(s().liveLoss?.train ?? NaN).toBeLessThan(start2);
});
