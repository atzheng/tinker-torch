// @vitest-environment jsdom
// The bias-variance presets (tinker-6h1.5): each shows exactly its controls, the tree depth edits on
// a locked graph, training loss falls toward 0 with depth and validation loss is U-shaped (fixed seed 5).
import { beforeAll, expect, test } from 'vitest';
import { act } from 'react';
import { createRoot } from 'react-dom/client';
import { setSetting } from '@tinker/core';
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

const s = () => useStore.getState();
const NAMES = [1, 2, 3].map((i) => `bias-variance-stage${i}`);

beforeAll(async () => {
  await loadEngine();
}, 60000);

function mount() {
  const host = document.createElement('div');
  document.body.appendChild(host);
  const root = createRoot(host);
  act(() => root.render(<App />));
  const has = (id: string) => host.querySelector(`[data-testid="${id}"]`) !== null;
  return {
    host,
    has,
    done: () => {
      act(() => root.unmount());
      host.remove();
    },
  };
}

function load(name: string) {
  expect(loadBundled(name), name).toBe(true);
  s().pause();
}

/** Final training and validation loss of a depth-`d` tree after the preset's training. */
function runDepth(d: number): { train: number; val: number } {
  const r = setSetting(s().graph, 'tree', 'depth', d, { locked: [] });
  act(() => s().apply(r));
  s().reset();
  for (let i = 0; i < 200; i++) s().stepOnce();
  const last = s().history.at(-1);
  return { train: last?.train ?? Number.NaN, val: last?.val ?? Number.NaN };
}

test('three presets: locked graph, DRAFT-free titles, prompt and steps, 30 training rows, sigma 1', () => {
  for (const name of NAMES) {
    const f = bundledPreset(name);
    if (!f || !('title' in f)) throw new Error(name);
    expect(f.prompt, name).toBeTruthy();
    expect((f.instructions ?? []).length, name).toBeGreaterThan(0);
    expect(f.graph.data).toMatchObject({ set: 'trend_sine', n: 43, noise: 0.6466, val: 0.3 });
    expect(f.show?.ui, name).toContain('tree-depth');
  }
  load(NAMES[0] as string);
  expect(s().dataset.nTrain).toBe(30);
});

test('stage 1: depth only, training curve without validation, no data controls but New sample', () => {
  load(NAMES[0] as string);
  const m = mount();
  expect(m.has('tree-depth')).toBe(false); // the inspector opens on a selection
  for (const id of ['btn-run', 'btn-reset', 'data-resample']) expect(m.has(id), id).toBe(true);
  for (const id of ['data-n', 'data-noise', 'data-seed', 'truth-toggle', 'loss-readout-val']) {
    expect(m.has(id), id).toBe(false);
  }
  expect(m.has('tab-loss')).toBe(false);
  expect(s().show?.ui).not.toContain('loss-val');
  m.done();
});

test('stage 1: raising the depth lowers the training loss toward 0', () => {
  load(NAMES[0] as string);
  const train = [1, 2, 4, 8].map((d) => runDepth(d).train);
  for (let i = 1; i < train.length; i++) expect(train[i]).toBeLessThan(train[i - 1] as number);
  expect(train.at(-1)).toBeLessThan(0.01);
});

test('stage 2: validation curve, n, noise, seed and the truth toggle; validation loss is U-shaped in depth', () => {
  load(NAMES[1] as string);
  const m = mount();
  for (const id of ['data-n', 'data-noise', 'data-seed', 'truth-toggle', 'loss-readout-val']) {
    expect(m.has(id), id).toBe(true);
  }
  expect(m.has('tab-loss')).toBe(false);
  m.done();
  const rows = [1, 2, 3, 4, 5, 6, 7, 8].map(runDepth);
  const val = rows.map((r) => r.val);
  const best = val.indexOf(Math.min(...val));
  expect(best).toBeGreaterThan(0); // not the shallowest
  expect(best).toBeLessThan(val.length - 1); // not the deepest
  expect(val[0]).toBeGreaterThan(val[best] as number);
  expect(val.at(-1)).toBeGreaterThan((val[best] as number) * 1.2);
  expect(rows.at(-1)?.train).toBeLessThan(0.01);
});

test('stage 3: minimum leaf size, the penalty on the loss tab, Resample x20 and pins', () => {
  load(NAMES[2] as string);
  const m = mount();
  for (const id of ['truth-toggle', 'data-n', 'tab-loss', 'loss-readout-val', 'resample-run']) {
    expect(m.has(id), id).toBe(true);
  }
  expect(s().graph.nodes.find((n) => n.id === 'pen')).toMatchObject({ type: 'scale', c: 0 });
  // everything but the penalty's Scale is locked: the Scale takes lambda, the Tree only its listed fields
  expect(s().preset?.locked).not.toContain('pen');
  expect(s().preset?.locked).toContain('tree');
  const lam = setSetting(s().graph, 'pen', 'c', 0.05, { locked: s().preset?.locked });
  expect(lam.ok).toBe(true);
  const leaf = setSetting(s().graph, 'tree', 'minLeaf', 4, { locked: [] });
  act(() => s().apply(leaf));
  expect(s().canTrain).toBe(true);
  m.done();
});
