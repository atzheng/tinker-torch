// @vitest-environment jsdom
// The tree classification presets (tinker-pc8): moons, n 120, noise 0.5, seed 5; each shows exactly its
// controls, the tree depth edits on a locked graph, training loss falls toward 0 with depth and
// validation loss is U-shaped (fixed seed).
import { beforeAll, expect, test } from 'vitest';
import { act } from 'react';
import { createRoot } from 'react-dom/client';
import { setSetting } from '@tinker/core';
import { App } from '../src/App.js';
import { loadEngine } from '../src/engine.js';
import { bundledPreset } from '../src/presets/index.js';
import { useStore } from '../src/state/store.js';
import { editSettingOpen } from '../src/canvas/drawings/fields.js';
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
const NAMES = [1, 2, 3].map((i) => `tree-class-stage${i}`);

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

test('three presets: locked graph, prompt and steps, 72 training rows, moons', () => {
  for (const name of NAMES) {
    const f = bundledPreset(name);
    if (!f || !('title' in f)) throw new Error(name);
    expect(f.prompt, name).toBeTruthy();
    expect((f.instructions ?? []).length, name).toBeGreaterThan(0);
    expect(f.graph.data).toMatchObject({ set: 'moons', n: 120, noise: 0.5, val: 0.4, seed: 5 });
    expect(f.show?.ui, name).toContain('tree-depth');
    expect(f.show?.ui, name).toContain('tab-model');
    expect(f.show?.ui, name).toContain('tab-loss');
  }
  load(NAMES[0] as string);
  expect(s().dataset.nTrain).toBe(72);
});

test('stage 1: depth only, training curve without validation', () => {
  load(NAMES[0] as string);
  const m = mount();
  for (const id of ['btn-run', 'btn-reset', 'data-resample']) expect(m.has(id), id).toBe(true);
  for (const id of ['data-n', 'data-noise', 'data-seed', 'truth-toggle', 'loss-readout-val']) {
    expect(m.has(id), id).toBe(false);
  }
  expect(s().show?.ui).not.toContain('loss-val');
  m.done();
});

test('stage 1: raising the depth lowers the training loss toward 0', () => {
  load(NAMES[0] as string);
  const train = [1, 2, 4, 8].map((d) => runDepth(d).train);
  for (let i = 1; i < train.length; i++) expect(train[i]).toBeLessThan(train[i - 1] as number);
  expect(train.at(-1)).toBeLessThan(0.01);
});

test('stage 2: validation loss is U-shaped in depth', () => {
  load(NAMES[1] as string);
  const m = mount();
  for (const id of ['data-n', 'data-noise', 'data-seed', 'truth-toggle', 'loss-readout-val']) {
    expect(m.has(id), id).toBe(true);
  }
  m.done();
  const rows = [1, 2, 3, 4, 5, 6, 7, 8].map(runDepth);
  const val = rows.map((r) => r.val);
  const best = val.indexOf(Math.min(...val));
  expect(best).toBeGreaterThan(0);
  expect(best).toBeLessThan(val.length - 1);
  expect(val[0]).toBeGreaterThan((val[best] as number) * 2);
  expect(val.at(-1)).toBeGreaterThan((val[best] as number) * 2);
});

test('stage 3: validation points toggle and minimum leaf size on a locked graph', () => {
  load(NAMES[2] as string);
  const m = mount();
  for (const id of ['val-points-toggle', 'truth-toggle', 'loss-readout-val']) {
    expect(m.has(id), id).toBe(true);
  }
  expect(s().preset?.locked).toContain('tree');
  editSettingOpen('tree', 'minLeaf', 4, 'tree-min-leaf');
  expect(s().graph.nodes.find((n) => n.id === 'tree')).toMatchObject({ minLeaf: 4 });
  editSettingOpen('tree', 'depth', 5, 'tree-depth');
  expect(s().graph.nodes.find((n) => n.id === 'tree')).toMatchObject({ depth: 5 });
  expect(s().canTrain).toBe(true);
  m.done();
});
