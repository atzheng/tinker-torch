// @vitest-environment jsdom
// Predicted-versus-observed class bars (tinker-6t7.16).
import { act } from 'react';
import { createRoot } from 'react-dom/client';
import { beforeAll, expect, test, vi } from 'vitest';
import { beginWeightEdit } from '../src/canvas/weightEdit.js';
import { loadEngine } from '../src/engine.js';
import { ClassBars } from '../src/results/ClassBars.js';
import { classBars } from '../src/results/classBarsModel.js';
import { PANEL_TAB } from '../src/results/panelLayout.js';
import { useStore } from '../src/state/store.js';
import { loadBundled } from '../src/topbar/PresetMenu.js';

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;
vi.stubGlobal('matchMedia', () => ({ addEventListener() {}, removeEventListener() {} }));
const s = () => useStore.getState();

beforeAll(async () => {
  await loadEngine();
}, 60000);

function mount() {
  const el = document.createElement('div');
  document.body.appendChild(el);
  const root = createRoot(el);
  act(() => root.render(<ClassBars />));
  const read = (t: string) =>
    Array.from(el.querySelectorAll(`[data-testid="${t}"]`)).map((c) =>
      Number((c as HTMLElement).dataset.value),
    );
  return { el, pred: () => read('class-bar-pred'), obs: () => read('class-bar-obs') };
}

test('classBars: sigmoid gives two classes, softmax one per column', () => {
  const y = [1, 0, 0, 1];
  const b = classBars(Float32Array.of(0.9, 0.1, 0.2, 0.6), 4, 1, y, 2, ['no', 'yes']);
  expect(b?.map((x) => x.name)).toEqual(['no', 'yes']);
  expect(b?.[1]?.predicted).toBeCloseTo(0.45);
  expect(b?.[0]?.predicted).toBeCloseTo(0.55);
  expect(b?.[1]?.observed).toBeCloseTo(0.5);
  const m = classBars(Float32Array.of(0.5, 0.3, 0.2, 0.1, 0.1, 0.8), 2, 3, [0, 2], 3);
  expect(m?.map((x) => x.predicted)).toEqual([
    expect.closeTo(0.3),
    expect.closeTo(0.2),
    expect.closeTo(0.5),
  ]);
  expect(m?.map((x) => x.observed)).toEqual([0.5, 0, 0.5]);
  expect(m?.[0]?.name).toBe('class 0');
  expect(classBars(Float32Array.of(2), 1, 1, [1], 2)).toBeNull(); // not a probability
  expect(classBars(Float32Array.of(0.5), 1, 1, [1], 3)).toBeNull(); // shape does not fit
});

test('panel is on the model tab', () => {
  expect(PANEL_TAB['class-bars']).toBe('model');
});

test('intercept-prob: predicted follows a hand edit, observed is the label share', () => {
  expect(loadBundled('intercept-prob')).toBe(true);
  const m = mount();
  const y = s().dataset.yTrain;
  const ones = y.reduce((a, v) => a + (v === 1 ? 1 : 0), 0) / y.length;
  act(() => {
    const e = beginWeightEdit({ kind: 'bias', node: 'bias' });
    e?.set(0.5);
    e?.commit();
  });
  const p = 1 / (1 + Math.exp(-0.5));
  expect(m.pred()[1]).toBeCloseTo(p, 4);
  expect(m.pred()[0]).toBeCloseTo(1 - p, 4);
  expect(m.obs()[1]).toBeCloseTo(ones, 4);
  expect(m.obs()[0]).toBeCloseTo(1 - ones, 4);
});

test('softmax preset: one bar pair per class, each side sums to 1', () => {
  expect(loadBundled('softmax')).toBe(true);
  const m = mount();
  const k = s().dataset.info.numClasses;
  expect(m.pred().length).toBe(k);
  expect(m.pred().reduce((a, v) => a + v, 0)).toBeCloseTo(1, 4);
  expect(m.obs().reduce((a, v) => a + v, 0)).toBeCloseTo(1, 4);
});
