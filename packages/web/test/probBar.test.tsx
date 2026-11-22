// @vitest-environment jsdom
import { act } from 'react';
import { createRoot } from 'react-dom/client';
import { beforeAll, expect, test } from 'vitest';
import { analyse } from '@tinker/core';
import { loadEngine } from '../src/engine.js';
import { useStore } from '../src/state/store.js';
import { loadBundled } from '../src/topbar/PresetMenu.js';
import { useProbBars } from '../src/state/derived.js';
import { probBarDue, probBarOf, probBars, probBarTargets } from '../src/canvas/probBarPure.js';

const s = () => useStore.getState();
const sigmoid = (z: number) => 1 / (1 + Math.exp(-z));

beforeAll(async () => {
  await loadEngine();
}, 60000);

test('probBarOf: the same value for every example is shown as is; differing ones are a mean', () => {
  expect(probBarOf(Float32Array.of(0.25), 1, 1)).toEqual({ values: [0.25], mean: false });
  expect(probBarOf(Float32Array.of(0.25, 0.25, 0.25), 3, 1)?.mean).toBe(false);
  const m = probBarOf(Float32Array.of(0.2, 0.6), 2, 1);
  expect(m?.mean).toBe(true);
  expect(m?.values[0]).toBeCloseTo(0.4, 6);
});

test('probBarOf: softmax gives class means; bad shapes and values give null', () => {
  const bar = probBarOf(Float32Array.of(0.5, 0.3, 0.2, 0.1, 0.5, 0.4), 2, 3);
  expect(bar?.values.map((v) => +v.toFixed(3))).toEqual([0.3, 0.4, 0.3]);
  expect(bar?.mean).toBe(true);
  expect(probBarOf(Float32Array.of(0.5), 2, 1)).toBeNull();
  expect(probBarOf(Float32Array.of(NaN), 1, 1)).toBeNull();
  expect(probBarOf(Float32Array.of(1.5), 1, 1)).toBeNull();
  expect(probBarOf(undefined, 1, 1)).toBeNull();
});

test('probBars only covers Links that carry a prediction pin', () => {
  const graph = {
    nodes: [
      { id: 'a', type: 'link', fn: 'sigmoid' },
      { id: 'b', type: 'link', fn: 'sigmoid' },
      { id: 'c', type: 'bias' },
    ],
    predictions: ['b'],
  } as never;
  const widths = { a: 1, b: 1, c: 1 };
  expect(probBarTargets(graph, widths)).toEqual([{ id: 'b', pin: 0, width: 1 }]);
  expect(Object.keys(probBars(graph, widths, [Float32Array.of(0.5)], 1))).toEqual(['b']);
});

test('no-input sigmoid preset: the Link bar equals sigmoid(b), live', () => {
  expect(loadBundled('intercept-prob')).toBe(true);
  s().pause();
  const analysis = analyse(s().graph, s().dataInfo);
  const out: { current: ReturnType<typeof useProbBars> } = { current: {} };
  function Probe() {
    out.current = useProbBars(analysis);
    return null;
  }
  const root = createRoot(document.createElement('div'));
  act(() => root.render(<Probe />));
  const result = out;
  act(() => {
    s().writeParams('bias', { b: Float32Array.of(2) });
  });
  const first = result.current['yhat'];
  expect(first?.mean).toBe(false);
  expect(first?.values[0]).toBeCloseTo(sigmoid(2), 5);
  act(() => {
    for (let i = 0; i < 5; i++) s().stepOnce();
  });
  const b = s().host?.bias?.b?.[0] as number;
  expect(b).not.toBe(2);
  expect(result.current['yhat']?.values[0]).toBeCloseTo(sigmoid(b), 5);
  act(() => root.unmount());
});

test('probBarDue: throttled only while running with many rows', () => {
  expect(probBarDue(100, 0, true, false)).toBe(false);
  expect(probBarDue(300, 0, true, false)).toBe(true);
  expect(probBarDue(10, 0, false, false)).toBe(true);
  expect(probBarDue(10, 0, true, true)).toBe(true);
});
