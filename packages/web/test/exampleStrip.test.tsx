// @vitest-environment jsdom
// Per-example strip (tinker-6t7.6): p of what happened for log loss, per-example loss otherwise.
import { act } from 'react';
import { createRoot } from 'react-dom/client';
import { beforeAll, expect, test, vi } from 'vitest';
import { beginWeightEdit } from '../src/canvas/weightEdit.js';
import { loadEngine } from '../src/engine.js';
import { ExampleStrip } from '../src/results/ExampleStrip.js';
import { formatCell, stripCells } from '../src/results/exampleStripModel.js';
import { useStore } from '../src/state/store.js';
import { loadBundled } from '../src/topbar/PresetMenu.js';

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;
vi.stubGlobal('matchMedia', () => ({ addEventListener() {}, removeEventListener() {} }));
const s = () => useStore.getState();

beforeAll(async () => {
  await loadEngine();
}, 60000);

function setB(v: number): void {
  act(() => {
    const e = beginWeightEdit({ kind: 'bias', node: 'bias' });
    e?.set(v);
    e?.commit();
  });
}

function mount() {
  const el = document.createElement('div');
  document.body.appendChild(el);
  const root = createRoot(el);
  act(() => root.render(<ExampleStrip />));
  const vals = () =>
    Array.from(el.querySelectorAll('[data-testid="example-cell"]')).map((c) =>
      Number((c as HTMLElement).dataset.value),
    );
  return { el, vals };
}

test('intercept-prob: each cell is p of the observed outcome and follows a hand edit', () => {
  expect(loadBundled('intercept-prob')).toBe(true);
  const m = mount();
  const y = s().dataset.yTrain;
  const check = (b: number) => {
    setB(b);
    const p = 1 / (1 + Math.exp(-b));
    const v = m.vals();
    expect(v.length).toBe(y.length);
    v.forEach((x, i) => expect(x).toBeCloseTo(y[i] === 1 ? p : 1 - p, 4));
  };
  check(0.5);
  check(-1.5);
  expect(m.el.textContent).toContain('p of what happened');
});

test('intercept-mean: each cell is the per-example loss', () => {
  expect(loadBundled('intercept-mean')).toBe(true);
  const m = mount();
  setB(1);
  const losses = s().exampleLosses('train');
  expect(losses).not.toBeNull();
  m.vals().forEach((x, i) => expect(x).toBeCloseTo(losses?.[i] ?? NaN, 4));
  expect(m.el.textContent).toContain('loss per example');
});

test('stripCells: probability mode and fractions', () => {
  const c = stripCells(Float32Array.of(0, Math.LN2), [1, 0], 'prob', true);
  expect(c[0]?.value).toBeCloseTo(1);
  expect(c[1]?.value).toBeCloseTo(0.5);
  expect(c[1]?.cls).toBe(0);
  const l = stripCells(Float32Array.of(1, 4), [0, 0], 'loss', false);
  expect(l[1]?.frac).toBe(1);
  expect(l[0]?.frac).toBeCloseTo(0.25);
});

test('non-finite loss gets the worst shade; the scale argument fixes the shade', () => {
  const l = stripCells(Float32Array.of(Number.NaN, Infinity, 1), [0, 0, 0], 'loss', false, 4);
  expect(l[0]?.frac).toBe(1);
  expect(l[1]?.frac).toBe(1);
  expect(l[2]?.frac).toBeCloseTo(0.25);
});

test('formatCell fits a small cell', () => {
  for (const [v, t] of [
    [0, '0'],
    [0.0004, '<.01'],
    [0.492, '0.49'],
    [1, '1.00'],
    [12.34, '12.3'],
    [1234.5, '1235'],
    [123456, '1e5'],
    [Infinity, 'inf'],
  ] as const)
    expect(formatCell(v)).toBe(t);
});
