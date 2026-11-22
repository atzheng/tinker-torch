// @vitest-environment jsdom
import { beforeAll, beforeEach, expect, test } from 'vitest';
import { act } from 'react';
import { createRoot } from 'react-dom/client';
import { loadEngine } from '../src/engine.js';
import {
  CURVE_MIN_HALF,
  listParams,
  nextAxisRange,
  sweepValues,
  tangent,
  visit,
  yRange,
} from '../src/plot/lossCurve1d.js';
import { LossCurve1D } from '../src/results/LossCurve1D.js';
import { ResultsPanels } from '../src/results/ResultsPanels.js';
import { useStore } from '../src/state/store.js';
import { loadBundled } from '../src/topbar/PresetMenu.js';
import { mlpGraph } from './mlpGraph.js';

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

beforeAll(async () => {
  await loadEngine();
}, 60000);

beforeEach(() => {
  s().pause();
});

test('listParams lists every scalar entry with a label from the piece name', () => {
  const host = {
    bias: { b: Float32Array.of(0.5) },
    lin: { W: Float32Array.of(1, 2), b: Float32Array.of(3) },
  };
  const ps = listParams(host, ['bias', 'lin']);
  expect(ps.map((p) => p.key)).toEqual(['bias:b:0', 'lin:W:0', 'lin:W:1', 'lin:b:0']);
  expect(ps.map((p) => p.value)).toEqual([0.5, 1, 2, 3]);
  expect(ps.map((p) => p.label)).toEqual(['bias $b$', 'lin $W_{1}$', 'lin $W_{2}$', 'lin $b$']);
});

test('the range grows to hold every visited value and the dot', () => {
  let v = visit(null, 0.5);
  let r = nextAxisRange(null, v ?? { min: 0, max: 0 }, 0.5);
  expect(r).toEqual([0.5 - CURVE_MIN_HALF, 0.5 + CURVE_MIN_HALF]);
  for (const x of [1, 2.5, 6, -4, 9]) {
    v = visit(v, x);
    r = nextAxisRange(r, v ?? { min: 0, max: 0 }, x);
    expect(r?.[0]).toBeLessThanOrEqual(v?.min ?? 0);
    expect(r?.[1]).toBeGreaterThanOrEqual(v?.max ?? 0);
  }
  // inside the range nothing changes
  const same = nextAxisRange(r, v ?? { min: 0, max: 0 }, 1);
  expect(same).toBe(r);
  // non-finite values are ignored
  expect(visit(v, Number.NaN)).toBe(v);
});

test('explored only sweeps only the visited interval; one visited value sweeps nothing', () => {
  expect(sweepValues([-3, 3], { min: 1, max: 1 }, true)).toHaveLength(0);
  const vals = sweepValues([-3, 3], { min: -1, max: 2 }, true);
  expect(vals[0]).toBeCloseTo(-1);
  expect(vals[vals.length - 1]).toBeCloseTo(2);
  const full = sweepValues([-3, 3], { min: -1, max: 2 }, false);
  expect(full[0]).toBeCloseTo(-3);
  expect(full[full.length - 1]).toBeCloseTo(3);
});

test('tangent and y range', () => {
  expect(tangent(1, 2, 3, 0.5)).toEqual({ x1: 0.5, y1: 0.5, x2: 1.5, y2: 3.5 });
  expect(tangent(1, 2, Number.NaN, 0.5)).toBeNull();
  const y = yRange(Float32Array.of(1, 2, Number.NaN, 3), 2.5);
  expect(y.lo).toBeLessThan(1);
  expect(y.hi).toBeGreaterThan(3 - 1e-9);
  expect(yRange(new Float32Array(0), null)).toEqual({ lo: 0, hi: 1 });
});

test('intercept-prob: the curve at the dot is the training loss, the slope is what Step applies', () => {
  expect(loadBundled('intercept-prob')).toBe(true);
  expect(s().canTrain).toBe(true);
  s().writeParams('bias', { b: Float32Array.of(0.7) });
  const axis = { node: 'bias', tensor: 'b' as const, index: 0 };
  const b0 = s().host?.bias?.b?.[0] ?? Number.NaN;
  const slope = s().lossSlope(axis);
  const at = s().lossCurve(axis, Float32Array.of(b0));
  expect(slope).not.toBeNull();
  const lr = s().graph.train.lr;
  expect(s().graph.train.optimizer).toBe('sgd');
  expect(s().graph.train.batch).toBe('full');
  s().stepOnce();
  // the loss Step reports is the loss before its update
  expect(at?.[0]).toBeCloseTo(s().trainLoss ?? Number.NaN, 5);
  const b1 = s().host?.bias?.b?.[0] ?? Number.NaN;
  expect((b0 - b1) / lr).toBeCloseTo(slope ?? Number.NaN, 4);
});

function mount(ui: React.ReactElement) {
  const host = document.createElement('div');
  document.body.appendChild(host);
  const root = createRoot(host);
  act(() => root.render(ui));
  return { host, done: () => act(() => root.unmount()) };
}

test('intercept-prob: the panel shows without an axis picker; the tangent toggle shows the slope', () => {
  loadBundled('intercept-prob');
  const m = mount(<LossCurve1D />);
  expect(m.host.querySelector('[data-testid="loss-curve-1d"]')).not.toBeNull();
  expect(m.host.querySelector('[data-testid="loss-1d-axis"]')).toBeNull(); // one parameter
  expect(m.host.querySelector('[data-testid="curve1d-slope"]')).toBeNull();
  const tan = m.host.querySelector<HTMLInputElement>('[data-testid="loss-1d-tangent"]');
  act(() => tan?.click());
  expect(m.host.querySelector('[data-testid="curve1d-slope"]')?.textContent).toMatch(/^slope -?\d/);
  m.done();
});

test('a model with several parameters lists them in the picker', () => {
  s().loadGraph(mlpGraph());
  const m = mount(<LossCurve1D />);
  const pick = m.host.querySelector<HTMLSelectElement>('[data-testid="loss-1d-axis"]');
  expect(pick).not.toBeNull();
  expect(pick?.options.length).toBe(s().analysis.totalParams);
  m.done();
});

test('the panel is opt-in: no ui list means no loss-1d panel; a ui list with panel-loss-1d shows it', () => {
  loadBundled('intercept-prob');
  expect(s().show).toBeNull();
  act(() => s().setViewTab('loss')); // the loss panels are on the loss tab
  const m = mount(<ResultsPanels />);
  expect(m.host.querySelector('[data-panel="loss"]')).not.toBeNull(); // the default panels are there
  expect(m.host.querySelector('[data-panel="loss-1d"]')).toBeNull();
  act(() => {
    useStore.setState({ show: { ui: ['panel-fit', 'panel-loss'] } });
  });
  expect(m.host.querySelector('[data-panel="loss-1d"]')).toBeNull();
  act(() => {
    useStore.setState({ show: { ui: ['panel-loss-1d'] } });
  });
  expect(m.host.querySelector('[data-panel="loss-1d"]')).not.toBeNull();
  expect(m.host.querySelector('[data-testid="loss-1d-tangent"]')).toBeNull(); // control not listed
  m.done();
  useStore.setState({ show: null });
});
