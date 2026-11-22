// @vitest-environment jsdom
import { beforeAll, expect, test } from 'vitest';
import { act } from 'react';
import { createRoot } from 'react-dom/client';
import { loadEngine } from '../src/engine.js';
import { beginWeightEdit } from '../src/canvas/weightEdit.js';
import {
  axisFor,
  histogramBins,
  lineDomain,
  logit,
  predictionChain,
  sigmoid,
  squareSides,
  valueToParam,
} from '../src/plot/histogramPure.js';
import { ResultsPanels } from '../src/results/ResultsPanels.js';
import { useRegionShown } from '../src/state/disclosure.js';
import { useStore } from '../src/state/store.js';
import { loadBundled } from '../src/topbar/PresetMenu.js';

(globalThis as { ResizeObserver?: unknown }).ResizeObserver = class {
  observe() {}
  unobserve() {}
  disconnect() {}
};

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;
window.matchMedia = ((q: string) => ({
  matches: false,
  media: q,
  addEventListener() {},
  removeEventListener() {},
})) as unknown as typeof window.matchMedia;
HTMLCanvasElement.prototype.getContext = (() =>
  null) as unknown as typeof HTMLCanvasElement.prototype.getContext;

/** Mounts `ui`; `q(testid)` finds an element in it. */
function mount(ui: React.ReactElement) {
  const host = document.createElement('div');
  document.body.appendChild(host);
  const root = createRoot(host);
  act(() => root.render(ui));
  return {
    q: (id: string) => host.querySelector(`[data-testid="${id}"]`),
    unmount: () => {
      act(() => root.unmount());
      host.remove();
    },
  };
}

const s = () => useStore.getState();

beforeAll(async () => {
  await loadEngine();
}, 60000);

test('bins count every value; binary gives two bars', () => {
  const b = histogramBins([0, 0, 1, 1, 1], true);
  expect(b.counts).toEqual([2, 3]);
  const r = histogramBins(
    Array.from({ length: 40 }, (_, i) => i),
    false,
  );
  expect(r.counts.reduce((a, c) => a + c, 0)).toBe(40);
});

test('squares have side |y - yhat|', () => {
  expect(squareSides([1, 3, 2], 2)).toEqual([1, 1, 0]);
});

test('squares axis is equal aspect and does not depend on the line', () => {
  const dom = lineDomain([0, 10], false);
  const a = axisFor(dom, 400, 100, true);
  // 1 unit is 100 / 12 px both ways; the widest possible square (range 12 units) fits the height
  expect(a.unit).toBeCloseTo(100 / 12);
  expect((a.xhi - a.xlo) * a.unit).toBeCloseTo(400);
  expect(axisFor(dom, 400, 100, false).unit).toBeCloseTo(400 / 12);
});

test('logit inverts sigmoid and the drag value is clamped off 0 and 1', () => {
  expect(sigmoid(logit(0.3))).toBeCloseTo(0.3);
  expect(Number.isFinite(valueToParam(0, true))).toBe(true);
  expect(Number.isFinite(valueToParam(1, true))).toBe(true);
  expect(valueToParam(0.7, false)).toBe(0.7);
});

test('prediction chain: bias, sigmoid(bias), and models with an input', () => {
  expect(loadBundled('intercept-mean')).toBe(true);
  expect(predictionChain(s().graph, 'bias')).toEqual({
    hasInput: false,
    param: { node: 'bias', sigmoid: false },
  });
  expect(loadBundled('intercept-prob')).toBe(true);
  expect(predictionChain(s().graph, 'yhat')).toEqual({
    hasInput: false,
    param: { node: 'bias', sigmoid: true },
  });
});

test('the panel shows on a no-input model and not on a model with an input', () => {
  expect(loadBundled('intercept-mean')).toBe(true);
  const m = mount(<ResultsPanels />);
  expect(m.q('target-histogram')).toBeTruthy();
  expect(m.q('histogram-yhat')?.textContent).toMatch(/ŷ = /);
  m.unmount();
  expect(loadBundled('linear')).toBe(true);
  const m2 = mount(<ResultsPanels />);
  expect(m2.q('target-histogram')).toBeNull();
  m2.unmount();
});

test('results region: a list naming only the histogram panel counts it only without inputs', () => {
  function Probe() {
    return <span data-testid="probe">{String(useRegionShown('results'))}</span>;
  }
  expect(loadBundled('intercept-mean')).toBe(true);
  useStore.setState({ show: { ui: ['panel-histogram'] } });
  const m = mount(<Probe />);
  expect(m.q('probe')?.textContent).toBe('true');
  m.unmount();
  expect(loadBundled('linear')).toBe(true);
  useStore.setState({ show: { ui: ['panel-histogram'] } });
  const m2 = mount(<Probe />);
  expect(m2.q('probe')?.textContent).toBe('false');
  m2.unmount();
});

test('setting the bias the way a drag does is one undo step and moves the line', () => {
  expect(loadBundled('intercept-mean')).toBe(true);
  s().pause();
  const edit = beginWeightEdit({ kind: 'bias', node: 'bias' });
  expect(edit).not.toBeNull();
  edit?.set(0.2);
  edit?.set(0.5);
  edit?.commit();
  expect(s().host?.bias?.b?.[0]).toBeCloseTo(0.5);
  expect(s().predictGrid(new Float32Array(200), 200)?.[0]).toBeCloseTo(0.5);
  s().undo();
  expect(s().host?.bias?.b?.[0]).not.toBeCloseTo(0.5);
});

test('a Bias with an incoming edge is not draggable (its output is parent + b)', () => {
  const g = {
    nodes: [
      { id: 'b1', type: 'bias' },
      { id: 'b2', type: 'bias' },
      { id: 's', type: 'link', fn: 'sigmoid' },
    ],
    edges: [
      ['b1', 'b2'],
      ['b2', 's'],
    ],
  } as unknown as Parameters<typeof predictionChain>[0];
  expect(predictionChain(g, 'b2').param).toBeNull();
  expect(predictionChain(g, 's').param).toBeNull();
  expect(predictionChain(g, 'b1').param).toEqual({ node: 'b1', sigmoid: false });
});
