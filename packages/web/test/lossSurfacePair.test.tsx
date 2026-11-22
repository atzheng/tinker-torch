// @vitest-environment jsdom
import { beforeAll, expect, test } from 'vitest';
import { act } from 'react';
import { createRoot } from 'react-dom/client';
import { loadEngine } from '../src/engine.js';
import { listParams } from '../src/plot/lossCurve1d.js';
import { SURFACE_N, defaultPair, resolvePair, surfaceParams } from '../src/plot/lossSurface.js';
import { LossSurface } from '../src/results/LossSurface.js';
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

beforeAll(async () => {
  await loadEngine();
}, 60000);

function mount(ui: React.ReactElement) {
  const host = document.createElement('div');
  document.body.appendChild(host);
  const root = createRoot(host);
  act(() => root.render(ui));
  return { host, done: () => act(() => root.unmount()) };
}

test('surfaceParams needs two parameters; the default pair prefers single-entry parameters', () => {
  const host = {
    lookup: { W: Float32Array.of(1, 2, 3, 4) },
    w: { W: Float32Array.of(0.5) },
    bias: { b: Float32Array.of(0.25) },
  };
  const ps = surfaceParams(host, ['lookup', 'w', 'bias']) ?? [];
  expect(ps).toHaveLength(6);
  expect(defaultPair(ps)).toEqual(['w:W:0', 'bias:b:0']);
  expect(surfaceParams({ w: { W: Float32Array.of(1) } }, ['w'])).toBeNull();
  // only Lookup entries: the first two
  const only = listParams({ l: { W: Float32Array.of(1, 2, 3) } }, ['l']);
  expect(defaultPair(only)).toEqual(['l:W:0', 'l:W:1']);
  // a pick equal to the other axis is moved off it; stale keys fall back to the default
  const pair = resolvePair(ps, ['w:W:0', 'w:W:0']);
  expect(pair?.[0].key).toBe('w:W:0');
  expect(pair?.[1].key).not.toBe('w:W:0');
  expect(resolvePair(ps, ['gone', 'gone2'])?.map((p) => p.key)).toEqual(['w:W:0', 'bias:b:0']);
});

test('lecture1-stage4: surface over the chosen pair, switching works, a grid point is the loss with the rest held', () => {
  expect(loadBundled('lecture1-stage4')).toBe(true);
  expect(s().analysis.totalParams).toBe(6);
  expect(s().canTrain).toBe(true);
  const zone = Object.keys(s().host ?? {}).find((k) => s().host?.[k]?.W?.length === 4) ?? '';
  expect(zone).not.toBe('');
  s().writeParams(zone, { W: Float32Array.of(0.3, -0.2, 0.1, 0.4) });
  const m = mount(<LossSurface />);
  expect(m.host.querySelector('[data-testid="loss-surface"]')).not.toBeNull();
  const px = m.host.querySelector<HTMLSelectElement>('[data-testid="surface-axis-x"]');
  const py = m.host.querySelector<HTMLSelectElement>('[data-testid="surface-axis-y"]');
  expect(px?.options.length).toBe(6);
  const params = listParams(s().host ?? {}, s().analysis.order);
  const [d0, d1] = defaultPair(params);
  expect(px?.value).toBe(d0);
  expect(py?.value).toBe(d1);
  expect(d0.startsWith(`${zone}:`)).toBe(false);
  expect(d1.startsWith(`${zone}:`)).toBe(false);
  act(() => {
    if (!py) return;
    const setter = Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype, 'value')?.set;
    setter?.call(py, `${zone}:W:2`);
    py.dispatchEvent(new Event('change', { bubbles: true }));
  });
  expect(py?.value).toBe(`${zone}:W:2`);
  expect(px?.value).toBe(d0);
  m.done();

  // the grid point (i, j) over (d0, zone row 2) equals the loss with d0 and the row set by hand
  const a0 = params.find((p) => p.key === d0);
  const a1 = params.find((p) => p.key === `${zone}:W:2`);
  if (!a0 || !a1) throw new Error('missing params');
  const v = Float32Array.from({ length: SURFACE_N }, (_, i) => -1 + (2 * i) / (SURFACE_N - 1));
  const grid = s().lossSurface([a0.axis, a1.axis], v, v);
  expect(grid).not.toBeNull();
  const i = 10;
  const j = 37;
  const expected = grid?.[i * SURFACE_N + j] ?? Number.NaN;
  // set d0 by hand; the row is swept by the 1-D curve with everything else at its current value
  const set = (a: typeof a0, x: number) =>
    s().writeParams(
      a.axis.node,
      a.axis.tensor === 'W'
        ? {
            W: Float32Array.from(s().host?.[a.axis.node]?.W ?? [], (y, k) =>
              k === a.axis.index ? x : y,
            ),
          }
        : { b: Float32Array.of(x) },
    );
  set(a0, v[i] ?? 0);
  const curve = s().lossCurve(a1.axis, Float32Array.of(v[j] ?? 0));
  expect(expected).toBeCloseTo(curve?.[0] ?? Number.NaN, 4);
});

test('a two-parameter model is unchanged: no pickers', () => {
  expect(loadBundled('intercept-prob')).toBe(true);
  let found = false;
  for (const name of ['linear', 'scalar', 'sine', 'logistic']) {
    loadBundled(name);
    if (s().analysis.totalParams === 2) {
      found = true;
      break;
    }
  }
  expect(found).toBe(true);
  const m = mount(<LossSurface />);
  expect(m.host.querySelector('[data-testid="loss-surface"]')).not.toBeNull();
  expect(m.host.querySelector('[data-testid="surface-axis-x"]')).toBeNull();
  expect(m.host.querySelector('[data-testid="surface-axis-y"]')).toBeNull();
  m.done();
});

test('stage 4: the grid is re-evaluated when training moves the held parameters', () => {
  expect(loadBundled('lecture1-stage4')).toBe(true);
  const real = s().lossSurface;
  const calls: Float32Array[] = [];
  useStore.setState({
    lossSurface: (axes, a, b) => {
      const g = real(axes, a, b);
      if (g) calls.push(g);
      return g;
    },
  });
  const m = mount(<LossSurface />);
  const n0 = calls.length;
  expect(n0).toBeGreaterThan(0);
  const zone = Object.keys(s().host ?? {}).find((k) => s().host?.[k]?.W?.length === 4) ?? '';
  const before = s().host?.[zone]?.W?.[0];
  act(() => {
    for (let i = 0; i < 5; i++) s().stepOnce();
  });
  expect(s().host?.[zone]?.W?.[0]).not.toBe(before); // a held parameter moved
  expect(calls.length).toBeGreaterThan(n0);
  expect(calls[calls.length - 1]).not.toEqual(calls[0]);
  m.done();
  useStore.setState({ lossSurface: real });
});

test('more than two parameters: the panel is opt-in through show.ui; two parameters always show', () => {
  loadBundled('mlp');
  const m = mount(<LossSurface />);
  expect(s().analysis.totalParams).toBeGreaterThan(2);
  expect(s().show?.ui).toBeUndefined();
  expect(m.host.querySelector('[data-testid="loss-surface"]')).toBeNull();
  act(() => {
    useStore.setState({ show: { ui: ['panel-surface'] } });
  });
  expect(m.host.querySelector('[data-testid="loss-surface"]')).not.toBeNull();
  m.done();
  useStore.setState({ show: null });
});
