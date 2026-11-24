// @vitest-environment jsdom
import { beforeAll, expect, test } from 'vitest';
import { act } from 'react';
import { createRoot } from 'react-dom/client';
import { App } from '../src/App.js';
import { loadEngine } from '../src/engine.js';
import { PAUSE_FIRST } from '../src/canvas/weightEdit.js';
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

function ptr(type: string, y: number): Event {
  const e = new Event(type, { bubbles: true, cancelable: true });
  Object.assign(e, { pointerId: 1, clientX: 10, clientY: y, button: 0, pointerType: 'touch' });
  return e;
}

function mount() {
  const host = document.createElement('div');
  document.body.appendChild(host);
  const root = createRoot(host);
  act(() => root.render(<App />));
  act(() => {
    s().setViewTab('model');
    s().setGrain('neuron');
  });
  return {
    host,
    pill: (id: string) => host.querySelector(`[data-testid="weight-pill-${id}"]`),
    done: () => {
      act(() => root.unmount());
      host.remove();
    },
  };
}

test('neuron view draws every Weight as a piece box with its live value pill', () => {
  expect(loadBundled('scalar')).toBe(true);
  s().pause();
  const m = mount();
  expect(m.host.querySelector('[data-testid="neuron-view"]')).not.toBeNull();
  expect(m.host.querySelector('[data-placeholder]')).toBeNull();
  const weights = s().graph.nodes.filter((n) => n.type === 'scalew');
  expect(weights.length).toBeGreaterThan(0);
  for (const n of weights) {
    const pill = m.pill(n.id);
    expect(pill).not.toBeNull();
    expect(Number(pill?.getAttribute('data-value'))).toBeCloseTo(
      s().host?.[n.id]?.W?.[0] ?? NaN,
      6,
    );
    expect(pill?.textContent).toMatch(/^w\s=\s-?\d+\.\d\d/);
  }
  m.done();
});

test('in neuron view a vertical drag on a Weight pill is one undo step; refused while running', () => {
  expect(loadBundled('scalar')).toBe(true);
  s().pause();
  const m = mount();
  const id = (s().graph.nodes.find((n) => n.type === 'scalew') as { id: string }).id;
  const el = m.pill(id) as Element;
  const start = Number(el.getAttribute('data-value'));
  const past = s().past.length;
  act(() => {
    el.dispatchEvent(ptr('pointerdown', 100));
    el.dispatchEvent(ptr('pointermove', 60)); // 40 px up = +0.40
    el.dispatchEvent(ptr('pointerup', 60));
  });
  expect(s().host?.[id]?.W?.[0]).toBeCloseTo(start + 0.4, 4);
  expect(Number(m.pill(id)?.getAttribute('data-value'))).toBeCloseTo(start + 0.4, 4);
  expect(s().past.length).toBe(past + 1);
  s().undo();
  expect(s().host?.[id]?.W?.[0]).toBeCloseTo(start, 4);
  useStore.setState({ running: true, error: null });
  const el2 = m.pill(id) as Element;
  act(() => {
    el2.dispatchEvent(ptr('pointerdown', 100));
    el2.dispatchEvent(ptr('pointermove', 60));
    el2.dispatchEvent(ptr('pointerup', 60));
  });
  expect(s().error).toBe(PAUSE_FIRST);
  useStore.setState({ running: false, error: null });
  m.done();
});
