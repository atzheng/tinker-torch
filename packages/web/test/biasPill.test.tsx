// @vitest-environment jsdom
import { beforeAll, expect, test } from 'vitest';
import { act } from 'react';
import { createRoot } from 'react-dom/client';
import { App } from '../src/App.js';
import { loadEngine } from '../src/engine.js';
import { pillHit } from '../src/canvas/ScalarValuePill.js';
import { beginWeightEdit, PAUSE_FIRST } from '../src/canvas/weightEdit.js';
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

function mount() {
  const host = document.createElement('div');
  document.body.appendChild(host);
  const root = createRoot(host);
  act(() => root.render(<App />));
  return {
    pill: () => host.querySelector('[data-testid="bias-pill-bias"]'),
    done: () => {
      act(() => root.unmount());
      host.remove();
    },
  };
}

function ptr(type: string, y: number): Event {
  const e = new Event(type, { bubbles: true, cancelable: true });
  Object.assign(e, { pointerId: 1, clientX: 10, clientY: y, button: 0, pointerType: 'touch' });
  return e;
}

test('the unwired Bias shows its value, follows Set value and training; a locked Bias can be set', () => {
  expect(loadBundled('lecture1-stage0')).toBe(true);
  const m = mount();
  act(() => s().setViewTab('model'));
  expect(m.pill()).not.toBeNull();
  expect(s().preset?.locked).toContain('bias');
  const shown = () => Number(m.pill()?.getAttribute('data-value'));
  expect(shown()).toBeCloseTo(s().host?.bias?.b?.[0] ?? NaN, 6);
  expect(m.pill()?.textContent).toMatch(/^b\s=\s-?\d+\.\d\d/);
  // Set value (the inspector's path) on a locked piece
  s().pause();
  const edit = beginWeightEdit({ kind: 'bias', node: 'bias' });
  expect(edit).not.toBeNull();
  act(() => {
    edit?.set(0.5);
    edit?.commit();
  });
  expect(shown()).toBeCloseTo(0.5, 6);
  expect(m.pill()?.textContent).toContain('0.50');
  // a training step moves it
  act(() => {
    s().stepOnce();
  });
  expect(shown()).toBeCloseTo(s().host?.bias?.b?.[0] ?? NaN, 6);
  expect(shown()).not.toBeCloseTo(0.5, 4);
  m.done();
});

test('a vertical drag on the pill sets the value as one undo step; refused while running', () => {
  expect(loadBundled('lecture1-stage0')).toBe(true);
  s().pause();
  const m = mount();
  act(() => s().setViewTab('model'));
  const el = m.pill() as Element;
  const start = Number(el.getAttribute('data-value'));
  const past = s().past.length;
  act(() => {
    el.dispatchEvent(ptr('pointerdown', 100));
    el.dispatchEvent(ptr('pointermove', 60)); // 40 px up = +0.40
    el.dispatchEvent(ptr('pointerup', 60));
  });
  expect(s().host?.bias?.b?.[0]).toBeCloseTo(start + 0.4, 4);
  expect(s().past.length).toBe(past + 1);
  s().undo();
  expect(s().host?.bias?.b?.[0]).toBeCloseTo(start, 4);
  // running: refused with the toast
  useStore.setState({ running: true, error: null });
  const el2 = m.pill() as Element;
  act(() => {
    el2.dispatchEvent(ptr('pointerdown', 100));
    el2.dispatchEvent(ptr('pointermove', 60));
    el2.dispatchEvent(ptr('pointerup', 60));
  });
  expect(s().error).toBe(PAUSE_FIRST);
  expect(s().host?.bias?.b?.[0]).toBeCloseTo(start, 4);
  useStore.setState({ running: false, error: null });
  m.done();
});

test('the pill press rectangle keeps the upper half of the body free at the floor scale', () => {
  for (const scale of [0.733, 1, 2]) {
    const r = pillHit(112, 56, scale);
    expect(r.y).toBeGreaterThanOrEqual(28); // below the middle of the 56-unit box
    expect(r.h * scale).toBeGreaterThanOrEqual(32 - 1e-9);
    expect(r.x + r.w).toBeLessThanOrEqual(112 - 16); // clear of the output handle
  }
});
