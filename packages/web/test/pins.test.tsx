// @vitest-environment jsdom
import { beforeAll, beforeEach, expect, test, vi } from 'vitest';
import { loadEngine } from '../src/engine.js';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { setLinearOut } from '@tinker/core';
import { initialGraph } from '../src/state/graphSlice.js';
import { useStore } from '../src/state/store.js';
import { Pins } from '../src/results/Pins.js';
import { PIN_LIMIT } from '../src/results/pinsModel.js';

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

// jsdom has no ResizeObserver or matchMedia; the thumbnails' canvas hook needs both.
class RO {
  observe() {}
  disconnect() {}
}
vi.stubGlobal('ResizeObserver', RO);
vi.stubGlobal('matchMedia', () => ({ addEventListener() {}, removeEventListener() {} }));

const s = () => useStore.getState();
let root: Root;
let host: HTMLDivElement;

const q = (id: string) => host.querySelectorAll<HTMLButtonElement>(`[data-testid="${id}"]`);
const click = (el: Element | undefined) => {
  act(() => {
    el?.dispatchEvent(new MouseEvent('click', { bubbles: true }));
  });
};

beforeAll(async () => {
  await loadEngine();
}, 60000);

beforeEach(() => {
  s().pause();
  s().loadGraph(initialGraph());
  s().reset();
  useStore.setState({ pins: [] });
  host = document.createElement('div');
  document.body.replaceChildren(host);
  root = createRoot(host);
  act(() => root.render(<Pins />));
});

const runSome = () => {
  act(() => {
    for (let i = 0; i < 5; i++) s().stepOnce();
  });
  expect(s().history.length).toBe(5);
};

test('Pin is disabled before any step, then adds a chip', () => {
  expect(q('pin-add')[0]?.disabled).toBe(true);
  runSome();
  expect(q('pin-add')[0]?.disabled).toBe(false);
  click(q('pin-add')[0]);
  expect(q('pin-chip').length).toBe(1);
  expect(s().pins[0]?.history.length).toBe(5);
});

test('a fourth pin is refused: the button is disabled', () => {
  runSome();
  for (let i = 0; i < PIN_LIMIT; i++) click(q('pin-add')[0]);
  expect(q('pin-chip').length).toBe(PIN_LIMIT);
  expect(q('pin-add')[0]?.disabled).toBe(true);
  expect(s().pins.length).toBe(PIN_LIMIT);
});

test('a 1-D pin stores a copy of the model curve on the fixed grid', () => {
  runSome();
  click(q('pin-add')[0]);
  const fit = s().pins[0]?.fit;
  expect(fit?.x.length).toBe(200);
  expect(fit?.y.length).toBe(200);
  expect(Array.from(fit?.y ?? []).every(Number.isFinite)).toBe(true);
  const y0 = fit?.y[100];
  act(() => s().stepOnce());
  expect(s().pins[0]?.fit?.y[100]).toBe(y0);
});

test('the x removes a pin and frees its slot', () => {
  runSome();
  click(q('pin-add')[0]);
  click(q('pin-add')[0]);
  const second = s().pins[1];
  click(q('pin-remove')[0]);
  expect(s().pins.map((p) => p.id)).toEqual([second?.id]);
  click(q('pin-add')[0]);
  expect(
    s()
      .pins.map((p) => p.slot)
      .sort(),
  ).toEqual([0, 1]);
});

test('pins survive graph edits, undo and data changes', () => {
  runSome();
  click(q('pin-add')[0]);
  s().apply(setLinearOut(s().graph, 'lin1', 6));
  s().setData({ noise: 0.5 });
  s().undo();
  expect(s().pins.length).toBe(1);
});

test('loading a pin restores its graph and weights, and is one undo step', () => {
  runSome();
  const w = Array.from(s().host?.lin1?.W ?? []);
  click(q('pin-add')[0]);
  s().apply(setLinearOut(s().graph, 'lin1', 6));
  expect(s().graph.nodes.find((n) => n.id === 'lin1')).toMatchObject({ out: 6 });
  click(q('pin-load')[0]);
  expect(s().graph.nodes.find((n) => n.id === 'lin1')).toMatchObject({ out: 4 });
  expect(Array.from(s().host?.lin1?.W ?? [])).toEqual(w);
  s().undo();
  expect(s().graph.nodes.find((n) => n.id === 'lin1')).toMatchObject({ out: 6 });
});
