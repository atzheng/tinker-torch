// @vitest-environment jsdom
import { afterEach, beforeAll, beforeEach, expect, test } from 'vitest';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { Inspector } from '../src/canvas/Inspector.js';
import { loadEngine } from '../src/engine.js';
import { mlpGraph as initialGraph } from './mlpGraph.js';
import { useStore } from '../src/state/store.js';

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;
const s = () => useStore.getState();
let el: HTMLDivElement;
let root: Root;

beforeAll(async () => {
  await loadEngine();
}, 60000);
beforeEach(() => {
  s().pause();
  s().loadGraph(initialGraph());
  el = document.createElement('div');
  document.body.appendChild(el);
  root = createRoot(el);
  act(() =>
    root.render(
      <Inspector
        selection={{ kind: 'edge', node: 'lin1', i: 0, j: 0 }}
        analysis={s().analysis}
        wireMessages={[]}
      />,
    ),
  );
});
afterEach(() => {
  act(() => root.unmount());
  el.remove();
});

function type(input: HTMLInputElement, value: string) {
  const set = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')?.set;
  act(() => {
    set?.call(input, value);
    input.dispatchEvent(new Event('input', { bubbles: true }));
  });
}
const field = () => el.querySelector('[data-testid="inspector-weight-input"]') as HTMLInputElement;
const key = (k: string) =>
  act(() => {
    field().dispatchEvent(new KeyboardEvent('keydown', { key: k, bubbles: true }));
  });

test('Escape commits nothing; Enter commits one undo step', () => {
  const past = s().past.length;
  field().focus();
  type(field(), '0.5');
  key('Escape');
  expect(s().past.length).toBe(past);
  expect(s().host?.lin1?.W?.[0]).not.toBeCloseTo(0.5, 6);
  field().focus();
  type(field(), '0.5');
  key('Enter');
  expect(s().past.length).toBe(past + 1);
  expect(s().host?.lin1?.W?.[0]).toBeCloseTo(0.5, 6);
});
