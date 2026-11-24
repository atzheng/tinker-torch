// @vitest-environment jsdom
import { afterEach, beforeAll, expect, test } from 'vitest';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { Inspector } from '../src/canvas/Inspector.js';
import { loadEngine } from '../src/engine.js';
import { mlpGraph } from './mlpGraph.js';
import { useStore } from '../src/state/store.js';

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;
const s = () => useStore.getState();
let el: HTMLDivElement;
let root: Root;

beforeAll(async () => {
  await loadEngine();
}, 60000);
afterEach(() => {
  act(() => root.unmount());
  el.remove();
});

function show(id: string, out?: number) {
  s().pause();
  const g = mlpGraph();
  if (out !== undefined) {
    g.nodes = g.nodes.map((n) => (n.id === id && n.type === 'linear' ? { ...n, out } : n));
  }
  s().loadGraph(g);
  el = document.createElement('div');
  document.body.appendChild(el);
  root = createRoot(el);
  act(() =>
    root.render(
      <Inspector selection={{ kind: 'node', id }} analysis={s().analysis} wireMessages={[]} />,
    ),
  );
}
const q = (t: string) => el.querySelector(`[data-testid="${t}"]`);

test('a small Linear shows W and the bias row with the model values', () => {
  show('lin1');
  const inW = s().analysis.inWidth.lin1 as number;
  const W = s().host?.lin1?.W as Float32Array;
  const cols = W.length / inW;
  expect(q('inspector-matrix')).not.toBeNull();
  expect(q('imcell-0-0')?.getAttribute('data-value')).toBe(String(W[0]));
  expect(q(`imcell-${inW - 1}-${cols - 1}`)).not.toBeNull();
  expect(q('imcell-b-0')).not.toBeNull();
  expect(q('inspector-matrix-large')).toBeNull();
});

test('the matrix follows host while paused', () => {
  show('lin1');
  const host = s().host as NonNullable<ReturnType<typeof s>['host']>;
  const W = Float32Array.from(host.lin1?.W as Float32Array, () => 0.25);
  act(() => useStore.setState({ host: { ...host, lin1: { ...host.lin1, W } } }));
  expect(q('imcell-0-0')?.getAttribute('data-value')).toBe('0.25');
  expect(q('imcell-0-0')?.textContent).toBe('0.25');
});

test('a large Linear shows its shape only', () => {
  show('lin1', 16);
  expect(q('inspector-matrix')).toBeNull();
  expect(q('inspector-matrix-large')?.textContent).toMatch(/to 16 units$/);
});
