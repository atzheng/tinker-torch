// @vitest-environment jsdom
import { afterEach, beforeEach, expect, test } from 'vitest';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { Canvas } from '../src/canvas/Canvas.js';
import { Palette } from '../src/palette/Palette.js';
import { deleteSelection } from '../src/canvas/actions.js';
import { mlpGraph as initialGraph } from './mlpGraph.js';
import { useStore } from '../src/state/store.js';

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

let host: HTMLDivElement;
let root: Root;
beforeEach(() => {
  useStore.getState().loadGraph(initialGraph());
  useStore.getState().setViewTab('model');
  useStore.getState().select(null);
  host = document.createElement('div');
  document.body.appendChild(host);
  root = createRoot(host);
});
afterEach(() => {
  act(() => root.unmount());
  host.remove();
});
const q = (id: string) => host.querySelector(`[data-testid="${id}"]`);
const press = (el: Element | null) => {
  if (!el) throw new Error('missing element');
  // jsdom has no PointerEvent: a plain event with the fields the gesture code reads
  const ev = (type: string) => {
    const e = new Event(type, { bubbles: true, cancelable: true });
    Object.assign(e, { pointerId: 1, clientX: 10, clientY: 10, button: 0, pointerType: 'touch' });
    return e;
  };
  act(() => {
    el.dispatchEvent(ev('pointerdown'));
    el.dispatchEvent(ev('pointerup'));
  });
};

test('regions are drawn behind the pieces: model and objective, none for dangling pieces', () => {
  act(() => root.render(<Canvas />));
  expect(q('region-model')).not.toBeNull();
  expect(q('region-objective')).not.toBeNull();
  // taking the L pin away leaves the model region only
  act(() => useStore.getState().loadGraph({ ...initialGraph(), loss: null }));
  expect(q('region-model')).not.toBeNull();
  expect(q('region-objective')).toBeNull();
});

test('the pins are the role pins; a selected piece offers add pins for the roles it lacks', () => {
  act(() => root.render(<Canvas />));
  expect(q('pin-pred-lin2')?.getAttribute('data-role')).toBe('pred');
  expect(host.querySelectorAll('[data-stub="true"]')).toHaveLength(1); // the stub of the Subtract
  expect(q('pin-loss-loss')).toBeNull(); // the L pin is on the loss tab
  act(() => useStore.getState().setViewTab('loss'));
  expect(q('pin-loss-loss')?.getAttribute('data-role')).toBe('loss');
  expect(q('pin-pred-lin2')?.getAttribute('data-role')).toBe('pred'); // on the stub of its piece
  act(() => useStore.getState().setViewTab('model'));
  act(() => useStore.getState().selectNode('act1'));
  expect(q('pin-add-pred-act1')).not.toBeNull();
  expect(q('pin-add-loss-act1')).not.toBeNull();
  // a node that already carries ŷ only offers L
  act(() => useStore.getState().selectNode('lin2'));
  expect(q('pin-add-pred-lin2')).toBeNull();
  expect(q('pin-add-loss-lin2')).not.toBeNull();
});

test('tapping an add pin pins the role there in one undo step and selects the new pin', () => {
  act(() => root.render(<Canvas />));
  act(() => useStore.getState().selectNode('act1'));
  press(q('pin-add-pred-act1'));
  expect(useStore.getState().graph.predictions).toEqual(['lin2', 'act1']);
  expect(useStore.getState().selection).toEqual({ kind: 'pin', role: 'pred', node: 'act1' });
  act(() => useStore.getState().undo());
  expect(useStore.getState().graph.predictions).toEqual(['lin2']);
  act(() => useStore.getState().setViewTab('loss'));
  act(() => useStore.getState().selectNode('err'));
  press(q('pin-add-loss-err'));
  expect(useStore.getState().graph.loss).toBe('err');
  act(() => useStore.getState().undo());
  expect(useStore.getState().graph.loss).toBe('loss');
});

test('Delete removes a selected ŷ pin, never the L pin', () => {
  act(() => root.render(<Canvas />));
  act(() => useStore.getState().setViewTab('loss'));
  act(() => useStore.getState().select({ kind: 'pin', role: 'loss', node: 'loss' }));
  deleteSelection();
  expect(useStore.getState().graph.loss).toBe('loss');
  act(() => useStore.getState().setViewTab('model'));
  act(() => useStore.getState().select({ kind: 'pin', role: 'pred', node: 'lin2' }));
  act(() => deleteSelection());
  expect(useStore.getState().graph.predictions).toEqual([]);
  expect(useStore.getState().selection).toBeNull();
});

test('a pin issue is drawn at the pin, with the pin in the error state and not the piece', () => {
  act(() => useStore.getState().loadGraph({ ...initialGraph(), predictions: ['act1'] }));
  act(() => root.render(<Canvas />));
  expect(q('pin-pred-act1')?.getAttribute('data-error')).toBe('true');
  expect(q('issue-pin-pred-act1')?.textContent).toMatch(/width/);
  expect(q('node-act1')?.getAttribute('data-error')).toBe('false');
  expect(q('issue-node-act1')).toBeNull();
});

test('the selected pin gets an inspector; the node inspector offers Pin as ŷ and Pin as L', () => {
  act(() => root.render(<Canvas />));
  act(() => useStore.getState().select({ kind: 'pin', role: 'pred', node: 'lin2' }));
  expect(q('inspector')?.textContent).toMatch(/Prediction/);
  expect(q('inspector-delete')).not.toBeNull();
  act(() => useStore.getState().setViewTab('loss'));
  act(() => useStore.getState().select({ kind: 'pin', role: 'loss', node: 'loss' }));
  expect(q('inspector')?.textContent).toMatch(/Loss/);
  expect(q('inspector-delete')).toBeNull();
  act(() => useStore.getState().setViewTab('model'));
  act(() => useStore.getState().selectNode('act1'));
  act(() => (q('inspector-pin-loss') as HTMLElement).click());
  expect(useStore.getState().graph.loss).toBe('act1');
});

test('the fit plot picker is only for several ŷ pins', async () => {
  const { PredictionPicker } = await import('../src/results/PredictionPicker.js');
  act(() => root.render(<PredictionPicker />));
  expect(q('prediction-picker')).toBeNull();
  act(() => useStore.getState().loadGraph({ ...initialGraph(), predictions: ['lin2', 'act1'] }));
  expect(q('prediction-picker')).not.toBeNull();
  const select = q('prediction-picker') as HTMLSelectElement;
  expect([...select.options].map((o) => o.value)).toEqual(['lin2', 'act1']);
  expect(select.value).toBe('lin2');
  act(() => useStore.getState().setPlotPrediction('act1'));
  expect((q('prediction-picker') as HTMLSelectElement).value).toBe('act1');
  // a pick that is no longer a ŷ falls back to the first
  act(() => useStore.getState().loadGraph({ ...initialGraph(), predictions: ['lin2', 'lin1'] }));
  expect((q('prediction-picker') as HTMLSelectElement).value).toBe('lin2');
  useStore.getState().setPlotPrediction(null);
});

test('palette groups collapse and keep their entries mounted; ready-made losses collect the macros', () => {
  act(() => useStore.getState().setViewTab('loss')); // the ready-made losses and Scale are on the loss tab
  act(() => root.render(<Palette />));
  const toggle = q('palette-group-math');
  expect(toggle?.getAttribute('aria-expanded')).toBe('false');
  expect(q('palette-scale')).not.toBeNull(); // mounted, only hidden
  expect(q('palette-scale')?.closest('[hidden]')).not.toBeNull();
  act(() => (toggle as HTMLElement).click());
  expect(toggle?.getAttribute('aria-expanded')).toBe('true');
  expect(q('palette-scale')?.closest('[hidden]')).toBeNull();
  const ready = q('palette-group-ready')?.closest('section');
  const keys = [...(ready?.querySelectorAll('[data-testid^="palette-loss_"]') ?? [])].length;
  expect(keys).toBe(5);
  expect(ready?.querySelector('[data-testid="palette-weight_penalty"]')).not.toBeNull();
  // the primitives stay out of it
  expect(ready?.querySelector('[data-testid="palette-log_loss"]')).toBeNull();
});
