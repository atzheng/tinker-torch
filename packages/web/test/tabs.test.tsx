// @vitest-environment jsdom
import { afterEach, beforeEach, expect, test } from 'vitest';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { App } from '../src/App.js';
import { deleteSelection } from '../src/canvas/actions.js';
import { mlpGraph } from './mlpGraph.js';
import { RESULTS_PANEL_IDS } from '@tinker/core';
import { PANEL_IDS } from '../src/results/panelLayout.js';
import { useStore } from '../src/state/store.js';
import { loadText, saveText } from '../src/topbar/FileMenu.js';

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

const s = () => useStore.getState();
let host: HTMLDivElement;
let root: Root;

function mount() {
  host = document.createElement('div');
  document.body.appendChild(host);
  root = createRoot(host);
  act(() => root.render(<App />));
}
beforeEach(() => {
  s().loadGraph(mlpGraph());
  s().setViewTab('model');
});
afterEach(() => {
  act(() => root.unmount());
  host.remove();
});

const q = (id: string) => host.querySelector(`[data-testid="${id}"]`);
const paletteKeys = () =>
  [...host.querySelectorAll('.tk-palette-entry')].map((e) =>
    (e.getAttribute('data-testid') ?? '').replace('palette-', ''),
  );
const press = (el: Element, key: string) =>
  act(() => void el.dispatchEvent(new KeyboardEvent('keydown', { key, bubbles: true })));
const presetText = (show: unknown) =>
  JSON.stringify({ ...JSON.parse(saveText(false)), title: 'T', show });

test('three tabs, opening on the model tab; each shows its own pieces, panels and palette', () => {
  mount();
  expect(q('tab-bar')?.getAttribute('role')).toBe('tablist');
  expect(q('tab-model')?.getAttribute('aria-selected')).toBe('true');
  expect(q('node-lin1')).not.toBeNull();
  expect(q('node-y')).toBeNull();
  expect(q('fit-plot')).not.toBeNull();
  expect(q('loss-curves')).toBeNull();
  expect(paletteKeys()).toContain('linear');
  expect(paletteKeys()).not.toContain('square');
  expect(q('grain-toggle')).not.toBeNull();

  act(() => (q('tab-loss') as HTMLElement).click());
  expect(q('tab-loss')?.getAttribute('aria-selected')).toBe('true');
  expect(q('node-lin1')).toBeNull();
  for (const id of ['y', 'diff', 'err', 'loss']) expect(q(`node-${id}`), id).not.toBeNull();
  expect(q('fit-plot')).toBeNull();
  expect(q('loss-curves')).not.toBeNull();
  expect(paletteKeys()).toContain('square');
  expect(paletteKeys()).not.toContain('linear');
  expect(q('grain-toggle')).toBeNull(); // the loss tab is drawn by layer

  // the full tab: the whole graph, every panel and palette entry, no stubs
  act(() => (q('tab-full') as HTMLElement).click());
  expect(q('tab-full')?.textContent).toBe('Full graph');
  for (const id of ['x', 'lin1', 'act1', 'lin2', 'y', 'diff', 'err', 'loss']) {
    expect(q(`node-${id}`), id).not.toBeNull();
  }
  expect(host.querySelector('[data-stub="true"]')).toBeNull();
  expect(q('fit-plot')).not.toBeNull();
  expect(q('loss-curves')).not.toBeNull();
  expect(paletteKeys()).toEqual(expect.arrayContaining(['linear', 'square']));
  expect(q('grain-toggle')).not.toBeNull();
});

test('the tabs are a tablist: arrow keys switch and move focus', () => {
  mount();
  const model = q('tab-model') as HTMLElement;
  expect(model.tabIndex).toBe(0);
  expect((q('tab-loss') as HTMLElement).tabIndex).toBe(-1);
  press(model, 'ArrowRight');
  expect(s().viewTab).toBe('loss');
  expect(document.activeElement).toBe(q('tab-loss'));
  press(q('tab-loss') as HTMLElement, 'ArrowRight');
  expect(s().viewTab).toBe('full');
  press(q('tab-full') as HTMLElement, 'ArrowRight');
  expect(s().viewTab).toBe('code');
  press(q('tab-code') as HTMLElement, 'ArrowRight'); // wraps
  expect(s().viewTab).toBe('model');
  press(q('tab-model') as HTMLElement, 'End');
  expect(s().viewTab).toBe('code');
});

test('the Code tab keeps the selection both ways; other switches clear it', () => {
  mount();
  act(() => s().selectNode('lin2'));
  act(() => s().setViewTab('code'));
  expect(s().selection).toEqual({ kind: 'node', id: 'lin2' });
  expect(q('code-view')).not.toBeNull();
  expect(host.querySelector('[data-region="palette"]')).toBeNull();
  expect(host.querySelector('[data-region="results"]')).toBeNull();
  expect(q('code-view')?.textContent).toContain('class Model(nn.Module)');
  expect(q('code-view')?.querySelector('[data-hl="true"]')).not.toBeNull(); // lin2's spans
  act(() => s().setViewTab('full'));
  expect(s().selection).toEqual({ kind: 'node', id: 'lin2' });
  act(() => s().setViewTab('loss'));
  expect(s().selection).toBeNull();
});

test('a wire across the boundary shows on the loss and full tabs only and is removed from either', () => {
  mount();
  // model tab: no stub of the Subtract and no wire into it
  expect(q('ghost-diff')).toBeNull();
  expect(q('node-diff')).toBeNull();
  expect(q('wire-3')).toBeNull();
  // loss tab: the same wire starts at a stub of the prediction
  act(() => s().setViewTab('loss'));
  expect(q('ghost-lin2')?.getAttribute('data-tab')).toBe('model');
  expect(q('wire-3')).not.toBeNull();
  const before = s().graph;
  act(() => s().select({ kind: 'wire', from: 'lin2', to: 'diff.a' }));
  act(() => deleteSelection());
  expect(s().graph.edges).toHaveLength(before.edges.length - 1);
  expect(q('ghost-lin2')).not.toBeNull(); // a ŷ node is always a stub on the loss tab
  // one undo history: undo brings the wire back on both tabs
  act(() => s().undo());
  expect(s().graph).toEqual(before);
  expect(q('ghost-lin2')).not.toBeNull();
  act(() => s().setViewTab('full'));
  expect(q('wire-3')).not.toBeNull();
  act(() => s().select({ kind: 'wire', from: 'lin2', to: 'diff.a' }));
  act(() => deleteSelection());
  expect(s().graph.edges).toHaveLength(before.edges.length - 1);
  act(() => s().undo()); // the one undo history
  expect(s().graph).toEqual(before);
});

test('tapping a stub goes to the other tab with that piece selected', () => {
  mount();
  act(() => s().setViewTab('loss'));
  const ghost = q('node-lin2') as Element;
  expect(ghost.getAttribute('data-stub')).toBe('true');
  act(
    () => void ghost.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true })),
  );
  expect(s().viewTab).toBe('model');
  expect(s().selection).toEqual({ kind: 'node', id: 'lin2' });
  expect(q('node-lin2')?.getAttribute('data-stub')).toBe('false');
});

test('a preset can hide a tab and choose the opening one; neither field shows both, on model', () => {
  mount();
  act(() => void loadText(presetText({ ui: ['tab-loss', 'panel-loss', 'btn-step'] })));
  expect(q('tab-bar')).toBeNull(); // one tab: no bar
  expect(s().viewTab).toBe('loss'); // the model tab is hidden: the loss tab opens
  expect(q('node-y')).not.toBeNull();
  expect(q('node-lin1')).not.toBeNull(); // one tab shown: it draws the whole graph
  expect(q('loss-curves')).not.toBeNull();

  act(() => void loadText(presetText({ tab: 'loss' })));
  expect(q('tab-bar')).not.toBeNull();
  expect(q('tab-loss')?.getAttribute('aria-selected')).toBe('true');

  act(() => void loadText(presetText({ ui: ['tab-model', 'tab-loss'] })));
  expect(q('tab-model')?.getAttribute('aria-selected')).toBe('true');
  expect(q('tab-full')).toBeNull(); // listed tabs only: the full graph tab is hidden
  expect(q('tab-loss')).not.toBeNull();

  // a preset can open on the full tab, or hide everything else
  act(() => void loadText(presetText({ tab: 'full' })));
  expect(q('tab-full')?.getAttribute('aria-selected')).toBe('true');
  expect(q('node-lin1')).not.toBeNull();
  expect(q('node-y')).not.toBeNull();
  act(() => void loadText(presetText({ ui: ['tab-full', 'btn-step'] })));
  expect(q('tab-bar')).toBeNull();
  expect(s().viewTab).toBe('full');

  // a preset that does not mention tabs at all (as every earlier preset)
  act(() => void loadText(presetText({ ui: ['btn-step', 'btn-run', 'panel-fit', 'panel-loss'] })));
  expect(q('tab-model')).not.toBeNull();
  expect(q('tab-loss')).not.toBeNull();
  expect(q('tab-full')).not.toBeNull();
  expect(q('tab-model')?.getAttribute('aria-selected')).toBe('true');
});

test('a bad tab in a preset is refused with one plain sentence and leaves the page alone', () => {
  mount();
  const r = loadText(presetText({ ui: ['tab-model'], tab: 'loss' }));
  expect(r.ok).toBe(false);
  expect(r.message).toBe('The preset opens on the loss tab, but its "show.ui" hides that tab.');
  expect(loadText(presetText({ tab: 'both' })).message).toBe(
    'The preset "show.tab" must be "model", "loss", "full" or "code".',
  );
});

test('a preset that carries show.tab is saved back with it', () => {
  mount();
  loadText(presetText({ tab: 'loss' }));
  expect(JSON.parse(saveText(false)).show).toEqual({ tab: 'loss' });
});

test('a loss-only preset draws the whole graph: every piece, panel and palette entry, no stubs', () => {
  mount();
  act(() => void loadText(presetText({ ui: ['tab-loss', 'panel-fit', 'panel-loss', 'btn-step'] })));
  expect(q('tab-bar')).toBeNull();
  for (const id of ['x', 'lin1', 'act1', 'lin2', 'y', 'diff', 'loss']) {
    expect(q(`node-${id}`), id).not.toBeNull();
  }
  expect(host.querySelector('[data-stub="true"]')).toBeNull();
  expect(q('fit-plot')).not.toBeNull();
  expect(q('loss-curves')).not.toBeNull();
  expect(paletteKeys()).toEqual(expect.arrayContaining(['linear', 'square']));
  expect(q('grain-toggle')).toBeNull(); // not listed by this preset
});

test('an unwired Add dropped on either tab stays in view on both', () => {
  mount();
  const g = s().graph;
  act(
    () =>
      void s().apply({
        ok: true,
        graph: { ...g, nodes: [...g.nodes, { id: 'free-add', type: 'add' }] },
      }),
  );
  expect(q('node-free-add')).not.toBeNull();
  act(() => s().setViewTab('loss'));
  expect(q('node-free-add')?.getAttribute('data-stub')).toBe('false');
});

test('show.panelTabs puts a panel on several tabs; unlisted panels keep their default tab', () => {
  mount();
  act(() => void loadText(presetText({ panelTabs: { loss: ['model', 'loss'] } })));
  expect(s().show?.panelTabs).toEqual({ loss: ['model', 'loss'] });
  expect(q('fit-plot')).not.toBeNull();
  expect(q('loss-curves')).not.toBeNull(); // the loss panel on the model tab as well
  act(() => (q('tab-loss') as HTMLElement).click());
  expect(q('loss-curves')).not.toBeNull();
  expect(q('fit-plot')).toBeNull();
  act(() => (q('tab-full') as HTMLElement).click());
  expect(q('loss-curves')).not.toBeNull();
  expect(q('fit-plot')).not.toBeNull();

  // moving a panel off its default tab
  act(() => void loadText(presetText({ panelTabs: { fit: ['loss'] } })));
  act(() => (q('tab-model') as HTMLElement).click());
  expect(q('fit-plot')).toBeNull();
  act(() => (q('tab-loss') as HTMLElement).click());
  expect(q('fit-plot')).not.toBeNull();

  // absent: as before
  act(() => void loadText(presetText({})));
  act(() => (q('tab-model') as HTMLElement).click());
  expect(q('fit-plot')).not.toBeNull();
  expect(q('loss-curves')).toBeNull();
});

test('show.panelTabs is saved back with the preset', () => {
  mount();
  loadText(presetText({ panelTabs: { loss: ['model', 'loss'] } }));
  expect(JSON.parse(saveText(false)).show).toEqual({ panelTabs: { loss: ['model', 'loss'] } });
});

test('core RESULTS_PANEL_IDS equals the web PANEL_IDS', () => {
  expect([...RESULTS_PANEL_IDS]).toEqual([...PANEL_IDS]);
});
