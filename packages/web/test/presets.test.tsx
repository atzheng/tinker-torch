// @vitest-environment jsdom
import { beforeEach, expect, test } from 'vitest';
import { act } from 'react';
import { createRoot } from 'react-dom/client';
import { analyse, deleteNode, layoutLayers, loadRealSet, sample, setSetting } from '@tinker/core';
import { LAYER_FIT_MIN_SCALE } from '../src/canvas/viewport.js';
import { initialGraph as defaultGraph } from '../src/state/graphSlice.js';
import { PRESET_NAMES, bundledPreset } from '../src/presets/index.js';
import { PromptLine } from '../src/presets/PromptLine.js';
import { InstructionsPanel } from '../src/presets/InstructionsPanel.js';
import { SuccessTick } from '../src/presets/SuccessTick.js';
import { mlpGraph as initialGraph } from './mlpGraph.js';
import { lockedIds } from '../src/state/locks.js';
import { useStore } from '../src/state/store.js';
import { loadText, saveText } from '../src/topbar/FileMenu.js';
import { loadBundled } from '../src/topbar/PresetMenu.js';

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const s = () => useStore.getState();

beforeEach(() => {
  s().loadGraph(initialGraph());
});

function presetText(extra: Record<string, unknown>): string {
  const g = JSON.parse(saveText(false)) as Record<string, unknown>;
  return JSON.stringify({ ...g, ...extra });
}

test('every bundled preset parses, is titled, has no prompt and analyses as trainable', async () => {
  await loadRealSet('iris');
  expect(PRESET_NAMES).toEqual([
    'linear',
    'logistic',
    'sine',
    'xor',
    'skip',
    'softmax',
    'scalar',
    'mlp',
    'iris',
    'intercept-prob',
    'intercept-mean',
  ]);
  for (const name of PRESET_NAMES) {
    const f = bundledPreset(name);
    expect(f, name).not.toBeNull();
    if (!f) continue;
    expect('title' in f).toBe(true);
    expect('prompt' in f).toBe(false);
    expect(f.graph.version, name).toBe(2); // bundled files are native version 2
    expect(f.graph.predictions, name).toHaveLength(1);
    expect(f.graph.loss, name).toBe('loss');
    const a = analyse(f.graph, sample(f.graph.data).info);
    expect(a.issues, name).toEqual([]);
    expect(a.trainable, name).toBe(true);
  }
  expect(bundledPreset('nope')).toBeNull();
  expect(bundledPreset('constructor')).toBeNull();
});

test('loading a bundled preset replaces the graph in one undo step', () => {
  const before = s().graph;
  expect(loadBundled('xor')).toBe(true);
  expect(s().graph.data.set).toBe('xor');
  expect(s().preset?.title).toMatch(/XOR/);
  expect(s().canUndo).toBe(true);
  s().undo();
  expect(s().graph).toBe(before);
  expect(loadBundled('nope')).toBe(false);
});

test('locked pieces refuse edits with the sentence; unknown ids are ignored', () => {
  const r = loadText(presetText({ title: 'T', locked: ['act1', 'ghost'] }));
  expect(r.ok).toBe(true);
  expect(lockedIds()).toEqual(['act1']);
  expect(s().apply(setSetting(s().graph, 'act1', 'fn', 'tanh', { locked: lockedIds() }))).toEqual({
    ok: false,
    error: 'This piece is locked.',
  });
  expect(s().error).toBe('This piece is locked.');
  expect(deleteNode(s().graph, 'act1', { locked: lockedIds() }).ok).toBe(false);
  expect(deleteNode(s().graph, 'lin2', { locked: lockedIds() }).ok).toBe(true);
});

test('a locked input refuses a column change through the data slice', () => {
  loadText(presetText({ title: 'T', locked: ['x'] }));
  const g = s().graph;
  s().setInputColumns('x', ['x1']);
  expect(s().graph).toBe(g);
  expect(s().error).toBe('This piece is locked.');
});

test('a plain file or loadGraph clears the preset', () => {
  loadText(presetText({ title: 'T', prompt: 'P', locked: ['act1'] }));
  expect(s().preset).toEqual({ title: 'T', prompt: 'P', locked: ['act1'] });
  loadText(plainText());
  expect(s().preset).toBeNull();
  loadText(presetText({ title: 'T' }));
  expect(s().preset).not.toBeNull();
  s().loadGraph(initialGraph());
  expect(s().preset).toBeNull();
  expect(lockedIds()).toEqual([]);
  expect(s().pastPresets).toEqual([]);
});

function plainText(): string {
  const { title: _t, prompt: _p, locked: _l, ...rest } = JSON.parse(saveText(false));
  return JSON.stringify(rest);
}

test('undo past a preset load restores the earlier graph with no preset; redo brings it back', () => {
  loadText(presetText({ title: 'T', prompt: 'P', locked: ['act1'] }));
  s().undo();
  expect(s().preset).toBeNull();
  expect(lockedIds()).toEqual([]);
  s().redo();
  expect(s().preset).toEqual({ title: 'T', prompt: 'P', locked: ['act1'] });
});

test('load plain file then undo returns the locked graph with its locks', () => {
  loadText(presetText({ title: 'T', locked: ['act1'] }));
  const locked = s().graph;
  loadText(plainText());
  expect(s().preset).toBeNull();
  s().undo();
  expect(s().graph).toBe(locked);
  expect(lockedIds()).toEqual(['act1']);
  expect(deleteNode(s().graph, 'act1', { locked: lockedIds() }).ok).toBe(false);
});

test('load B then undo gives A with A locks', () => {
  loadText(presetText({ title: 'A', prompt: 'pa', locked: ['act1'] }));
  loadText(presetText({ title: 'B', locked: ['lin1'] }));
  expect(lockedIds()).toEqual(['lin1']);
  s().undo();
  expect(s().preset).toEqual({ title: 'A', prompt: 'pa', locked: ['act1'] });
  s().redo();
  expect(s().preset?.title).toBe('B');
});

test('an edit keeps the preset in history', () => {
  loadText(presetText({ title: 'T', locked: ['act1'] }));
  s().apply(setSetting(s().graph, 'act1', 'fn', 'relu'));
  s().undo();
  expect(lockedIds()).toEqual(['act1']);
  expect(s().pastPresets.length).toBe(s().past.length);
});

test('a dataset switch blocked by a locked Input says so', () => {
  loadText(presetText({ title: 'T', locked: ['x'] }));
  s().setData({ set: 'xor' });
  expect(s().error).toBe(
    'This dataset needs different input columns, but the Input piece is locked.',
  );
  expect(s().graph.data.set).toBe('sine');
});

test('Save of a preset keeps title, prompt and locked', () => {
  loadText(presetText({ title: 'T', prompt: 'P', locked: ['act1'] }));
  const doc = JSON.parse(saveText(false)) as Record<string, unknown>;
  expect(doc).toMatchObject({ title: 'T', prompt: 'P', locked: ['act1'] });
  s().setPreset(null);
  expect(JSON.parse(saveText(false))).not.toHaveProperty('title');
});

test('the prompt shows verbatim and can be dismissed', () => {
  const host = document.createElement('div');
  document.body.appendChild(host);
  const root = createRoot(host);
  act(() => root.render(<PromptLine />));
  expect(host.querySelector('[data-testid="preset-prompt"]')).toBeNull();
  act(() => {
    loadText(presetText({ title: 'T', prompt: 'Make it fit.' }));
  });
  expect(host.querySelector('[data-testid="preset-prompt"]')?.textContent).toContain(
    'Make it fit.',
  );
  act(() => {
    host.querySelector<HTMLButtonElement>('[data-testid="prompt-dismiss"]')?.click();
  });
  expect(host.querySelector('[data-testid="preset-prompt"]')).toBeNull();
  act(() => root.unmount());
});

test('instructions: shown as steps, collapsible, replace the prompt line, saved back', () => {
  const host = document.createElement('div');
  document.body.appendChild(host);
  const root = createRoot(host);
  const q = (id: string) => host.querySelector<HTMLElement>(`[data-testid="${id}"]`);
  act(() =>
    root.render(
      <>
        <PromptLine />
        <InstructionsPanel />
      </>,
    ),
  );
  expect(q('instructions')).toBeNull();
  const steps = ['Drag $p$ up.', 'Press Step.'];
  act(() => {
    loadText(presetText({ title: 'T', prompt: 'Find the best p.', instructions: steps }));
  });
  expect(host.querySelectorAll('[data-step]').length).toBe(2);
  expect(q('instructions-toggle')?.textContent).toContain('Find the best p.');
  expect(q('instructions-toggle')?.textContent).toContain('2 steps');
  expect(q('preset-prompt')).toBeNull();
  act(() => q('instructions-toggle')?.click());
  expect(q('instructions-steps')).toBeNull();
  expect(JSON.parse(saveText(false))).toMatchObject({ instructions: steps });
  act(() => {
    loadText(presetText({ title: 'U', instructions: steps }));
  });
  expect(q('instructions-steps')).not.toBeNull();
  act(() => {
    loadText(presetText({ title: 'V', prompt: 'Only prompt.', instructions: undefined }));
  });
  expect(q('instructions')).toBeNull();
  expect(q('preset-prompt')).not.toBeNull();
  act(() => root.unmount());
});

test('success checks: ticks follow the live loss and parameters, are saved back and can be hidden', () => {
  const host = document.createElement('div');
  document.body.appendChild(host);
  const root = createRoot(host);
  const q = (id: string) => host.querySelector<HTMLElement>(`[data-testid="${id}"]`);
  act(() =>
    root.render(
      <>
        <InstructionsPanel />
        <SuccessTick />
      </>,
    ),
  );
  const checks = [
    { kind: 'lossBelow', value: 0.3, step: 0 },
    { kind: 'paramNear', node: 'lin2', tensor: 'b', value: 0.25, tol: 0.01 },
  ];
  act(() => {
    loadText(presetText({ title: 'T', instructions: ['Lower the loss.'], checks }));
  });
  const live = (train: number | null, b: number) =>
    act(() =>
      useStore.setState({
        liveLoss: { train, val: null },
        host: { lin2: { W: Float32Array.of(0), b: Float32Array.of(b) } },
      }),
    );
  live(0.5, 0);
  expect(q('success-check-step')).toBeNull();
  expect(q('success-check')).toBeNull();
  live(0.2, 0);
  expect(q('success-check-step')).not.toBeNull();
  expect(q('success-check')).toBeNull();
  live(0.2, 0.251);
  expect(q('success-check')).not.toBeNull();
  live(0.9, 0.251);
  expect(q('success-check-step')).toBeNull();
  expect(JSON.parse(saveText(false))).toMatchObject({ checks });
  // No instructions: the preset-level tick stands alone.
  act(() => {
    loadText(presetText({ title: 'U', checks: [checks[1]] }));
  });
  live(0.9, 0.25);
  expect(q('success-check')).not.toBeNull();
  // Instructions hidden, success-check shown: the shared tick stands alone.
  act(() => {
    loadText(
      presetText({
        title: 'W',
        instructions: ['a'],
        checks: [checks[1]],
        show: { ui: ['success-check'] },
      }),
    );
  });
  live(0.9, 0.25);
  expect(q('instructions')).toBeNull();
  expect(q('success-check')).not.toBeNull();
  act(() => {
    loadText(presetText({ title: 'V', checks: [checks[1]], show: { ui: ['btn-step'] } }));
  });
  live(0.9, 0.25);
  expect(q('success-check')).toBeNull();
  act(() => root.unmount());
});

test('the first-open graph is the scalar preset and its whole drawing fits at names >= 8 px', () => {
  const f = bundledPreset('scalar');
  if (!f) throw new Error('no scalar preset');
  expect(defaultGraph()).toEqual(f.graph);
  const g = f.graph;
  const l = layoutLayers(g, analyse(g, sample(g.data).info));
  // canvas widths (CONTRACTS section 10 viewports): 1024x768 touch is the narrowest, 561 px, 24 px margins
  for (const w of [561, 647, 702, 1054]) {
    expect((w - 48) / l.size[0], `canvas ${w}`).toBeGreaterThanOrEqual(LAYER_FIT_MIN_SCALE);
  }
  expect((400 - 48) / l.size[1]).toBeGreaterThanOrEqual(LAYER_FIT_MIN_SCALE);
});
