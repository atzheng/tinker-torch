// @vitest-environment jsdom
import { beforeEach, expect, test } from 'vitest';
import { act } from 'react';
import { createRoot } from 'react-dom/client';
import { analyse, deleteNode, sample, setSetting } from '@tinker/core';
import { PRESET_NAMES, bundledPreset } from '../src/presets/index.js';
import { PromptLine } from '../src/presets/PromptLine.js';
import { initialGraph } from '../src/state/graphSlice.js';
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

test('every bundled preset parses, is titled, has no prompt and analyses as trainable', () => {
  expect(PRESET_NAMES).toEqual(['linear', 'logistic', 'sine', 'xor', 'skip']);
  for (const name of PRESET_NAMES) {
    const f = bundledPreset(name);
    expect(f, name).not.toBeNull();
    if (!f) continue;
    expect('title' in f).toBe(true);
    expect('prompt' in f).toBe(false);
    expect(f.graph.version, name).toBe(2); // the bundled files are version 1 and open migrated
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
