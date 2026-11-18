// @vitest-environment jsdom
import { beforeEach, expect, test, vi } from 'vitest';
import { act } from 'react';
import { createRoot } from 'react-dom/client';
import { paramsToJson, parseFileText, setLinearOut, serializeGraph } from '@tinker/core';
import { initialGraph } from '../src/state/graphSlice.js';
import { useStore } from '../src/state/store.js';
import { FileMenu, loadText, saveText } from '../src/topbar/FileMenu.js';

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const s = () => useStore.getState();

beforeEach(() => {
  s().loadGraph(initialGraph());
});

test('saveText with weights round-trips through loadText', () => {
  expect(s().host).not.toBeNull();
  const text = saveText(true);
  const before = serializeGraph(s().graph);
  const w = Array.from(s().host?.lin1?.W ?? []);
  expect(parseFileText(text).ok).toBe(true);
  s().apply(setLinearOut(s().graph, 'lin1', 6));
  expect(s().graph.nodes.find((n) => n.id === 'lin1')).toMatchObject({ out: 6 });
  const r = loadText(text);
  expect(r).toEqual({ ok: true, message: null });
  expect(serializeGraph(s().graph)).toBe(before);
  expect(Array.from(s().host?.lin1?.W ?? [])).toEqual(w);
  expect(s().canUndo).toBe(false); // a load replaces the graph and clears history
  expect(s().step).toBe(0);
});

test('saveText without weights has no params', () => {
  expect(JSON.parse(saveText(false))).not.toHaveProperty('params');
});

test('a broken file gives a sentence and leaves the graph alone', () => {
  const g = s().graph;
  for (const bad of ['{nope', '[]', '{"kind":"tinker","version":9}']) {
    const r = loadText(bad);
    expect(r.ok).toBe(false);
    expect(r.message).toMatch(/\.$/);
  }
  expect(s().graph).toBe(g);
});

test('weights for a resized layer load with a notice', () => {
  const g = s().graph;
  const old = JSON.parse(saveText(true)) as Record<string, unknown>;
  s().apply(setLinearOut(g, 'lin1', 6));
  const resized = JSON.parse(saveText(false)) as Record<string, unknown>;
  const mixed = { ...resized, params: old['params'] };
  const r = loadText(JSON.stringify(mixed));
  expect(r.ok).toBe(true);
  expect(r.message).toContain('"lin1"');
  expect(Object.keys(paramsToJson(s().host ?? {})).sort()).toEqual(['lin1', 'lin2']);
});

test('Save downloads through a Blob URL that is revoked; Load reads a file', async () => {
  const create = vi.fn(() => 'blob:x');
  const revoke = vi.fn();
  Object.assign(URL, { createObjectURL: create, revokeObjectURL: revoke });
  const clicks: string[] = [];
  vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(function (
    this: HTMLAnchorElement,
  ) {
    clicks.push(this.download);
  });
  const el = document.createElement('div');
  document.body.appendChild(el);
  const root = createRoot(el);
  await act(async () => root.render(<FileMenu />));
  const q = (id: string) => el.querySelector(`[data-testid="${id}"]`) as HTMLElement;
  await act(async () => q('file-save').click());
  expect(clicks).toEqual(['tinker-model.tinker.json']);
  expect(create).toHaveBeenCalledTimes(1);
  await new Promise((r) => setTimeout(r, 20));
  expect(revoke).toHaveBeenCalledWith('blob:x');

  const input = q('file-input') as HTMLInputElement;
  const file = { text: async () => '{bad' }; // jsdom's File has no text()
  Object.defineProperty(input, 'files', { value: [file], configurable: true });
  await act(async () => {
    input.dispatchEvent(new Event('change', { bubbles: true }));
    await new Promise((r) => setTimeout(r, 20));
  });
  expect(q('file-note').textContent).toContain('not valid JSON');
  await act(async () => root.unmount());
});
