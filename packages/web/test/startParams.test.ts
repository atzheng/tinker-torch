// @vitest-environment jsdom
import { beforeAll, beforeEach, expect, test } from 'vitest';
import { parseFile, setLinearOut } from '@tinker/core';
import scalar from '../src/presets/scalar.json';
import { loadEngine } from '../src/engine.js';
import { useStore } from '../src/state/store.js';
import { loadParsed, loadText } from '../src/topbar/FileMenu.js';
import { mlpGraph } from './mlpGraph.js';

const s = () => useStore.getState();

beforeAll(async () => {
  await loadEngine();
}, 60000);

beforeEach(() => {
  s().pause();
  s().loadGraph(mlpGraph());
  s().reset();
});

function preset(params: unknown) {
  const r = parseFile({ ...scalar, params });
  if (!r.ok) throw new Error(r.error);
  return r.file;
}
const w = (id: string) => Array.from(s().host?.[id]?.W ?? []);
const b = () => Array.from(s().host?.bias?.b ?? []);

test('a preset with a Weight and a Bias loads exactly those values and Reset restores them', () => {
  expect(loadParsed(preset({ w1: 0, w2: 2, bias: 0.5 }))).toEqual({ ok: true, message: null });
  expect([w('w1'), w('w2'), b()]).toEqual([[0], [2], [0.5]]);
  for (let i = 0; i < 20; i++) s().stepOnce();
  expect(w('w1')).not.toEqual([0]);
  s().reset();
  expect([w('w1'), w('w2'), b()]).toEqual([[0], [2], [0.5]]);
  expect(s().step).toBe(0);
  s().reset();
  expect([w('w1'), w('w2'), b()]).toEqual([[0], [2], [0.5]]);
});

test('partial params: listed nodes are fixed, the rest come from the seed', () => {
  loadParsed(preset({ bias: 0 }));
  const w1 = w('w1');
  s().reset();
  expect(b()).toEqual([0]);
  expect(w('w1')).not.toEqual(w1); // a new seed draws new values for the unlisted Weight
});

test('a preset without params reseeds on Reset', () => {
  loadParsed(preset(undefined));
  const w1 = w('w1');
  s().reset();
  expect(w('w1')).not.toEqual(w1);
});

test('loading a file without params clears the start values; undo of the load restores them', () => {
  loadParsed(preset({ w1: 1, w2: 1, bias: 1 }));
  loadParsed(preset(undefined));
  expect(s().startParams).toBeNull();
  s().reset();
  expect(w('w1')).not.toEqual([1]);
  s().undo();
  expect(s().startParams).not.toBeNull();
  s().reset();
  expect([w('w1'), b()]).toEqual([[1], [1]]);
  s().redo();
  expect(s().startParams).toBeNull();
});

test('a new graph clears them', () => {
  loadParsed(preset({ w1: 1, w2: 1, bias: 1 }));
  s().loadGraph(mlpGraph());
  expect(s().startParams).toBeNull();
});

test('values whose shape no longer fits are ignored by Reset, the others stay', () => {
  const g = mlpGraph();
  const r = loadText(
    JSON.stringify({
      kind: 'tinker',
      version: 2,
      graph: g,
      params: {
        lin1: { W: Array(3).fill(0.25), b: Array(3).fill(0) },
        lin2: { W: [1, 1, 1], b: [0] },
      },
    }),
  );
  expect(r.ok).toBe(true);
  s().apply(setLinearOut(s().graph, 'lin1', 4));
  s().reset();
  expect(s().host?.lin1?.W?.length).toBe(4);
  expect(Array.from(s().host?.lin1?.W ?? []).every((v) => v === 0.25)).toBe(false);
});
