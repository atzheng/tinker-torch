// @vitest-environment jsdom
import { beforeAll, beforeEach, expect, test } from 'vitest';
import { beginWeightEdit, PAUSE_FIRST } from '../src/canvas/weightEdit.js';
import { getEngine, loadEngine } from '../src/engine.js';
import { initialGraph } from '../src/state/graphSlice.js';
import { useStore } from '../src/state/store.js';

const s = () => useStore.getState();

beforeAll(async () => {
  await loadEngine();
}, 60000);

beforeEach(() => {
  s().pause();
  s().loadGraph(initialGraph());
});

test('a hand-set weight is written into the model and is one undo step', () => {
  expect(s().canTrain).toBe(true);
  const before = s().host?.lin1?.W[2];
  const pastBefore = s().past.length;
  const edit = beginWeightEdit({ kind: 'W', node: 'lin1', i: 0, j: 2 });
  expect(edit).not.toBeNull();
  expect(edit?.start).toBe(before);
  edit?.set(0.25);
  edit?.set(0.5); // a drag: many previews, no history yet
  expect(s().past.length).toBe(pastBefore);
  expect(s().host?.lin1?.W[2]).toBeCloseTo(0.5, 6);
  edit?.commit();
  expect(s().past.length).toBe(pastBefore + 1);
  expect(s().host?.lin1?.W[2]).toBeCloseTo(0.5, 6);
  s().undo();
  expect(s().host?.lin1?.W[2]).toBe(before);
  s().redo();
  expect(s().host?.lin1?.W[2]).toBeCloseTo(0.5, 6);
});

test('a published host snapshot is never written in place', () => {
  const snap = s().host;
  const copy = Float32Array.from(snap?.lin1?.W ?? []);
  const edit = beginWeightEdit({ kind: 'W', node: 'lin1', i: 0, j: 0 });
  edit?.set(3);
  expect(Array.from(snap?.lin1?.W ?? [])).toEqual(Array.from(copy));
  expect(s().host).not.toBe(snap);
  edit?.cancel();
});

test('a bias can be set and cancel restores it', () => {
  const b = s().host?.lin1?.b?.[1];
  const edit = beginWeightEdit({ kind: 'b', node: 'lin1', j: 1 });
  edit?.set(-1.5);
  expect(s().host?.lin1?.b?.[1]).toBeCloseTo(-1.5, 6);
  edit?.cancel();
  expect(s().host?.lin1?.b?.[1]).toBe(b);
});

test('hand edits are refused while running and for locked pieces', () => {
  useStore.setState({ running: true });
  expect(beginWeightEdit({ kind: 'W', node: 'lin1', i: 0, j: 0 })).toBeNull();
  expect(s().error).toBe(PAUSE_FIRST);
  useStore.setState({ running: false, error: null });
  useStore.setState({ preset: { title: 't', prompt: 'p', locked: ['lin1'] } });
  expect(beginWeightEdit({ kind: 'W', node: 'lin1', i: 0, j: 0 })).toBeNull();
  expect(s().error).toBe('This piece is locked.');
  useStore.setState({ preset: null });
});

test('many writes keep the live array count flat', () => {
  const live = () => getEngine()?.liveArrays() ?? -1;
  const edit = beginWeightEdit({ kind: 'W', node: 'lin1', i: 0, j: 0 });
  edit?.set(0.1);
  const n = live();
  for (let k = 0; k < 20; k++) edit?.set(k / 10);
  expect(live()).toBe(n);
  edit?.cancel();
});
