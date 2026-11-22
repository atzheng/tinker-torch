// @vitest-environment jsdom
import { beforeAll, expect, test } from 'vitest';
import { loadEngine } from '../src/engine.js';
import { useStore } from '../src/state/store.js';
import { loadBundled } from '../src/topbar/PresetMenu.js';

const s = () => useStore.getState();

beforeAll(async () => {
  await loadEngine();
}, 60000);

function trainTo(): number[] {
  s().pause();
  for (let i = 0; i < 600; i++) s().stepOnce();
  expect(s().diverged).toBe(false);
  return Array.from(s().host?.bias?.b ?? []);
}

test('intercept-prob trains to the sample proportion of the training target', () => {
  expect(loadBundled('intercept-prob')).toBe(true);
  expect(s().canTrain).toBe(true);
  const y = s().dataset.yTrain;
  const p = y.reduce((a, b) => a + b, 0) / y.length;
  // the data are balanced, so start away from the answer: the bias is written to 2 (p = 0.88)
  expect(s().writeParams('bias', { b: Float32Array.of(2) })).toBe(true);
  const [z] = trainTo();
  expect(1 / (1 + Math.exp(-(z as number)))).toBeCloseTo(p, 4);
});

test('intercept-mean trains to the mean of the training target', () => {
  expect(loadBundled('intercept-mean')).toBe(true);
  expect(s().writeParams('bias', { b: Float32Array.of(2) })).toBe(true);
  expect(s().canTrain).toBe(true);
  const y = s().dataset.yTrain;
  const m = y.reduce((a, b) => a + b, 0) / y.length;
  const [b] = trainTo();
  expect(b as number).toBeCloseTo(m, 4);
});
