import { describe, expect, test } from 'vitest';
import {
  evaluateCheck,
  evaluateChecks,
  parseFile,
  parseGraph,
  serializeFile,
  type Check,
  type HostParams,
} from '../src/index.js';

const graph = () => {
  const r = parseGraph({
    version: 2,
    nodes: [
      { id: 'bias', type: 'bias' },
      { id: 'yhat', type: 'link', fn: 'sigmoid' },
      { id: 'y', type: 'target' },
      { id: 'err', type: 'log_loss' },
      { id: 'loss', type: 'mean' },
    ],
    edges: [
      ['bias', 'yhat'],
      ['yhat', 'err.pred'],
      ['y', 'err.target'],
      ['err', 'loss'],
    ],
    predictions: ['yhat'],
    loss: 'loss',
    data: { set: 'circle', n: 20, noise: 0.1, val: 0.3, seed: 1, standardize: false },
    train: { optimizer: 'sgd', lr: 0.5, batch: 'full', seed: 1 },
  });
  if (!r.ok) throw new Error(r.error);
  return r.graph;
};
const doc = (extra: Record<string, unknown>) => ({
  ...JSON.parse(serializeFile(graph())),
  title: 'T',
  ...extra,
});
const host = (b: number): HostParams =>
  ({ bias: { b: Float32Array.of(b) } }) as unknown as HostParams;

describe('evaluateCheck', () => {
  const lossBelow: Check = { kind: 'lossBelow', value: 0.3 };
  test('lossBelow follows the live loss, strictly, with no latch', () => {
    const at = (train: number | null, val: number | null = null) => ({
      liveLoss: { train, val },
      host: null,
    });
    expect(evaluateCheck(lossBelow, at(0.29))).toBe(true);
    expect(evaluateCheck(lossBelow, at(0.3))).toBe(false);
    expect(evaluateCheck(lossBelow, at(0.5))).toBe(false); // lost again: tick removed
    expect(evaluateCheck(lossBelow, at(null))).toBe(false);
    expect(evaluateCheck(lossBelow, at(Number.NaN))).toBe(false);
    expect(evaluateCheck(lossBelow, { liveLoss: null, host: null })).toBe(false);
    expect(evaluateCheck({ ...lossBelow, split: 'val' }, at(0.1, 0.4))).toBe(false);
    expect(evaluateCheck({ ...lossBelow, split: 'val' }, at(0.9, 0.1))).toBe(true);
  });
  test('paramNear: within tol of the value, default tensor and index', () => {
    const c: Check = { kind: 'paramNear', node: 'bias', value: 0.25, tol: 0.01 };
    const at = (b: number) => ({ liveLoss: null, host: host(b) });
    expect(evaluateCheck(c, at(0.255))).toBe(true);
    expect(evaluateCheck(c, at(0.27))).toBe(false);
    expect(evaluateCheck({ ...c, tensor: 'b' }, at(0.25))).toBe(true);
    expect(evaluateCheck({ ...c, tensor: 'W' }, at(0.25))).toBe(false);
    expect(evaluateCheck({ ...c, index: 1 }, at(0.25))).toBe(false);
    expect(evaluateCheck({ ...c, node: 'nope' }, at(0.25))).toBe(false);
    expect(evaluateCheck(c, { liveLoss: null, host: null })).toBe(false);
    expect(evaluateCheck(c, at(Number.NaN))).toBe(false);
  });
  test('evaluateChecks groups by step and keeps unstepped ones together', () => {
    const checks: Check[] = [
      lossBelow,
      { kind: 'paramNear', node: 'bias', value: 0.25, tol: 0.01 },
      { kind: 'lossBelow', value: 1, step: 1 },
    ];
    const r = evaluateChecks(checks, { liveLoss: { train: 0.2, val: null }, host: host(0.9) });
    expect(r.general).toBe(false);
    expect([...r.steps]).toEqual([[1, true]]);
    expect(evaluateChecks([], { liveLoss: null, host: null }).general).toBeNull();
    const ok = evaluateChecks(checks, { liveLoss: { train: 0.2, val: null }, host: host(0.25) });
    expect(ok.general).toBe(true);
  });
});

describe('parseFile checks', () => {
  test('valid checks are kept', () => {
    const checks = [
      { kind: 'lossBelow', value: 0.3, split: 'val', step: 0 },
      { kind: 'paramNear', node: 'bias', tensor: 'b', index: 0, value: 0.25, tol: 0.01 },
    ];
    const r = parseFile(doc({ instructions: ['a'], checks }));
    expect(r.ok && 'checks' in r.file && r.file.checks).toEqual(checks);
    const r2 = parseFile(doc({}));
    expect(r2.ok && 'checks' in r2.file).toBe(false);
  });
  test.each([
    ['not a list', 'x'],
    ['not an object', [3]],
    ['unknown kind', [{ kind: 'magic', value: 1 }]],
    ['missing value', [{ kind: 'lossBelow' }]],
    ['non-finite value', [{ kind: 'lossBelow', value: null }]],
    ['bad split', [{ kind: 'lossBelow', value: 1, split: 'test' }]],
    ['unknown node', [{ kind: 'paramNear', node: 'zzz', value: 1, tol: 0.1 }]],
    ['missing node', [{ kind: 'paramNear', value: 1, tol: 0.1 }]],
    ['missing tol', [{ kind: 'paramNear', node: 'bias', value: 1 }]],
    ['negative tol', [{ kind: 'paramNear', node: 'bias', value: 1, tol: -1 }]],
    ['bad tensor', [{ kind: 'paramNear', node: 'bias', value: 1, tol: 1, tensor: 'Q' }]],
    ['bad index', [{ kind: 'paramNear', node: 'bias', value: 1, tol: 1, index: 1.5 }]],
    ['step out of range', [{ kind: 'lossBelow', value: 1, step: 1 }]],
  ])('%s is refused with a sentence', (_n, checks) => {
    const r = parseFile(doc({ instructions: ['only one'], checks }));
    expect(r.ok).toBe(false);
    if (!r.ok) expect(r.error).toMatch(/^(The preset checks|Check \d)/);
  });
  test('a step without instructions is refused', () => {
    expect(parseFile(doc({ checks: [{ kind: 'lossBelow', value: 1, step: 0 }] })).ok).toBe(false);
  });
});
