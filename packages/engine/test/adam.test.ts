// Regression for tinker-df1: plain Adam makes the loss spike at irregular, roughly evenly spaced
// intervals (up to ~800x the running minimum) once the gradients vanish. The default Adam is
// AMSGrad (CONTRACTS.md section 7), which never spikes on the same run.
import { analyse, initParams, parseGraph, sample } from '@tinker/core';
import { beforeAll, expect, test } from 'vitest';
import { createEngine } from '../src/index.js';
import type { Engine } from '../src/index.js';

let engine: Engine;
beforeAll(async () => {
  engine = await createEngine();
});

/** Noise-free sine fit, 1-8-8-1 tanh MLP, full batch, Adam lr 0.05, seed 1 (so no batching effects). */
function run(steps: number): number[] {
  const cols = ['x1'];
  const lin = (id: string, out: number) => ({ id, type: 'linear', out, bias: true });
  const act = (id: string) => ({ id, type: 'activation', fn: 'tanh' });
  const g = parseGraph({
    version: 1,
    nodes: [
      { id: 'x', type: 'input', columns: cols },
      lin('lin1', 8),
      act('act1'),
      lin('lin2', 8),
      act('act2'),
      lin('lin3', 1),
      { id: 'yhat', type: 'output', link: 'identity' },
      { id: 'y', type: 'target' },
      { id: 'loss', type: 'loss', fn: 'squared_error' },
    ],
    edges: [
      ['x', 'lin1'],
      ['lin1', 'act1'],
      ['act1', 'lin2'],
      ['lin2', 'act2'],
      ['act2', 'lin3'],
      ['lin3', 'yhat'],
      ['yhat', 'loss.pred'],
      ['y', 'loss.target'],
    ],
    data: { set: 'sine', n: 40, noise: 0, val: 0, seed: 1, standardize: false },
    train: { optimizer: 'adam', lr: 0.05, batch: 'full', seed: 1 },
  });
  if (!g.ok) throw new Error(g.error);
  const ds = sample(g.graph.data);
  const data = { columns: cols, kind: 'regression' as const, numClasses: 1 };
  const model = engine.compile(g.graph, data);
  const host = initParams(g.graph, analyse(g.graph, data), 1);
  let state = engine.startTraining(model, engine.fromHost(model, host), 1);
  const batch = { x: ds.xTrain, y: ds.yTrain, n: ds.yTrain.length };
  const losses: number[] = [];
  for (let i = 0; i < steps; i++) {
    const r = engine.step(model, state, batch);
    state = r.state;
    losses.push(r.loss);
    expect(r.diverged).toBe(false);
  }
  engine.dispose(state);
  return losses;
}

test('Adam does not spike late in training (loss never exceeds 3x its running minimum)', () => {
  const losses = run(2000);
  let runMin = Infinity;
  let worst = 0;
  losses.forEach((l, i) => {
    if (i > 300) worst = Math.max(worst, l / runMin);
    runMin = Math.min(runMin, l);
  });
  expect(runMin).toBeLessThan(1e-4); // the run does converge, so the check is on the flat part
  expect(worst).toBeLessThan(3);
});
