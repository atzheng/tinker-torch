// The resampling overlay's work (resampleRun.ts) on the Wasm engine in Node: bias and variance of
// two architectures on a sine, and that repeated overlays leave no live buffers behind.
import { installProbe } from '@tinker/engine/probe';
import { expect, test, vi } from 'vitest';
import { analyse, biasVariance, grid1d, sample, truth, type Graph } from '@tinker/core';
import sine from '../src/presets/sine.json';
import { adaptEngine } from '../src/legacyEngine.js';
import { runResample } from '../src/results/resampleRun.js';

installProbe();
vi.setConfig({ testTimeout: 120_000 });
const { createEngine } = await import('@tinker/engine');
const engine = adaptEngine(await createEngine()); // temporary v1 -> v2 adapter (tinker-8xw.4)

const base = (sine as unknown as { graph: Graph }).graph;

/** The sine preset with `width` hidden tanh units, or no hidden layer (a straight line) for 0. */
function graphOf(width: number, n: number): Graph {
  const data = { ...base.data, n, val: 0, noise: 0.2 };
  const nodes = base.nodes.filter((x) => width > 0 || (x.id !== 'act1' && x.id !== 'lin2'));
  const edges =
    width > 0
      ? base.edges
      : ([
          ['x', 'lin1'],
          ['lin1', 'yhat'],
          ['yhat', 'loss.pred'],
          ['y', 'loss.target'],
        ] as Graph['edges']);
  return {
    ...base,
    data,
    nodes: nodes.map((x) => (x.id === 'lin1' ? { ...x, out: width > 0 ? width : 1 } : x)),
    edges,
  } as Graph;
}

function bv(g: Graph, steps: number) {
  const r = runResample(engine, { graph: g, steps, k: 20, seed: 5, gridN: 200 });
  if (!r.ok) throw new Error(r.message);
  const info = analyse(g, sample(g.data).info);
  expect(info.trainable).toBe(true);
  const grid = grid1d(sample(g.data).domain, 200);
  return { r, ...biasVariance(r.curves, truth(g.data.set, grid.xRaw)) };
}

test('a line on a sine: high bias, low variance; a 1-32-1 network at n = 20: the reverse', () => {
  const line = bv(graphOf(0, 20), 400);
  const net = bv(graphOf(32, 20), 400);
  expect(line.k).toBe(20);
  expect(net.k).toBe(20);
  expect(line.bias2).toBeGreaterThan(line.variance * 3);
  expect(net.variance).toBeGreaterThan(net.bias2 * 1.5);
  expect(line.bias2).toBeGreaterThan(net.bias2);
});

test('copies differ (new data and new init) and diverged copies are counted, not drawn', () => {
  const r = runResample(engine, {
    graph: { ...graphOf(8, 20), train: { ...base.train, optimizer: 'sgd', lr: 1e6 } },
    steps: 20,
    k: 6,
    seed: 1,
    gridN: 50,
  });
  expect(r.ok).toBe(true);
  if (!r.ok) return;
  expect(r.curves.length + r.diverged).toBe(6);
  expect(r.diverged).toBeGreaterThan(0);
  const ok = runResample(engine, { graph: graphOf(8, 20), steps: 5, k: 3, seed: 1, gridN: 50 });
  if (!ok.ok) throw new Error(ok.message);
  expect(ok.curves[0]).not.toEqual(ok.curves[1]);
});

test('repeated overlays leave the live buffer count flat', () => {
  const g = graphOf(8, 20);
  const run = () => runResample(engine, { graph: g, steps: 30, k: 20, seed: 2, gridN: 200 });
  run();
  const before = engine.liveArrays();
  for (let i = 0; i < 3; i++) expect(run().ok).toBe(true);
  expect(engine.liveArrays()).toBe(before);
});

test('a graph the engine cannot train comes back as a sentence, with nothing left behind', () => {
  const before = engine.liveArrays();
  const g = { ...graphOf(8, 20), edges: base.edges.slice(1) };
  const r = runResample(engine, { graph: g, steps: 5, k: 3, seed: 1, gridN: 50 });
  expect(r.ok).toBe(false);
  expect(engine.liveArrays()).toBe(before);
});
