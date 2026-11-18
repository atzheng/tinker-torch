// Opt-in timing (TINKER_BENCH=1 pnpm --filter @tinker/engine test many.bench): 20 copies x 1000
// steps on the 1-16-1 tanh MLP, compared with docs/spike.md (0.48 s at n = 40, 3.6 s at n = 500 in
// Chrome). Not part of the default run.
import { describe, expect, test } from 'vitest';
import { loadFixture } from './util.js';
import { createEngine } from '../src/index.js';

const bench = process.env['TINKER_BENCH'] ? describe : describe.skip;

bench('trainMany timing', () => {
  for (const [n, optimizer] of [
    [40, 'sgd'],
    [500, 'sgd'],
    [40, 'adam'],
  ] as const) {
    test(`k = 20, 1000 steps, n = ${n}, ${optimizer}`, async () => {
      const engine = await createEngine();
      const { graph, data: info } = loadFixture('mlp_tanh_sgd_5steps');
      // the spike's 1-16-1 MLP: widen the first layer of the fixture graph (1 hidden layer)
      const nodes = graph.nodes.map((nd) =>
        nd.id === 'lin1' && nd.type === 'linear' ? { ...nd, out: 16 } : nd,
      );
      const train = { ...graph.train, optimizer, lr: 0.02 };
      const model = engine.compile({ ...graph, nodes, train }, info);
      const d = info.columns.length;
      const sets = Array.from({ length: 20 }, (_, i) => ({
        x: Float32Array.from({ length: n * d }, (_, j) => Math.sin(j * 0.37 + i)),
        y: Float32Array.from({ length: n }, (_, j) => Math.sin(j * 0.11 + i)),
        n,
      }));
      const seeds = sets.map((_, i) => i + 1);
      engine.trainMany(model, seeds, sets, 5); // trace + warm up
      const t0 = performance.now();
      const out = engine.trainMany(model, seeds, sets, 1000);
      const ms = performance.now() - t0;
      expect(out.some((r) => r.diverged)).toBe(false);
      throw new Error(`TIMING n=${n} ${optimizer}: ${ms.toFixed(0)} ms`);
    }, 120_000);
  }
});
