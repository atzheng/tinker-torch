// Timings for spike item 3. Runs the same code in Node and in the browser page.
import './probe.ts'; // must be the first import (see probe.ts)
import { jit, numpy as np, tree } from '@jax-js/jax';
import {
  forward,
  initParams,
  initJax,
  linearData,
  makeManyTrainer,
  makeTrainer,
  sineData,
  wasmShared,
} from './lib.ts';
import type { OptName } from './lib.ts';

export type BenchResult = {
  env: string;
  sharedMemory: boolean;
  crossOriginIsolated: boolean | null;
  hardwareConcurrency: number | null;
  compileMs: Record<string, number>;
  msPerStep: Record<string, number>;
  predictGrid4096Ms: { jitFirstCall: number; jitMedian: number; eagerMedian: number };
  trainMany: Record<string, number>;
};

const now = () => performance.now();

function median(xs: number[]): number {
  const s = [...xs].sort((a, b) => a - b);
  return s[Math.floor(s.length / 2)]!;
}

/** ms per full-batch step: first call (compile) separately, then the median of 5 blocks of `steps`. */
function timeSteps(
  sizes: number[],
  tanh: boolean,
  opt: OptName,
  lr: number,
  n: number,
  steps: number,
  data = tanh ? sineData(n, 1) : linearData(n, 1),
): { compileMs: number; msPerStep: number } {
  const t = makeTrainer(sizes, tanh, opt, lr, data, 1);
  const c0 = now();
  t.run(1); // trace + compile + first execution
  const compileMs = now() - c0;
  t.run(20); // warm up
  const blocks: number[] = [];
  for (let b = 0; b < 5; b++) {
    const t0 = now();
    t.run(steps);
    blocks.push((now() - t0) / steps);
  }
  t.dispose();
  return { compileMs, msPerStep: median(blocks) };
}

export async function runBench(
  log: (s: string) => void,
  opts: { env: string; quick?: boolean },
): Promise<BenchResult> {
  await initJax();
  const steps = opts.quick ? 100 : 1000;
  const g = globalThis as unknown as {
    crossOriginIsolated?: boolean;
    navigator?: { hardwareConcurrency?: number };
  };
  const res: BenchResult = {
    env: opts.env,
    sharedMemory: false,
    crossOriginIsolated: g.crossOriginIsolated ?? null,
    hardwareConcurrency: g.navigator?.hardwareConcurrency ?? null,
    compileMs: {},
    msPerStep: {},
    predictGrid4096Ms: { jitFirstCall: 0, jitMedian: 0, eagerMedian: 0 },
    trainMany: {},
  };

  // Full-batch steps.
  const cases: [string, number[], boolean, OptName, number, number][] = [
    ['linear n=40 sgd', [1, 1], false, 'sgd', 0.1, 40],
    ['mlp 1-16-1 n=40 sgd', [1, 16, 1], true, 'sgd', 0.05, 40],
    ['mlp 1-16-1 n=40 momentum', [1, 16, 1], true, 'momentum', 0.05, 40],
    ['mlp 1-16-1 n=40 adam', [1, 16, 1], true, 'adam', 0.02, 40],
    ['mlp 1-16-1 n=40 adam-optax', [1, 16, 1], true, 'adam-optax', 0.02, 40],
    ['mlp 1-16-1 n=500 adam', [1, 16, 1], true, 'adam', 0.02, 500],
    ['mlp 1-64-64-1 n=500 adam', [1, 64, 64, 1], true, 'adam', 0.02, 500],
  ];
  for (const [name, sizes, tanh, opt, lr, n] of cases) {
    const r = timeSteps(sizes, tanh, opt, lr, n, steps);
    res.compileMs[name] = r.compileMs;
    res.msPerStep[name] = r.msPerStep;
    log(`${name}: ${r.msPerStep.toFixed(3)} ms/step, first call ${r.compileMs.toFixed(0)} ms`);
  }

  // Predict on a 4096-point grid (jitted and eager forward), including the dataSync readback.
  {
    const params = initParams([1, 16, 1], 1);
    const grid = new Float32Array(4096);
    for (let i = 0; i < 4096; i++) grid[i] = -3 + (6 * i) / 4095;
    const pj = jit((p: typeof params, x: ReturnType<typeof np.array>) => forward(p, x, true));
    const f = () =>
      Float32Array.from(
        pj(tree.ref(params), np.array(grid, { shape: [4096, 1] })).dataSync() as Float32Array,
      );
    const e = () =>
      Float32Array.from(
        forward(
          tree.ref(params),
          np.array(grid, { shape: [4096, 1] }),
          true,
        ).dataSync() as Float32Array,
      );
    let t0 = now();
    f();
    const first = now() - t0;
    const jt: number[] = [];
    const et: number[] = [];
    for (let i = 0; i < 21; i++) {
      t0 = now();
      f();
      jt.push(now() - t0);
      t0 = now();
      e();
      et.push(now() - t0);
    }
    pj.dispose();
    tree.dispose(params);
    res.predictGrid4096Ms = {
      jitFirstCall: first,
      jitMedian: median(jt),
      eagerMedian: median(et),
    };
    log(
      `predict 4096 grid: jit first ${first.toFixed(1)} ms, jit median ${median(jt).toFixed(3)} ms, eager median ${median(et).toFixed(3)} ms`,
    );
  }

  // trainMany: k = 20 copies, each with its own data and params; 1000 steps of adam.
  for (const [n, label] of [
    [40, 'k=20 n=40'],
    [500, 'k=20 n=500'],
  ] as const) {
    const k = 20;
    const datas = Array.from({ length: k }, (_, i) => sineData(n, 10 + i));
    const seeds = Array.from({ length: k }, (_, i) => i + 1);
    const t0 = now();
    const m = makeManyTrainer([1, 16, 1], true, 'adam', 0.02, datas, seeds);
    const setup = now() - t0;
    const t1 = now();
    m.run(1);
    const compile = now() - t1;
    const t2 = now();
    m.run(1000);
    const total = now() - t2;
    m.read();
    m.dispose();
    res.trainMany[`${label} setup ms`] = setup;
    res.trainMany[`${label} first step (compile) ms`] = compile;
    res.trainMany[`${label} 1000 steps ms`] = total;
    res.trainMany[`${label} ms/step`] = total / 1000;
    log(
      `trainMany ${label}: 1000 steps ${total.toFixed(0)} ms (${(total / 1000).toFixed(3)} ms/step), compile ${compile.toFixed(0)} ms`,
    );
  }
  // Pitfall: jax-js is lazy, so unread steps stack up as one growing graph. Same k=20 n=40
  // workload, reading the loss every 1 / 25 / 100 steps (the engine reads it every step).
  for (const sync of [1, 25, 100]) {
    const k = 20;
    const datas = Array.from({ length: k }, (_, i) => sineData(40, 10 + i));
    const seeds = Array.from({ length: k }, (_, i) => i + 1);
    const m = makeManyTrainer([1, 16, 1], true, 'adam', 0.02, datas, seeds, sync);
    m.run(sync);
    const t0 = now();
    m.run(200);
    const per = (now() - t0) / 200;
    m.dispose();
    res.trainMany[`k=20 n=40 loss read every ${sync} steps, ms/step`] = per;
    log(`trainMany k=20 n=40, loss read every ${sync} steps: ${per.toFixed(2)} ms/step`);
  }
  res.sharedMemory = wasmShared();
  return res;
}
