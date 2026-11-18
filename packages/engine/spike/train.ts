// Spike item 1: train linear regression (40 points) and a 1-16-1 tanh MLP on sine with sgd,
// sgd+momentum and adam, using grad/valueAndGrad + jit over a params object keyed by node id.
import './probe.ts'; // must be the first import (see probe.ts)
import { initJax, linearData, makeTrainer, sineData } from './lib.ts';
import type { OptName } from './lib.ts';

export type TrainRow = {
  problem: string;
  optimizer: string;
  steps: number;
  lossAt: Record<number, number>;
  finalLoss: number;
  expected: string;
  ok: boolean;
  detail: string;
};

/** Least squares y = a x + b in closed form, and its mean squared residual. */
function lstsq(x: Float32Array, y: Float32Array): { a: number; b: number; mse: number } {
  const n = x.length;
  let sx = 0,
    sy = 0,
    sxx = 0,
    sxy = 0;
  for (let i = 0; i < n; i++) {
    sx += x[i]!;
    sy += y[i]!;
    sxx += x[i]! * x[i]!;
    sxy += x[i]! * y[i]!;
  }
  const a = (n * sxy - sx * sy) / (n * sxx - sx * sx);
  const b = (sy - a * sx) / n;
  let mse = 0;
  for (let i = 0; i < n; i++) mse += (y[i]! - a * x[i]! - b) ** 2;
  return { a, b, mse: mse / n };
}

export async function runTraining(log: (s: string) => void): Promise<TrainRow[]> {
  await initJax();
  const rows: TrainRow[] = [];

  // Linear regression: converges to the closed-form least-squares line.
  const lin = linearData(40, 1);
  const ls = lstsq(lin.x, lin.y);
  const linCases: [OptName, number, number][] = [
    ['sgd', 0.1, 500],
    ['momentum', 0.05, 300],
    ['adam', 0.05, 500],
  ];
  for (const [opt, lr, steps] of linCases) {
    const t = makeTrainer([1, 1], false, opt, lr, lin, 1);
    const losses = t.run(steps);
    const p = t.read().lin1!;
    t.dispose();
    const da = Math.abs(p.W[0]! - ls.a);
    const db = Math.abs(p.b[0]! - ls.b);
    const ok = da < 0.01 && db < 0.01 && losses[steps - 1]! < ls.mse * 1.01 + 1e-6;
    const row: TrainRow = {
      problem: 'linear, n=40',
      optimizer: `${opt} lr=${lr}`,
      steps,
      lossAt: { 1: losses[0]!, [steps]: losses[steps - 1]! },
      finalLoss: losses[steps - 1]!,
      expected: `loss -> least-squares floor ${ls.mse.toFixed(5)}, W -> ${ls.a.toFixed(4)}, b -> ${ls.b.toFixed(4)}`,
      ok,
      detail: `W=${p.W[0]!.toFixed(4)} b=${p.b[0]!.toFixed(4)}`,
    };
    rows.push(row);
    log(
      `${ok ? 'ok  ' : 'FAIL'} ${row.problem} ${row.optimizer}: loss ${losses[0]!.toFixed(4)} -> ${row.finalLoss.toFixed(5)}; ${row.detail}; ${row.expected}`,
    );
  }

  // MLP on sine: loss must fall by more than 20x and monotonically at the checkpoints.
  const sine = sineData(40, 2);
  const mlpCases: [OptName, number, number][] = [
    ['sgd', 0.05, 4000],
    ['momentum', 0.02, 2000],
    ['adam', 0.01, 2000],
  ];
  for (const [opt, lr, steps] of mlpCases) {
    const t = makeTrainer([1, 16, 1], true, opt, lr, sine, 3);
    const checkpoints = [1, steps / 4, steps / 2, steps];
    const lossAt: Record<number, number> = {};
    let done = 0;
    let first = 0;
    for (const c of checkpoints) {
      const l = t.run(c - done);
      done = c;
      lossAt[c] = l[l.length - 1]!;
      if (c === 1) first = l[0]!;
    }
    t.dispose();
    const vals = checkpoints.map((c) => lossAt[c]!);
    const mono = vals.every((v, i) => i === 0 || v <= vals[i - 1]! * 1.001);
    const ok = mono && lossAt[steps]! < first / 20 && lossAt[steps]! < 0.02;
    const row: TrainRow = {
      problem: 'sine, 1-16-1 tanh, n=40',
      optimizer: `${opt} lr=${lr}`,
      steps,
      lossAt,
      finalLoss: lossAt[steps]!,
      expected: 'monotone loss, final < 0.02 and < 1/20 of initial',
      ok,
      detail: checkpoints.map((c) => `${c}:${lossAt[c]!.toFixed(4)}`).join(' '),
    };
    rows.push(row);
    log(`${ok ? 'ok  ' : 'FAIL'} ${row.problem} ${row.optimizer}: ${row.detail}`);
  }
  return rows;
}
