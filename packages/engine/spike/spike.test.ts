// Fast regression guard for the spike: every operation check, every training run and a short leak
// loop. The full 10,000-step leak test and the timings run through `pnpm spike`.
import './probe.ts'; // must be the first import (see probe.ts)
import { expect, test } from 'vitest';
import { initJax } from './lib.ts';
import { runLeakTests } from './leak.ts';
import { checkOps } from './ops.ts';
import { runTraining } from './train.ts';

test('operations behave (known jax-js defects are listed separately)', async () => {
  await initJax();
  const results = checkOps();
  const failures = results.filter((r) => !r.pass && !r.known).map((r) => `${r.name}: ${r.detail}`);
  expect(failures).toEqual([]);
});

test('training loss falls as expected for linear and MLP with sgd, momentum, adam', async () => {
  const rows = await runTraining(() => {});
  const bad = rows.filter((r) => !r.ok).map((r) => `${r.problem} ${r.optimizer}: ${r.detail}`);
  expect(bad).toEqual([]);
});

test('live buffer count is flat over 1,000 steps (and a deliberate leak is detected)', async () => {
  const results = await runLeakTests(() => {}, 1_000);
  const real = results.filter((r) => !r.control);
  expect(real.filter((r) => !r.flat).map((r) => r.name)).toEqual([]);
  expect(results.filter((r) => r.control).every((r) => !r.flat)).toBe(true);
}, 60_000);
