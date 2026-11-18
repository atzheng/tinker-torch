// Node entry for the spike. Usage (from packages/engine):
//   pnpm spike                      all four parts, human-readable log, then one JSON line
//   pnpm spike -- --only=ops        one of: train, ops, bench, leak
//   pnpm spike -- --quick           100-step timing blocks and 1,000-step leak loops
// Needs Node >= 22.6 (runs the .ts files directly with --experimental-strip-types).
import './probe.ts'; // must be the first import (see probe.ts)
import { runBench } from './bench.ts';
import { runLeakTests } from './leak.ts';
import { checkOps } from './ops.ts';
import { initJax } from './lib.ts';
import { runTraining } from './train.ts';

// Minimal typing for the two Node globals used here (the package has no @types/node).
declare const process: {
  argv: string[];
  version: string;
  stdout: { write(s: string): void };
  exitCode?: number;
};

const args = process.argv.slice(2);
const only = args.find((a) => a.startsWith('--only='))?.slice(7);
const quick = args.includes('--quick');
const want = (part: string) => !only || only === part;
const log = (s: string) => process.stdout.write(s + '\n');

await initJax();
const out: Record<string, unknown> = { node: process.version };
let failed = false;

if (want('train')) {
  log('== training ==');
  const rows = await runTraining(log);
  out.train = rows;
  failed ||= rows.some((r) => !r.ok);
}
if (want('ops')) {
  log('== operations ==');
  const results = checkOps();
  for (const r of results) {
    const tag = r.pass ? 'ok    ' : r.known ? 'DEFECT' : 'FAIL  ';
    log(`${tag} ${r.name} | ${r.detail}`);
  }
  out.ops = results;
  failed ||= results.some((r) => !r.pass && !r.known);
}
if (want('bench')) {
  log('== timings ==');
  out.bench = await runBench(log, { env: 'node ' + process.version, quick });
}
if (want('leak')) {
  log('== leak test ==');
  const leaks = await runLeakTests(log, quick ? 1_000 : 10_000);
  out.leak = leaks;
  failed ||= leaks.some((r) => (r.control ? r.flat : !r.flat));
}
log('JSON ' + JSON.stringify(out));
process.exitCode = failed ? 1 : 0;
