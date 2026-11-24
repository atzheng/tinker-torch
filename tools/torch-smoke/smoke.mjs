// `pnpm torch-smoke`: opt-in (needs uv and network, never part of `pnpm test`). Dumps the torch script of
// every bundled preset and engine fixture graph to a temp dir (through vitest, to import the TypeScript
// sources), then runs each under uv with torch and checks it (run.py). The scripts are kept for inspection.
import { spawnSync } from 'node:child_process';
import { mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = fileURLToPath(new URL('../..', import.meta.url));
const out = mkdtempSync(join(tmpdir(), 'torch-smoke-'));
const run = (cmd, args, env = {}) =>
  spawnSync(cmd, args, { cwd: root, stdio: 'inherit', env: { ...process.env, ...env } }).status ??
  1;

console.log(`scripts in ${out}`);
const dump = run(
  'pnpm',
  ['exec', 'vitest', 'run', '--config', 'tools/torch-smoke/vitest.config.ts'],
  { SMOKE_OUT: out },
);
if (dump !== 0) process.exit(dump);
process.exit(run('uv', ['run', 'tools/torch-smoke/run.py', join(out, 'manifest.json')]));
