// Bare page for the spike (not part of the app). Query: ?auto=1 runs everything on load;
// &only=ops|train|bench|leak picks one part; &quick=1 shortens the loops.
// The result object is left in window.__spike for scripted runs (see docs/spike.md).
import '../probe.ts'; // must be the first import (see probe.ts)
import { runBench } from '../bench.ts';
import { initJax, wasmShared } from '../lib.ts';
import { runLeakTests } from '../leak.ts';
import { checkOps } from '../ops.ts';
import { runTraining } from '../train.ts';

const logEl = document.getElementById('log')!;
const log = (s: string) => {
  logEl.textContent += s + '\n';
};
const q = new URLSearchParams(location.search);

async function runPart(part: string, out: Record<string, unknown>, quick: boolean) {
  log(`== ${part} ==`);
  // Yield so the page paints between parts.
  await new Promise((r) => setTimeout(r, 0));
  if (part === 'train') out.train = await runTraining(log);
  if (part === 'ops') {
    const results = checkOps();
    for (const r of results) {
      log(`${r.pass ? 'ok    ' : r.known ? 'DEFECT' : 'FAIL  '} ${r.name} | ${r.detail}`);
    }
    out.ops = results;
  }
  if (part === 'bench') out.bench = await runBench(log, { env: navigator.userAgent, quick });
  if (part === 'leak') out.leak = await runLeakTests(log, quick ? 1_000 : 10_000);
}

async function run(parts: string[]) {
  const quick = q.get('quick') === '1';
  const w = window as unknown as { __spike?: Record<string, unknown>; __spikeDone?: boolean };
  w.__spikeDone = false;
  const out: Record<string, unknown> = {};
  w.__spike = out;
  try {
    await initJax();
    out.env = {
      userAgent: navigator.userAgent,
      crossOriginIsolated: window.crossOriginIsolated,
      sharedArrayBuffer: typeof SharedArrayBuffer !== 'undefined',
      wasmMemoryShared: wasmShared(),
      hardwareConcurrency: navigator.hardwareConcurrency,
    };
    document.getElementById('env')!.textContent = JSON.stringify(out.env);
    for (const p of parts) await runPart(p, out, quick);
  } catch (e) {
    out.error = e instanceof Error ? (e.stack ?? e.message) : String(e);
    log('ERROR ' + String(out.error));
  }
  w.__spikeDone = true;
  log('done');
}

const all = ['train', 'ops', 'bench', 'leak'];
document.getElementById('all')!.onclick = () => void run(all);
document.querySelectorAll<HTMLButtonElement>('button[data-part]').forEach((b) => {
  b.onclick = () => void run([b.dataset.part!]);
});
if (q.get('auto') === '1') void run(q.get('only') ? [q.get('only')!] : all);
