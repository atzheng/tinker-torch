/// <reference types="node" />
// Dump step of `pnpm torch-smoke` (tools/torch-smoke/smoke.mjs): writes torchScript output (50 steps)
// for every bundled preset and every engine fixture graph to $SMOKE_OUT, plus manifest.json.
// Not collected by `pnpm test` (see vitest.config.ts); it only runs from smoke.mjs.
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { test } from 'vitest';
import {
  SETS,
  analyse,
  classCount,
  fitTrees,
  initParams,
  isBundled,
  loadRealSet,
  parseFile,
  parseGraph,
  sample,
  torchScript,
  treeLeafCounts,
  type DataInfo,
  type Dataset,
  type Graph,
} from '../../packages/core/src/index.js';

const root = join(import.meta.dirname, '../..');
const presetDir = join(root, 'packages/web/src/presets');
const fixtureDir = join(root, 'packages/engine/test/fixtures');
const out = process.env.SMOKE_OUT ?? '';

/** Presets whose start graph is meant to diverge (not failures). Keep this list short and justified. */
const EXPECTED_NAN: Record<string, string> = {
  'lecture1-stage2':
    'bias straight into log loss at lr 0.5; the lesson has the student add a sigmoid',
  'lecture1-stage0':
    'the app starts the bias outside (0, 1) for a raw log loss: step 0 is NaN there too',
  'lecture1-stage1':
    'the app starts the weight outside (0, 1) for a raw log loss: step 0 is NaN there too',
  prob_direct_log_loss:
    'the app starts a raw probability outside (0, 1) for a raw log loss: NaN there too',
};

type Entry = { name: string; script: string | null; skip?: string; expectedNaN?: string };

function scriptOf(graph: Graph, ds: Dataset): string | null {
  const a = analyse(graph, ds.info, treeLeafCounts(fitTrees(graph, ds)));
  // the engine's initial parameters (initParams), so the strict load_state_dict is exercised for every case
  return (
    torchScript(graph, a, ds, { steps: 50, params: initParams(a, graph.train.seed) })?.text ?? null
  );
}

type Fixture = {
  name: string;
  graph: unknown;
  categories?: Record<string, number>;
  batch: { x: number[]; y: number[]; n: number };
};

/** The engine tests' dataset for a fixture: the batch is the training set, columns x1..xd (util.ts loadFixture). */
function fixtureDataset(fx: Fixture, graph: Graph): Dataset {
  const d = fx.batch.x.length / fx.batch.n;
  const columns = Array.from({ length: d }, (_, i) => `x${i + 1}`);
  const spec = SETS[graph.data.set];
  const info: DataInfo = {
    columns,
    kind: spec.kind,
    numClasses: classCount(graph.data.set),
    ...(fx.categories
      ? {
          columnKinds: columns.map((c) => (fx.categories?.[c] ? 'categorical' : 'numeric')),
          categories: Object.fromEntries(
            Object.entries(fx.categories).map(([c, k]) => [
              c,
              Array.from({ length: k }, (_, i) => `c${i}`),
            ]),
          ),
        }
      : {}),
  };
  const xTrain = Float32Array.from(fx.batch.x);
  const domain = columns.map((_, c): [number, number] => {
    const col = Array.from({ length: fx.batch.n }, (_, i) => xTrain[i * d + c] ?? 0);
    return [Math.min(...col), Math.max(...col)];
  });
  return {
    info,
    xTrain,
    yTrain: Float32Array.from(fx.batch.y),
    nTrain: fx.batch.n,
    xVal: new Float32Array(0),
    yVal: new Float32Array(0),
    nVal: 0,
    standardization: null,
    domain,
  };
}

test('dump scripts', async () => {
  if (out === '') throw new Error('SMOKE_OUT is not set: run `pnpm torch-smoke`.');
  mkdirSync(out, { recursive: true });
  const entries: Entry[] = [];
  const add = async (
    kind: string,
    name: string,
    make: () => Promise<string | null> | string | null,
    extra?: Partial<Entry>,
  ) => {
    const label = `${kind}:${name}`;
    let text: string | null;
    try {
      text = await make();
    } catch (e) {
      entries.push({ name: label, script: null, skip: `dump failed: ${String(e)}` });
      return;
    }
    if (text === null) {
      entries.push({ name: label, script: null, skip: 'torchScript returned null' });
      return;
    }
    const path = join(out, `${kind}-${name}.py`);
    writeFileSync(path, text);
    entries.push({ name: label, script: path, ...extra });
  };

  for (const f of readdirSync(presetDir)
    .filter((f) => f.endsWith('.json'))
    .sort()) {
    const name = f.replace(/\.json$/, '');
    const why = EXPECTED_NAN[name];
    await add(
      'preset',
      name,
      async () => {
        const r = parseFile(JSON.parse(readFileSync(join(presetDir, f), 'utf8')));
        if (!r.ok) throw new Error(r.error);
        // the preset's own dataset and seed (the fixed dataset seed of the case: graph.data.seed)
        if (isBundled(r.file.graph.data.set)) await loadRealSet(r.file.graph.data.set); // bundled real sets load their rows on demand
        return scriptOf(r.file.graph, sample(r.file.graph.data));
      },
      why ? { expectedNaN: why } : {},
    );
  }
  for (const f of readdirSync(fixtureDir)
    .filter((f) => f.endsWith('.json'))
    .sort()) {
    const name = f.replace(/\.json$/, '');
    const nan = EXPECTED_NAN[name];
    await add(
      'fixture',
      name,
      () => {
        const fx = JSON.parse(readFileSync(join(fixtureDir, f), 'utf8')) as Fixture;
        const p = parseGraph(fx.graph); // version 1 fixtures are migrated
        if (!p.ok) throw new Error(p.error);
        return scriptOf(p.graph, fixtureDataset(fx, p.graph));
      },
      nan ? { expectedNaN: nan } : {},
    );
  }
  writeFileSync(join(out, 'manifest.json'), JSON.stringify(entries, null, 1));
});
