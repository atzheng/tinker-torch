// Every bundled real dataset trains without diverging on the settings a user gets when choosing
// it: standardise on, a one-layer model, the app's default SGD (lr 0.03) and the iris preset's Adam.
import { readFileSync } from 'node:fs';
import {
  BUNDLED,
  SETS,
  analyse,
  batchSchedule,
  gatherBatch,
  initParams,
  loadRealSet,
  parseFile,
  sample,
  type Graph,
} from '@tinker/core';
import { expect, test } from 'vitest';
import { createEngine } from '../src/index.js';

const engine = await createEngine();

function presetGraph(name: string): Graph {
  const url = new URL(`../../web/src/presets/${name}.json`, import.meta.url);
  const f = parseFile(JSON.parse(readFileSync(url, 'utf8')));
  if (!f.ok) throw new Error(f.error);
  return f.file.graph;
}

const cases = BUNDLED.flatMap((set) =>
  (['sgd', 'adam'] as const).map((optimizer) => [set, optimizer] as const),
);

test.each(cases)(
  '%s trains with %s and standardise on: finite and decreasing',
  async (set, opt) => {
    await loadRealSet(set);
    const regression = SETS[set].kind === 'regression';
    const base = presetGraph(regression ? 'linear' : 'iris');
    const d = SETS[set].d;
    const g: Graph = {
      ...base,
      nodes: base.nodes.map((n) =>
        n.type === 'input'
          ? { ...n, columns: Array.from({ length: d }, (_, i) => `x${i + 1}`) }
          : n.type === 'linear' && !regression
            ? { ...n, out: SETS[set].numClasses ?? 3 }
            : n,
      ),
      data: { ...base.data, set, standardize: true },
      train: { ...base.train, optimizer: opt, lr: opt === 'sgd' ? 0.03 : 0.05 },
    };
    const ds = sample(g.data);
    const analysis = analyse(g, ds.info);
    expect(analysis.issues).toEqual([]);
    const model = engine.compile(g, ds.info);
    let state = engine.startTraining(
      model,
      engine.fromHost(model, initParams(analysis, g.train.seed)),
      g.train.seed,
    );
    const batches = batchSchedule(ds.nTrain, g.train.batch, g.train.seed);
    const losses: number[] = [];
    for (let i = 0; i < 300; i++) {
      const idx = batches.next().value;
      const r = engine.step(model, state, gatherBatch(ds.xTrain, ds.yTrain, d, idx));
      state = r.state;
      expect(r.diverged).toBe(false);
      losses.push(r.loss);
    }
    expect(losses.every(Number.isFinite)).toBe(true);
    expect(losses[299] ?? Infinity).toBeLessThan(losses[0] ?? 0);
    engine.dispose(state);
    engine.disposeModel(model);
  },
);
