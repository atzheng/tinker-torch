// The bundled presets are native graph v2 files (tinker-8xw.5). Their training loss at steps 1, 50
// and 300 was recorded before the cutover (version 1 graphs through the migration adapter); the same
// numbers must come out of the native v2 files.
import { readFileSync } from 'node:fs';
import { batchSchedule, gatherBatch, initParams, analyse, parseFile, sample } from '@tinker/core';
import { expect, test } from 'vitest';
import { createEngine } from '../src/index.js';

const BEFORE: Record<string, { params: number; losses: [number, number, number] }> = {
  linear: { params: 2, losses: [3.641615629196167, 0.032944925129413605, 0.032927874475717545] },
  logistic: { params: 3, losses: [0.6768960356712341, 0.12601564824581146, 0.026935765519738197] },
  sine: { params: 25, losses: [1.3797086477279663, 0.330300897359848, 0.004551100078970194] },
  skip: { params: 97, losses: [0.12895630300045013, 0.07969650626182556, 0.010925737209618092] },
  scalar: { params: 3, losses: [1.2799723148345947, 0.0032104565761983395, 0.0031482100021094084] },
  mlp: { params: 13, losses: [0.7887354493141174, 0.5219921469688416, 0.5090423822402954] },
  xor: { params: 105, losses: [0.7204835414886475, 0.06934435665607452, 0.0006533119012601674] },
};

const engine = await createEngine();

test.each(Object.keys(BEFORE))(
  'preset %s trains to the same losses as before the cutover',
  (name) => {
    const want = BEFORE[name];
    if (!want) throw new Error(name);
    const url = new URL(`../../web/src/presets/${name}.json`, import.meta.url);
    const file = parseFile(JSON.parse(readFileSync(url, 'utf8')));
    if (!file.ok) throw new Error(file.error);
    const g = file.file.graph;
    expect(g.version).toBe(2);
    const ds = sample(g.data);
    const analysis = analyse(g, ds.info);
    expect(analysis.issues).toEqual([]);
    expect(analysis.totalParams).toBe(want.params);
    const model = engine.compile(g, ds.info);
    let state = engine.startTraining(
      model,
      engine.fromHost(model, initParams(analysis, g.train.seed)),
      g.train.seed,
    );
    const batches = batchSchedule(ds.nTrain, g.train.batch, g.train.seed);
    const d = ds.info.columns.length;
    const got: number[] = [];
    for (let i = 0; i < 300; i++) {
      const idx = batches.next().value;
      const r = engine.step(model, state, gatherBatch(ds.xTrain, ds.yTrain, d, idx));
      state = r.state;
      if (i === 0 || i === 49 || i === 299) got.push(r.loss);
    }
    // logistic and xor use the log loss, rewritten constant-free (tinker-oob): last float32 digits moved
    const loose = name === 'logistic' || name === 'xor';
    want.losses.forEach((l, k) => {
      if (loose) expect(Math.abs(((got[k] ?? NaN) - l) / l)).toBeLessThanOrEqual(1e-6);
      else expect(got[k]).toBe(l);
    });
    engine.dispose(state);
    engine.disposeModel(model);
  },
);
