/**
 * The work of one "Resample x 20" run, written against an Engine so it runs in the worker
 * (resampleWorker.ts) and in Node tests. No jax array leaves this function: every model and Params
 * it makes is disposed before it returns, also when something throws.
 */
import {
  fitTrees,
  grid1d,
  grid2dSlice,
  heldValues,
  makeRng,
  sample,
  type Graph,
} from '@tinker/core';
import type { Engine } from '@tinker/engine';

export type ResampleRequest = {
  graph: Graph;
  /** Full-batch steps per copy. */
  steps: number;
  /** Number of copies. */
  k: number;
  /** Base seed: every copy's data seed and init seed derive from it. */
  seed: number;
  /** Grid points: 1-D data uses grid1d(domain, gridN), 2-D grid2d(domain, gridN). */
  gridN: number;
  /** Raw value each column is held at in the 2-D slice (`heldValues` of the plotted dataset); default: of each copy. */
  held?: number[];
  /** Which ŷ pin to predict (index into `graph.predictions`); default 0. */
  prediction?: number;
};

export type ResampleResponse =
  | {
      ok: true;
      /** One curve (1-D, length gridN) or surface (2-D, gridN * gridN) per copy that did not diverge. */
      curves: Float32Array[];
      /** Copies that diverged (or predicted non-finite values), excluded from `curves`. */
      diverged: number;
    }
  | { ok: false; message: string };

/**
 * Trains `k` copies of the graph, each on a fresh training set from the same generator (a forked
 * data seed, same n) and a new initialisation, then predicts each on the fixed grid. Each copy is
 * standardised with its own training set, the grid with the same numbers, so every curve is the
 * model as a function of the raw x. A graph with Tree nodes fits each copy's own trees (trainMany does it,
 * on that copy's training rows) and each copy is predicted with ITS OWN tree: a model compiled with the
 * copy's fits. Borrows the engine.
 */
export function runResample(engine: Engine, req: ResampleRequest): ResampleResponse {
  const { graph, steps, k, seed, gridN } = req;
  const base = makeRng(seed);
  const sets = Array.from({ length: k }, (_, i) =>
    sample({ ...graph.data, seed: base.fork('data' + i).int(2 ** 31) }),
  );
  const first = sets[0];
  if (!first) return { ok: false, message: 'There are no copies to train.' };
  const seeds = sets.map((_, i) => base.fork('init' + i).int(2 ** 31));
  const d = first.info.columns.length;
  let model: ReturnType<Engine['compile']> | null = null;
  try {
    // the first copy's fits give the Tree widths of the model that trainMany pads; the copies refit their own
    model = engine.compile(graph, first.info, fitTrees(graph, first));
    const results = engine.trainMany(
      model,
      seeds,
      sets.map((s) => ({ x: s.xTrain, y: s.yTrain, n: s.nTrain })),
      steps,
    );
    const curves: Float32Array[] = [];
    let diverged = 0;
    results.forEach((r, i) => {
      const ds = sets[i];
      if (!ds || r.diverged || !model) {
        diverged++;
        return;
      }
      const grid =
        d === 1
          ? grid1d(ds.domain, gridN, ds.standardization)
          : grid2dSlice(ds.domain, req.held ?? heldValues(ds), 0, 1, gridN, ds.standardization);
      // with trees the params have this copy's leaf counts: predict with a model made from its fits
      const own = Object.keys(r.fits).length > 0 ? engine.compile(graph, ds.info, r.fits) : model;
      let out: Float32Array | undefined;
      try {
        const params = engine.fromHost(own, r.params);
        try {
          out = engine.predict(own, params, grid.xStd, grid.n)[req.prediction ?? 0];
        } finally {
          engine.dispose(params);
        }
      } finally {
        if (own !== model) engine.disposeModel(own);
      }
      if (!out) throw new Error('No prediction selected.');
      const width = Math.max(1, Math.round(out.length / grid.n));
      // a binary model with two outputs: P(class 1) is the last column, as in the fit plots
      // a multi-class model keeps every class probability: the curve is row-major [n, k]
      const multi = first.info.kind === 'multiclass';
      const col = d === 1 ? 0 : Math.min(1, width - 1);
      const y = multi ? Float32Array.from(out) : new Float32Array(grid.n);
      let finite = true;
      if (multi) {
        for (const v of y) if (!Number.isFinite(v)) finite = false;
      } else {
        for (let j = 0; j < grid.n; j++) {
          const v = out[j * width + col] ?? Number.NaN;
          if (!Number.isFinite(v)) finite = false;
          y[j] = v;
        }
      }
      if (finite) curves.push(y);
      else diverged++;
    });
    return { ok: true, curves, diverged };
  } catch (e) {
    return {
      ok: false,
      message: e instanceof Error ? e.message : 'The copies could not be trained.',
    };
  } finally {
    if (model) engine.disposeModel(model);
  }
}
