// Compares the engine with the PyTorch fixtures (CONTRACTS.md section 11). Fixtures whose pieces are
// not implemented yet are skipped by name; the bead that implements them moves the name from PENDING
// to IMPLEMENTED. Gradients are not part of the Engine interface, so they are read from one SGD
// step with learning rate 1: grad = (W before - W after).
import { beforeAll, describe, expect, test } from 'vitest';
import { debugTrainPass } from '../src/compile.js';
import { peekParams } from '../src/convert.js';
import { createEngine } from '../src/index.js';
import type { Engine } from '../src/index.js';
import { compare, fixtureNames, loadFixture, toBatch, toHostParams, withLr } from './util.js';
import type { JsonParams } from './util.js';

/** Compared now. Value: tolerance overrides (CONTRACTS.md section 11, "Engine exceptions"). */
const IMPLEMENTED: Record<string, { tol?: number }> = {
  linear_nobias_d1: {},
  linear_bias_d1: {},
  linear_nobias_d2: {},
  linear_bias_d2: {},
  act_sigmoid: {},
  act_tanh: {},
  act_relu: {},
  act_gelu: { tol: 1e-4 },
  link_identity: {},
  link_sigmoid: {},
  loss_squared_error: {},
  loss_squared_error_sigmoid: {},
  two_param: {},
  mlp_tanh_sgd_5steps: {},
  feature_deg3_sin_cos_products: {},
  add_skip: {},
  'dropout_p0.5': {},
};

/** Skipped until the named bead lands. */
const PENDING: Record<string, string> = {
  link_softmax: 'tinker-hj4.2 (needs cross_entropy)',
  loss_absolute_error: 'tinker-hj4.2',
  loss_huber_delta1: 'tinker-hj4.2',
  'loss_huber_delta0.5': 'tinker-hj4.2',
  loss_log_loss: 'tinker-hj4.2',
  loss_cross_entropy: 'tinker-hj4.2',
  penalty_l2: 'tinker-hj4.2',
  penalty_l1: 'tinker-hj4.2',
  mlp_tanh_momentum_5steps: 'tinker-hj4.4',
  mlp_tanh_adam_5steps: 'tinker-hj4.4 (compare at 5e-3)',
};

/** Never compared (owner decision, tinker-e1h.2): they pin torch's at-zero subgradients. */
const SKIPPED = ['act_relu_zero_preact', 'penalty_l1_zero_weight'];

function flat(p: JsonParams, ids: string[]): number[] {
  return ids.flatMap((id) => [...(p[id]?.W ?? []), ...(p[id]?.b ?? [])]);
}

let engine: Engine;
beforeAll(async () => {
  engine = await createEngine();
});

test('every fixture is accounted for', () => {
  const all = new Set([...Object.keys(IMPLEMENTED), ...Object.keys(PENDING), ...SKIPPED]);
  expect(fixtureNames().sort()).toEqual([...all].sort());
});

describe('fixtures', () => {
  for (const name of fixtureNames()) {
    if (SKIPPED.includes(name) || name in PENDING) {
      test.skip(`${name} (${PENDING[name] ?? 'skipped by decision'})`, () => {});
      continue;
    }
    test(name, () => {
      const tol = IMPLEMENTED[name]?.tol ?? 1e-5;
      const { fx, graph, data } = loadFixture(name);
      const ids = Object.keys(fx.params);
      const batch = toBatch(fx);
      const report: string[] = [];

      const model = engine.compile(graph, data);
      const params = engine.fromHost(model, toHostParams(fx.params));

      // Dropout cases: pred, loss and grads are the dropout-ON pass with the fixture's mask
      // (injected through the test-only debugTrainPass); predict/evaluate must give the dropout-off values.
      const mask = fx.expect.dropoutMask;
      if (mask) {
        const dropId = graph.nodes.find((n) => n.type === 'dropout')?.id ?? '';
        const on = debugTrainPass(model, peekParams(params), batch, { [dropId]: mask });
        const cp = compare(on.pred, fx.expect.pred, tol);
        const cl = compare([on.loss], [fx.expect.loss], tol);
        const gotG = ids.flatMap((id) => [...(on.grads[id]?.W ?? []), ...(on.grads[id]?.b ?? [])]);
        const cg = compare(gotG, flat(fx.expect.grads, ids), tol);
        const pe = engine.predict(model, params, batch.x, batch.n);
        const cpe = compare(pe, fx.expect.predEval ?? [], tol);
        const cle = compare(
          [engine.evaluate(model, params, batch)],
          [fx.expect.lossEval ?? NaN],
          tol,
        );
        console.log(
          `[ref] ${name} tol ${tol}: pred ${cp.maxAbs.toExponential(2)}, loss ${cl.maxAbs.toExponential(2)}, grads ${cg.maxAbs.toExponential(2)}, predEval ${cpe.maxAbs.toExponential(2)}, lossEval ${cle.maxAbs.toExponential(2)}`,
        );
        expect(cp.ok, `pred max abs error ${cp.maxAbs}`).toBe(true);
        expect(cl.ok, `loss ${on.loss} vs ${fx.expect.loss}`).toBe(true);
        expect(cg.ok, `grads max abs error ${cg.maxAbs}`).toBe(true);
        expect(cpe.ok, `predEval max abs error ${cpe.maxAbs}`).toBe(true);
        expect(cle.ok, `lossEval max abs error ${cle.maxAbs}`).toBe(true);
        engine.dispose(params);
        return;
      }

      // pred and loss
      const pred = engine.predict(model, params, batch.x, batch.n);
      const cp = compare(pred, fx.expect.pred, tol);
      report.push(`pred ${cp.maxAbs.toExponential(2)}`);
      expect(cp.ok, `pred max abs error ${cp.maxAbs}`).toBe(true);
      const loss = engine.evaluate(model, params, batch);
      const cl = compare([loss], [fx.expect.loss], tol);
      report.push(`loss ${cl.maxAbs.toExponential(2)}`);
      expect(cl.ok, `loss ${loss} vs ${fx.expect.loss}`).toBe(true);

      // grads, from one step with lr 1 (borrows `params`; startTraining consumes a clone)
      const m1 = engine.compile(withLr(graph, 1), data);
      const p1 = engine.fromHost(m1, toHostParams(fx.params));
      const s1 = engine.startTraining(m1, p1, 0);
      const r1 = engine.step(m1, s1, batch);
      expect(r1.diverged).toBe(false);
      const after = engine.toHost(r1.state.params);
      const grads = ids.flatMap((id) => {
        const before = toHostParams(fx.params)[id];
        const now = after[id];
        if (!before || !now) return [];
        return [
          ...Array.from(before.W, (w, i) => w - (now.W[i] ?? NaN)),
          ...(before.b ? Array.from(before.b, (w, i) => w - (now.b?.[i] ?? NaN)) : []),
        ];
      });
      const cg = compare(grads, flat(fx.expect.grads, ids), tol);
      report.push(`grads ${cg.maxAbs.toExponential(2)}`);
      expect(cg.ok, `grads max abs error ${cg.maxAbs}`).toBe(true);
      engine.dispose(r1.state);

      // optimizer cases
      const exp = fx.expect.after;
      if (exp) {
        const st0 = engine.startTraining(model, engine.fromHost(model, toHostParams(fx.params)), 0);
        let state = st0;
        const losses: number[] = [];
        for (let k = 0; k < exp.steps; k++) {
          const r = engine.step(model, state, batch);
          expect(r.diverged).toBe(false);
          losses.push(r.loss);
          state = r.state;
        }
        const tolAfter = 1e-4;
        const cL = compare(losses, exp.losses, tolAfter);
        const host = engine.toHost(state.params);
        const got = ids.flatMap((id) => [...(host[id]?.W ?? []), ...(host[id]?.b ?? [])]);
        const cA = compare(got, flat(exp.params, ids), tolAfter);
        report.push(
          `after.losses ${cL.maxAbs.toExponential(2)}`,
          `after.params ${cA.maxAbs.toExponential(2)}`,
        );
        expect(cL.ok, `after.losses max abs error ${cL.maxAbs}`).toBe(true);
        expect(cA.ok, `after.params max abs error ${cA.maxAbs}`).toBe(true);
        engine.dispose(state);
      }
      engine.dispose(params);
      console.log(`[ref] ${name} tol ${tol}: ${report.join(', ')}`);
    });
  }
});
