import { readdirSync, readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { expect, test } from 'vitest';

type Params = Record<string, { W: number[]; b?: number[] }>;
type Fixture = {
  name: string;
  graph: {
    nodes: { id: string; type: string; out?: number; bias?: boolean }[];
    edges: [string, string][];
  };
  params: Params;
  batch: { x: number[]; y: number[]; n: number };
  expect: {
    pred: number[];
    loss: number;
    grads: Params;
    after?: { steps: number; params: Params; losses: number[] };
    dropoutMask?: number[][];
  };
};

function need<T>(v: T | undefined): T {
  if (v === undefined) throw new Error('missing value');
  return v;
}

const dir = fileURLToPath(new URL('./fixtures/', import.meta.url));
const files = readdirSync(dir)
  .filter((f) => f.endsWith('.json'))
  .sort();

test('fixtures exist for every listed case', () => {
  const names = files.map((f) => f.replace(/\.json$/, ''));
  for (const must of [
    'linear_nobias_d1',
    'linear_bias_d1',
    'linear_nobias_d2',
    'linear_bias_d2',
    'act_sigmoid',
    'act_tanh',
    'act_relu',
    'act_gelu',
    'link_identity',
    'link_sigmoid',
    'link_softmax',
    'loss_squared_error',
    'loss_absolute_error',
    'loss_huber_delta1',
    'loss_huber_delta0.5',
    'loss_log_loss',
    'loss_cross_entropy',
    'loss_squared_error_sigmoid',
    'penalty_l2',
    'penalty_l1',
    'feature_deg3_sin_cos_products',
    'add_skip',
    'dropout_p0.5',
    'mlp_tanh_sgd_5steps',
    'mlp_tanh_momentum_5steps',
    'mlp_tanh_adam_5steps',
    'two_param',
  ]) {
    expect(names).toContain(must);
  }
});

for (const file of files) {
  test(`${file} parses and is shape-consistent`, () => {
    const fx = JSON.parse(readFileSync(dir + file, 'utf8')) as Fixture;
    expect(fx.name + '.json').toBe(file);
    const { n } = fx.batch;
    const input = fx.graph.nodes.find((x) => x.type === 'input') as unknown as {
      columns: string[];
    };
    expect(fx.batch.x).toHaveLength(n * input.columns.length);
    expect(fx.batch.y).toHaveLength(n);
    const linears = fx.graph.nodes.filter((x) => x.type === 'linear');
    expect(Object.keys(fx.params).sort()).toEqual(linears.map((l) => l.id).sort());
    for (const l of linears) {
      const p = need(fx.params[l.id]);
      expect(p.W.length % need(l.out)).toBe(0);
      expect(p.b !== undefined).toBe(l.bias);
      if (l.bias) expect(p.b).toHaveLength(need(l.out));
      for (const set of [fx.expect.grads, fx.expect.after?.params]) {
        if (!set) continue;
        expect(need(set[l.id]).W).toHaveLength(p.W.length);
        expect(need(set[l.id]).b?.length).toBe(p.b?.length);
      }
    }
    expect(fx.expect.pred.length % n).toBe(0);
    expect(Number.isFinite(fx.expect.loss)).toBe(true);
    for (const v of [
      ...fx.expect.pred,
      ...Object.values(fx.expect.grads).flatMap((g) => [...g.W, ...(g.b ?? [])]),
    ]) {
      expect(Number.isFinite(v)).toBe(true);
    }
    if (fx.expect.after) {
      expect(fx.expect.after.losses).toHaveLength(fx.expect.after.steps);
      expect(fx.expect.after.losses[0]).toBeCloseTo(fx.expect.loss, 12);
    }
    if (fx.expect.dropoutMask) expect(fx.expect.dropoutMask).toHaveLength(n);
  });
}
