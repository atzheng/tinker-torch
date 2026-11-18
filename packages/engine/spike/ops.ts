// Operation checks against hand-computed values (spike item 2). Each check is isolated: a thrown
// error or a mismatch is recorded, never fatal. Every check disposes the arrays it creates.
import './probe.ts'; // must be the first import (see probe.ts)
import { grad, jit, nn, numpy as np, random, tree, valueAndGrad, vmap } from '@jax-js/jax';
import type { Array as JArray } from '@jax-js/jax';
import * as optax from '@jax-js/optax';
import { liveArrays } from './probe.ts';
import { adamExact, initParams, makeManyTrainer, makeTrainer, sineData } from './lib.ts';

/** `known` marks a check that documents a jax-js defect: pass=false is then the expected outcome. */
export type OpResult = { name: string; pass: boolean; detail: string; known?: boolean };

/** Read an array to a plain number list. Consumes `a`. */
function vals(a: JArray): number[] {
  return Array.from(a.dataSync() as ArrayLike<number>, Number);
}

type Check = { got: number[]; want: number[]; tol?: number; note?: string };

function run(name: string, fn: () => Check, known = false): OpResult {
  const before = liveArrays();
  try {
    const { got, want, tol = 1e-5, note } = fn();
    const bad = got.length !== want.length || got.some((g, i) => !(Math.abs(g - want[i]!) <= tol));
    const leak = liveArrays() - before;
    const fmt = (xs: number[]) => '[' + xs.map((x) => +x.toPrecision(6)).join(', ') + ']';
    return {
      name,
      known,
      pass: !bad && leak === 0,
      detail:
        `got ${fmt(got)} want ${fmt(want)}` +
        (note ? ` (${note})` : '') +
        (leak !== 0 ? ` LEAKED ${leak} buffers` : ''),
    };
  } catch (e) {
    return {
      name,
      known,
      pass: false,
      detail: 'threw: ' + (e instanceof Error ? e.message : String(e)),
    };
  }
}

export function checkOps(): OpResult[] {
  const out: OpResult[] = [];
  const add = (name: string, fn: () => Check) => out.push(run(name, fn));
  /** A check that records a defect in a built-in: expected to fail; its workaround follows it. */
  const defect = (name: string, fn: () => Check) => out.push(run('DEFECT ' + name, fn, true));

  add('softmax [1,2,3]', () => ({
    got: vals(nn.softmax(np.array([1, 2, 3]))),
    want: [0.09003057, 0.24472847, 0.66524096],
  }));
  add('softmax stable at [1000,1001]', () => ({
    got: vals(nn.softmax(np.array([1000, 1001]))),
    want: [0.26894142, 0.73105858],
  }));
  add('softmax axis=0 on [2,2]', () => ({
    got: vals(
      nn.softmax(
        np.array([
          [0, 0],
          [Math.log(3), 0],
        ]),
        0,
      ),
    ),
    want: [0.25, 0.5, 0.75, 0.5],
  }));
  add('logSoftmax [1,2,3]', () => ({
    got: vals(nn.logSoftmax(np.array([1, 2, 3]))),
    want: [-2.4076059, -1.4076059, -0.4076059],
  }));
  add('logsumexp [1,2,3]', () => ({
    got: vals(nn.logsumexp(np.array([1, 2, 3]))),
    want: [3.4076059],
  }));
  add('logsumexp axis=1 stable [[1000,1000],[0,0]]', () => ({
    got: vals(
      nn.logsumexp(
        np.array([
          [1000, 1000],
          [0, 0],
        ]),
        1,
      ),
    ),
    want: [1000.693147, 0.693147],
    tol: 1e-3,
  }));
  add('sigmoid [-2,0,2]', () => ({
    got: vals(nn.sigmoid(np.array([-2, 0, 2]))),
    want: [0.11920292, 0.5, 0.88079708],
  }));
  add('sigmoid extremes [-100,100] finite', () => ({
    got: vals(nn.sigmoid(np.array([-100, 100]))),
    want: [0, 1],
  }));
  add('gelu default (tanh form) at ±1', () => ({
    got: vals(nn.gelu(np.array([1, -1]))),
    want: [0.841192, -0.158808],
    note: 'default is approximate=true (tanh); torch nn.GELU() default is exact',
  }));
  add('gelu {approximate:false} (exact erf form) at ±1', () => ({
    got: vals(nn.gelu(np.array([1, -1]), { approximate: false })),
    want: [0.8413447, -0.1586553],
    tol: 2e-5,
  }));
  add('relu [-1,0,2]', () => ({
    got: vals(nn.relu(np.array([-1, 0, 2]))),
    want: [0, 0, 2],
  }));
  defect('nn.relu gradient at [-1,0,2] (torch: 0 at 0)', () => ({
    got: vals(grad((x: JArray) => nn.relu(x).sum())(np.array([-1, 0, 2]))),
    want: [0, 0, 1],
  }));
  add('relu via where(x>0,x,0): gradient at [-1,0,2] matches torch', () => ({
    got: vals(grad((x: JArray) => np.where(x.ref.greater(0), x, 0).sum())(np.array([-1, 0, 2]))),
    want: [0, 0, 1],
  }));
  add('abs [-3,0,2]', () => ({
    got: vals(np.abs(np.array([-3, 0, 2]))),
    want: [3, 0, 2],
  }));
  defect('np.abs gradient at [-3,0,2] (torch: 0 at 0)', () => ({
    got: vals(grad((x: JArray) => np.abs(x).sum())(np.array([-3, 0, 2]))),
    want: [-1, 0, 1],
  }));
  add('abs via x*sign(x): gradient at [-3,0,2] matches torch', () => ({
    got: vals(grad((x: JArray) => x.ref.mul(np.sign(x)).sum())(np.array([-3, 0, 2]))),
    want: [-1, 0, 1],
  }));
  add('where(c, a, b)', () => ({
    got: vals(np.where(np.array([true, false, true]), np.array([1, 2, 3]), np.array([10, 20, 30]))),
    want: [1, 20, 3],
  }));
  add('where gradient: d/dx where(x>0, x^2, 0) at [-1,2]', () => ({
    got: vals(
      grad((x: JArray) => np.where(x.ref.greater(0), np.square(x), 0).sum())(np.array([-1, 2])),
    ),
    want: [0, 4],
  }));
  add('isfinite [1,NaN,Inf,-Inf]', () => ({
    got: vals(np.isfinite(np.array([1, NaN, Infinity, -Infinity])).astype(np.float32)),
    want: [1, 0, 0, 0],
  }));
  add('mean over axis', () => ({
    got: vals(
      np.mean(
        np.array([
          [1, 2],
          [3, 6],
        ]),
        1,
      ),
    ),
    want: [1.5, 4.5],
  }));
  add('var_ of [1,2,3,4] (population, as torch layer_norm)', () => ({
    got: vals(np.var_(np.array([1, 2, 3, 4]))),
    want: [1.25],
  }));
  add('var_ with correction=1', () => ({
    got: vals(np.var_(np.array([1, 2, 3, 4]), undefined, { correction: 1 })),
    want: [1.6666667],
  }));
  add('layer norm composed from mean/var_, row [1,2,3,4]', () => {
    const x = np.array([[1, 2, 3, 4]]);
    const m = x.ref.mean(1, { keepdims: true });
    const v = np.var_(x.ref, 1, { keepdims: true });
    return {
      got: vals(x.sub(m).div(np.sqrt(v.add(1e-5)))),
      want: [-1.3416355, -0.4472118, 0.4472118, 1.3416355],
      tol: 1e-4,
    };
  });
  add('nn.standardize [1,2,3,4] (eps 1e-5)', () => ({
    got: vals(nn.standardize(np.array([1, 2, 3, 4]))),
    want: [-1.3416355, -0.4472118, 0.4472118, 1.3416355],
    tol: 1e-4,
  }));
  add('oneHot([0,2,1], 3)', () => ({
    got: vals(nn.oneHot(np.array([0, 2, 1], { dtype: np.int32 }), 3).astype(np.float32)),
    want: [1, 0, 0, 0, 0, 1, 0, 1, 0],
  }));
  add('take rows (embedding lookup) axis=0', () => ({
    got: vals(
      np.take(
        np.array([
          [1, 2],
          [3, 4],
          [5, 6],
        ]),
        np.array([2, 0, 2], { dtype: np.int32 }),
        0,
      ),
    ),
    want: [5, 6, 1, 2, 5, 6],
  }));
  add('take gradient accumulates duplicates (idx [0,0,1])', () => ({
    got: vals(
      grad((w: JArray) => np.take(w, np.array([0, 0, 1], { dtype: np.int32 }), 0).sum())(
        np.array([
          [1, 2],
          [3, 4],
          [5, 6],
        ]),
      ),
    ),
    want: [2, 2, 1, 1, 0, 0],
  }));
  add('argmax -> int32 dataSync', () => {
    const a = np.argmax(
      np.array([
        [0.1, 0.9],
        [0.8, 0.2],
      ]),
      1,
    );
    const d = a.dataSync();
    return {
      got: Array.from(d as ArrayLike<number>, Number),
      want: [1, 0],
      note: d.constructor.name,
    };
  });
  add('cross-entropy composed: -sum(onehot * logSoftmax)', () => {
    const logits = np.array([[1, 2, 3]]);
    const t = nn.oneHot(np.array([2], { dtype: np.int32 }), 3);
    return { got: vals(nn.logSoftmax(logits).mul(t).sum().neg()), want: [0.4076059] };
  });
  // Note: a 1-element array takes a different (more accurate) path than a vector, so these use 2+.
  defect('nn.softplus([1,100]) overflows to Infinity (float32 exp), vector path only', () => ({
    got: vals(nn.softplus(np.array([1, 100]))),
    want: [1.3132617, 100],
    tol: 1e-4,
  }));
  // softplus(x) = logsumexp([x, 0]): stable, smooth (no kink at 0, so the gradient at 0 is 0.5).
  const softplusStable = (x: JArray) => nn.logsumexp(np.stack([x.ref, np.zerosLike(x)]), 0);
  add('BCE-with-logits as logsumexp([x,0]) - x*y, x=[-100,0,100] y=[1,1,0]', () => {
    const x = np.array([-100, 0, 100]);
    const y = np.array([1, 1, 0]);
    const l = softplusStable(x.ref).sub(x.mul(y));
    return { got: vals(l), want: [100, 0.6931472, 100], tol: 1e-3 };
  });
  add('BCE-with-logits gradient at x=[-100,0,100] y=[1,1,0] is sigmoid(x)-y', () => ({
    got: vals(
      grad((x: JArray) => {
        const y = np.array([1, 1, 0]);
        return softplusStable(x.ref).sub(x.mul(y)).sum();
      })(np.array([-100, 0, 100])),
    ),
    want: [-1, -0.5, 1],
    tol: 1e-4,
  }));
  defect('nn.sigmoid gradient at x=-100 is NaN', () => ({
    got: vals(grad((x: JArray) => nn.sigmoid(x).sum())(np.array([-100, 0, 100]))),
    want: [0, 0.25, 0],
  }));
  add('sigmoid as 0.5*(1+tanh(x/2)): gradient finite at ±100', () => ({
    got: vals(
      grad((x: JArray) => np.tanh(x.mul(0.5)).add(1).mul(0.5).sum())(np.array([-100, 0, 100])),
    ),
    want: [0, 0.25, 0],
  }));
  add('huber composed with where (delta 1) at [0.5,3]', () => {
    const r = np.array([0.5, 3]);
    const a = np.abs(r.ref);
    const l = np.where(a.ref.lessEqual(1), np.square(r).mul(0.5), a.sub(0.5));
    return { got: vals(l), want: [0.125, 2.5] };
  });

  // Random numbers.
  add('random.bernoulli: same key gives the same mask', () => {
    const k = random.key(7);
    const a = vals(random.bernoulli(k.ref, 0.5, [8]).astype(np.float32));
    const b = vals(random.bernoulli(k, 0.5, [8]).astype(np.float32));
    return { got: a, want: b };
  });
  add('random.split gives different, reproducible keys', () => {
    const [k1, k2] = np.unstack(random.split(random.key(1), 2));
    const a = vals(random.uniform(k1!, [1]));
    const b = vals(random.uniform(k2!, [1]));
    const [j1, j2] = np.unstack(random.split(random.key(1), 2));
    j2!.dispose();
    const c = vals(random.uniform(j1!, [1]));
    return { got: [a[0] !== b[0] ? 1 : 0, a[0] === c[0] ? 1 : 0], want: [1, 1] };
  });
  add('random.bernoulli(p=0.7) mean over 20000', () => ({
    got: vals(random.bernoulli(random.key(3), 0.7, [20000]).astype(np.float32).mean()),
    want: [0.7],
    tol: 0.01,
  }));
  add('dropout mask: keep prob 0.8, scaled 1/(1-p); mean of output ~1', () => {
    const p = 0.2;
    const x = np.ones([20000]);
    const mask = random.bernoulli(random.key(5), 1 - p, [20000]);
    return { got: vals(np.where(mask, x.div(1 - p), 0).mean()), want: [1], tol: 0.02 };
  });
  add('bernoulli inside jit with key as argument', () => {
    const f = jit((key: JArray, x: JArray) =>
      np.where(random.bernoulli(key, 0.5, [x.shape[0]!]), x, 0),
    );
    const k = random.key(1);
    const r1 = vals(f(k.ref, np.ones([16])));
    const r2 = vals(f(k, np.ones([16])));
    f.dispose();
    return { got: r1, want: r2, note: 'same key, same mask' };
  });
  add('normal init: mean ~0, std ~1', () => {
    const x = random.normal(random.key(9), [20000]);
    const m = vals(x.ref.mean())[0]!;
    const s = vals(np.std(x))[0]!;
    return { got: [m, s], want: [0, 1], tol: 0.03 };
  });

  // Transformations.
  add('valueAndGrad over a nested object (arrays inside objects)', () => {
    const f = (p: { a: { w: JArray }; list: JArray[] }) =>
      p.a.w.mul(p.list[0]!).sum().add(p.list[1]!.sum());
    const [v, g] = valueAndGrad(f)({
      a: { w: np.array([1, 2]) },
      list: [np.array([3, 4]), np.array([5, 6])],
    });
    return {
      got: [...vals(v), ...vals(g.a.w), ...vals(g.list[0]!), ...vals(g.list[1]!)],
      want: [11 + 11, 3, 4, 1, 2, 1, 1],
    };
  });
  add('valueAndGrad with hasAux-free scalar loss: d/dw mean((w*x-y)^2)', () => {
    const f = (w: JArray, x: JArray, y: JArray) => np.square(w.mul(x).sub(y)).mean();
    const [v, g] = valueAndGrad(f)(np.array(2), np.array([1, 2]), np.array([1, 1]));
    // mean((2-1)^2,(4-1)^2)=5; grad = mean(2(wx-y)x)=mean(2*1*1, 2*3*2)=7
    return { got: [...vals(v), ...vals(g)], want: [5, 7] };
  });
  add('vmap over a function with per-copy args', () => {
    const f = vmap((a: JArray, b: JArray) => a.mul(b).sum(), 0);
    return {
      got: vals(
        f(
          np.array([
            [1, 2],
            [3, 4],
          ]),
          np.array([
            [1, 1],
            [2, 2],
          ]),
        ),
      ),
      want: [3, 14],
    };
  });
  add('vmap(grad) per-sample gradients', () => {
    const f = vmap(
      grad((w: JArray, x: JArray) => np.square(w.mul(x)).sum()),
      [null, 0],
    );
    // d/dw (w x)^2 = 2 w x^2 at w=3: x=1 -> 6, x=2 -> 24
    return { got: vals(f(np.array(3), np.array([1, 2]))), want: [6, 24] };
  });

  // Optimizers: one step against hand values. params [1], grad of 0.5*w^2 is w.
  const optStep = (opt: optax.GradientTransformation, n: number) => {
    let w = np.array([1]);
    let st = opt.init(tree.ref(w));
    for (let i = 0; i < n; i++) {
      const g = tree.ref(w); // gradient of 0.5 w^2 at w
      const [u, s] = opt.update(g, st);
      st = s;
      w = optax.applyUpdates(w, u);
    }
    tree.dispose(st);
    return vals(w);
  };
  add('optax.sgd(0.1), 1 step from w=1 (0.9)', () => ({
    got: optStep(optax.sgd(0.1), 1),
    want: [0.9],
  }));
  add('optax.sgd(0.1, momentum 0.9), 2 steps (torch: v=g+0.9v)', () => {
    // step1: v=1, w=1-0.1*1=0.9; step2: g=0.9, v=0.9+0.9*1=1.8, w=0.9-0.18=0.72
    return { got: optStep(optax.sgd(0.1, { momentum: 0.9 }), 2), want: [0.72] };
  });
  const adamRef = (steps: number, lr = 0.1) => {
    let w = 1,
      m = 0,
      v = 0;
    for (let t = 1; t <= steps; t++) {
      const g = w; // gradient of 0.5 w^2, double precision, torch.optim.Adam recipe
      m = 0.9 * m + 0.1 * g;
      v = 0.999 * v + 0.001 * g * g;
      w -= (lr * (m / (1 - 0.9 ** t))) / (Math.sqrt(v / (1 - 0.999 ** t)) + 1e-8);
    }
    return w;
  };
  defect('optax.adam(0.1) after 1 step is 0.9 (float32 pow cancellation)', () => ({
    got: optStep(optax.adam(0.1, { b1: 0.9, b2: 0.999, eps: 1e-8 }), 1),
    want: [adamRef(1)],
    tol: 1e-5,
  }));
  add('adamExact(0.1) after 1 step', () => ({
    got: optStep(adamExact(0.1), 1),
    want: [adamRef(1)],
    tol: 1e-6,
  }));
  add('adamExact(0.1) after 2 and 50 steps vs double-precision torch recipe', () => ({
    got: [optStep(adamExact(0.1), 2)[0]!, optStep(adamExact(0.1), 50)[0]!],
    want: [adamRef(2), adamRef(50)],
    tol: 1e-5,
  }));

  // Whole-step checks.
  add('vmapped training step: 3 copies equal 3 separate trainings after 30 steps', () => {
    const k = 3;
    const sizes = [1, 4, 1];
    const datas = Array.from({ length: k }, (_, i) => sineData(20, 100 + i));
    const seeds = [1, 2, 3];
    const many = makeManyTrainer(sizes, true, 'adam', 0.02, datas, seeds);
    many.run(30);
    const stacked = many.read();
    many.dispose();
    const got: number[] = [];
    const want: number[] = [];
    for (let i = 0; i < k; i++) {
      const one = makeTrainer(sizes, true, 'adam', 0.02, datas[i]!, seeds[i]!);
      one.run(30);
      const r = one.read();
      one.dispose();
      for (const id of Object.keys(r)) {
        const W = stacked[id]!.W;
        const per = W.length / k;
        got.push(...W.slice(i * per, (i + 1) * per));
        want.push(...r[id]!.W);
      }
    }
    return { got, want, tol: 1e-4, note: 'jit(vmap(jit step)) vs one trainer each' };
  });
  add('NaN loss is detectable after dataSync', () => {
    const l = np.log(np.array([-1]));
    const v = vals(l)[0]!;
    return { got: [Number.isFinite(v) ? 1 : 0], want: [0] };
  });
  add('initParams is keyed by node id and leak-free', () => {
    const p = initParams([1, 3, 1], 1);
    const ids = Object.keys(p);
    tree.dispose(p);
    return { got: [ids.length], want: [2], note: ids.join(',') };
  });

  return out;
}
