# DRAFT upstream issue for @jax-js/jax (not posted; for the owner to post)

## Title

Memory leak: global jaxpr caches grow without bound, and `partialEvalFlat` never pops its trace frame (0.1.27)

## Summary

Repeatedly building and discarding the same model (`jit` of a `valueAndGrad` step, and a `vmap` of it) keeps
about 0.3 MB (single step) to 1 MB (with `vmap`) of JS heap alive per repetition, even after the jitted
function is disposed and `gc()` has run. 1000 repetitions retain about 1 GB. Three causes, all in `dist/index.js`:

1. `partialEvalFlat` calls `newMain(PartialEvalTrace)` and never disposes the returned object, so `traceStack`
   gets one more frame per call. Every other `newMain` call site uses `using` / `_usingCtx`.
2. `Jaxpr.getHash()` says it ignores Var identity, but it hashes `JSON.stringify(eqn.params)`. Primitives such as
   `custom_jvp`-style calls (for example `tanh_jvp_peval`) carry a nested `Jaxpr` in `params`, and its JSON includes
   Var ids, which change on every trace. The key of `jitCompileCache` is therefore different on each retrace of an
   identical function, so one new `JitProgram` is added per trace and none is ever reused or freed.
3. `vmapJaxprCache`, `jvpJaxprCache` and `transposeJaxprCache` are `Map`s keyed by the `Jaxpr` object, so every
   Jaxpr ever batched, differentiated or transposed stays reachable. They should be `WeakMap`s.

## Minimal repro

```ts
import { jit, numpy as np, valueAndGrad, nn } from '@jax-js/jax';

const w0 = np.ones([8, 3]);
for (let i = 0; i < 2000; i++) {
  // new closure each time, as when an app recompiles on every edit
  const step = jit((w, x) => valueAndGrad((w) => nn.tanh(np.matmul(x, w)).sum())(w)[1]);
  const x = np.ones([4, 8]);
  step(w0.ref, x).dispose();
  step.dispose?.();
  if (i % 500 === 0) {
    await new Promise((r) => setTimeout(r)); // V8 holds WeakRef targets until the job ends
    globalThis.gc(); // run with node --expose-gc
    console.log(i, (process.memoryUsage().heapUsed / 1e6).toFixed(1), 'MB');
  }
}
```

(Adapt to the exact API; the point is a new traced function each iteration.)

## Observed vs expected

- Observed (0.1.27): `heapUsed` after `gc()` grows linearly with the number of iterations; `jitCompileCache.size`
  grows by one per retrace; `traceStack.length` grows by one per `partialEvalFlat` call.
- Expected: flat. The same function retraced gives the same `jitCompileCache` key (cache hit), and no module-level
  cache holds a strong reference to a Jaxpr that the caller has dropped.

## Fix (what we apply locally with `pnpm patch`)

```diff
-	const trace = new PartialEvalTrace(newMain(PartialEvalTrace));
-	...
+	const main = newMain(PartialEvalTrace);
+	try { const trace = new PartialEvalTrace(main); ...; return {...}; } finally { main[Symbol.dispose](); }

-			hasher.update(JSON.stringify(eqn.params));
+			hasher.update(JSON.stringify(eqn.params, (_k, v) => v instanceof Jaxpr ? String(v.getHash()) : v));

-const vmapJaxprCache = new Map();        // same for jvpJaxprCache, transposeJaxprCache
+const vmapJaxprCache = new WeakMap();
```

The one-line fix for cause 1 alone is `using main = newMain(PartialEvalTrace);` in `partialEvalFlat`
(the pop happens after the jaxpr has been extracted). Cause 1 retains only small objects; causes 2 and 3 are the
large ones.

Also worth considering: `jitCompileCache` is still unbounded across distinct programs.
