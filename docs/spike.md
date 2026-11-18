# jax-js spike (bead tinker-e1h.1)

Question: can `@jax-js/jax` 0.1.27 and `@jax-js/optax` 0.1.3 carry version 1 of the tool?

**Recommendation: GO**, with the workarounds listed below built into the engine from the start. All four gate thresholds pass with a wide margin on the machine used. Five library defects were found; each has a verified workaround, and none blocks a piece. One pitfall (lazy evaluation across steps) shapes the engine's step loop and is the main thing the implementer must get right.

| Gate                           | Threshold  | Measured                         | Result |
| ------------------------------ | ---------- | -------------------------------- | ------ |
| MLP step, n = 40, Chrome       | under 2 ms | 0.18 ms (adam), 0.10 ms (sgd)    | pass   |
| trainMany, k = 20 x 1000 steps | under 3 s  | 0.48 s (n = 40 each)             | pass   |
| Leak, 10,000 steps             | flat       | flat in 7 loops, Node and Chrome | pass   |
| Every op present or composable | yes        | yes, 5 defects with workarounds  | pass   |

Not covered: Firefox, Safari, real tablets, WebGPU. Numbers below are one Apple M2 Pro laptop (10 cores), headless Chrome 157, Node 23.1, so a touch tablet will be slower by some factor that nobody has measured. See "What was not measured".

## How to run

Everything lives in `packages/engine/spike/`. Needs Node 22.6 or newer for the Node scripts (they run the `.ts` files directly).

```bash
pnpm install
pnpm --filter @tinker/engine spike                  # all four parts in Node, then one JSON line
pnpm --filter @tinker/engine spike --only=ops       # one part: train | ops | bench | leak
pnpm --filter @tinker/engine spike --quick          # shorter timing blocks and 1,000-step leak loops
pnpm --filter @tinker/engine spike:web              # bare Vite page on http://localhost:5199/
SPIKE_ISOLATED=1 pnpm --filter @tinker/engine spike:web   # same page with COOP/COEP headers
pnpm test                                           # includes spike/spike.test.ts (ops, training, 1,000-step leak)
```

In the page, open `/?auto=1` to run everything on load, `&only=bench` for one part, `&quick=1` to shorten it. The result object is left in `window.__spike` and `window.__spikeDone` turns true at the end. The Chrome numbers here came from headless Chrome driven by `playwright-core` installed in a temp directory outside the repo, loading `http://localhost:5199/?auto=1&only=<part>`.

Files: `probe.ts` (live-buffer probe), `lib.ts` (MLP over a params object keyed by node id, jitted step, vmapped many-trainer, exact Adam), `ops.ts` (operation checks), `train.ts`, `bench.ts`, `leak.ts`, `node.ts` (Node entry), `web/` (page), `spike.test.ts`.

## 1. Training works

Params are `{ lin1: { W, b }, lin2: { W, b } }` (keyed by node id). The step is `jit((params, optState, x, y) => [newParams, newOptState, loss])` around `valueAndGrad(loss)(params, x, y)`, `opt.update` and `applyUpdates`.

Linear regression on 40 points, y = 2x + 1 plus noise. All three optimizers reach the closed-form least-squares line (W 2.0113, b 0.9974, loss 0.00405) to four digits.

1-16-1 tanh MLP on sine, n = 40, loss before the update at the given step:

| Optimizer                 | Steps | 1      | 500    | 1000   | 2000   | 4000   |
| ------------------------- | ----- | ------ | ------ | ------ | ------ | ------ |
| sgd lr 0.05               | 4000  | 0.3268 |        | 0.0102 | 0.0055 | 0.0024 |
| sgd momentum 0.9, lr 0.02 | 2000  | 0.3268 | 0.0057 | 0.0025 | 0.0013 |        |
| adam lr 0.01              | 2000  | 0.3268 | 0.0018 | 0.0008 | 0.0006 |        |

The loss is monotone at the checkpoints and ends under 0.02 in all cases (asserted in `spike.test.ts`). Optimizer one-step and two-step values match hand computation: `optax.sgd` exactly; `optax.sgd(lr, { momentum: 0.9 })` equals torch SGD with momentum 0.9 (v = g + 0.9 v). `optax.adam` does **not** match torch (defect 5).

## 2. Operations

Each check compares against a hand-computed value and also checks that it leaked no buffer. 53 checks (5 of them record defects); run them with `--only=ops`. The same results come out in Node and in Chrome (plain and cross-origin isolated).

| Operation                                                                           | Behaves?                              | Notes                                                                                                                                                                                                                                            |
| ----------------------------------------------------------------------------------- | ------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `nn.softmax`, `nn.logSoftmax`, `nn.logsumexp`                                       | yes                                   | Stable at [1000, 1001]; `axis` works. `logsumexp` with no axis reduces over everything.                                                                                                                                                          |
| `nn.sigmoid` forward                                                                | yes                                   | Finite at +-100. **Gradient is NaN for x below about -88 (defect 3).**                                                                                                                                                                           |
| `nn.gelu`                                                                           | yes                                   | Default is `approximate: true`, the tanh form (gelu(1) = 0.841192). `{ approximate: false }` is the exact erf form (0.841345), which is what `torch.nn.GELU()` computes. **The engine must pass `approximate: false`** to match torch's default. |
| `nn.relu`                                                                           | forward yes                           | **Gradient at exactly 0 is 0.5, torch gives 0 (defect 1).**                                                                                                                                                                                      |
| `np.abs`                                                                            | forward yes                           | **Gradient at exactly 0 is 1, torch gives 0 (defect 2).**                                                                                                                                                                                        |
| `np.where`                                                                          | yes                                   | Forward and gradient (value taken from the selected branch only).                                                                                                                                                                                |
| `np.isfinite`                                                                       | yes                                   | `[1, NaN, Inf, -Inf]` gives `[1, 0, 0, 0]`. The engine can also just test the read-back loss with `Number.isFinite`.                                                                                                                             |
| mean, `np.var_`                                                                     | yes                                   | `var_` is population variance by default (torch layer_norm convention); `{ correction: 1 }` gives the sample variance. Layer norm composed from `mean` and `var_` and `nn.standardize` both match. The function is named `var_`, not `var`.      |
| `nn.oneHot`                                                                         | yes                                   | Needs an int32 array: `np.array(idx, { dtype: np.int32 })`.                                                                                                                                                                                      |
| `np.take(W, idx, 0)`                                                                | yes                                   | Embedding lookup. Gradient scatter-adds duplicate indices correctly.                                                                                                                                                                             |
| `random.key`, `random.split`, `random.bernoulli`, `random.uniform`, `random.normal` | yes                                   | Same key gives the same mask; works inside `jit` with the key as an argument and as part of the carried state (dropout). Mean of `bernoulli(0.7)` over 20,000 draws is 0.702.                                                                    |
| `valueAndGrad` over a nested object                                                 | yes                                   | Objects and arrays of arrays, including mixed.                                                                                                                                                                                                   |
| `vmap` over a whole training step                                                   | yes                                   | k copies, each with its own data and params; 3 copies match 3 separate trainings to 1e-4 after 30 steps.                                                                                                                                         |
| `jit(vmap(step))`                                                                   | yes                                   | The step is itself jitted; nesting works.                                                                                                                                                                                                        |
| `dataSync()`                                                                        | yes                                   | Returns `Float32Array` for float32 and `Int32Array` for int32. It consumes the array.                                                                                                                                                            |
| `nn.softplus`                                                                       | forward yes, **overflows (defect 4)** | `softplus([1, 100])` gives Infinity for 100.                                                                                                                                                                                                     |

### Defects and workarounds

Each is a `DEFECT` line in the ops output (expected to fail), followed by a passing check for the workaround.

1. **`nn.relu` and `np.maximum(x, 0)` gradient at 0 is 0.5.** Use `np.where(x.greater(0), x, 0)`: gradient 0 at 0 as in torch.
2. **`np.abs` gradient at 0 is 1.** Use `x.mul(np.sign(x))` (gradient 0 at 0, as torch's L1). Matters for the L1 penalty and Huber.
3. **`nn.sigmoid` gradient is NaN for large negative inputs** (x = -100, -89 in a vector). Use `0.5 * (1 + tanh(x / 2))`: value and gradient are finite at +-100. Only matters for a Sigmoid hidden activation, because the fused output path below never differentiates through it.
4. **`nn.softplus` and `nn.logSigmoid` overflow in float32** (`softplus(x)` is Infinity for x above about 88; `logSigmoid(-100)` is -Infinity). So BCE with logits must not be written as `softplus(x) - x * y`. Use `nn.logsumexp(np.stack([x, zeros]), 0) - x * y`: finite at +-100, gradient equal to sigmoid(x) - y including at 0 (smooth, no kink). Verified at x = [-100, 0, 100].
5. **`optax.adam` 0.1.3 is inexact in early steps.** It computes `1 - pow(b2, t)` in float32, which cancels: the first step is 0.100609 instead of 0.1 (0.6% too large), and after 2 steps w is 0.7992 instead of 0.8004; the error decays roughly as 1/t. Fixed in `lib.ts` as `adamExact`: carry `bc = 1 - b^t` in the state via `bc' = b * bc + (1 - b)`, which has no cancellation. It matches a double-precision torch recipe to 1e-6 (1, 2 and 50 steps). It is the same `GradientTransformation` interface, so the step function does not change. `optax.sgd` with and without momentum needs no fix. Note (tinker-hj4.4): the inaccuracy is eager-only. Inside jit, which is how the engine runs optax, `optax.adam` matches torch to about 1e-6 over 5 steps, so the engine uses it as is.

Also noted:

- A one-element array takes a different, more accurate numeric path than a vector: `softplus([100])` is fine, `softplus([1, 100])` is not. Test numerics on vectors only.
- Destructuring a jax `Array` (`const [a, b] = random.split(key, 2)`) iterates it and leaks the parent buffer. Use `np.unstack(random.split(key, 2))`, which consumes the parent. Found by the leak probe; `initParams` leaked 7 buffers until fixed.
- `optax.*.update(updates, state, params?)` **disposes `params`** when given. We do not pass it.
- jax-js keeps a few constant buffers in library-global caches (for example the jitted `random.uniform`). They appear the first time a function is used and then stay constant, so a leak test must warm up first.
- Ownership as documented held everywhere: functions consume their array arguments, `.ref` adds a reference, `jit` and `vmap` consume the arguments of the function they wrap, and `tree.ref` / `tree.dispose` handle whole params trees.

### The lazy-evaluation pitfall (important)

jax-js is lazy. If a loop calls the jitted step repeatedly and never reads anything, step i + 1 is built on top of the unevaluated outputs of step i, and the cost per step grows with the length of the chain. Measured on k = 20 copies, n = 40, same workload, reading the loss every N steps:

| Loss read every                  | Node ms/step                                                               | Chrome ms/step |
| -------------------------------- | -------------------------------------------------------------------------- | -------------- |
| 1 step                           | 0.60                                                                       | 0.51           |
| 25 steps                         | 1.19                                                                       | 0.82           |
| 100 steps                        | 3.06                                                                       | 2.19           |
| 1000 steps (one read at the end) | 44 (and "Allocation exceeds the 4 GiB memory32 limit" on the n = 500 case) | not run        |

So the engine must realise every step: `step` reads the loss (`item()` / `dataSync()`), which it needs for `diverged` anyway, and `trainMany` reads the stacked loss each step. This is a requirement for CONTRACTS section 7, not an optimisation.

## 3. Measurements

Median of 5 blocks of 1,000 full-batch steps after a 20-step warm-up. `step` includes the loss read-back (`item()`). The model is 1-16-1 tanh unless stated. "First call" is trace + Wasm compile + first execution of a cold jitted step (separate process, bench only).

| Case                         | Node 23 ms/step | Chrome 157 ms/step | Chrome, COOP/COEP ms/step |
| ---------------------------- | --------------- | ------------------ | ------------------------- |
| linear, n = 40, sgd          | 0.091           | 0.044              | 0.056                     |
| MLP n = 40, sgd              | 0.159           | 0.098              | 0.106                     |
| MLP n = 40, momentum         | 0.199           | 0.121              | 0.128                     |
| **MLP n = 40, adam** (exact) | 0.230           | **0.183**          | 0.190                     |
| MLP n = 40, `optax.adam`     | 0.291           | 0.182              | 0.199                     |
| **MLP n = 500, adam**        | 0.303           | 0.258              | 0.266                     |
| MLP 1-64-64-1, n = 500, adam | 1.200           | 1.141              | 1.165                     |

| Case                                                         | Node                      | Chrome                   | Chrome, COOP/COEP         |
| ------------------------------------------------------------ | ------------------------- | ------------------------ | ------------------------- |
| predict on 4096-point grid, jitted (median, incl. read-back) | 0.48 ms                   | 0.50 ms                  | 0.50 ms                   |
| predict on 4096-point grid, not jitted                       | 0.70 ms                   | 0.70 ms                  | 0.74 ms                   |
| predict, first jitted call                                   | 5.9 ms                    | 4.0 ms                   | 3.9 ms                    |
| **trainMany k = 20, n = 40, 1000 steps**                     | 0.51 s                    | **0.48 s**               | 0.50 s                    |
| trainMany k = 20, n = 500, 1000 steps                        | 3.49 s                    | 3.61 s                   | 3.66 s                    |
| jit compile + first step, MLP n = 40 (cold)                  | 18 ms (adam), 26 ms (sgd) | 8 ms (adam), 13 ms (sgd) | 10 ms (adam), 15 ms (sgd) |
| jit(vmap(step)) compile + first step, k = 20                 | 57 ms                     | 26 ms                    | 28 ms                     |

Page bundle (`vite build` of the spike page, including jax-js, optax and the spike code): 293 kB, 90 kB gzipped; the WebGPU and WebGL backends are separate lazy chunks (52 kB and 14 kB) that the page never loads.

**Multi-threading with COOP/COEP.** Served with `Cross-Origin-Opener-Policy: same-origin` and `Cross-Origin-Embedder-Policy: require-corp`, Chrome reports `crossOriginIsolated: true`, `SharedArrayBuffer` is defined and the Wasm memory is shared, so jax-js's worker pool is available. Every synchronous read, the whole ops suite and the 10,000-step leak test still work in that mode. There was no measurable speedup for any workload here (the largest was trainMany k = 20, n = 500, with 160,000 elements per layer activation: 3.66 s against 3.61 s). Without those headers `SharedArrayBuffer` is undefined and the single-threaded path is used. So the tool does not need the headers; at these sizes they change nothing. Node has no global `Worker`, so the Node numbers are single-threaded.

## 4. Leak test

**Method.** jax-js exposes no public count of live buffers (`Array.refCount` is per array; `setDebug` only logs compilation; the backend object is not exported). The Wasm backend keeps its live buffers in a private `Map<slot, {ptr, size, ref}>`. `probe.ts` wraps `Map.prototype.set` until the first buffer is allocated, recognises that map by the `{ ptr, size, ref }` value, keeps a reference to it and restores `Map.prototype.set`. `liveArrays()` is the size of that map: one entry per live buffer. Blind spot: arrays with fewer than 128 elements whose values are all equal (scalars, zeros/ones, 1-element arrays) are kept by jax-js as lazy symbolic constants with no backend buffer, so they are not counted; they use no memory, so leaking them is harmless, and real buffers they wrap are counted. It also wraps `WebAssembly.Memory` so that the linear-memory size (`wasmBytes()`) is available as a second signal (this needs `probe.ts` imported before `@jax-js/jax`).

**Result.** After a 50-step warm-up, 10,000 steps of each loop. Live buffer count and Wasm memory are identical at the end and at every 100-step sample, and zero buffers remain after disposing the loop:

| Loop                                                  | Steps  | Live buffers | Wasm memory |
| ----------------------------------------------------- | ------ | ------------ | ----------- |
| linear, sgd, train + predict every step               | 10,000 | 4 -> 4       | flat        |
| MLP, sgd                                              | 10,000 | 6 -> 6       | flat        |
| MLP, momentum                                         | 10,000 | 10 -> 10     | flat        |
| MLP, adam (exact)                                     | 10,000 | 16 -> 16     | flat        |
| MLP, `optax.adam`                                     | 10,000 | 15 -> 15     | flat        |
| MLP, adam, dropout with the key carried through `jit` | 10,000 | 17 -> 17     | flat        |
| trainMany, k = 20, `jit(vmap(step))`                  | 2,000  | 17 -> 17     | flat        |
| negative control: one array not disposed per step     | 1,000  | 50 -> 1000   | grows       |

The negative control is reported as a leak, so the check can fail. The same results came out in Node, in Chrome, and in Chrome with COOP/COEP.

**This becomes `Engine.liveArrays()`.** Proposed implementation: `createEngine()` installs the `Map.prototype.set` hook, then loads jax-js with a dynamic `import()` (so no jax import precedes it), and `liveArrays()` returns the captured map's size. It is only for tests: the engine leak test warms up once, then asserts the count after N steps equals the count after the warm-up. The hook is fragile because it depends on a private data structure of jax-js 0.1.27, so the pinned version matters and the test fails loudly (count stays 0) if the structure changes. An upstream request for a public live-buffer count would remove the hack.

## 5. Proposed changes to CONTRACTS.md section 7

Section 7 is not impossible as written, so it is not edited here. Proposed text for the engine bead to adopt:

1. **Optimizers.** Replace "adam = `optax.adam(lr)`" with: adam is implemented in the engine as a `GradientTransformation` that carries `bc = 1 - b^t` in its state (see `adamExact` in `spike/lib.ts`), because `optax.adam` 0.1.3 is up to 0.6% off in the first steps. sgd and momentum stay `optax.sgd`.
2. **Loss and activation compositions** (add to the fused-forms bullet): relu is `where(x > 0, x, 0)`; abs is `x * sign(x)`; a Sigmoid hidden activation is `0.5 * (1 + tanh(x / 2))`; BCE with logits is `logsumexp([x, 0]) - x * y`, never `softplus`; GELU uses `{ approximate: false }`.
3. **`step` and `trainMany` evaluate every step.** `step` reads the loss each step (it already must for `diverged`); `trainMany` reads the stacked loss every step. Never run more than a step ahead of a read.
4. **`liveArrays()`.** As in section 4 above: the size of the Wasm backend's slot table, found by a `Map.prototype.set` hook installed in `createEngine()` before jax-js loads.
5. **Ownership notes for the contract:** a jitted step consumes all its array arguments; `tree.ref` before reuse; `optax` `update` disposes a `params` argument (do not pass it); destructure split keys with `np.unstack`, never with `const [a, b] = ...`.
6. **Pin.** Keep `@jax-js/jax` 0.1.27 and `@jax-js/optax` 0.1.3 exactly: the probe and several defect workarounds are tied to them.

## 6. What was not measured

- Firefox and Safari (including iOS Safari), and any real tablet. Chrome numbers are headless Chrome 157 on an M2 Pro, which is fast. If a tablet is 5x slower, trainMany k = 20, n = 40 would take about 2.5 s, which is inside the 3 s gate but close, and the n = 500 case about 18 s. This should be checked on a real touch tablet before the resampling overlay is built.
- The SPEC's 300 ms limit for the resampling overlay: trainMany at k = 20 and n = 40 takes about 0.5 s for 1000 steps on this laptop, so a worker (the follow-up noted in SPEC) is likely needed unless the overlay uses fewer steps. Not tested in a worker.
- WebGPU (not needed in version 1).
- Models bigger than 1-64-64-1; attention and layer norm training steps. The ops they rely on (`var_`, `standardize`, `take`, `matmul`, `softmax`) were checked numerically; a full layer-norm or attention model was not trained.
- Gradients and losses against PyTorch. That is the job of the fixtures in CONTRACTS section 11; here only hand-computed values were used.
- Dev-server code was measured, not a minified production build (jax-js is loaded unminified from `node_modules` by the dev server; the build output has the same compute kernels).

## Deviations from the bead

- `packages/engine/package.json` gains two scripts (`spike`, `spike:web`); `packages/engine/tsconfig.json` references the new `spike/tsconfig.json` so `pnpm typecheck` covers the spike; `eslint.config.js` turns off `no-non-null-assertion` and `no-this-alias` for `packages/engine/spike/**` only.
- Node scripts need Node 22.6+; the repo's `engines` says 20.19+. Only the spike is affected.
