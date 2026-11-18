# Tinker contracts

The interfaces that beads share. SPEC.md says what the tool does; this file fixes names, shapes and file locations so that beads built separately in parallel worktrees fit together. If a bead needs to change something here, change this file in the same commit and say so in the PR body.

## 1. Layout

```
packages/core/src/            pure TypeScript, no DOM, no jax-js
  graph.ts        Graph, Node, Edge types and zod schema; parseGraph / serializeGraph (section 2)
  registry.ts     PieceType, CORE_PIECES: Record<PieceType, CorePiece> (section 3)
  analyse.ts      analyse(graph, dataInfo) -> Analysis: widths, issues, param counts (section 4)
  rng.ts          seeded PRNG (section 5)
  params.ts       HostParams, initParams, countParams (section 5)
  data/sets.ts    the 12 dataset generators and their true functions (section 6)
  data/sample.ts  sample(config) -> Dataset; split, standardise (section 6)
  data/grid.ts    plot grids for 1-D and 2-D fit plots
  data/batches.ts mini-batch schedule (section 6)
  edit.ts         pure graph edits (section 8)
  layout.ts       layer-view and neuron-view geometry (section 9)
  stats.ts        biasVariance and other small numeric helpers
  file.ts         save file and preset formats (section 12)
  index.ts        re-exports everything above
packages/engine/src/          the only package that imports @jax-js/*
  index.ts        createEngine(): Engine (section 7)
  compile.ts      Graph + Analysis -> Model (predict / loss closures, jit)
  pieces/<type>.ts  one file per PieceType; ENGINE_PIECES table in pieces/index.ts
  losses.ts       loss functions incl. fused link+loss forms
  penalty.ts      L1/L2 over Linear weights
  optim.ts        sgd / momentum / adam via @jax-js/optax
  many.ts         trainMany (vmap)
  surface.ts      lossGrid for the loss-surface view
  convert.ts      HostParams <-> jax arrays; the only place arrays cross the boundary
  types.ts        the public types of section 7 (Engine, Model, Params, TrainState, ...)
  probe.ts        test-only live-buffer probe behind liveArrays()
packages/engine/test/
  fixtures/*.json   PyTorch reference values (section 11), generated, committed
  reference.test.ts  compares every fixture to the engine
  leak.test.ts       10,000 steps, flat live-array count
packages/engine/spike/        milestone 1 only; kept for reference after the spike
tools/torch-ref/gen.py        uv script that writes the fixtures (section 11)
packages/web/src/
  main.tsx  App.tsx            shell: five regions, theme, portrait notice
  theme.css                    CSS variables, light and dark
  state/store.ts               zustand store composed from the slices below
  state/graphSlice.ts          graph + undo/redo history
  state/dataSlice.ts           data config, sampled Dataset
  state/trainSlice.ts          engine model, TrainState, run loop, loss history, best
  state/viewSlice.ts           grain, selection, hover, toggles, pins
  engine.ts                    the single Engine instance and its React hook
  gestures/drag.ts             Pointer Events drag helper used by every gesture
  topbar/TopBar.tsx  DataControls.tsx  FileMenu.tsx  PresetMenu.tsx
  palette/Palette.tsx
  canvas/Canvas.tsx  LayerView.tsx  NeuronView.tsx  Wire.tsx  Inspector.tsx
  canvas/drawings/<type>.tsx   one file per PieceType; WEB_PIECES table in drawings/index.ts
  results/FitPlot.tsx  FitPlot2D.tsx  LossCurves.tsx  LossSurface.tsx  ResampleOverlay.tsx  Pins.tsx
  plot/axes.ts  plot/canvas.ts plot/color.ts   shared plotting helpers (canvas 2D)
  trainbar/TrainingBar.tsx
  presets/*.json               bundled presets (section 12)
```

Skeleton beads create placeholder files at these paths (each drawing, each engine piece, each results view, each state slice) so later beads fill in a file instead of editing a shared one.

Libraries, fixed: pnpm, TypeScript strict, vitest, zod, `@jax-js/jax` 0.1.27 and `@jax-js/optax` 0.1.3 (exact versions, no caret), React, zustand, plain CSS modules with CSS variables (no Tailwind), Vite, vite-plugin-pwa, wrangler (deploy only), playwright-core (end-to-end only). Plots and heat maps are drawn with canvas 2D; the model canvas is SVG. No charting or graph-layout library. Do not add other runtime dependencies without a reason written in the PR body.

## 2. Graph format (`core/graph.ts`)

The JSON document in SPEC.md "Graph format" is the source of truth. Types:

```ts
type NodeId = string; // [a-z][a-z0-9_]*, unique
type ActivationFn = 'sigmoid' | 'tanh' | 'relu' | 'gelu';
type OutputLink = 'identity' | 'sigmoid' | 'softmax';
type LossFn = 'squared_error' | 'absolute_error' | 'huber' | 'log_loss' | 'cross_entropy';
type Penalty = { kind: 'l2' | 'l1'; lambda: number }; // lambda >= 0

type Node =
  | { id: NodeId; type: 'input'; columns: string[] }          // subset of the dataset's columns, in order
  | { id: NodeId; type: 'feature'; degree: number; sin: boolean; cos: boolean; products: boolean } // degree 1..5
  | { id: NodeId; type: 'linear'; out: number; bias: boolean }  // out 1..64
  | { id: NodeId; type: 'activation'; fn: ActivationFn }
  | { id: NodeId; type: 'dropout'; p: number }                  // 0 <= p < 1
  | { id: NodeId; type: 'add' }
  | { id: NodeId; type: 'output'; link: OutputLink }
  | { id: NodeId; type: 'target' }
  | { id: NodeId; type: 'loss'; fn: LossFn; delta?: number; penalty?: Penalty }; // delta only for huber, default 1.0

type PortRef = string;          // "node" (its default port) or "node.port"
type Edge = [from: PortRef, to: PortRef];

type DataConfig = {
  set: DatasetName; n: number; noise: number; val: number; seed: number;
  standardize: boolean;         // default false (SPEC decision)
};
type TrainConfig = {
  optimizer: 'sgd' | 'momentum' | 'adam'; lr: number; batch: 'full' | number; seed: number;
};
type Graph = { version: 1; nodes: Node[]; edges: Edge[]; data: DataConfig; train: TrainConfig };
```

- Also exported from `graph.ts`: `DatasetName` and the `*_NAMES`/`*_FNS` const lists, `NodeBody`, `parsePort(ref) -> {node, port | null}`, `resolveEdge(nodes, edge)` (fills default ports, one sentence on a bad edge) and `inputPortsOf(type)`. Types are inferred from the zod schema.
- `parseGraph(json: unknown): { ok: true; graph: Graph } | { ok: false; error: string }`. Fills defaults (`standardize: false`, huber `delta: 1`). Unknown fields are rejected. The error is one plain sentence.
- `serializeGraph(graph)`: stable key order, two-space JSON.
- The weight penalty is the `penalty` field on the loss node. In the palette it is a piece ("Weight penalty"); dropping it on the loss sets this field. It has no node of its own.
- Ports. Every node has at most one output port, `out`. Input ports: `add` has `a` and `b`; `loss` has `pred` and `target`; `input` and `target` have none; every other piece has one input port `in`. A bare node id in an edge means `out` as a source and the single input port as a destination.
- Every input port takes exactly one edge. An output port may feed several input ports (this is how a skip connection is made: `lin1 -> act1` and `lin1 -> add.b`).

## 3. Piece registry (split by layer)

`PieceType = 'input' | 'feature' | 'linear' | 'activation' | 'dropout' | 'add' | 'output' | 'target' | 'loss'`. The 14 palette pieces map onto these types: the four activations are one type with `fn`, the five losses are one type with `fn`, and Weight penalty edits the loss node.

```ts
// core/registry.ts
type Family = 'inputs' | 'features' | 'linear' | 'activations' | 'structure' | 'outputs' | 'losses';
type CorePiece<N extends Node> = {
  family: Family;                         // drop-to-swap is allowed only within a family
  inputPorts: string[];                   // section 2
  defaults: () => Omit<N, 'id'>;
  outWidth(node: N, inWidths: Record<string, number>, ctx: WidthCtx): number | WidthError;
  paramShapes(node: N, inWidth: number): { W?: [rows: number, cols: number]; b?: [number] }; // W is [in, out]
  torch(node: N, inWidth?: number): string; // e.g. "nn.Linear(4, 8, bias=True)" (needs inWidth); used by the later code view
};
type WidthCtx = { dataColumns: number; numClasses: number };
type WidthError = { error: string };      // one plain sentence, e.g. "Add needs both inputs to have the same width: 4 and 8."
const CORE_PIECES: { [T in PieceType]: CorePiece<Extract<Node, { type: T }>> };

// a palette entry (one per row of the SPEC pieces table, 14 entries)
type PaletteEntry = { key: string; label: string; family: Family | 'penalties'; make: () => NodeBody | Penalty };
// NodeBody = Omit<Node, 'id'> distributed over the union. Target is in family 'outputs'.
// The weight penalty's make() gives { kind: 'l2', lambda: 0.01 }.
const PALETTE: PaletteEntry[];

// engine/src/pieces/index.ts
const ENGINE_PIECES: { [T in PieceType]: EnginePiece<Extract<Node, { type: T }>> };

// web/src/canvas/drawings/index.ts
const WEB_PIECES: { [T in PieceType]: WebPiece<Extract<Node, { type: T }>> }; // layer drawing, neuron drawing, inspector fields
```

Width rules: input = number of selected columns; feature = d·degree + (sin ? d : 0) + (cos ? d : 0) + (products ? d(d−1)/2 : 0), where power 1 is x itself; linear = `out`; activation, dropout, output = input width; add = shared input width (error if unequal); target = 1 for regression and binary sets, and the loss checks against classes; loss has no output.

Feature output layout (the order of columns; the engine and the torch fixtures use it): `x^p` for p = 1..degree (power-major, then column, so power 1 is x itself), then `sin(x)` if `sin`, then `cos(x)` if `cos`, then `x_i * x_j` for i < j in row-major pair order if `products`. sin and cos apply to the raw column values (no frequency scaling).

Parameter count of a Linear: in·out + (bias ? out : 0). Every other piece has 0.

## 4. Analysis (`core/analyse.ts`)

```ts
// DataInfo is defined in core/graph.ts (not analyse.ts) so that data and analysis both import it.
type DataInfo = { columns: string[]; kind: 'regression' | 'binary'; numClasses: number }; // from the dataset
type Issue = { at: { node?: NodeId; edge?: number }; message: string };  // edge = index into graph.edges
type Analysis = {
  order: NodeId[];                          // topological order of nodes on the path to the loss
  inWidth: Record<NodeId, number>;          // width entering each node (single-input pieces)
  outWidth: Record<NodeId, number>;
  edgeWidth: (number | null)[];             // per graph.edges index, for the wire labels
  params: Record<NodeId, number>; totalParams: number;
  predNode: NodeId | null; lossNode: NodeId | null;
  issues: Issue[];                          // empty when trainable
  trainable: boolean;
};
function analyse(graph: Graph, data: DataInfo): Analysis;
```

Trainable means: no cycles; exactly one loss; every input port connected; a path input -> ... -> output -> loss.pred and target -> loss.target; widths resolve; the output/loss/target combination is in the table below; total parameters <= 10,000. Each failure gives one Issue with one plain sentence.

`analyse` does not assume `parseGraph` rejected anything beyond unknown nodes/ports: cycles, self-edges, doubly-fed ports, duplicate or unknown Input columns and bad edges all become issues. Behaviour fixed here for later beads:

- `order`: with exactly one loss that is not on a cycle, the ancestors of the loss (the loss last) in topological order, so dangling nodes are left out. With no loss, several losses, or a loss on a cycle, every node not blocked by a cycle, in topological order. Nodes on or downstream of a cycle are never in `order`.
- `inWidth` and `outWidth` have no key where the width is unknown (never NaN or null). `inWidth` is absent for Input and Target; for Add it is the shared width and absent when the two inputs differ or one is missing; for the loss it is the width of `pred`. `outWidth` of a loss is absent.
- `params` has a key for every node whose input width is known, including nodes off the loss path; `totalParams` sums only nodes in `order`. Consumers (init/reconcile, engine) iterate `order`, not the keys of `params`. The records are prototype-free, so lookups by node id are safe for any id.
- Issues are reported at the root cause only: a node whose upstream width is unknown gets no issue of its own.
- An Add width mismatch is one issue with both `node` (the Add) and `edge` (the wire into `b`). Other issues about one wire carry `edge`, and `node` too when a node is the cause.
- A doubly-fed input port: the first wire (lowest edge index) is used for widths and order; each later wire gets an issue with its `edge` and the receiving `node`, and `edgeWidth` for it is null.
- A self-edge or any cycle: one issue at the first node on the cycle; those nodes have unknown widths.
- Wiring rules: Output may only feed `loss.pred` and Target only `loss.target` (issue on the edge otherwise); `loss.pred` must come from an Output and `loss.target` from a Target; an Input listing a column twice or a column the dataset lacks is an issue at the Input node.
- `lossNode` and `predNode` can be non-null while the loss is not in `order` (a loss downstream of a cycle). Consumers may rely on `order` ending in `lossNode` only when `trainable` is true.
- Duplicate node ids (only possible for an unparsed graph) give an issue at each later duplicate; the first node with the id is analysed.
- `predNode` is the Output feeding `loss.pred` and `lossNode` the loss; both are null unless there is exactly one loss (`predNode` also null if `pred` is unconnected or not from an Output).

| Loss | Output link | Prediction width | Target |
| --- | --- | --- | --- |
| squared_error, absolute_error, huber | identity or sigmoid | 1 | regression or binary (as 0/1 numbers) |
| log_loss | sigmoid | 1 | binary |
| cross_entropy | softmax | numClasses (2 for all v1 sets) | binary (as class index) |

## 5. Parameters, initialisation, randomness

```ts
// core/rng.ts
type Rng = { next(): number; normal(): number; int(n: number): number; fork(label: string): Rng };
function makeRng(seed: number): Rng;   // sfc32 or splitmix32; same seed -> same sequence on every platform

// core/params.ts
type HostParams = Record<NodeId, { W: Float32Array; b?: Float32Array }>; // W row-major [in, out]; only linear nodes
function initParams(graph: Graph, analysis: Analysis, seed: number): HostParams;
function reconcileParams(old: HostParams, oldGraph: Graph, oldAnalysis: Analysis, graph: Graph, analysis: Analysis, seed: number): HostParams;
function countParams(host: HostParams): number;
function paramsToJson(host: HostParams): Record<NodeId, { W: number[]; b?: number[] }>; // prototype-free object
function paramsFromJson(json: unknown, graph: Graph, analysis: Analysis): { params: HostParams; dropped: NodeId[] };
```

Initialisation follows `nn.Linear`: W and b both uniform on ±1/√in. Different seeds give different values; the same seed always gives the same values. Note the layout: torch stores `weight` as [out, in]; HostParams stores [in, out] so that the forward pass is `x @ W + b`. Fixtures (section 11) are written in HostParams layout.

HostParams has an entry for exactly the linear nodes on `analysis.order` with a known input width (an entry has `b` iff the node has bias). Each node draws W then b from `makeRng(seed).fork(nodeId)`, so adding a node does not change the others' init. `reconcileParams` takes the graph and analysis the old params belong to (flat arrays alone cannot tell 2x3 from 3x2). Per node: W (the same array object) is kept when its old and new [in, out] are equal, even if bias was toggled; b is kept when both have it, drawn fresh from the node's rng stream when bias is turned on, and dropped when turned off; if [in, out] changed, the node is re-initialised; nodes that left the path are dropped; `old` is never mutated. `paramsFromJson` drops malformed entries (non-arrays, non-finite numbers, holes), shape-mismatched entries and unknown nodes, listing their ids in `dropped`; linear nodes with no saved entry are absent, so callers follow it with `reconcileParams`.

HostParams arrays are immutable once created: `reconcileParams` shares unchanged arrays with `old` (and so with undo history), so nobody writes into them in place. The engine's `fromHost` copies into jax arrays and `toHost` creates new arrays.

All data sampling, initialisation and mini-batch shuffling use `core/rng.ts`, never `Math.random` and never jax random. Only dropout masks use jax random, keyed from `TrainState.key`.

## 6. Data (`core/data/*`)

```ts
type DatasetName = 'line' | 'quadratic' | 'sine' | 'step' | 'wiggle'
                 | 'plane' | 'bowl' | 'saddle'
                 | 'blobs' | 'circle' | 'xor' | 'moons';
type Dataset = {
  info: DataInfo;
  xTrain: Float32Array; yTrain: Float32Array; nTrain: number;   // x row-major [n, d]
  xVal: Float32Array;   yVal: Float32Array;   nVal: number;
  standardization: { mean: number[]; std: number[] } | null;    // from training rows, applied to val and grids
  domain: [lo: number, hi: number][];                           // per column, in raw units
};
function sample(config: DataConfig): Dataset;
function truth(set: DatasetName, xRaw: Float32Array): Float32Array; // regression: f*(x); classification: P(y=1|x) of the clean generator
```

| Set | d | Domain | f* or rule |
| --- | --- | --- | --- |
| line | 1 | [-3, 3] | 0.8x + 0.5 |
| quadratic | 1 | [-3, 3] | 0.5x² − 1 |
| sine | 1 | [-3, 3] | sin(1.5x) |
| step | 1 | [-3, 3] | x < 0 ? −1 : 1 |
| wiggle | 1 | [-3, 3] | 0.5 sin(3x) + 0.3x |
| plane | 2 | [-2, 2]² | 0.6x₁ − 0.4x₂ |
| bowl | 2 | [-2, 2]² | 0.5(x₁² + x₂²) − 1 |
| saddle | 2 | [-2, 2]² | 0.5(x₁² − x₂²) |
| blobs | 2 | [-3, 3]² | two Gaussians at (−1, −1) and (1, 1), std 0.6 |
| circle | 2 | [-2, 2]² | class 1 inside radius 1, class 0 in the ring 1.3 to 2 |
| xor | 2 | [-2, 2]² | x uniform; class 1 when x₁x₂ > 0 |
| moons | 2 | [-1.5, 2.5]×[-1, 1.5] | the two interleaved half circles of `sklearn.datasets.make_moons` |

- Regression noise: y = f*(x) + noise · s · ε, ε ~ N(0, 1), where s is the standard deviation of f* over a uniform sample of the domain (a constant per set, computed once).
- Classification noise: Gaussian jitter of point positions with std noise · 0.5 (blobs: noise scales its std, 0.6 · (0.5 + noise)). Classes are balanced.
- n is the total; the last round(n · val) points are validation. Columns are named `x1`, `x2`.
- Standardisation: when `config.standardize` is on, `xTrain` and `xVal` are stored already standardised, (x - mean) / std per column, with mean and population std (divide by n) of the raw training rows; a constant column gets std 1. `standardization` holds those numbers (null when off) and `domain` stays in raw units. `standardizeRows(x, d, st)` and `unstandardizeRows(x, d, st)` convert row-major arrays.
- `truth` takes raw (unstandardised) x. Regression returns f*(x). Classification returns P(y=1|x) of the clean generator: blobs the exact logistic of the two Gaussians, circle (cut at r = 1.15, mid-gap), xor and moons (class 1 = inner moon, the nearer half circle) 0 or 1. All v1 classification sets are binary, so `info.kind` is `'binary'` with `numClasses` 2; regression has `numClasses` 1. `DataInfo` is defined in `core/graph.ts`.
- Classification labels alternate 0, 1, 0, 1, ... so the train and validation parts are each balanced.
- `data/grid.ts`: `grid1d(domain, m = 200, standardization = null)` and `grid2d(domain, m = 64, standardization = null)` return `{ n, d, xRaw, xStd }` (row-major Float32Arrays; xStd equals xRaw when standardization is null). Both span the domain inclusive of the ends. grid2d row index is `iy * m + ix`, x1 varying fastest.
- Web data slice (`state/dataSlice.ts`): the slice holds `dataset: Dataset` and `dataInfo: DataInfo` (`dataset.info`), always equal to `sample(graph.data)`; a store subscription re-samples whenever `graph.data` changes identity (data action, undo, redo, load). Actions: `setData(patch)` (clamps to the schema ranges; one undo step per call via `graphSlice.apply`, because history holds whole graphs including `graph.data`; the top bar commits number fields on blur or Enter, not per keystroke), `resample()` (new seed from `makeRng(seed).fork('resample').int(2 ** 31)`), `setInputColumns(id, columns)`. Switching to a set with different columns resets every Input piece to all columns of the new set.
- `data/batches.ts`: `batchSchedule(nTrain, batch, seed)` is an infinite generator of fresh `Int32Array` index arrays, one per step; each epoch shuffles with `makeRng(seed).fork('epoch' + e)` and cuts the order into batches, keeping a smaller last batch so every row appears once per epoch. `batch: 'full'` yields all rows in order every step. `stepsPerEpoch(nTrain, batch)` gives steps per epoch. `gatherBatch(xTrain, yTrain, d, idx)` copies those rows into a `Batch` (section 7).

## 7. Engine (`engine/src/index.ts`)

The SPEC interface, refined. All arrays crossing the interface are plain Float32Arrays; jax arrays never leave the package.

```ts
type Batch = { x: Float32Array; y: Float32Array; n: number };      // x row-major [n, d], already standardised if on
type Model = { readonly graph: Graph; readonly analysis: Analysis }; // plus private compiled closures
type Params = { readonly __brand: 'Params' };                        // opaque, owns jax arrays
type TrainState = { params: Params; opt: unknown; step: number; key: unknown };
type StepResult = { state: TrainState; loss: number; diverged: boolean }; // loss before the update; diverged if non-finite

interface Engine {
  compile(graph: Graph, data: DataInfo): Model;               // throws Error(message) when !analysis.trainable
  init(model: Model, seed: number): Params;                   // = fromHost(initParams(...))
  fromHost(model: Model, host: HostParams): Params;
  toHost(params: Params): HostParams;                          // SPEC's read(); copies
  write(params: Params, nodeId: NodeId, values: { W?: Float32Array; b?: Float32Array }): Params; // consumes params
  startTraining(model: Model, params: Params, seed: number): TrainState; // consumes params; optimizer state from graph.train
  step(model: Model, state: TrainState, batch: Batch): StepResult;       // consumes state; dropout on
  predict(model: Model, params: Params, x: Float32Array, n: number): Float32Array; // dropout off; output after the link; [n, width]
  evaluate(model: Model, params: Params, batch: Batch): number;          // mean loss + penalty, dropout off
  lossGrid(model: Model, params: Params, batch: Batch, axes: [ParamAxis, ParamAxis], a: Float32Array, b: Float32Array): Float32Array; // [a.length, b.length]
  trainMany(model: Model, seeds: number[], data: Batch[], steps: number): HostParams[]; // vmap over k copies
  clone(params: Params): Params;
  dispose(x: Params | TrainState): void;
  liveArrays(): number;                                        // for the leak test; method chosen by the spike
}
type ParamAxis = { node: NodeId; tensor: 'W' | 'b'; index: number };
function createEngine(): Promise<Engine>; // awaits jax init(); Wasm device; never WebGPU in v1
```

- "Consumes" means the caller must not use that value again; the engine disposes it or reuses its arrays. Everything else borrows. Callers that need to keep a Params (best-so-far, pins) take `clone` or `toHost`.
- The objective is mean loss over the batch plus λ·R(w), with R summed over Linear `W` only: L2 = Σw², L1 = Σ|w|.
- Fused forms: output sigmoid + log_loss is computed as BCE with logits on the pre-link value; output softmax + cross_entropy is computed as log-softmax. Other combinations apply the link and then the loss. `predict` always returns post-link values.
- Huber follows `nn.HuberLoss(delta)`. Dropout follows `nn.Dropout` (inverted scaling 1/(1−p)).
- Optimizers: sgd = `optax.sgd(lr)`; momentum = `optax.sgd(lr, { momentum: 0.9 })` (equal to torch SGD with momentum 0.9, dampening 0); adam = `optax.adam(lr)` with b1 0.9, b2 0.999, eps 1e-8 (torch defaults).
- `step` never throws on divergence; it returns `diverged: true` and the state is unchanged.
- jax-js decisions (go/no-go `tinker-e1h.2`, from `docs/spike.md`; jax-js 0.1.27 and optax 0.1.3 stay pinned exactly):
  - Use `optax.adam` as is. It is accurate inside jit (matches torch Adam to about 1e-6 on the 5-step fixture); it is only inexact when run eagerly (float32 bias correction), which the engine never does. Switching optimizer means `compile` + `startTraining` again; `step` throws "This training state was started for a different optimizer; start training again." for a state from another optimizer, before consuming anything.
  - Sigmoid is computed as `0.5 * (1 + tanh(x / 2))` (jax-js `nn.sigmoid` has a NaN gradient below about −88), and the fused sigmoid + log_loss as `logsumexp([z, 0]) − z·y` (jax-js `softplus`/`logSigmoid` overflow in float32). These keep large logits from reporting a false "diverged".
  - ReLU, abs and GELU use the jax-js built-ins (`nn.relu`, `np.abs`, `nn.gelu` with its default tanh approximation). Their at-zero gradients (relu'(0) = 0.5, |x|'(0) = 1) differ from torch; accepted.
  - Laziness: jax-js evaluates lazily, so `step` and `trainMany` read the loss back every step. Without a read the pending computation grows with each step and the run slows and runs out of memory.
  - `step` keeps a `.ref` of the pre-step state until the loss is read, so a diverged step can return the state unchanged; the leak test includes a diverged step.
  - `liveArrays()` counts the Wasm backend's live buffers through the test-only hook in `docs/spike.md` (it finds jax-js's private slot map by intercepting `Map.prototype.set` until the first allocation). It is tied to the pinned version and is never used by the app: the hook lives in `engine/src/probe.ts` and is off unless a test calls `installProbe()` and then loads the engine with a dynamic `import()` (see `test/leak.test.ts`); without it `liveArrays()` returns 0. The package exports it as `@tinker/engine/probe` so a dev build of the app can do the same.
  - Engine structure fixed by `tinker-hj4.1` (later engine beads fill files and flags, they do not change these):
    - `EnginePiece<N> = { implemented: boolean; forward(node, inputs: Record<port, JArray>, layer, ctx): JArray }` in `pieces/types.ts`; `forward` consumes its inputs and layer. `compile` walks `analysis.order`; a piece, loss (`losses.ts` `LOSSES`), penalty (`penalty.ts` `PENALTY_IMPLEMENTED`) or optimizer (`optim.ts` `OPTIMIZERS`) with `implemented: false` makes `compile` throw "The <thing> is not implemented yet."
    - In loss mode the Output piece returns the pre-link value and the loss applies or fuses the link itself (`LossImpl.compute(node, z, y, link)`: z pre-link, y `[n]`, result the mean loss); in predict mode the Output applies the link. The penalty is added in `compile.ts` `walk` (hj4.2).
    - Batch x is `[n, dataColumns]` (all dataset columns, `DataInfo.columns`); the Input piece selects its own `columns`.
    - A consumed `Params` or `TrainState` throws "already consumed or disposed" instead of touching freed arrays; `dispose` is idempotent. `step` consumes the state and its `params`: after a successful step the old `state.params` is dead; after a diverged step the returned state reuses the same `Params` object.
    - `diverged` is true when the loss is non-finite or any updated parameter is non-finite (a non-finite update is never kept); `loss` is then the value read, which may be finite. A batch that does not fit the model throws before anything is consumed.
    - `TrainState.key` is created from the seed; each `step` splits it (`np.unstack(random.split(key, 2))`), samples one keep mask per Dropout node on the loss path inside the jitted step but outside `valueAndGrad` (`PieceCtx.dropoutMask(nodeId)` hands it to the piece; random ops traced under `valueAndGrad`, and `np.take` index arrays, leave a constant buffer per compiled model that is never freed) and stores the next key in the new state. Pieces must not create array constants (index arrays, `random.*`) inside the differentiated walk. A diverged step returns the old key. `predict` and `evaluate` run with `PieceCtx.train = false` (Dropout is the identity).
    - Jitted functions are built lazily per model and never disposed (the interface has no model disposal), so each `compile` leaves one set until garbage collection.
  - Ownership pitfalls found by the spike: optax `update` disposes its `params` argument; destructuring a jax Array leaks the parent (use `np.unstack` for split keys).

## 8. Edits (`core/edit.ts`)

Every canvas action is one pure function `(graph, args) -> EditResult`, with `type EditResult = { ok: true; graph: Graph; select?: NodeId } | { ok: false; error: string }`. The store applies `ok` results through the undo history and shows `error` as a toast; no edit mutates its input.

```ts
insertOnWire(g, edgeIndex, piece: Omit<Node,'id'>)       // new id = `${type}${k}`, smallest free k
swapPiece(g, nodeId, piece: Omit<Node,'id'>)              // same family only; carry over settings with the same name and valid value
connect(g, from: PortRef, to: PortRef)                    // replaces whatever fed `to`; rejects cycles
deleteNode(g, nodeId)                                     // reconnects upstream to downstream when the node had one input and widths still match
deleteEdge(g, edgeIndex)
setSetting(g, nodeId, key, value)                         // validated against the node schema
setLinearOut(g, nodeId, out)                              // neuron view +/-; 1..64
setPenalty(g, lossNodeId, penalty | null)
setColumns(g, inputNodeId, columns)
// all edits refuse to change a node listed in `locked` (preset field, section 12): error "This piece is locked."
```

Choices fixed by the implementation (`core/edit.ts`, exports `EditResult` and `EditOptions`):

- Every edit takes a last optional argument `opts?: { locked?: readonly NodeId[] }`; the store passes the preset's `locked`. A node counts as changed when its own settings change or a wire into it is added, removed or rewired (so `deleteNode` refuses when the piece or any piece it feeds is locked, and `insertOnWire`, `connect` and `deleteEdge` refuse when the wire's destination is locked). Source nodes are not changed by a new wire out of them.
- Every ok graph has been through `parseGraph` (defaults filled, huber `delta` 1), so it always re-parses. Edits never check trainability; `analyse` reports it.
- `insertOnWire`: the piece must have an input and an output (not Input, Target or a loss). The wire `a -> b` becomes `a -> new` at the same index and `new -> b` right after it; an Add goes in on `a`. `select` is the new id.
- `swapPiece`: same family, except Output and Target never swap for each other. Dropout and Add (both `structure`) can swap; wires move to the new piece's input ports by position. A swap that would drop a wire (e.g. a connected Add `b` when becoming a Dropout) is refused with one sentence, never silently. Same-type swaps carry over each setting that exists on the new type and still validates, except the identity key (`fn` of an activation or loss). A loss keeps its `penalty`; `delta` carries on huber -> huber, is set to 1 when swapping to huber from another loss, and is dropped otherwise. `select` is the node.
- `connect`, `deleteEdge`, `insertOnWire` and `deleteNode` can shift edge indices, so the store and canvas must not hold edge indices across edits: re-derive them from the new graph (or use `select`).
- `connect`: stored in short form (`x`, `loss.pred`); replaces every wire into that port, appended at the end of `edges`. Refuses a bad port (resolveEdge's sentence), a self-wire or cycle, and an identical wire that already exists. It does not enforce the Output/Target wiring rules (analyse does).
- `deleteNode`: closes the gap only when the piece has a single input port with exactly one wire in, at least one wire out, and the width arriving equals the width it passes on (or either is not yet known; widths come from `analyse` on a data info built from the graph's Input columns). Each outgoing wire keeps its index and its source becomes the upstream node. Otherwise the wires in and out are removed.
- `setSetting(g, id, key, value)`: `key` must be a setting of that piece type (`delta` and `penalty` for losses); `id` and `type` cannot be set; `undefined` removes an optional setting; setting a loss `fn` other than huber drops `delta`. The value is checked by the node schema.
- `setPenalty(g, lossId, null)` removes the penalty. `setColumns` refuses repeated columns but allows an empty list.
Params after an edit: the train slice re-initialises the parameters of every node whose param shapes changed and keeps the others (`reconcileParams(oldHost, oldGraph, oldAnalysis, newGraph, newAnalysis, seed)` in `core/params.ts`).

## 9. Layout geometry (`core/layout.ts`)

```ts
type Box = { x: number; y: number; w: number; h: number };      // abstract units; the canvas scales to fit
type LayerLayout = { nodes: Record<NodeId, Box & { col: number }>; wires: { edge: number; points: [number, number][] }[]; size: [number, number] };
function layoutLayers(g: Graph, a: Analysis): LayerLayout;      // columns by longest path from inputs; target sits in the loss's column minus one, below the prediction; skip wires route above the main row
type NeuronLayout = { units: { node: NodeId; index: number; kind: 'in' | 'sum' | 'act' | 'out' | 'target'; x: number; y: number }[];
                      edges: { node: NodeId; i: number; j: number; from: [number, number]; to: [number, number] }[];  // one per weight W[i][j]
                      collapsed: NodeId[]; boxes: LayerLayout['nodes']; wires: LayerLayout['wires']; size: [number, number] };
function layoutNeurons(g: Graph, a: Analysis): NeuronLayout;    // a Linear expands only when in <= 8 and out <= 8; an activation directly after an expanded Linear is drawn as per-unit circles
```

Layout is deterministic and stored nowhere.

Choices fixed by `layout.ts` (constants in the exported `LAYOUT`; all in abstract units):

- Both functions lay out the whole node set, not `analysis.order`. Duplicate ids: the first node wins. An edge that does not resolve (`resolveEdge` fails) gets no wire. Records are prototype-free.
- Columns: longest path; the edge from a target into a loss does not count; a cycle is broken at its first node in graph order; a node with no wires (other than Input and Target) goes in the last column. Rows: a node follows its first-listed source's row, moving down past taken rows; a target sits one row below the prediction in the prediction's column.
- A wire's `points` run from the right-middle of the source box to the left side of the destination box, entering at height `(k+1)/(ports+1)` of the box for input port k (so `add.a` and `add.b` differ). Adjacent columns: one elbow in the gap. Any other wire (skip, same column, backwards) rises into a lane above every box (shorter spans nearer the boxes), runs across, and drops in the gap left of the destination, so no segment enters a box interior. Each wire also has `labelAt: [x, y]`, the midpoint of its longest horizontal segment (first on ties); the web puts the width label there. Edges that do not resolve get no wire geometry (their issues must be shown at node or global level); doubly-fed edges, extra losses and cycle edges do get wires.
- Neuron view uses a wider column gap. Unit circles per node: Input (`in`, one per column), expanded Linear (`sum`, one per output), an Activation whose first source is an expanded Linear (`act`, same count), Output (`out`), Target (`target`, one); only when the width is known and at most 8. Everything else is a box. `boxes` covers every node (a unit column's box is the bounding box of its circles); `wires` connect boxes in both views.
- `edges[].from` is the unit centre of the upstream node when it has as many units as the Linear has inputs, else a point on the upstream box's right side spread evenly. `to` is the `sum` unit centre. Endpoints are centres; the web trims by the circle radius. Biases are not edges. `edges` is ordered by node (graph order), then i-major then j, so a node's edge index is `i*out + j`, matching HostParams `W` row-major `[in, out]`. `collapsed` lists every Linear that is not expanded, including ones with unknown widths.

## 10. Web conventions

- Store slices (section 1) are owned by beads; other beads read them through selectors and call their actions, and do not add fields to a slice they do not own without saying so in the PR body.
- Every gesture (palette drag, wire drag, edge weight drag, unit +/−, sliders) uses Pointer Events through `gestures/drag.ts` (pointer capture, a 4 px slop before a drag starts, cancel on pointercancel). No HTML5 drag and drop: it does not work with touch.
- Hit targets are at least 32 px on touch devices (`@media (pointer: coarse)`). Hover information (weight value, wire width) is also shown on tap, in the inspector.
- Delete is the Delete or Backspace key, and a Delete button in the inspector for touch. Undo and redo are Ctrl/Cmd+Z, Shift+Ctrl/Cmd+Z, and two buttons in the top bar.
- Required viewports: 1280×720 and 1920×1080 (mouse), 1024×768 and 1180×820 (touch, `hasTouch: true`). No page scroll at any of them. Below 1024 px wide, or in portrait, show only the notice "Tinker needs a wider screen. Rotate your tablet or use a laptop."
- Themes: light and dark from `prefers-color-scheme`, all colours as CSS variables in `theme.css`. The weight colours (positive and negative) and the train and validation series colours are variables too and must read in both themes.
- The run loop is `requestAnimationFrame`: each frame runs up to `speed` steps within an 12 ms budget, then redraws once. Results views redraw at most once per frame.
- `data-testid` on: every palette entry (`palette-<key>`), every node (`node-<id>`), every wire (`wire-<index>`), `btn-step`, `btn-run`, `btn-reset`, `step-count`, `param-count`, `status-diverged`, `fit-plot`, `loss-curves`, `grain-toggle`.

## 11. PyTorch reference fixtures

`tools/torch-ref/gen.py` is a uv script with inline metadata (`# /// script` with `dependencies = ["torch==<current stable>", "numpy"]`), run as `uv run tools/torch-ref/gen.py`. It writes `packages/engine/test/fixtures/<name>.json`, one file per case, and is opt-in: tests never run Python.

```ts
type Fixture = {
  name: string;                       // e.g. "linear_bias", "loss_huber_delta0.5", "mlp_relu_adam_5steps"
  graph: Graph;                       // a complete trainable graph in the section 2 format
  params: Record<NodeId, { W: number[]; b?: number[] }>;   // HostParams layout [in, out], float64 written in full
  batch: { x: number[]; y: number[]; n: number };
  expect: {
    pred: number[];                   // predict(), post-link, [n, width] row-major
    loss: number;                     // evaluate()
    grads: Record<NodeId, { W: number[]; b?: number[] }>;  // d loss / d params
    after?: { steps: number; params: Record<NodeId, { W: number[]; b?: number[] }>; losses: number[] }; // optimizer cases
    dropoutMask?: number[][];         // dropout cases: the engine test injects this mask instead of sampling
    predEval?: number[];              // dropout cases: dropout-off forward (what predict() returns)
    lossEval?: number;                // dropout cases: dropout-off objective (what evaluate() returns)
  };
};
```

Details: in dropout cases `pred`, `loss` and `grads` are the dropout-ON training forward and backward with the given mask (how the engine test injects the mask is the engine's choice); `predEval` and `lossEval` are the dropout-off values that `predict` and `evaluate` must return. At-zero conventions are pinned by fixtures: the L1 penalty has subgradient sign(0) = 0 (`penalty_l1_zero_weight`) and relu'(0) = 0 (`act_relu_zero_preact`). `dropoutMask` is a 0/1 keep mask shaped [n][width of the dropout node]; the engine multiplies by mask/(1−p). The engine test injects it through the internal `debugTrainPass(model, params, batch, { [dropoutNodeId]: mask })` in `engine/src/compile.ts` (non-jitted dropout-ON forward, loss and gradients; not on `Engine`), and compares `predEval` / `lossEval` with `predict` / `evaluate`. Classification targets in `batch.y` are class indices stored as floats (0 or 1). `after.losses[k]` is the objective before update k+1 (so `losses[0] == loss`). GELU is the exact erf form (`nn.GELU()`); the engine uses jax-js `gelu` in its default tanh form instead (see "Engine exceptions" below).

Gradients are not part of `Engine`, so the engine reference test reads them from one SGD step with learning rate 1 (grad = W before - W after).

Tolerance: 1e-5 absolute and relative on pred, loss and grads; 1e-4 on `after.params` after 5 steps. The case list is in the fixtures bead (`tinker-7x7.3`).

Engine exceptions (decided in `tinker-e1h.2`, because the engine uses jax-js built-ins and optax rather than torch-exact forms): `act_relu_zero_preact` and `penalty_l1_zero_weight` are skipped by the engine comparison test (they pin torch's at-zero subgradients); `act_gelu` is compared at 1e-4 (tanh-form GELU differs by about 3e-5). The fixtures stay as generated.

## 12. Files and presets (`core/file.ts`)

```ts
type SaveFile = { kind: 'tinker'; version: 1; graph: Graph; params?: Record<NodeId, { W: number[]; b?: number[] }> };
type Preset = SaveFile & { title: string; prompt?: string; locked?: NodeId[] };
function parseFile(json: unknown): { ok: true; file: SaveFile | Preset } | { ok: false; error: string };
```

Saved params whose shapes do not match the graph are dropped with a notice, not an error. Bundled presets live in `web/src/presets/*.json`; `?preset=<name>` loads a bundled preset on start, and a preset file can also be opened through Load.

## 13. Commands

From the repo root: `pnpm install`, `pnpm test`, `pnpm typecheck`, `pnpm lint`, `pnpm build`, `pnpm dev` (Vite on :5173), `pnpm e2e` (playwright-core against `pnpm preview`, added by `tinker-ysl.5`), `pnpm torch-ref` (runs the uv script; needs uv and network). Per package: `pnpm --filter @tinker/<core|engine|web> <script>`.
