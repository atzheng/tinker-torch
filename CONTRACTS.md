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
```

Initialisation follows `nn.Linear`: W and b both uniform on ±1/√in. Different seeds give different values; the same seed always gives the same values. Note the layout: torch stores `weight` as [out, in]; HostParams stores [in, out] so that the forward pass is `x @ W + b`. Fixtures (section 11) are written in HostParams layout.

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
- `data/grid.ts`: `grid1d(domain, 200)` and `grid2d(domain, 64)` return raw x plus standardised x.
- `data/batches.ts`: `batchSchedule(nTrain, batch, seed)` yields index arrays; each epoch shuffles with `makeRng(seed).fork('epoch' + e)`; `batch: 'full'` yields all rows in order every step.

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

Params after an edit: the train slice re-initialises the parameters of every node whose param shapes changed and keeps the others (`reconcileParams(oldHost, newGraph, newAnalysis, seed)` in `core/params.ts`).

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
  };
};
```

Details: `dropoutMask` is a 0/1 keep mask shaped [n][width of the dropout node]; the engine multiplies by mask/(1−p). Classification targets in `batch.y` are class indices stored as floats (0 or 1). `after.losses[k]` is the objective before update k+1 (so `losses[0] == loss`). GELU is the exact erf form (`nn.GELU()`); jax-js `gelu` defaults to the tanh approximation, so the engine must pass `approximate: false`.

Tolerance: 1e-5 absolute and relative on pred, loss and grads; 1e-4 on `after.params` after 5 steps. The case list is in the fixtures bead (`tinker-7x7.3`).

## 12. Files and presets (`core/file.ts`)

```ts
type SaveFile = { kind: 'tinker'; version: 1; graph: Graph; params?: Record<NodeId, { W: number[]; b?: number[] }> };
type Preset = SaveFile & { title: string; prompt?: string; locked?: NodeId[] };
function parseFile(json: unknown): { ok: true; file: SaveFile | Preset } | { ok: false; error: string };
```

Saved params whose shapes do not match the graph are dropped with a notice, not an error. Bundled presets live in `web/src/presets/*.json`; `?preset=<name>` loads a bundled preset on start, and a preset file can also be opened through Load.

## 13. Commands

From the repo root: `pnpm install`, `pnpm test`, `pnpm typecheck`, `pnpm lint`, `pnpm build`, `pnpm dev` (Vite on :5173), `pnpm e2e` (playwright-core against `pnpm preview`, added by `tinker-ysl.5`), `pnpm torch-ref` (runs the uv script; needs uv and network). Per package: `pnpm --filter @tinker/<core|engine|web> <script>`.
