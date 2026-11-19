# Tinker contracts

The interfaces that beads share. SPEC.md says what the tool does; this file fixes names, shapes and file locations so that beads built separately in parallel worktrees fit together. If a bead needs to change something here, change this file in the same commit and say so in the PR body.

## 1. Layout

```
packages/core/src/            pure TypeScript, no DOM, no jax-js
  graph.ts        Graph v2, Node, Edge types and zod schema; parseGraph (migrates v1) / serializeGraph (section 2)
  registry.ts     PieceType, CORE_PIECES, PALETTE (section 3)
  macros.ts       ready-made loss and penalty groups as primitive subgraphs (section 2)
  migrate.ts      version 1 -> 2 migration (section 2)
  legacy/         version 1 graph, registry and file, used by analyse, edit, layout, params, engine, web until tinker-8xw.2-.5 (the engine already uses version 2; the web reaches it through `web/src/legacyEngine.ts`, a temporary adapter that tinker-8xw.4 removes)
  common.ts       data/train schemas, ports and ids shared by both versions
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
  compile.ts      Graph v2 + Analysis -> Model: predict and loss plans, stable fusion, jit (section 7)
  pieces/<type>.ts  one file per PieceType (18); ENGINE_PIECES table in pieces/index.ts; log_loss.ts and cross_entropy.ts hold the fused forms
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

## 2. Graph format (`core/graph.ts`), version 2

The graph is an arbitrary DAG of operations. The user pins one or more outputs as predictions (ŷ) and exactly one as the loss (L). Training minimises L; `predict()` returns the ŷ ports. Roles are pins on output ports, never nodes. Model and objective regions are derived from the pins (ancestors of ŷ are the model; what L needs beyond that is the objective), never a fixed layout zone.

**Migration status.** `core/graph.ts`, `core/registry.ts`, `core/macros.ts`, `core/migrate.ts` and `core/file.ts` are version 2 and are exported as the namespace `v2` from `@tinker/core`. Analyse, edit, layout, params, the engine and the web app still consume version 1 (`core/legacy/*`, exported under the old names) until beads tinker-8xw.2 to .5 move them; then `legacy/` is deleted and the `v2` namespace is flattened into the root exports. `core/analyseV2.ts` (section 4) is also version 2, as `v2.analyse`, and `core/paramsV2.ts` as `v2.initParams(analysis, seed)` (section 5: the same init as version 1 for the same ids, over `analysis.paramNodes`, so a Linear read only through a Weights tap gets parameters); the engine (section 7) is version 2 too; section 8 describes version 1 until its bead rewrites it. Version 1 documents are still readable: `v2.parseGraph` migrates them.

```ts
type NodeId = string; // [a-z][a-z0-9_]*, unique
type ActivationFn = 'sigmoid' | 'tanh' | 'relu' | 'gelu';
type LinkFn = 'sigmoid' | 'softmax';
type NormKind = 'l2' | 'l1';

type Node =
  | { id: NodeId; type: 'input'; columns: string[] }          // subset of the dataset's columns, in order
  | { id: NodeId; type: 'target' }                             // the dataset's y
  | { id: NodeId; type: 'feature'; degree: number; sin: boolean; cos: boolean; products: boolean } // degree 1..5
  | { id: NodeId; type: 'linear'; out: number; bias: boolean }  // out 1..64
  | { id: NodeId; type: 'activation'; fn: ActivationFn }
  | { id: NodeId; type: 'dropout'; p: number }                  // 0 <= p < 1
  | { id: NodeId; type: 'add' }                                 // a + b, same shape
  | { id: NodeId; type: 'link'; fn: LinkFn }                    // sigmoid or softmax, drawn as a link piece
  | { id: NodeId; type: 'subtract' }                            // a - b, same shape
  | { id: NodeId; type: 'square' } | { id: NodeId; type: 'abs' }
  | { id: NodeId; type: 'huber'; delta: number }                // elementwise Huber of a difference, delta > 0, default 1
  | { id: NodeId; type: 'log_loss' }                            // per example: -(y log p + (1-y) log(1-p)); pred is a probability
  | { id: NodeId; type: 'cross_entropy' }                       // per example: -log p[class]; pred is probabilities, target a class index
  | { id: NodeId; type: 'mean' }                                // reduction over the batch (and elements): per-example -> single number
  | { id: NodeId; type: 'scale'; c: number }                    // c * x, any finite c
  | { id: NodeId; type: 'weights'; of: NodeId }                 // tap: reads W of the Linear `of` (not a bias)
  | { id: NodeId; type: 'norm'; kind: NormKind };               // sum w^2 (l2) or sum |w| (l1) of a weight matrix -> single number

type PortRef = string;          // "node" (its output) or "node.port"
type Edge = [from: PortRef, to: PortRef];

type Graph = {
  version: 2;
  nodes: Node[]; edges: Edge[];
  predictions: PortRef[];       // the ŷ pins, in order; may be empty
  loss: PortRef | null;         // the L pin
  data: DataConfig; train: TrainConfig;   // unchanged from version 1
};
```

- Also exported (under `v2`): `DatasetName` and the `*_NAMES`/`*_FNS`/`LINK_FNS`/`NORM_KINDS`/`NODE_TYPES` const lists, `NodeBody`, `NodeOf`, `parsePort`, `resolveEdge(nodes, edge)`, `resolvePin(nodes, ref)` (a pin must name a node and, if it has a port, `out`), `inputPortsOf(type)`, `outputPortsOf(type)`. Types come from the zod schema.
- `parseGraph(json)`: version 2 as written, or version 1 migrated (below). Fills defaults (`standardize: false`, `huber.delta: 1`, `predictions: []`, `loss: null`). Rejects unknown fields and types, duplicate ids, a `weights.of` that is not a Linear, a pin that is not a node's `out`, and the same node pinned twice as ŷ. One plain sentence on failure. It does not check the pins' widths, that L is a single number or that L depends on a parameter: that is analyse (tinker-8xw.2).
- `serializeGraph(graph)`: always version 2, stable key order (`version, nodes, edges, predictions, loss, data, train`), two-space JSON. `loss: null` is written, not omitted.
- Ports. Every node has the one output port `out` (a port ref still allows more later). Input ports: `add` and `subtract` have `a` and `b`; `log_loss` and `cross_entropy` have `pred` and `target`; `input`, `target` and `weights` have none; every other piece has one input port `in`. A bare node id means `out` as a source and the single input port as a destination.
- Every input port takes exactly one edge; an output port may feed several.
- `weights.of` is a node field, not an edge: the tap reads a parameter, so it creates no data dependency and no cycle. The canvas draws it as a dashed wire from the Linear to the Weights node. Extension point for the scalar-level bead (tinker-9io): scalar weights on wires, a variadic Sum, a Bias and per-column inputs are new node types or a richer `of`; nothing here forbids them.
- ŷ and L are pins, so the same node can be both a ŷ and part of the loss path (the usual case: ŷ feeds the loss). A graph with no ŷ still trains when L is valid. Rules that analyse enforces: every ŷ width equals the dataset target (1 for regression and binary; numClasses for softmax classification); L is a single number that depends on at least one parameter.

### Version 1 migration

`parseGraph` migrates a version 1 document to a version 2 graph with identical math, widths and parameter counts (`core/migrate.ts`; tests run every engine fixture graph and bundled preset):

| Version 1 | Version 2 |
| --- | --- |
| Output(sigmoid or softmax) | `link` node, same id and fn; it is the ŷ pin |
| Output(identity) | removed; its source is wired to its consumers and the ŷ pin sits on that source |
| Loss(squared_error / absolute_error / huber(delta)) | `subtract(pred, target)` `{id}_diff` -> `square` / `abs` / `huber` `{id}_err` -> `mean` |
| Loss(log_loss / cross_entropy) | `log_loss` / `cross_entropy` `{id}_err` -> `mean` |
| the Loss's own id | the group's final node (the `mean`, or the penalty's `add`) and the L pin |
| penalty (kind, lambda) | for each Linear feeding the loss, in node order: `weights` `{id}_w_{lin}` -> `norm` `{id}_n_{lin}`; summed by `add` `{id}_sum{k}`; `scale(lambda)` `{id}_pen`; `add(a = mean, b = pen)`. The mean is then `{id}_mean`. A penalty with no Linear feeding the loss is dropped (it was 0) |
| everything else | unchanged, same id (so saved parameters keep their keys) |

The ŷ pin is the Output feeding the first loss's `pred`; with no loss, or a `pred` not fed by an Output, every Output is pinned. Only the first Loss gets the L pin. Generated ids are made unique against all existing ids (`_2`, `_3`, ...). Edges keep their order, with rewritten ends; the group's internal edges follow. Fusion (sigmoid then log loss, softmax then cross-entropy) is the engine's job; the diagram stays semantic, so `log_loss` takes the probability.

### Ready-made groups: macros, not group nodes

The five classic losses and the weight penalty are palette macros (`PaletteEntry.kind === 'macro'`): inserting one adds ordinary primitive nodes and edges (`lossSubgraph`, `weightPenaltySubgraph` in `core/macros.ts`, which migration also uses). There is no group node type, so analyse and the engine see only primitives. A macro returns `{ nodes, edges, inputs, out }`: `inputs` names the destination ports to wire into (`pred`, `target`; `total` for the penalty) and `out` is the node whose `out` is the result. The caller (web gestures, tinker-8xw.4) supplies a `Namer` that makes unused ids, wires the inputs, and sets the L pin.

## 3. Piece registry (`core/registry.ts`)

`PieceType` is the `type` of the nodes above (18 types). The palette (`PALETTE`) has one `piece` entry per type, with Sigmoid and Softmax as two entries of `link` and L2 and L1 as two entries of `norm`, plus six `macro` entries: `loss_squared_error`, `loss_absolute_error`, `loss_huber`, `loss_log_loss`, `loss_cross_entropy` and `weight_penalty`. The four activations are one type with `fn`.

```ts
// core/registry.ts
type Family = 'inputs' | 'features' | 'linear' | 'activations' | 'structure' | 'links'
  | 'errors' | 'losses' | 'reductions' | 'math' | 'parameters';   // drop-to-swap only within a family
// inputs: input, target. structure: dropout, add. links: link. errors: square, abs, huber.
// losses: log_loss, cross_entropy. reductions: mean. math: subtract, scale. parameters: weights, norm.

// What flows along a wire.
type Shape =
  | { kind: 'example'; width: number }          // one row of `width` numbers per training example
  | { kind: 'scalar' }                          // a single number
  | { kind: 'param'; rows: number; cols: number }; // a weight matrix, from a Weights tap
type ShapeError = { error: string };            // one plain sentence
type ShapeCtx = {
  dataColumns: number; numClasses: number;
  weightShape(linearId: NodeId): [rows: number, cols: number] | undefined; // [in, out] of that Linear's W
};
type CorePiece<N extends Node> = {
  family: Family;
  inputPorts: string[];                         // section 2
  defaults: () => Omit<N, 'id'>;
  outShape(node: N, inShapes: Record<string, Shape>, ctx: ShapeCtx): Shape | ShapeError; // missing key = port not connected
  paramShapes(node: N, inWidth: number): { W?: [rows: number, cols: number]; b?: [number] }; // W is [in, out]; Linear only
  torch(node: N, inWidth?: number): string;
};
const CORE_PIECES: { [T in PieceType]: CorePiece<Extract<Node, { type: T }>> };

type PaletteEntry =
  | { kind: 'piece'; key: string; label: string; family: Family; make: () => NodeBody }
  | { kind: 'macro'; key: string; label: string; family: 'losses' | 'parameters';
      make: (name: Namer, opts?: { linears?: readonly NodeId[] }) => Subgraph };
const PALETTE: PaletteEntry[];

// engine/src/pieces/index.ts
const ENGINE_PIECES: { [T in PieceType]: EnginePiece<Extract<Node, { type: T }>> };   // tinker-8xw.3
// web/src/canvas/drawings/index.ts
const WEB_PIECES: { [T in PieceType]: WebPiece<Extract<Node, { type: T }>> };          // tinker-8xw.4
```

Shape rules (`outShape`):

- input: `example(columns)`, error with no column. target: `example(1)`.
- feature, linear, activation, dropout, link: need an `example` input (a scalar or weight matrix is an error sentence). feature `example(d·degree + (sin ? d : 0) + (cos ? d : 0) + (products ? d(d−1)/2 : 0))`; linear `example(out)`; the rest keep the width.
- add, subtract: both inputs must have the same shape (width for examples); the result has it. This is how the penalty total is formed from two single numbers.
- square, abs, huber, scale: elementwise, any shape passes through.
- log_loss: `pred` and `target` are `example`; pred width 1; result `example(1)`. cross_entropy: same, pred width any; result `example(1)`. Whether pred width matches numClasses and the target is analyse's job.
- mean: `example(w)` -> `scalar` (mean over the batch and the elements).
- weights: `param(rows, cols)` from `ctx.weightShape(of)`; error when that Linear's input width is unknown. norm: `param` -> `scalar`.

Feature output layout (the order of columns; the engine and the torch fixtures use it): `x^p` for p = 1..degree (power-major, then column, so power 1 is x itself), then `sin(x)` if `sin`, then `cos(x)` if `cos`, then `x_i * x_j` for i < j in row-major pair order if `products`. sin and cos apply to the raw column values (no frequency scaling).

Parameter count of a Linear: in·out + (bias ? out : 0). Every other piece has 0; the penalty's norms read Linear weights but own nothing. The penalty covers Linear weight matrices only, never biases.

## 4. Analysis (`core/analyseV2.ts`, exported as `v2.analyse`)

Graph v2 analysis. The version 1 `core/analyse.ts` (root export `analyse`) stays until the engine and web move (tinker-8xw.3, .4); then this file becomes `core/analyse.ts`.

```ts
// DataInfo is defined in core/common.ts and re-exported by graph.ts.
type DataInfo = { columns: string[]; kind: 'regression' | 'binary'; numClasses: number };
type Pin = { role: 'loss' } | { role: 'prediction'; index: number };       // index into graph.predictions
type Issue = { at: { node?: NodeId; edge?: number; pin?: Pin }; message: string }; // edge = index into graph.edges
type Region = 'model' | 'objective';
type ParamShapes = { W?: [rows: number, cols: number]; b?: [number] };       // W is [in, out]
type Analysis = {
  order: NodeId[];                          // topological; what the roles need (below)
  shape: Record<NodeId, Shape>;             // output shape per node; absent where unknown
  inWidth: Record<NodeId, number>;          // example width entering the first input port, when known
  outWidth: Record<NodeId, number>;         // example width out; absent for scalar / param / unknown
  edgeShape: (Shape | null)[]; edgeWidth: (number | null)[];   // per graph.edges index
  params: Record<NodeId, number>;           // per node whose input width is known (0 for most)
  paramShapes: Record<NodeId, ParamShapes>; // per Linear whose input width is known
  paramNodes: NodeId[];                     // Linears with live parameters, in node order
  totalParams: number;                      // sum of params over paramNodes
  predictions: NodeId[];                    // nodes carrying the ŷ pins, in pin order
  lossNode: NodeId | null;                  // node carrying the L pin
  region: Record<NodeId, Region>;           // absent = neither
  modelOrder: NodeId[];                     // order restricted to the model region
  paramWires: { linear: NodeId; weights: NodeId }[];   // weights.of wires, drawn dashed
  fusable: { link: NodeId; loss: NodeId }[];           // sigmoid -> log_loss, softmax -> cross_entropy
  issues: Issue[]; trainable: boolean;      // trainable = no issues
};
function analyse(graph: Graph, data: DataInfo): Analysis;
```

Trainable means no issues. Each failure is one Issue with one plain sentence, at the root cause only (a node whose upstream shape is unknown gets no issue of its own). `analyse` does not assume `parseGraph` ran: bad edges, unknown pins, a missing `weights.of`, duplicate ids and cycles all become issues.

Shapes come from the registry (`outShape`, section 3). Add and Subtract need equal shapes, no broadcasting; the mismatch is one issue at the node and the wire into `b`. A doubly fed port uses the first wire; each later wire is an issue with its `edge` and the receiving `node`. A cycle is one issue at its first node; those nodes and anything downstream have no shape and are never in `order`.

Roles, each an issue when it fails:

- L: a pin must be set ("No loss is pinned ..." with `at: {}`) and name a node (`at.pin`). Its shape must be a single number (an example-shaped L says to add a Mean; a weight matrix is also refused), and its ancestors must include a parameter (a Linear, or a Weights tap of one). Pin issues carry `pin` and the pinned `node`.
- ŷ: several pins are allowed (the fit plot picks one, default the first); `modelOrder` is the union of their ancestors. A pin naming no node, or the same node pinned twice (one issue on the duplicate), is an issue. Each pin is checked on its own: it must be per-example with width 1, or `numClasses` when the pinned node is a softmax link; a softmax ŷ needs a `binary` (classification) dataset. A graph with no ŷ pin still trains.
- Log loss and cross-entropy take probabilities: their `pred` must come from a `link` of the matching fn (sigmoid for log loss, softmax for cross-entropy; issue at the node and the pred wire otherwise), the dataset must be `binary`, and cross-entropy's pred width must be `numClasses`. A matching pair is listed in `fusable`.
- The 10,000 parameter limit applies to `totalParams` (issue with `at: {}`).

Regions and order. Ancestry follows data wires (the first wire into each port). Model = ancestors of the ŷ pins (including the pinned nodes). Objective = ancestors of L that are not in the model. `order` is the topological order of model plus objective when L or a ŷ resolves, so dangling nodes are left out; otherwise every acyclic node. `modelOrder` is what `predict()` runs (dropout off). Wires from a Linear to a Weights tap (`weights.of`) are not data wires: the tap reads parameters, so it pulls in neither the Linear's inputs nor an ordering constraint, and a Linear read only through a tap is not in `order` (the engine does not run its forward pass). Such a Linear is still live: `paramNodes` lists every Linear in `order` plus every Linear read by a Weights node in `order`, and `totalParams` and the engine's parameter tables use that list, not the keys of `params`. `paramWires` lists every Weights node whose `of` is a Linear, for the dashed wire.

`inWidth`/`outWidth`/`shape` have no key where unknown (never NaN or null). The records are prototype-free, so lookups by node id are safe for any id.

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
type Model = { readonly graph: v2.Graph; readonly analysis: v2.Analysis }; // graph version 2; plus private compiled closures
type Params = { readonly __brand: 'Params' };                        // opaque, owns jax arrays
type TrainState = { params: Params; opt: unknown; step: number; key: unknown };
type StepResult = { state: TrainState; loss: number; diverged: boolean }; // loss before the update; diverged if non-finite

interface Engine {
  compile(graph: v2.Graph, data: DataInfo): Model;            // throws Error(first issue) when !analysis.trainable
  init(model: Model, seed: number): Params;                   // = fromHost(v2.initParams(analysis, seed)); one entry per analysis.paramNodes
  fromHost(model: Model, host: HostParams): Params;
  toHost(params: Params): HostParams;                          // SPEC's read(); copies
  write(params: Params, nodeId: NodeId, values: { W?: Float32Array; b?: Float32Array }): Params; // consumes params
  startTraining(model: Model, params: Params, seed: number): TrainState; // consumes params; optimizer state from graph.train
  step(model: Model, state: TrainState, batch: Batch): StepResult;       // consumes state; dropout on
  predict(model: Model, params: Params, x: Float32Array, n: number): Float32Array[]; // dropout off; one [n, width] array per ŷ pin, in pin order; [] with no ŷ pin
  evaluate(model: Model, params: Params, batch: Batch): number;          // the L pin's value (a single number), dropout off
  lossGrid(model: Model, params: Params, batch: Batch, axes: [ParamAxis, ParamAxis], a: Float32Array, b: Float32Array): Float32Array; // [a.length, b.length]
  trainMany(model: Model, seeds: number[], data: Batch[], steps: number): ManyResult[]; // vmap over k copies
  disposeModel(model: Model): void;                            // frees the jitted functions; idempotent
  clone(params: Params): Params;
  dispose(x: Params | TrainState): void;
  liveArrays(): number;                                        // for the leak test; method chosen by the spike
}
type ParamAxis = { node: NodeId; tensor: 'W' | 'b'; index: number };
type ManyResult = { params: HostParams; diverged: boolean };         // one trained copy of trainMany
function createEngine(): Promise<Engine>; // awaits jax init(); Wasm device; never WebGPU in v1
```

- "Consumes" means the caller must not use that value again; the engine disposes it or reuses its arrays. Everything else borrows. Callers that need to keep a Params (best-so-far, pins) take `clone` or `toHost`.
- The objective is the value at the L pin, a single number: whatever the graph computes (Mean of an error, plus a Scale of Norms of Weights taps, ...). `step` minimises it; `evaluate` reads it with dropout off; `predict` returns the ŷ pins' values with dropout off (a link is applied, so they are probabilities). The engine knows no loss names and no "penalty": a Weights tap reads the Linear's `W` (never `b`), so gradients from the objective reach `W` through the tap as well as through the forward pass. Shapes: per-example values are `[n, width]`; `target` is `[n, 1]` (a class index as a float for classification); Mean and Norm give a scalar; log_loss and cross_entropy give `[n, 1]`.
- Two plans are made at compile time. The predict plan is `analysis.modelOrder` (ancestors of the ŷ pins). The loss plan is `analysis.order` restricted to what the L pin reaches (a ŷ branch the loss does not use is not run, and its Linear is not read, in a loss pass). A Target in the predict plan reads zeros. Parameters come from `analysis.paramNodes` and `paramShapes`, not from `order`: a Linear read only through a Weights tap has parameters, is initialised and trained, and has no forward pass. A parameter a pass does not read is disposed by that pass; its gradient is zero.
- Stable fusion, in the loss plan only (the graph and the predict plan stay semantic): for each pair in `analysis.fusable` the loss node reads the link's INPUT, the logits, instead of the probability. log_loss then computes `logsumexp([z, 0]) − z·y` and cross_entropy `−logSoftmax(z)[class]`; both are finite for any finite logit (checked at ±5,000 in the loss, the prediction and the gradient). The link is still computed in the loss pass when something else uses its output. A log_loss or cross_entropy whose `pred` is not a link (analyse refuses it, so not reachable when trainable) clips the probability to [1e-7, 1−1e-7] before the log. There is no log-sigmoid primitive, so the "log(sigmoid)" form beyond log_loss has nothing to fuse yet.
- Huber follows `nn.HuberLoss(delta)`. Dropout follows `nn.Dropout` (inverted scaling 1/(1−p)).
- Optimizers: sgd = `optax.sgd(lr)`; momentum = `optax.sgd(lr, { momentum: 0.9 })` (equal to torch SGD with momentum 0.9, dampening 0); adam = `optax.adam(lr)` with b1 0.9, b2 0.999, eps 1e-8 (torch defaults). Adam is plain and can spike periodically near convergence (`tinker-df1`): on a noise-free sine fit (n=40, 1-8-8-1 tanh MLP, full batch, seed 1, lr 0.01 or 0.05) the loss, once near 1e-4, jumps every ~100-200 steps by up to 69x (lr 0.01) or 770x (lr 0.05) its running minimum and decays within a few steps. PyTorch 2.15 float32 `Adam` on the same data and init does the same (58x); SGD does not. Owner decision: do not damp it. Measured mitigations: larger eps (1e-7 to 1e-3), smaller b2 (0.99, 0.95) and 1/(1+t/k) lr decay (k 1000, 500) did not reliably remove it or removed it only by killing the lr (k 500); torch `amsgrad=True` still spiked early; a running max of the bias-corrected second moment removed it but was 3x-7x slower to reach small losses on presets. The UI carries no explanatory text. `mlp_tanh_adam_300steps` checks a long plain-Adam run against PyTorch.
- `step` never throws on divergence; it returns `diverged: true` and the state is unchanged.
- jax-js decisions (go/no-go `tinker-e1h.2`, from `docs/spike.md`; jax-js 0.1.27 and optax 0.1.3 stay pinned exactly):
  - Use `optax.adam` as is. It is accurate inside jit (matches torch Adam to about 1e-6 on the 5-step fixture); it is only inexact when run eagerly (float32 bias correction), which the engine never does. Switching optimizer means `compile` + `startTraining` again; `step` throws "This training state was started for a different optimizer; start training again." for a state from another optimizer, before consuming anything.
  - Sigmoid is computed as `0.5 * (1 + tanh(x / 2))` (jax-js `nn.sigmoid` has a NaN gradient below about −88), and the fused sigmoid + log_loss as `logsumexp([z, 0]) − z·y` (jax-js `softplus`/`logSigmoid` overflow in float32). These keep large logits from reporting a false "diverged".
  - ReLU, abs and GELU use the jax-js built-ins (`nn.relu`, `np.abs`, `nn.gelu` with its default tanh approximation). Their at-zero gradients (relu'(0) = 0.5, |x|'(0) = 1) differ from torch; accepted.
  - Laziness: jax-js evaluates lazily, so `step` and `trainMany` read the loss back every step. Without a read the pending computation grows with each step and the run slows and runs out of memory.
  - `step` keeps a `.ref` of the pre-step state until the loss is read, so a diverged step can return the state unchanged; the leak test includes a diverged step.
  - `liveArrays()` counts the Wasm backend's live buffers through the test-only hook in `docs/spike.md` (it finds jax-js's private slot map by intercepting `Map.prototype.set` until the first allocation). It is tied to the pinned version and is never used by the app: the hook lives in `engine/src/probe.ts` and is off unless a test calls `installProbe()` and then loads the engine with a dynamic `import()` (see `test/leak.test.ts`); without it `liveArrays()` returns 0. It does not count arrays of fewer than 128 elements with all values equal (jax-js keeps those as symbolic constants without a buffer; no memory, so harmless). The package exports it as `@tinker/engine/probe` so a dev build of the app can do the same.
  - Engine structure fixed by `tinker-hj4.1` (later engine beads fill files and flags, they do not change these):
    - `EnginePiece<N> = { forward(node, inputs: Record<port, JArray>, layer, ctx): JArray }` in `pieces/types.ts`; `forward` consumes its inputs and layer. `compile` walks the plan of each pass (above); an optimizer (`optim.ts` `OPTIMIZERS`) with `implemented: false` makes `compile` throw "The <name> optimizer is not implemented yet." `layer` is the node's own Linear's parameters, or for a Weights tap the parameters of the Linear it reads. `walk` hands out one reference per read (the Linear's forward pass and each tap) and `.ref`s a node's value for all but its last consumer.
    - `PieceCtx.fused(nodeId)` tells a log_loss / cross_entropy piece that its `pred` input is the logits (the compiler rewired the edge, see above).
    - Batch x is `[n, dataColumns]` (all dataset columns, `DataInfo.columns`); the Input piece selects its own `columns`.
    - A consumed `Params` or `TrainState` throws "already consumed or disposed" instead of touching freed arrays; `dispose` is idempotent. `step` consumes the state and its `params`: after a successful step the old `state.params` is dead; after a diverged step the returned state reuses the same `Params` object.
    - `diverged` is true when the loss is non-finite or any updated parameter is non-finite (a non-finite update is never kept); `loss` is then the value read, which may be finite. A batch that does not fit the model throws before anything is consumed.
    - `TrainState.key` is created from the seed; each `step` splits it (`np.unstack(random.split(key, 2))`), samples one keep mask per Dropout node on the loss path inside the jitted step but outside `valueAndGrad` (`PieceCtx.dropoutMask(nodeId)` hands it to the piece; random ops traced under `valueAndGrad` and `np.take` index arrays would become constants of the compiled function, which `disposeModel` frees with the jits) and stores the next key in the new state. Pieces must not create array constants (index arrays, `random.*`) inside the differentiated walk. A diverged step returns the old key. `predict` and `evaluate` run with `PieceCtx.train = false` (Dropout is the identity).
    - Jitted functions (predict, evaluate, step, the vmapped `trainMany` step, the `lossGrid` functions) are built lazily per model and freed by `disposeModel(model)` (`tinker-klv`); a model that is dropped without it leaves its set until garbage collection. `@jax-js/jax` 0.1.27 is patched locally (`patches/`, `docs/spike.md` section 4b) because it leaked jaxprs through global caches; re-check the patch whenever the pin changes. `disposeModel` is idempotent; afterwards every method that takes the model throws "This model was disposed.". Params and TrainStates made for it stay valid arrays (`toHost`, `write`, `clone`, `dispose` work) but cannot be stepped or evaluated without a new model. The web app disposes the old model whenever it recompiles.
  - `trainMany` (`tinker-hj4.6`): copy i is `init(model, seeds[i])` + `startTraining(model, .., seeds[i])` + `steps` full-batch `step` calls on `data[i]`, vectorised: `jit(vmap(step))` over params, optimizer state (the scalar step count is broadcast to `[k]`), the dropout key (`random.key(seeds[i])`, so each copy has its own masks) and `x [k, n, d]`, `y [k, n]`. All `data[i]` must have the same `n` (it throws "All copies need training sets of the same size." otherwise); there is no padding or mini-batching. Everything is borrowed. The finite check reads one `[k]` flag every step (laziness). A copy whose loss or updated parameters become non-finite freezes at its last finite parameters, optimizer state and key and is returned with `diverged: true`; the other copies are unaffected (bit-identical to a run without it); when every copy has diverged the loop stops early. Arrays are passed to the jitted function as arguments, never captured; all arrays are disposed before returning (also on a throw). The vmapped step is cached per compiled model like the other jitted functions.
  - Ownership pitfalls found by the spike: optax `update` disposes its `params` argument; destructuring a jax Array leaks the parent (use `np.unstack` for split keys).

## 8. Edits (`core/edit.ts`)

Every canvas action is one pure function `(graph, args) -> EditResult`, with `type EditResult = { ok: true; graph: Graph; select?: NodeId } | { ok: false; error: string }`. The store applies `ok` results through the undo history and shows `error` as a toast; no edit mutates its input.

```ts
insertOnWire(g, edgeIndex, piece: Omit<Node,'id'>)       // new id = `${type}${k}`, smallest free k
swapPiece(g, nodeId, piece: Omit<Node,'id'>)              // same family only; carry over settings with the same name and valid value
connect(g, from: PortRef, to: PortRef)                    // replaces whatever fed `to`; rejects cycles
deleteNode(g, nodeId)                                     // reconnects upstream to downstream when the node had one input and widths still match
deleteEdge(g, edgeIndex)
rewireEdge(g, edgeIndex, end: 'from'|'to', ref)           // move one end of a wire; one edit = one undo step; like `connect`, REPLACES any other wire already feeding the new port (the UI dims that wire while dragging); rejects cycles; old and new destination must not be locked
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
- The train slice exposes `host` (immutable HostParams snapshot, at most once per frame, null while the graph is invalid) and `predictGrid(x, n): Float32Array | null` (current parameters, dropout off, post-link, `[n, width]`; null when there is no compiled model). Keep `n` fixed per dataset: each new `n` retraces once. It also exposes `lossSurface(axes: [ParamAxis, ParamAxis], a, b): Float32Array | null` (`engine.lossGrid` on the current model and params over the whole training set, dropout off; `[a.length, b.length]` row-major; null when there is no compiled model or the call fails). The loss-surface view keeps the grid fixed at 50×50.
- The run loop is `requestAnimationFrame`: each frame runs up to `speed` steps within an 12 ms budget, then redraws once. Results views redraw at most once per frame.
- Canvas choices fixed by tinker-27m.3 (`packages/web/src/canvas`, `state/viewSlice.ts`, `state/derived.ts`):
  - Selection is `viewSlice.selection: { kind: 'node'; id } | { kind: 'wire'; from; to } | null` with `select(sel)` and `selectNode(id)`. A wire is named by its stored endpoints, never by edge index; read it through `useSelection()` (null when it no longer exists) and `edgeIndexOf(graph, from, to)`. Building gestures (27m.4) should call `selectNode(result.select)` after an edit.
  - Hit-testing: every piece is `<g data-testid="node-<id>">` and every wire has a line `<g data-testid="wire-<i>">` (secondary target, fat transparent stroke) and a width label `<g data-testid="wire-label-<i>">` that is its sibling (not nested), painted above all pieces: the label is the primary wire target. The hit rectangles are owned by `LayerView` (`layerHitShapes` in `canvas/pure.ts`, tested in Node): a label rectangle is at least 32 screen px each way (moved up when two labels overlap), a piece rectangle is its box grown to 32 px tall and never reaches into the gap where labels sit; drawings must not add their own hit shapes. Presses go through `pointerDownWith(handlers)` (`canvas/tap.ts`, a wrapper on `startDrag` that also stops the event so the background does not clear the selection); 27m.4 and ran.1 add `onStart/onMove/onEnd` there. The svg background tap clears the selection; Escape too.
  - `WebPiece.Layer` now takes `LayerProps` (`node`, `w`, `h`, `inWidth?`, `outWidth?`, `paramCount`, `weights?: { W, rows, cols }`) and is drawn with its box's top-left at (0, 0) (LayerView does the translate). Shapes use the `tk-*` classes in `canvas/canvas.css` so error and selected states recolour them. `Fields` edit only through core edits and `graphSlice.apply`.
  - Weights for drawing come from `useWeights(analysis)` (`state/derived.ts`): the train slice's `host` when it has one (read structurally as `s.host`, so tinker-27m.2 should name its HostParams snapshot `host`), else `initParams(graph, analysis, graph.train.seed)`. Missing entries and arrays whose length does not match `in * out` are skipped, never an error.
  - Issue placement (`placeIssues`): an issue with an edge that has a wire goes on the wire; only the Output-versus-loss compatibility issues (link does not suit the loss, output width) go on the Output's wire into `loss.pred` (`wireOfNode`); "needs an input", cycles, duplicate ids and every other node issue stay at the piece, and the wire inspector shows exactly the sentences the callout shows; other node issues sit under the piece; anything else is one global line at the top of the canvas. Every piece named by an issue gets a red dashed outline.
- Building gestures (tinker-27m.4, `canvas/dnd.ts`, `canvas/actions.ts`, `palette/Palette.tsx`): a gesture in progress lives in a small external store `dnd` (not a slice): `begin/move/end/cancel`, and the canvas registers `setHandler({ resolve, commit })` while mounted. `resolveDrop(payload, graph, layout, widths, scale, point, hitPx)` (`hitPx` 20 mouse, 32 touch) and `editForDrop(graph, payload, target, opts?)` (`opts.locked`, from `lockedIds()` in `state/locks.ts`, which every web edit call site passes) are pure. Drop targets use `layerHitShapes` (a wire's label first, the wire line within 10 px (mouse) or 16 px (touch) as fallback; pieces before wire lines). `layerHitShapes` takes an optional `outputs` set and then adds a `{ kind: 'port', node }` shape (32 px square, centred `PORT_INSET` units inside the right edge of the piece so it stays clear of the label), painted between pieces and labels; `data-testid` `port-out-<id>` is the handle to drag a wire from. A drop applies the edit through `graphSlice.apply` and selects `result.select`; a refused edit shows its sentence in `<Toast/>` (`data-testid="toast"`, reads `graphSlice.error`). An edit during a run is applied as usual and the train slice recompiles and pauses on a structural change. On touch the palette scrolls vertically (`touch-action: pan-y`) and a drag begins on mostly horizontal movement. Delete/Backspace and the inspector `inspector-delete` button call `deleteSelection()`; Ctrl/Cmd+Z, Shift+Ctrl/Cmd+Z and `btn-undo`/`btn-redo` in the top bar.
- Neuron view (tinker-ran.1, `canvas/NeuronView.tsx`, `canvas/neuronPure.ts`): `viewSlice.grain: 'layer' | 'neuron'` with `setGrain` (clears an edge or unit selection, never touches the graph); `grain-toggle` is a pressed-state button ("Neurons", `aria-pressed`, at least 32 px) in a 44 px header strip at the top of the canvas region (`Canvas.tsx` `HEADER_H`: the drawing is fitted below it, so it never covers a piece or wire); it is not in the top bar. A store subscription clears an edge or unit selection as soon as a graph or data change makes it stop fitting (`neuronSelectionLive` in `canvas/pure.ts`), so undo or regrowing a layer cannot resurrect it. `HostParams` has no shapes, so the store remembers the graph each `host` snapshot was published with (`state/hostShape.ts`) and `weightAt` refuses a weight whose [in, out] differs from the shape the snapshot was built for. `Selection` gains `{ kind: 'edge'; node; i; j }` (weight `W[i][j]` of a Linear, key into `NeuronLayout.edges` and `HostParams` index `i*out+j`) and `{ kind: 'unit'; node; index }` (a circle of a unit column, `index` as in `NeuronLayout.units`); `useSelection(analysis)` drops an edge that no longer fits the widths. The canvas passes `{ nodes: boxes, wires, size }` of the neuron layout wherever a `LayerLayout` is expected (drops, callouts, inspector anchor). Edge lines are painted imperatively from `host` (no React render per frame; `useWeights(analysis, false)` does not follow `host`, `useInitialWeights` gives step 0). Hit-testing: a press on an edge picks the edge whose line is nearest (`nearestEdge`, within half the 20/32 px hit width), units win over edges (`neuronHitShapes`: a unit's cell is the full column pitch tall and at least 32 px wide, so cells tile at any scale), then pieces. Mouse hover over an edge shows its value (`edge-tip`); tap selects it and the inspector shows it (`inspector-weight`, `inspector-bias`). `data-testid`: `neuron-view`, `edge-<node>-<i>-<j>` (its `.tk-edge-line` carries `data-sign` `pos|neg|none` and `stroke-width`), `unit-<node>-<index>`, `edges-<node>` (the "N edges" label of a collapsed Linear), `edge-selected`, `edge-tip`. Delete does nothing for an edge or unit selection. Unit +/- (tinker-ran.2): `unit-minus-<node>` / `unit-plus-<node>` buttons in one row under the whole drawing (below the lowest box, so no edge or unit ever lies under them), each pair centred under its Linear's column (also a collapsed one, so a column wider than 8 can shrink back), `setLinearOut` with `{ locked: lockedIds() }` through `apply` (one undo step); the canvas reserves `UNIT_BTN_ROOM` extra height under the neuron drawing. Hand-set weights: a vertical drag on an edge (or on a plus circle for its bias; 0.01 per screen px, up raises it, within ±100) or a value typed in the inspector (`inspector-weight-input`, `inspector-bias-input`, committed on Enter or blur) goes through `canvas/weightEdit.ts` `beginWeightEdit`; it refuses with a toast while running ("Pause training to set a weight by hand."), after a divergence ("Reset the run before setting a weight by hand.") and for a locked piece. A drag that begins on a non-plus unit cell hands over to the nearest edge; if the graph changes mid-drag (Ctrl+Z) the preview is taken back and the edit ends. The drag is a tap-or-drag by the 4 px slop (a drag never selects through the tap path; it selects the edge or unit when it starts). Previews call the one train-slice action `writeParams(node, { W?, b? })` (`engine.write`, optimizer state and loss history kept, `host` republished, never writes into a published array), at most once per frame; the release adds ONE undo step: the graph is copied (`{ ...graph }`, same content, new object) and applied, and `state/weightHistory.ts` records the layer's values before and after, written back by `writeParams` when undo or redo crosses that entry. Undo restores the layer as it was when the drag began (weights trained in between are not kept). The write-back happens only when undo or redo crosses the record, never for the original apply. `weightHistory` keeps a generation counter bumped by Reset, Restore best, loadHost and a fresh start after a divergence (`resync` fresh): a record of an older generation is skipped on undo and redo (the graph still moves), because the weights it describes no longer exist. Known and accepted: the same graph reached by different undo paths can show different weights after reconcile (weights are not part of graph history). `LOCKED_MESSAGE` is exported from core `edit.ts`. Palette drops work on boxes in neuron view; wire dragging from output ports is layer view only. Pan and zoom (tinker-qvj, `canvas/viewport.ts`, `canvas/useViewGestures.ts`; neuron view only, layer view is unchanged): on a coarse pointer the view never starts below the scale that makes a unit cell 32 px tall (`touchScaleFloor(LAYOUT.unitPitch)`, about 0.73; left-aligned, the user pans to the rest), on a fine pointer it starts at the fit. Two touch pointers pinch to zoom and pan together anywhere (the second press never reaches a piece, a one-finger drag already begun is cancelled); Ctrl or Cmd with the wheel zooms about the cursor (a trackpad pinch), the plain wheel pans; a drag on empty background pans with a finger or the mouse (a tap there still clears the selection). Scale is limited to the fit scale and 3; `clampView` keeps the drawing reachable. `reset-view` ("Reset view", 32 px, in the header strip beside the grain toggle) appears while the view is moved and restores the start view; switching grain resets it. Hit shapes are computed from the on-screen scale, so cells and edge widths keep their 32 px size at any zoom. Block names (tinker-71r): a piece drawn as a unit column shows its name (Input, Linear, the activation's function, Output, Target; `unit-name-<node>`) centred in a strip (text kept within the 36-unit column, ellipsis if needed) of `LAYOUT.unitLabelH` (20) units that `layoutNeurons` reserves at the top of that box (the box is `unitLabelH` taller and unit `i` sits at `box.y + unitLabelH + (i + 0.5) * unitPitch`); the strip is part of the node's hit shape, so a tap on the name selects the piece, and no edge, wire, unit or other box crosses it (tested). Collapsed Linears and other boxes name themselves through `NodeShell`. `LAYOUT` gains only `unitLabelH`.
- `data-testid` on: every palette entry (`palette-<key>`), every node (`node-<id>`), every wire (`wire-<index>`) and its label (`wire-label-<index>`), `btn-step`, `btn-run`, `btn-reset`, `step-count`, `param-count`, `status-diverged`, `fit-plot`, `loss-curves`, `grain-toggle`.
- Discoverability and rewiring (tinker-6f8, `canvas/dnd.ts`, `palette/Palette.tsx`, `hints.ts`):
  - *Hint exception.* The "no lesson text" rule (SPEC principle 5) has one exception, decided by the owner: one-line gesture hints. They are plain, one sentence each, few, and all live in `web/src/hints.ts` (`HINTS`): the palette header line, three canvas lines shown until the first edit (`canUndo` false), and the empty-graph line. The canvas lines show until the first edit of the session (a `canUndo` latch, not `canUndo` itself) and the connect and rewire lines are hidden in neuron view. Tooltips (`title`) on palette entries (`ENTRY_TITLES`, one phrase: what it is and what it connects to) and on the port and wire-end handles are the same kind of text. No other explanatory text may be added without the owner.
  - *Palette header.* A sticky "Pieces" title with the hint above the groups.
  - *Rewire.* `DragPayload` gains `{ kind: 'rewire'; edge; from; to }`. The destination end of a wire is a handle (`wire-end-<edge>`) that exists only while that wire is hovered (mouse, with a 200 ms grace so the pointer can cross onto it) or selected (touch: tap the wire or its label, then drag its end). It is a square `hitPx` wide (20 mouse, 32 touch) placed just outside the destination piece, ending at the piece's left edge, painted above wires and labels, and never over a piece. A tap on it selects the wire. Dragging it shows the input ports and the preview line from the wire's source; the original wire is drawn faint (`data-lifted`). Dropping on another piece's input port runs `rewireEdge(g, edge, 'to', port)`; dropping on the port it already ends at, on its own source or on empty canvas does nothing. Dropping on a port that already has a wire replaces that wire, as `connect` does; while dragging (rewire or new wire) the wire that would be replaced is drawn dimmed, red and dashed (`data-replace`). The source end is not draggable (drag from the output dot for a new wire, then delete the old one). Locked destinations refuse with the usual toast.
  - *Precedence.* The piece body always wins a press over wire handles: wire-end handles and wire hit strokes never overlap a piece box, and later handles (piece dragging in tinker-ong, role pins at output ports in tinker-8xw.4) must also sit outside the box or on the piece's own ports. A handle that must be touched is made reachable by selecting its wire or piece first, not by growing over a neighbour.
  - *Tap-to-add.* `dnd.arm(entry)` (same entry again, Escape, a tap that fits nothing, or a press outside the palette entries and the drawing cancels), `dnd.armed()`, `dnd.tapAt(x, y)`: a tap on the drawing resolves with the same `resolveDrop` as a drag drop and commits once, then disarms. A keyboard Enter or Space on an entry arms it. Any graph change (edit, undo, redo, preset or file load), a grain switch and a pan or zoom disarm it (Canvas effect on `graph`, `grain`, `loadEpoch`, `userView`). While armed the entry has `aria-pressed`, a banner says what to do, and the candidates are lit.
  - *Candidates.* `dropCandidates(entry, graph)` (pure) lists the wires and pieces an entry can land on; Canvas passes it to `LayerView` as `candidates` for a palette drag or an armed entry (for a wire or rewire drag: every piece with an input except the source). Lit wires and pieces carry `data-valid`; the one under the pointer also carries `data-drop`.
- Pinned runs (tinker-iej.2, `results/Pins.tsx`, `results/pinsModel.ts`): `viewSlice.pins: Pin[]` (oldest first) with `addPin` (refused, returns false, at `PIN_LIMIT` = 3: the Pin button is disabled rather than replacing the oldest) and `removePin`. A `Pin` is plain data: `{ id, slot, label, graph, host, history, finalLoss }`; `history` is decimated to at most `PIN_POINTS` = 200 points (`decimateHistory`: last point of each run, last non-null validation value of the run) and has no batch series; `label` is "activations · hidden widths · optimizer lr". The pin's `host` is the weights at pin time, so its fit is restored from them. For 1-D data the pin also stores `fit: { x, y }` (two plain Float32Arrays of the fixed 200-point grid, raw x, from `snapshotFit` / `predictGrid`; null for 2-D data, where nothing is overlaid) and `FitPlot` draws it thin under the live curve in the pin colour. A pin taken on another dataset is still drawn (in its own raw x, clipped to the plot's domain). Pins live outside the graph and its history: they survive edits, undo/redo, dataset changes and file loads (a pin from other data is still a loss curve to compare, but its numbers are on that data). Clicking a chip does `apply({ ok: true, graph })` (one undo step) then `loadHost(host)`, which starts a new run (history cleared; Pin is disabled until a step is taken). The loss curves draw each pin's train (alpha 0.3) and val (alpha 0.7) in `--pin-1..3` (`PlotColors.pins`) under the current curves and widen the axes to include them. `data-testid`: `pins`, `pin-add`, `pin-chip`, `pin-load`, `pin-remove`.

- **Visual language (tinker-zpr).** One node style; the type is told apart by name and glyph, never by shape or colour.
  - *Component.* `canvas/drawings/NodeShell.tsx` is the only thing that draws a node. A piece's `Layer` returns `<NodeShell piece name glyph? badge? slot? secondary? tag? role? inPorts outPort scale />`; it never draws its own outline, port or state colour. Neuron view draws collapsed Linears and other boxes through the same shell; its unit circles use the same tokens (`.tk-shape`) and the roles below. A later bead adding a piece (loss primitives, scalar weights and sums, link pieces) adds a `Layer` that calls `NodeShell` and nothing else.
  - *Anatomy* (box 112 x 56 abstract units, `LAYOUT.boxW/boxH`): body rounded rect `rx` 8 (`NODE_RADIUS`), 1.5 border (2 for a role); **name** top left, 15 px, weight 600, always drawn (tinker-71r: it is the piece's PALETTE name; an Activation shows its function (Sigmoid, Tanh, ReLU, GELU), a Loss its loss name; it is shrunk (to 10 units at least) to fit the box width and, if 10 units is still too wide, cut with an ellipsis (`fitName` in `canvas/pure.ts`), and below the fit scale raised, within that fit, so it stays at least `MIN_TEXT_PX` (8) on screen: it is a label, `pointer-events: none`, so a tap on it reaches the node's own hit area and selects the piece); **tag** top right, 11 px muted (qualifier such as Feature options), hidden when illegible; **glyph** right slot, at most the name's size (15 px, 13 when longer than 4 characters; baseline `h/2 + 11`, so it clears the name), for the operation (`ReLU`, `p=0.2`, `+`), or a custom `slot` (the Linear heat map) that must stay inside `slotBox(w, h)` = x `w - 52`, y `h/2 - 10`, width 40, height 32 node-local units (a KaTeX `foreignObject` goes there, right-aligned at `w - 12`; it must not catch pointer events); **secondary** bottom left, 13 px muted, one or two lines (widths, parameter count, settings; the first line is always drawn, the second only when legible); **ports**: an input dot on the left edge per input port, spread evenly as `inputPortAt`, an output dot at the right edge middle, all `PORT_R` 4.5 with the same fill and border. The draggable output handle (`port-out-<id>`, hit rectangle unchanged, `PORT_INSET`) is drawn over the static dot at the same spot. Padding is 12 units; text is system UI stack, never a network font.
  - *Roles* (the only special nodes): `data-role` on the node group, set by `roleOfNode(node)` in `NodeShell.tsx` (the single place that maps graph data to roles; graph v2 changes only it; neuron unit circles use it too, for their input, output and target units): `input` (Input and Target: data entering), `pred` (Output, the prediction y-hat), `loss` (Loss). The role variables hang on a bare `[data-role]` selector, so a role pin that is not a node can use them. A role changes only the two variables `--role` and `--role-fill` (border and tint, from `--role-input|pred|loss` and `--role-*-fill`) and shows a filled badge with its letter (x, y, y-hat, L) in place of the glyph; shape, size, ports and anatomy stay the same, so graph v2 (tinker-8xw.4) can move the same style onto a role pin by setting `data-role` on the pin. Palette entries carry the same `data-role` and preview the node (body, border, ports, badge).
  - *States* are applied by one rule each on `.tk-node` in `canvas.css`, never per type: **selected** (`data-selected`) accent border plus a 2 px accent ring (`.tk-ring`, 4 units outside the body); **issue** (`data-error`) red border plus a dashed red ring; **drop target** (`data-drop`) 3 px accent ring; **locked** (`data-locked`) the padlock badge on the top right corner; **hover** (pointer devices only) `--node-fill-hover`; **pressed** `--node-fill-press`; **focus** the focus ring. Wires use the same words: `data-selected` accent line and halo, `data-error` red, `data-drop` accent, 3 px.
  - *Wires.* One style: `--wire` line, 2 px (2.5 px with an issue), arrowhead. Width labels are diagnostics (tinker-t6o): monospace (`--font-mono`), `--diag-fg` on `--diag-bg` with a `--diag-border` outline, never a node colour; selected and issue states recolour the outline and text.
  - *Tokens* (`theme.css`, light and dark): node `--node-fill|-hover|-press`, `--node-border`, `--node-name`, `--node-secondary`; roles `--role-input|pred|loss`, `--role-*-fill`, `--on-role` (badge glyph); diagnostics `--diag-fg|bg|border`; type `--font`, `--font-mono`, scale `--fs-xs 11 / sm 12 / md 13 / lg 14 / xl 16`; spacing `--sp-1..4` (4, 8, 12, 16); elevation `--shadow-1|2`; region headers `--header-fg` with `.tk-region-title` / `.tk-panel-title` (11 px, 600, uppercase, 0.06em).
  - *Contrast* (WCAG, computed from the tokens; checked for light and dark): name text on every node fill at least 9.7:1; secondary text on node, hover, press and role fills at least 4.6:1; badge glyph on its role colour at least 5.9:1; node border, role borders and the accent ring against the canvas at least 4.6:1 (3:1 is the non-text minimum). Roles also differ in hue family (teal, violet, ochre), lightness and glyph, so none depends on colour alone; weights stay blue and orange, issues red with a dashed outline.

## 11. PyTorch reference fixtures

`tools/torch-ref/gen.py` is a uv script with inline metadata (`# /// script` with `dependencies = ["torch==<current stable>", "numpy"]`), run as `uv run tools/torch-ref/gen.py`. It writes `packages/engine/test/fixtures/<name>.json`, one file per case, and is opt-in: tests never run Python.

```ts
type Fixture = {
  name: string;                       // e.g. "linear_bias", "loss_huber_delta0.5", "mlp_relu_adam_5steps"
  graph: Graph;                       // a complete trainable graph in the section 2 format
  params: Record<NodeId, { W: number[]; b?: number[] }>;   // HostParams layout [in, out], float64 written in full
  batch: { x: number[]; y: number[]; n: number };
  expect: {
    pred: number[];                   // predict()[0], post-link, [n, width] row-major (empty when the graph has no ŷ pin)
    predictions?: number[][];         // version 2 cases: every ŷ pin in order (`pred` is the first)
    loss: number;                     // evaluate()
    grads: Record<NodeId, { W: number[]; b?: number[] }>;  // d loss / d params
    after?: { steps: number; params: Record<NodeId, { W: number[]; b?: number[] }>; losses: number[] }; // optimizer cases
    dropoutMask?: number[][];         // dropout cases: the engine test injects this mask instead of sampling
    predEval?: number[];              // dropout cases: dropout-off forward (what predict() returns)
    lossEval?: number;                // dropout cases: dropout-off objective (what evaluate() returns)
  };
};
```

Details: in dropout cases `pred`, `loss` and `grads` are the dropout-ON training forward and backward with the given mask (how the engine test injects the mask is the engine's choice); `predEval` and `lossEval` are the dropout-off values that `predict` and `evaluate` must return. At-zero conventions are pinned by fixtures: the L1 penalty has subgradient sign(0) = 0 (`penalty_l1_zero_weight`) and relu'(0) = 0 (`act_relu_zero_preact`). `dropoutMask` is a 0/1 keep mask shaped [n][width of the dropout node]; the engine multiplies by mask/(1−p). The engine test injects it through the internal `debugTrainPass(model, params, batch, { [dropoutNodeId]: mask })` in `engine/src/compile.ts` (non-jitted dropout-ON forward, loss and gradients; not on `Engine`), and compares `predEval` / `lossEval` with `predict` / `evaluate`. Version 1 graphs in fixtures are read with `v2.parseGraph` (migrated), so every version 1 case also checks the migration; the `prim_*` cases are version 2 graphs (primitive losses, Weights taps and norms, a Linear read only by a tap, no ŷ pin, two ŷ pins, an auxiliary objective, a link that also feeds something else, a DAG with dropout, a skip and a feature piece, a 300-step Adam run). Their torch reference evaluates the graph literally, primitive by primitive (a log loss on `sigmoid` probabilities), so it also checks the engine's fused forms. Classification targets in `batch.y` are class indices stored as floats (0 or 1). `after.losses[k]` is the objective before update k+1 (so `losses[0] == loss`). GELU is the exact erf form (`nn.GELU()`); the engine uses jax-js `gelu` in its default tanh form instead (see "Engine exceptions" below).

Gradients are not part of `Engine`, so the engine reference test reads them from one SGD step with learning rate 1 (grad = W before - W after).

Tolerance: 1e-5 absolute and relative on pred, loss and grads; 1e-4 on `after.params` after 5 steps. The case list is in the fixtures bead (`tinker-7x7.3`).

Engine exceptions (decided in `tinker-e1h.2`, because the engine uses jax-js built-ins and optax rather than torch-exact forms): `act_relu_zero_preact` and `penalty_l1_zero_weight` are skipped by the engine comparison test (they pin torch's at-zero subgradients); `act_gelu` is compared at 1e-4 (tanh-form GELU differs by about 3e-5). The fixtures stay as generated.

## 12. Files and presets (`core/file.ts`)

```ts
type SaveFile = { kind: 'tinker'; version: 2; graph: Graph; params?: Record<NodeId, { W: number[]; b?: number[] }> };
type Preset = SaveFile & { title: string; prompt?: string; locked?: NodeId[] };
function parseFile(json: unknown): { ok: true; file: SaveFile | Preset } | { ok: false; error: string };
function parseFileText(text: string): ReturnType<typeof parseFile>;   // adds "This file is not valid JSON."
function serializeFile(graph: Graph, params?: Record<NodeId, { W: number[]; b?: number[] }>): string; // stable, 2-space, trailing newline
// TrainSlice: setRunSeed(seed): void; loadHost(host: HostParams): boolean  (false: engine ready but the model did not build)
```

`v2.parseFile` opens file versions 1 and 2 (a version 1 file's graph is migrated; `params` keys are unchanged because Linear ids are kept) and `v2.serializeFile` writes version 2. The root-level `parseFile` is still the version 1 reader until the web app moves (tinker-8xw.4).

A file with `title`, `prompt` or `locked` is a Preset. Save writes `tinker-model.tinker.json`; "Weights" saves the live parameters (`host`), not the best snapshot. Load sets the run seed to the file's `train.seed`, then replaces the graph as one undo step (`graphSlice.apply`); undo returns the previous graph with its weights reconciled as for any edit. The preset (title, prompt, locked; null for a plain file) travels with the graph in the undo history (`pastPresets`/`futurePresets` parallel to `past`/`future`; `apply(edit, preset?)` sets it with the graph): undo and redo restore the preset together with the graph, and `loadGraph` (a new graph) clears it. A locked list names ids of that graph only; ids it lacks are dropped on load. Loaded weights are installed with `loadHost` (new run: history and best cleared).

Saved params whose shapes do not match the graph are dropped with a notice, not an error. Bundled presets live in `web/src/presets/*.json`; `?preset=<name>` loads a bundled preset on start, and a preset file can also be opened through Load.

## 13. Commands

From the repo root: `pnpm install`, `pnpm test`, `pnpm typecheck`, `pnpm lint`, `pnpm build`, `pnpm dev` (Vite on :5173), `pnpm e2e` (playwright-core against `pnpm preview`, added by `tinker-ysl.5`), `pnpm torch-ref` (runs the uv script; needs uv and network). Per package: `pnpm --filter @tinker/<core|engine|web> <script>`.
