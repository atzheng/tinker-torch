# Spec: Tinker, a visual neural network construction kit

Nov 18, 2026 · @Andy

## Summary

Tinker is a single web page where a student snaps pieces together to build a model, attaches a small dataset and a loss, presses train, and watches the fit. It runs entirely in the browser on [jax-js](https://github.com/ekzhang/jax-js), with no server and no accounts. "Tinker" is a working name.

|  | Version 1 | Later |
| --- | --- | --- |
| Models | Linear and logistic regression, feature expansions, multilayer networks, skip connections | Embeddings, attention, transformer blocks |
| Grain | Neuron view and layer view of the same model | Block view |
| Data | Synthetic 1-D and 2-D regression and classification sets with a known true function | Toy sequence tasks, pasted CSV |
| Training | SGD, momentum, Adam; full batch or mini-batch; step or run | Learning-rate schedules |
| Views | Fit plot, loss curves, weights on the canvas, resampling overlay, pinned-run comparison | Attention patterns |
| Code | None | Read-only torch view, then two-way editing |
| Lessons | None; the tool is generic | Presets with prompts and checks |

Not in scope at any version: showing intermediate activations or the backward pass node by node. Other resources cover that.

## Design principles

1. **Architecture first.** The canvas is the largest thing on screen. Data, loss and training are attached to the architecture, and every result view is a consequence of it.
2. **The slide notation is the interface.** A blue edge is one parameter. A circle applies a function. A plus circle sums its inputs. The loss is a node that takes the prediction and the target. A student who has seen the lecture slides should recognise the canvas without instruction.
3. **Everything swappable is a piece.** Activations, losses, penalties and feature transforms are parts on the canvas. Swapping one is the same gesture for all of them: drop the new piece on the old one.
4. **One model, several grains.** The neuron view and the layer view are two drawings of the same stored graph. Zooming changes the picture and never the model.
5. **Generic tool, no lesson text.** The tool contains no explanations or tasks. A lesson is a preset loaded from outside: a graph, a dataset, a list of locked pieces and a prompt.
6. **Small and immediate.** Models stay under roughly 10,000 parameters in version 1, and a training run finishes in seconds on a student laptop. The page works offline once loaded.
7. **Ready for the code view.** Each piece corresponds to exactly one torch module or function, recorded in the piece registry now, so that code generation later is a lookup and not a redesign.

## Screen layout

One screen, five regions, no scrolling at laptop size (1280 by 720 and up).

| Region | Position | Contents |
| --- | --- | --- |
| Top bar | Top | Dataset picker, sample size, noise, Resample, grain toggle, save, load, presets |
| Palette | Left, narrow | Pieces grouped as Inputs, Features, Linear, Activations, Structure, Outputs, Losses, Penalties |
| Canvas | Centre, about 55% of width | The model, drawn left to right: inputs, model pieces, prediction, loss with the target feeding in |
| Results | Right, about 30% of width | Fit plot above, loss curves below |
| Training bar | Bottom | Step, Run or Pause, Reset, learning rate, batch size, optimizer, step counter, parameter count |

Selecting a piece opens a small inspector beside it with that piece's settings and its parameter count. There are no modal dialogs.

## Pieces

Version 1 ships 14 piece types. Each has typed ports, a rule for its output width, a parameter count, and one torch counterpart recorded for the later code view.

| Piece | Drawn as | Settings | Parameters | Torch counterpart |
| --- | --- | --- | --- | --- |
| Input | Labelled dots x1 to xd | Which data columns are on | 0 | Input tensor |
| Feature | Small box with its formula | Powers up to degree k, sin, cos, products | 0 | Tensor operations |
| Linear | Layer view: box "d to m". Neuron view: d times m blue edges into m plus circles | Output width m, bias on or off | d times m, plus m | `nn.Linear` |
| Activation | Circle with its symbol | Sigmoid, tanh, ReLU, GELU | 0 | `nn.Sigmoid`, `nn.Tanh`, `nn.ReLU`, `nn.GELU` |
| Dropout | Dashed circle | Drop probability p | 0 | `nn.Dropout` |
| Add | Plus circle with two black inputs | None | 0 | `+` (skip connection) |
| Output | Dot labelled with y-hat | Raw value, sigmoid probability, or softmax probabilities | 0 | Final activation |
| Target | Dot labelled y | None | 0 | Label tensor |
| Squared error | Loss circle | None | 0 | `nn.MSELoss` |
| Absolute error | Loss circle | None | 0 | `nn.L1Loss` |
| Huber | Loss circle | Threshold delta | 0 | `nn.HuberLoss` |
| Log loss | Loss circle | None | 0 | `nn.BCEWithLogitsLoss` |
| Cross-entropy | Loss circle | None | 0 | `nn.CrossEntropyLoss` |
| Weight penalty | Tag attached to the loss | L2 or L1, strength lambda | 0 | `weight_decay`, or an added term |

One detail matters for the later code view. On the canvas the sigmoid or softmax is drawn as part of the Output piece and the loss takes probabilities, as on the slides. A loss whose prediction is a probability from anywhere else (for example a Bias pinned as ŷ, optimised by hand first) is computed as it is; if the prediction leaves (0, 1) the page says so in one sentence and stops the run. When a sigmoid or softmax is in front, the engine computes the combined, numerically stable form, which is what `BCEWithLogitsLoss` and `CrossEntropyLoss` do, so picture and code will agree.

Pieces specified now and built later: Embedding, Position, Attention head, Layer norm, Transformer block (a group of Attention, Add, Layer norm and a small network), and Unembed.

## Canvas behaviour

The canvas lays itself out left to right; students choose what connects to what, never where things sit.

**Building**

- Drag a piece from the palette onto a wire to insert it there.
- Drop a piece onto another of the same family to swap it. Activation replaces activation, loss replaces loss. Settings carry over where they still apply.
- Drag from an output port to an input port to add a wire. This is how a skip connection into an Add piece is made.
- Delete with the keyboard. The wire closes over the gap when widths still match.
- Undo and redo cover every canvas action.

**Widths**

- Every wire carries a width, shown as a small number on the wire in layer view.
- A Linear piece reads its input width from upstream. The student sets only the output width.
- A mismatch turns the wire red and shows one plain sentence, for example "Add needs both inputs to have the same width: 4 and 8."
- Training is disabled until the graph is valid: one loss, a connected path from inputs to prediction to loss, and a target type the loss accepts.

**Grain**

- Layer view draws each Linear as one box. Neuron view expands it into inputs, blue edges, plus circles and activation circles, exactly as on the slides.
- A Linear piece expands only while it has at most 8 units on each side. Wider layers stay collapsed and show their edge count.
- In neuron view, a plus button under a column adds a unit and a minus button removes one. This edits the width of the stored Linear piece.

**Parameters on the canvas**

- In neuron view, edge colour shows the sign of the weight and thickness shows its size. Hovering shows the value.
- In layer view, the Linear box shows a small heat map of its weight matrix.
- While training is paused, dragging on an edge changes its weight by hand, and the fit plot and loss update live.
- The parameter count is always visible, in total and per piece.

## Data

All version 1 datasets are synthetic, drawn from a known true function plus noise, so that the truth can be plotted and resampling is free.

| Kind | Sets |
| --- | --- |
| 1-D regression | Line, quadratic, sine, step, wiggle |
| 2-D regression | Plane, bowl, saddle |
| 2-D classification | Two blobs, circle, XOR, two moons |

| Control | Range | Default |
| --- | --- | --- |
| Sample size n | 10 to 500 | 40 |
| Noise level | 0 to 1, in units of the target's spread | 0.2 |
| Validation share | 0 to 50% | 30% |
| Seed | Shown, editable | Random |
| Resample | Draws a fresh training set from the same generator |  |

- The dataset appears on the canvas as Input dots and a Target dot. Columns can be switched off at the Input piece.
- Inputs are left in raw units by default, with a toggle to standardise them using training-set statistics.
- The true function is hidden by default and shown by a toggle on the fit plot.

## Training

The objective is the mean of the loss piece over the batch, plus the weight penalty if one is attached.

```latex
L(\theta) = \frac{1}{|B|} \sum_{i \in B} \ell\big(f_\theta(x_i),\, y_i\big) + \lambda\, R(w)
```

| Control | Options | Default |
| --- | --- | --- |
| Optimizer | SGD, SGD with momentum, Adam | SGD |
| Learning rate | 0.0001 to 1, log slider | 0.03 |
| Batch size | 1 to n, or full batch | Full batch |
| Step | One parameter update |  |
| Run or Pause | Continuous updates, with a speed setting |  |
| Reset | New initial parameters from a new seed |  |

- Initial weights follow the torch default for `nn.Linear`: uniform on plus or minus one over the square root of the input width.
- A learning rate that is too large is allowed. When the loss becomes non-finite the run stops, the canvas shows "diverged", and Reset restores a working state. The page never crashes on divergence.
- Dropout is active during training steps and off for the fit plot and the validation loss.
- The loss curve marks the step with the lowest validation loss. A "Restore best" button returns the parameters from that step.

## Result views

Five views, all updating live during a run at up to 30 frames per second.

| View | Shows | When |
| --- | --- | --- |
| Fit plot | 1-D: training points filled, validation points hollow, the model's curve, optionally the true function. 2-D classification: decision surface with points. 2-D regression: heat map | Always |
| Loss curves | Training and validation loss against step, log scale toggle, lowest validation loss marked | Always |
| Resampling overlay | Many fits of the same architecture on fresh training sets, drawn faintly, with their average in bold | On "Resample x 20" |
| Loss surface | Contours of the training loss over two parameters, with the path taken so far | When the model has exactly two parameters, for example one weight and one bias |
| Pinned runs | Up to 3 earlier runs kept as thumbnails with their curves overlaid on the current one | On "Pin" |

**Resampling overlay.** This is the bias and variance view. It trains k copies of the current architecture, each on a new training set and a new initialisation, for the same number of steps as the current run. With the true function shown, the gap between the bold average and the truth is the bias, and the spread of the faint curves is the variance. Two numbers are reported, averaged over the plotted input range:

```latex
\mathbb{E}\big[(\hat f(x) - f^*(x))^2\big] = \big(\mathbb{E}\,\hat f(x) - f^*(x)\big)^2 + \operatorname{Var}\,\hat f(x)
```

The k copies train in one vectorised call, which jax-js supports through `vmap`.

**Pinned runs.** A pin stores the graph, the data settings, the loss curves and the final fit. This is how a student compares two activations, two losses or two widths side by side.

## Graph format

One JSON document is the source of truth. The canvas draws it, the engine compiles it, save and load exchange it, presets are instances of it, and the later code view will translate to and from it.

```json
{
  "version": 1,
  "nodes": [
    {"id": "x",    "type": "input",  "columns": ["x1"]},
    {"id": "lin1", "type": "linear", "out": 4, "bias": true},
    {"id": "act1", "type": "activation", "fn": "relu"},
    {"id": "lin2", "type": "linear", "out": 1, "bias": true},
    {"id": "yhat", "type": "output", "link": "identity"},
    {"id": "y",    "type": "target"},
    {"id": "loss", "type": "loss",   "fn": "squared_error",
     "penalty": {"kind": "l2", "lambda": 0.0}}
  ],
  "edges": [
    ["x", "lin1"], ["lin1", "act1"], ["act1", "lin2"],
    ["lin2", "yhat"], ["yhat", "loss.pred"], ["y", "loss.target"]
  ],
  "data":  {"set": "sine", "n": 40, "noise": 0.2, "val": 0.3, "seed": 7},
  "train": {"optimizer": "sgd", "lr": 0.03, "batch": "full", "seed": 11}
}
```

Rules:

- The graph is acyclic. Input widths are never stored; they are inferred from upstream on load.
- Neuron view is a drawing of the graph and is not stored.
- Parameter values are stored separately, keyed by node id, and are optional in a saved file.
- Layout positions are not stored, because the canvas lays itself out.

**Piece registry.** Each piece type is one registry entry with six fields: its ports, its width rule, its parameter shapes and initialiser, its forward function, its drawing in each grain, and its torch counterpart. Adding a piece means adding one entry. Attention and the other later pieces arrive this way, with no change to the canvas or the engine.

## Engine

The engine is [jax-js](https://github.com/ekzhang/jax-js), wrapped in a single module so that nothing else in the tool imports it.

**How the graph runs.** The engine walks the graph in order and composes each piece's forward function into two pure functions of a parameter tree keyed by node id: `predict(params, x)` and `loss(params, batch)`. Gradients come from `grad`, compilation from `jit`, the resampling overlay from `vmap`, and the optimizers from the companion package `@jax-js/optax`.

```ts
interface Engine {
  compile(graph: Graph): Model;                 // throws a readable error on an invalid graph
  init(model: Model, seed: number): Params;
  predict(model: Model, params: Params, x: Float32Array): Float32Array;
  evaluate(model: Model, params: Params, batch: Batch): number;        // loss only
  step(model: Model, state: TrainState, batch: Batch): TrainState;     // one optimizer update
  read(params: Params): Record<string, Float32Array>;                  // for drawing weights
  write(params: Params, nodeId: string, values: Float32Array): Params; // hand-set weights
  trainMany(model: Model, seeds: number[], data: Batch[], steps: number): { params: HostParams; diverged: boolean }[];
}
```

**Backend.** Use the WebAssembly backend, which is the jax-js default. Models here are tiny, so WebGPU is not needed in version 1 and stays available for larger models later.

**Facts taken from the jax-js README** ([version 0.1.11](https://cdn.jsdelivr.net/npm/@jax-js/jax@0.1.11/README.md)): pure JavaScript with no dependencies; `grad`, `vmap` and `jit` over nested objects; arrays are reference counted, a function takes ownership of what it is passed, and `.ref` keeps an array alive for reuse; WebAssembly runs in every current browser, and is multi-threaded only when `SharedArrayBuffer` is available.

| Risk | Handling |
| --- | --- |
| Young library; the API may change | Pin one exact version and ship its bundle with the page, with no CDN at run time |
| Ownership rules make leaks and use-after-free easy | All array handling stays inside the engine module; a test trains 10,000 steps and checks that live array count is flat |
| The README does not confirm softmax, gather for embeddings, layer norm, a random-number API, or a combined value-and-gradient call | Check each in the first spike; where missing, compose from primitives (exp and sum, mean and variance, one-hot times matrix) |
| Multi-threading needs cross-origin isolation headers on the host | Do not depend on it; single-threaded speed is the target |
| A library defect blocks a piece | The engine interface is small enough to re-implement on another library without touching the canvas |

**Correctness.** For each piece and each loss, a test compares the engine's output and gradients with values precomputed in PyTorch on fixed inputs, to 5 decimal places. This is what guarantees the later torch view describes what the tool actually computes.

## Milestones

Six steps, each ending in something a student could use.

1. **Spike.** Train linear regression and a two-layer network on jax-js in a bare page. Confirm the unverified operations listed under Engine, measure time per step, and run the leak test. Exit: a go or no-go on jax-js.
2. **Core loop.** Graph format, piece registry, engine module. Layer view with Input, Linear, Activation, Output, Target and squared error. 1-D datasets, Step and Run, fit plot, loss curves.
3. **Neuron view.** Grain toggle, weights drawn on edges, add and remove units, hand-set weights, parameter count, loss surface for two-parameter models.
4. **Swapping.** The remaining losses, weight penalty, Dropout, Feature and Add pieces, the three optimizers, mini-batches, 2-D datasets with the decision surface.
5. **Comparison.** Resample, the resampling overlay with bias and variance numbers, pinned runs.
6. **Save and presets.** Save and load to a file, and a preset loader that later lessons will use.

Version 1 is complete at step 6. Each step is tested against the PyTorch reference values before the next begins.

## Later versions and open questions

| Version | Adds |
| --- | --- |
| 2 | Block grain: Embedding, Position, Attention head, Layer norm, Transformer block, Unembed. Toy sequence tasks (copy, reverse, sort, modular addition). An attention-pattern view |
| 3 | Torch view: read-only code generated from the graph, with selection linked both ways. Then editing: a restricted torch subset parsed back into the graph |
| 4 | Lessons: presets with a prompt, locked pieces and automatic checks. Pasted CSV data |

**Open questions**

- [x] Hosting: Cloudflare (static assets deployed with wrangler). The bundle, including the jax-js Wasm, ships with the page. A `_headers` file sets cross-origin isolation so multi-threaded Wasm is available, but nothing depends on it.
- [x] Devices: laptops and tablets, mouse, trackpad and touch. Wide windows use the side-by-side layout; portrait tablets and narrow windows stack the regions (CONTRACTS section 10). Every gesture uses Pointer Events. Screens under 600 px wide or tall show a "use a larger screen" notice.
- [ ] Wiring freedom: version 1 allows branches only through the Add piece. Is free wiring between any two ports needed earlier?
- [x] Hand-set weights: kept in version 1 (milestone 3).
- [ ] Student work: is file export enough, or will submissions need to be collected? Version 1 ships file export only.
- [x] Inputs standardised by default: no. Raw scales are the default so that students meet the problem themselves; a toggle standardises inputs using training-set statistics.
- [ ] Name: "Tinker" is a placeholder.

**Decisions made while planning (Nov 18, 2026)**

- jax-js is pinned to 0.1.27 (and `@jax-js/optax` 0.1.3), not 0.1.11. Its type declarations include `softmax`, `logSoftmax`, `logsumexp`, `valueAndGrad`, `random` (uniform, normal, bernoulli), `oneHot`, `take` and `dataSync`; the spike confirms they behave, rather than discovering whether they exist.
- The weight penalty applies to Linear weight matrices only, never biases. The later code view emits it as an explicit added term, not `weight_decay`.
- PyTorch reference values are generated by an opt-in Python script (`uv run`) and committed as JSON; JS tests read only the JSON.
- The piece registry is split by layer (core, engine, web), each a `Record<PieceType, ...>` so that adding a piece type fails to typecheck until every layer has its entry. See CONTRACTS.md section 3.
- The engine runs on the main thread, synchronously (Wasm, `dataSync`). If the spike shows the resampling overlay blocking the page for more than about 300 ms, a follow-up moves it to a Web Worker.
- Sample size n is the total number of points; the validation set is the last `round(n * val)` of them.
- The resampling overlay draws faint curves for 1-D datasets only; for 2-D datasets it reports the two numbers and draws the mean surface.

**Sources**

- [jax-js repository](https://github.com/ekzhang/jax-js) and [README for version 0.1.11](https://cdn.jsdelivr.net/npm/@jax-js/jax@0.1.11/README.md)
- Course slides, "Intro to Deep Learning" (BAIT 509), for the notation
- Prior tools reviewed: [TensorFlow Playground](https://playground.tensorflow.org/), [neural-viz](https://github.com/MrZoller/neural-viz), [DL-Playground](https://github.com/dsgiitr/DL-Playground), [nFlow](https://huggingface.co/krystv/nflow), [MLU-Explain on bias and variance](https://mlu-explain.github.io/bias-variance)
