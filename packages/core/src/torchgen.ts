/**
 * A whole runnable PyTorch script from a graph (CONTRACTS.md section 14). Pure text: no DOM, no jax-js.
 * The script is pedagogical, not a bit-exact copy of the app: torch default init seeded with the
 * graph's seed, a plain shuffled DataLoader, F.dropout. `spans` map text ranges to nodes (statements
 * to pieces) and `folds` mark the data literals.
 */
import type { Analysis } from './analyse.js';
import type { NodeId } from './common.js';
import { unstandardizeRows, type Dataset } from './data/index.js';
import type { Graph, Node } from './graph.js';
import { CORE_PIECES, type CorePiece, type TorchCtx } from './registry.js';

export type TorchSpan = { node: NodeId; from: number; to: number };
export type TorchFold = { from: number; to: number; label: string };
export type TorchCode = { text: string; spans: TorchSpan[]; folds: TorchFold[] };

const PY_KEYWORDS = [
  'False', 'None', 'True', 'and', 'as', 'assert', 'async', 'await', 'break', 'case', 'class',
  'continue', 'def', 'del', 'elif', 'else', 'except', 'finally', 'for', 'from', 'global', 'if',
  'import', 'in', 'is', 'lambda', 'match', 'nonlocal', 'not', 'or', 'pass', 'raise', 'return',
  'try', 'type', 'while', 'with', 'yield',
]; // prettier-ignore
/** Names the script itself uses (and builtins it calls). */
const SCRIPT_NAMES = [
  'x', 'y', 'self', 'model', 'torch', 'nn', 'F', 'np', 'optimizer', 'step', 'steps', 'loader',
  'batches', 'cycle', 'xb', 'yb', 'x_train', 'y_train', 'x_val', 'y_val', 'x_mean', 'x_std',
  'objective', 'Model', 'batch_loss', 'train_loss', 'val_loss', 'DataLoader', 'TensorDataset',
  'DecisionTreeRegressor', 'DecisionTreeClassifier', 'print', 'range', 'iter', 'next', 'super',
]; // prettier-ignore
/** Attributes of nn.Module: `self.<id> = ...` must not overwrite one. */
const MODULE_ATTRS = [
  'training', 'forward', 'train', 'eval', 'parameters', 'named_parameters', 'buffers',
  'named_buffers', 'children', 'named_children', 'modules', 'named_modules', 'state_dict',
  'load_state_dict', 'zero_grad', 'apply', 'to', 'cpu', 'cuda', 'float', 'double', 'half',
  'bfloat16', 'requires_grad_', 'add_module', 'register_buffer', 'register_parameter',
  'register_module', 'get_parameter', 'get_submodule', 'get_buffer', 'extra_repr', 'share_memory',
  'compile', 'to_empty', 'T_destination', 'get_extra_state', 'set_extra_state',
  'register_forward_hook', 'register_forward_pre_hook', 'register_full_backward_hook',
  'register_full_backward_pre_hook', 'register_backward_hook', 'register_state_dict_pre_hook',
  'register_load_state_dict_post_hook', 'ipu', 'xpu', 'mtia', 'call_super_init', 'dump_patches',
]; // prettier-ignore
const RESERVED: ReadonlySet<string> = new Set([...PY_KEYWORDS, ...SCRIPT_NAMES, ...MODULE_ATTRS]);

/** Reserved names, and any `register_*` (nn.Module methods) or leading-underscore name. */
const reserved = (s: string): boolean =>
  RESERVED.has(s) || s.startsWith('register_') || s.startsWith('_');

/** Whether a node id needs a trailing `_` to be a safe Python variable (used by the piece cards' torch line). */
export const isReservedPyName = reserved;

const num = (v: number): string => String(Number(v.toPrecision(7)));
const pyFloat = (x: number): string => (Number.isInteger(x) ? x.toFixed(1) : String(x));

/** The Python script for `graph`, or null while `analysis` has issues. Never throws. */
export function torchScript(
  graph: Graph,
  analysis: Analysis,
  dataset: Dataset,
  opts?: { steps?: number },
): TorchCode | null {
  try {
    return build(graph, analysis, dataset, opts?.steps ?? 1000);
  } catch {
    return null;
  }
}

type Stmt = { nodes: NodeId[]; text: string };

function build(
  graph: Graph,
  analysis: Analysis,
  dataset: Dataset,
  steps: number,
): TorchCode | null {
  if (!analysis.trainable || analysis.lossNode === null) return null;
  const lossNode = analysis.lossNode;
  const info = dataset.info;
  const d = info.columns.length;
  const byId = new Map<NodeId, Node>(graph.nodes.map((n) => [n.id, n]));

  // ---- names: node ids, with a trailing `_` where one collides
  const names = new Map<NodeId, string>();
  const taken = new Set<string>(graph.nodes.map((n) => n.id));
  const fresh = (base: string): string => {
    let s = base;
    while (RESERVED.has(s) || taken.has(s)) s += '_';
    taken.add(s);
    return s;
  };
  for (const n of graph.nodes) if (!reserved(n.id)) names.set(n.id, n.id);
  for (const n of graph.nodes) if (!names.has(n.id)) names.set(n.id, fresh(n.id));
  const name = (id: NodeId): string => names.get(id) ?? id;

  // ---- regions, wires, readers
  const region = analysis.region;
  const modelIds = analysis.order.filter((id) => region[id] === 'model');
  const objectiveIds = analysis.order.filter((id) => region[id] === 'objective');
  const wires = (id: NodeId, port: string): NodeId[] =>
    (analysis.inputs[id]?.[port] ?? []).map((w) => w.from);
  const readerCount = new Map<NodeId, number>();
  const bump = (id: NodeId) => readerCount.set(id, (readerCount.get(id) ?? 0) + 1);
  for (const id of analysis.order)
    for (const port of Object.keys(analysis.inputs[id] ?? {}))
      for (const from of wires(id, port)) bump(from);
  for (const id of analysis.predictions) bump(id);
  bump(lossNode);
  const readers = (id: NodeId): number => readerCount.get(id) ?? 0;

  // ---- emitters
  const trees = analysis.order
    .map((id) => byId.get(id))
    .filter((n): n is Node & { type: 'tree' } => n?.type === 'tree');
  const treeNames = new Map<NodeId, { estimator: string; leaves: string; columns: number[] }>();
  for (const t of trees) {
    const src = byId.get(wires(t.id, 'in')[0] ?? '');
    if (src?.type !== 'input') return null;
    treeNames.set(t.id, {
      estimator: fresh(`${name(t.id)}_tree`),
      leaves: fresh(`${name(t.id)}_leaves`),
      columns: src.columns.map((c) => info.columns.indexOf(c)),
    });
  }
  // A scalar parameter that reaches the `pred` of an unfused log loss or cross-entropy through
  // parameterless pieces only (no link) starts inside (0, 1), so the loss is defined at step 0.
  const consumers = new Map<NodeId, { id: NodeId; port: string }[]>();
  for (const id of analysis.order)
    for (const port of Object.keys(analysis.inputs[id] ?? {}))
      for (const from of wires(id, port)) {
        const list = consumers.get(from) ?? [];
        list.push({ id, port });
        consumers.set(from, list);
      }
  const feedsRawProbability = (start: NodeId): boolean => {
    const seen = new Set<NodeId>();
    const stack = [start];
    while (stack.length > 0) {
      const cur = stack.pop() as NodeId;
      for (const c of consumers.get(cur) ?? []) {
        const n = byId.get(c.id);
        if (n === undefined || seen.has(c.id)) continue;
        if (n.type === 'log_loss' || n.type === 'cross_entropy') {
          if (c.port === 'pred' && !analysis.fusable.some((f) => f.loss === c.id)) return true;
          continue;
        }
        if (n.type === 'link' || n.type === 'linear' || n.type === 'lookup') continue;
        if (n.type === 'scalew' || n.type === 'bias') continue;
        seen.add(c.id);
        stack.push(c.id);
      }
    }
    return false;
  };
  const emit = (id: NodeId, owner: 'self' | 'model') => {
    const node = byId.get(id) as Node;
    const piece = CORE_PIECES[node.type] as unknown as CorePiece<Node>;
    const ins: Record<string, string[]> = {};
    for (const port of piece.inputPorts) ins[port] = wires(id, port).map(name);
    const firstIn = wires(id, piece.inputPorts[0] ?? '')[0];
    const shape = firstIn === undefined ? undefined : analysis.shape[firstIn];
    const init = analysis.paramInit[id];
    const node0 = node.type;
    const ctx: TorchCtx = {
      ins,
      inWidth: analysis.inWidth[id] ?? 0,
      dataColumns: info.columns,
      categories: shape?.kind === 'example' ? (shape.category?.count ?? 0) : 0,
      bound: init?.kind === 'uniform' ? init.bound : 1,
      initRange:
        (node0 === 'scalew' || node0 === 'bias') && feedsRawProbability(id)
          ? [0.25, 0.75]
          : undefined,
      owner,
      training: `${owner}.training`,
      self: `${owner}.${name(id)}`,
      name,
      tree: treeNames.get(id),
    };
    return piece.torch(node, ctx);
  };

  // ---- the objective: idioms first, then one statement per remaining piece
  type Idiom = { expr: string; covered: NodeId[]; uses: NodeId[] };
  const idiomFor = (m: Node): Idiom | null => {
    if (m.type !== 'mean') return null;
    const pid = wires(m.id, 'in')[0];
    const p = pid === undefined ? undefined : byId.get(pid);
    if (p === undefined || region[p.id] !== 'objective' || readers(p.id) !== 1) return null;
    if (p.type === 'square' || p.type === 'abs' || p.type === 'huber') {
      const s = byId.get(wires(p.id, 'in')[0] ?? '');
      if (s?.type !== 'subtract' || region[s.id] !== 'objective' || readers(s.id) !== 1) {
        return null;
      }
      const a = wires(s.id, 'a')[0];
      const b = wires(s.id, 'b')[0];
      if (a === undefined || b === undefined) return null;
      const args = `${name(a)}, ${name(b)}`;
      const expr =
        p.type === 'square'
          ? `F.mse_loss(${args})`
          : p.type === 'abs'
            ? `F.l1_loss(${args})`
            : `F.huber_loss(${args}, delta=${pyFloat(p.delta)})`;
      return { expr, covered: [p.id, s.id], uses: [a, b] };
    }
    if (p.type === 'log_loss' || p.type === 'cross_entropy') {
      const fus = analysis.fusable.find((f) => f.loss === p.id);
      if (fus === undefined) return null;
      const linkInObjective = region[fus.link] === 'objective';
      if (linkInObjective && readers(fus.link) !== 1) return null;
      const z = wires(fus.link, 'in')[0];
      const t = wires(p.id, 'target')[0];
      if (z === undefined || t === undefined) return null;
      const expr =
        p.type === 'log_loss'
          ? `F.binary_cross_entropy_with_logits(${name(z)}, ${name(t)})`
          : `F.cross_entropy(${name(z)}, ${name(t)}.long().squeeze(1))`;
      return { expr, covered: linkInObjective ? [p.id, fus.link] : [p.id], uses: [z, t] };
    }
    return null;
  };
  const idioms = new Map<NodeId, Idiom>();
  const skipped = new Set<NodeId>();
  for (const id of objectiveIds) {
    const idiom = idiomFor(byId.get(id) as Node);
    if (idiom === null) continue;
    idioms.set(id, idiom);
    for (const c of idiom.covered) skipped.add(c);
  }
  const used = new Set<NodeId>();
  const objective: Stmt[] = [];
  for (const id of objectiveIds) {
    if (skipped.has(id)) continue;
    const idiom = idioms.get(id);
    if (idiom !== undefined) {
      for (const u of idiom.uses) used.add(u);
      objective.push({ nodes: [id, ...idiom.covered], text: `${name(id)} = ${idiom.expr}` });
      continue;
    }
    const piece = CORE_PIECES[(byId.get(id) as Node).type] as unknown as CorePiece<Node>;
    for (const port of piece.inputPorts) for (const from of wires(id, port)) used.add(from);
    objective.push({ nodes: [id], text: `${name(id)} = ${emit(id, 'model').expr}` });
  }
  // Model nodes the objective reads besides the ŷ pins come back from forward as extra values.
  const returned = [
    ...analysis.predictions,
    ...modelIds.filter((id) => used.has(id) && !analysis.predictions.includes(id)),
  ];

  // ---- text
  const out = new Writer();
  const regressor = trees.some((t) => t.criterion.endsWith('error'));
  const classifier = trees.some((t) => !t.criterion.endsWith('error'));
  out.line('import torch');
  out.line('import torch.nn as nn');
  out.line('import torch.nn.functional as F');
  if (graph.train.batch !== 'full')
    out.line('from torch.utils.data import DataLoader, TensorDataset');
  if (trees.length > 0) {
    const cls = [
      classifier ? 'DecisionTreeClassifier' : '',
      regressor ? 'DecisionTreeRegressor' : '',
    ].filter((s) => s !== '');
    out.line(`from sklearn.tree import ${cls.join(', ')}`);
  }
  out.blank();
  out.line(`torch.manual_seed(${graph.train.seed})`);
  out.blank();

  // data: raw rows as literals, standardisation as explicit lines
  const st = graph.data.standardize ? dataset.standardization : null;
  const rawTrain = st ? unstandardizeRows(dataset.xTrain, d, st) : dataset.xTrain;
  const rawVal = st ? unstandardizeRows(dataset.xVal, d, st) : dataset.xVal;
  out.tensor('x_train', rawTrain, dataset.nTrain, d);
  out.tensor('y_train', dataset.yTrain, dataset.nTrain, 1);
  out.tensor('x_val', rawVal, dataset.nVal, d);
  out.tensor('y_val', dataset.yVal, dataset.nVal, 1);
  if (st) {
    const categorical: number[] = [];
    const constant: number[] = [];
    for (let j = 0; j < d; j++) {
      if (info.columnKinds?.[j] === 'categorical') {
        categorical.push(j);
        continue;
      }
      let same = dataset.nTrain > 0;
      for (let i = 1; i < dataset.nTrain && same; i++) {
        same = num(rawTrain[i * d + j] as number) === num(rawTrain[j] as number);
      }
      if (same) constant.push(j);
    }
    out.blank();
    out.line('x_mean = x_train.mean(dim=0)');
    out.line('x_std = x_train.std(dim=0, unbiased=False)');
    if (categorical.length > 0) {
      out.line(`x_mean[[${categorical.join(', ')}]] = 0.0`);
      out.line(`x_std[[${categorical.join(', ')}]] = 1.0`);
    }
    if (constant.length > 0) out.line(`x_std[[${constant.join(', ')}]] = 1.0`);
    out.line('x_train = (x_train - x_mean) / x_std');
    out.line('x_val = (x_val - x_mean) / x_std');
  }
  out.blank();

  // tree fits, on the training rows, before the model
  if (trees.length > 0) {
    for (const t of trees) {
      const pre = emit(t.id, 'self').pre;
      if (pre !== undefined) for (const l of pre.split('\n')) out.line(l, 0, [t.id]);
    }
    out.blank();
  }

  // Model
  const forward: Stmt[] = modelIds.map((id) => ({
    nodes: [id],
    text: `${name(id)} = ${emit(id, 'self').expr}`,
  }));
  out.line('class Model(nn.Module):');
  out.line('def __init__(self):', 1);
  out.line('super().__init__()', 2);
  for (const id of analysis.paramNodes) {
    const init = emit(id, 'self').init;
    if (init !== undefined) out.line(`self.${name(id)} = ${init}`, 2, [id]);
  }
  if (forward.length > 0) {
    out.blank();
    out.line('def forward(self, x):', 1);
    for (const s of forward) out.line(s.text, 2, s.nodes);
    out.line(`return ${returned.map(name).join(', ')}`, 2);
  }
  out.blank(2);

  // objective
  out.line('def objective(model, x, y):');
  if (forward.length > 0) out.line(`${returned.map(name).join(', ')} = model(x)`, 1);
  for (const s of objective) out.line(s.text, 1, s.nodes);
  out.line(`return ${name(lossNode)}`, 1);
  out.blank(2);

  // training
  const tr = graph.train;
  const lr = pyFloat(tr.lr);
  out.line('model = Model()');
  out.line(
    tr.optimizer === 'adam'
      ? `optimizer = torch.optim.Adam(model.parameters(), lr=${lr})`
      : tr.optimizer === 'momentum'
        ? `optimizer = torch.optim.SGD(model.parameters(), lr=${lr}, momentum=0.9)`
        : `optimizer = torch.optim.SGD(model.parameters(), lr=${lr})`,
  );
  out.line(`steps = ${steps}`);
  if (tr.batch !== 'full') {
    out.line(
      `loader = DataLoader(TensorDataset(x_train, y_train), batch_size=${tr.batch}, shuffle=True)`,
    );
    out.blank(2);
    out.line('def cycle(loader):');
    out.line('while True:', 1);
    out.line('yield from loader', 2);
    out.blank(2);
    out.line('batches = cycle(loader)');
  }
  out.blank();
  out.line('for step in range(1, steps + 1):');
  out.line('model.train()', 1);
  if (tr.batch !== 'full') {
    out.line('xb, yb = next(batches)', 1);
    out.line('batch_loss = objective(model, xb, yb)', 1);
  } else {
    out.line('batch_loss = objective(model, x_train, y_train)', 1);
  }
  out.line('optimizer.zero_grad()', 1);
  out.line('batch_loss.backward()', 1);
  out.line('optimizer.step()', 1);
  out.line(`if step % ${Math.max(1, Math.floor(steps / 10))} == 0 or step == steps:`, 1);
  out.line('model.eval()', 2);
  out.line('with torch.no_grad():', 2);
  out.line('train_loss = objective(model, x_train, y_train)', 3);
  const hasVal = dataset.nVal > 0;
  if (hasVal) out.line('val_loss = objective(model, x_val, y_val)', 3);
  out.line(
    hasVal
      ? 'print(f"step {step}: train loss {train_loss.item():.4f}, val loss {val_loss.item():.4f}")'
      : 'print(f"step {step}: train loss {train_loss.item():.4f}")',
    2,
  );
  return out.result();
}

/** Builds the text with offsets, so spans and folds are exact. */
class Writer {
  private text = '';
  private readonly spans: TorchSpan[] = [];
  private readonly folds: TorchFold[] = [];

  line(content: string, indent = 0, nodes: readonly NodeId[] = []): void {
    this.text += '    '.repeat(indent);
    const from = this.text.length;
    this.text += content;
    const to = this.text.length;
    this.text += '\n';
    for (const node of nodes) this.spans.push({ node, from, to });
  }

  blank(n = 1): void {
    this.text += '\n'.repeat(n);
  }

  /** `<name> = torch.tensor([...rows...], dtype=torch.float32)`; the bracketed rows are one fold. */
  tensor(name: string, v: Float32Array, rows: number, cols: number): void {
    if (rows === 0) {
      this.line(`${name} = torch.zeros((0, ${cols}))`);
      return;
    }
    this.text += `${name} = torch.tensor(`;
    const from = this.text.length;
    this.text += '[\n';
    for (let i = 0; i < rows; i++) {
      const row: string[] = [];
      for (let j = 0; j < cols; j++) row.push(num(v[i * cols + j] as number));
      this.text += `    [${row.join(', ')}],\n`;
    }
    this.text += ']';
    this.folds.push({ from, to: this.text.length, label: `${rows} rows` });
    this.text += ', dtype=torch.float32)\n';
  }

  result(): TorchCode {
    return { text: this.text, spans: this.spans, folds: this.folds };
  }
}
