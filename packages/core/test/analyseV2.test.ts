/// <reference types="node" />
import { readdirSync, readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { describe, expect, test } from 'vitest';
import { analyse as analyseV1, parseGraph as parseGraphV1, sample, v2 } from '../src/index.js';

const { analyse, parseGraph, weightPenaltySubgraph } = v2;
type Graph = v2.Graph;
type Node = v2.Node;
type Edge = v2.Edge;
type DataInfo = v2.DataInfo;

const reg: DataInfo = { columns: ['x1'], kind: 'regression', numClasses: 2 };
const bin: DataInfo = { columns: ['x1', 'x2'], kind: 'binary', numClasses: 2 };

function g(
  nodes: Node[],
  edges: Edge[],
  predictions: string[] = [],
  loss: string | null = null,
): Graph {
  return {
    version: 2,
    nodes,
    edges,
    predictions,
    loss,
    data: { set: 'line', n: 100, noise: 0.1, val: 0.2, seed: 1, standardize: false },
    train: { optimizer: 'adam', lr: 0.01, batch: 'full', seed: 1 },
  };
}
const input = (columns = ['x1']): Node => ({ id: 'in', type: 'input', columns });
const lin = (id: string, out: number, bias = true): Node => ({ id, type: 'linear', out, bias });
const act = (id: string): Node => ({ id, type: 'activation', fn: 'tanh' });
const tgt: Node = { id: 'y', type: 'target' };
const messages = (a: v2.Analysis) => a.issues.map((i) => i.message);

/** The SPEC example: 1 -> 4 -> 4 -> 1, squared error: sub -> sq -> loss (mean). ŷ = l2, L = loss. */
function spec(): Graph {
  return g(
    [
      input(),
      lin('l1', 4),
      act('a1'),
      lin('l2', 1),
      tgt,
      { id: 'diff', type: 'subtract' },
      { id: 'err', type: 'square' },
      { id: 'loss', type: 'mean' },
    ],
    [
      ['in', 'l1'],
      ['l1', 'a1'],
      ['a1', 'l2'],
      ['l2', 'diff.a'],
      ['y', 'diff.b'],
      ['diff', 'err'],
      ['err', 'loss'],
    ],
    ['l2'],
    'loss',
  );
}

describe('analyse v2: the SPEC example', () => {
  test('shapes, parameters, order, regions', () => {
    const a = analyse(spec(), reg);
    expect(a.issues).toEqual([]);
    expect(a.trainable).toBe(true);
    expect(a.order).toEqual(['in', 'y', 'l1', 'a1', 'l2', 'diff', 'err', 'loss']);
    expect(a.shape['l1']).toEqual({ kind: 'example', width: 4 });
    expect(a.shape['loss']).toEqual({ kind: 'scalar' });
    expect(a.outWidth).toMatchObject({ in: 1, l1: 4, a1: 4, l2: 1, y: 1, diff: 1, err: 1 });
    expect(a.outWidth['loss']).toBeUndefined();
    expect(a.inWidth).toMatchObject({ l1: 1, a1: 4, l2: 4, diff: 1, loss: 1 });
    expect(a.params).toMatchObject({ l1: 8, l2: 5, a1: 0 });
    expect(a.paramShapes['l1']).toEqual({ W: [1, 4], b: [4] });
    expect(a.paramNodes).toEqual(['l1', 'l2']);
    expect(a.totalParams).toBe(13);
    expect(a.predictions).toEqual(['l2']);
    expect(a.lossNode).toBe('loss');
    expect(a.region).toEqual({
      in: 'model',
      l1: 'model',
      a1: 'model',
      l2: 'model',
      y: 'objective',
      diff: 'objective',
      err: 'objective',
      loss: 'objective',
    });
    expect(a.modelOrder).toEqual(['in', 'l1', 'a1', 'l2']);
    expect(a.edgeWidth).toEqual([1, 4, 4, 1, 1, 1, 1]);
    expect(a.edgeShape[6]).toEqual({ kind: 'example', width: 1 });
    expect(a.paramWires).toEqual([]);
    expect(a.fusable).toEqual([]);
  });

  test('a graph with no ŷ pin still trains', () => {
    const gr = spec();
    gr.predictions = [];
    const a = analyse(gr, reg);
    expect(a.trainable).toBe(true);
    expect(a.modelOrder).toEqual([]);
    expect(a.order).toContain('loss');
    expect(a.region['l2']).toBe('objective');
  });

  test('order is topological for any node order and leaves out dangling nodes', () => {
    const gr = spec();
    gr.nodes.reverse();
    gr.nodes.push(lin('stray', 8));
    gr.edges.push(['in', 'stray']);
    const a = analyse(gr, reg);
    expect(a.trainable).toBe(true);
    expect(a.order).not.toContain('stray');
    expect(a.params['stray']).toBe(16);
    expect(a.totalParams).toBe(13);
    const pos = new Map(a.order.map((id, i) => [id, i]));
    for (const [from, to] of gr.edges) {
      const f = pos.get(v2.parsePort(from).node);
      const t = pos.get(v2.parsePort(to).node);
      if (f !== undefined && t !== undefined) expect(f).toBeLessThan(t);
    }
  });

  test('a node shared by model and objective stays in the model', () => {
    const a = analyse(spec(), reg);
    expect(a.region['l2']).toBe('model');
    expect(a.region['diff']).toBe('objective');
  });
});

describe('analyse v2: loss pin', () => {
  test('no loss pinned', () => {
    const gr = spec();
    gr.loss = null;
    const a = analyse(gr, reg);
    expect(a.trainable).toBe(false);
    expect(a.lossNode).toBeNull();
    expect(a.issues).toEqual([
      {
        at: {},
        message:
          'No loss is pinned; pin a single-number output as the loss so there is something to train.',
      },
    ]);
    expect(a.order).toEqual(['in', 'l1', 'a1', 'l2']);
  });

  test('a loss that is one value per example needs a Mean', () => {
    const gr = spec();
    gr.loss = 'err';
    const a = analyse(gr, reg);
    expect(a.trainable).toBe(false);
    expect(a.issues).toEqual([
      {
        at: { node: 'err', pin: { role: 'loss' } },
        message:
          'The loss must be a single number, but "err" gives 1 per example; add a Mean to average over the examples.',
      },
    ]);
  });

  test('a loss that is a weight matrix', () => {
    const gr = spec();
    gr.nodes.push({ id: 'w', type: 'weights', of: 'l1' });
    gr.loss = 'w';
    const a = analyse(gr, reg);
    expect(messages(a)).toEqual([
      'The loss must be a single number, but "w" is a 1 by 4 weight matrix.',
    ]);
    expect(a.issues[0]?.at.pin).toEqual({ role: 'loss' });
  });

  test('a loss that depends on no parameter', () => {
    const gr = g(
      [tgt, { id: 'sq', type: 'square' }, { id: 'loss', type: 'mean' }],
      [
        ['y', 'sq'],
        ['sq', 'loss'],
      ],
      [],
      'loss',
    );
    const a = analyse(gr, reg);
    expect(a.issues).toEqual([
      {
        at: { node: 'loss', pin: { role: 'loss' } },
        message: 'The loss "loss" does not depend on any parameter, so there is nothing to train.',
      },
    ]);
  });

  test('a loss depending on parameters only through a Weights tap is fine', () => {
    const gr = g(
      [
        input(),
        lin('l1', 1),
        { id: 'w', type: 'weights', of: 'l1' },
        { id: 'n', type: 'norm', kind: 'l2' },
      ],
      [
        ['w', 'n'],
        ['in', 'l1'],
      ],
      [],
      'n',
    );
    const a = analyse(gr, reg);
    expect(a.issues).toEqual([]);
    expect(a.totalParams).toBe(2);
  });

  test('a loss pin naming no node', () => {
    const gr = spec();
    gr.loss = 'ghost';
    const a = analyse(gr, reg);
    expect(a.trainable).toBe(false);
    expect(a.issues[0]?.at).toEqual({ pin: { role: 'loss' } });
    expect(a.lossNode).toBeNull();
  });

  test('a loss on a cycle is left out of order', () => {
    const gr = g(
      [lin('a', 1), { id: 'b', type: 'mean' }],
      [
        ['a', 'b'],
        ['b', 'a'],
      ],
      [],
      'b',
    );
    const a = analyse(gr, reg);
    expect(a.trainable).toBe(false);
    expect(a.order).toEqual([]);
  });
});

describe('analyse v2: prediction pins', () => {
  test('a prediction cannot depend on the target', () => {
    const gr = spec();
    gr.predictions = ['diff']; // lin2 - y
    const a = analyse(gr, reg);
    expect(a.issues).toEqual([
      {
        at: { node: 'diff', pin: { role: 'prediction', index: 0 } },
        message: 'A prediction cannot depend on the target, but "diff" does.',
      },
    ]);
    expect(a.trainable).toBe(false);
  });

  test('width must match the target: 1 for regression and binary', () => {
    const gr = spec();
    gr.nodes[3] = lin('l2', 3);
    const a = analyse(gr, reg);
    expect(a.issues.find((i) => i.at.pin)).toEqual({
      at: { node: 'l2', pin: { role: 'prediction', index: 0 } },
      message:
        'This dataset\'s target has width 1, so the prediction needs width 1, but "l2" has width 3.',
    });
  });

  test('softmax prediction needs numClasses and a classification dataset', () => {
    const sm: Node = { id: 'sm', type: 'link', fn: 'softmax' };
    const build = (width: number, cols: string[]) =>
      g(
        [input(cols), lin('l', width), sm],
        [
          ['in', 'l'],
          ['l', 'sm'],
        ],
        ['sm'],
        null,
      );
    const bad = analyse(build(3, ['x1', 'x2']), bin);
    expect(bad.issues.find((i) => i.at.pin)?.at).toEqual({
      node: 'sm',
      pin: { role: 'prediction', index: 0 },
    });
    expect(messages(bad)).toContain(
      'This dataset has 2 classes, so the softmax prediction needs width 2, but "sm" has width 3.',
    );
    expect(messages(analyse(build(2, ['x1', 'x2']), bin))).toEqual([
      'No loss is pinned; pin a single-number output as the loss so there is something to train.',
    ]);
    expect(messages(analyse(build(2, ['x1']), reg))).toContain(
      'A softmax prediction needs a dataset with classes, but this dataset is regression.',
    );
  });

  test('a prediction that is a single number', () => {
    const gr = spec();
    gr.predictions = ['loss'];
    expect(messages(analyse(gr, reg))).toEqual([
      'A prediction must be one row per example, but "loss" is a single number.',
    ]);
  });

  test('several prediction pins are fine; each width is checked and modelOrder is the union', () => {
    const gr = spec();
    gr.predictions = ['l2', 'a1'];
    const a = analyse(gr, reg);
    // a1 has width 4, so only that pin complains
    expect(a.issues.map((i) => i.at.pin)).toEqual([{ role: 'prediction', index: 1 }]);
    expect(a.predictions).toEqual(['l2', 'a1']);
    const b = g(
      [input(), lin('p', 1), lin('q', 1), tgt],
      [
        ['in', 'p'],
        ['in', 'q'],
      ],
      ['p', 'q'],
    );
    const r = analyse(b, reg);
    expect(r.modelOrder).toEqual(['in', 'p', 'q']);
    expect(r.issues.some((i) => i.at.pin)).toBe(false);
  });

  test('the same node pinned twice is one issue on the duplicate', () => {
    const gr = spec();
    gr.predictions = ['l2', 'l2.out'];
    const a = analyse(gr, reg);
    expect(a.issues).toEqual([
      {
        at: { node: 'l2', pin: { role: 'prediction', index: 1 } },
        message: '"l2" is pinned as a prediction twice.',
      },
    ]);
  });

  test('a pin naming no node', () => {
    const gr = spec();
    gr.predictions = ['ghost'];
    const a = analyse(gr, reg);
    expect(a.issues[0]?.at).toEqual({ pin: { role: 'prediction', index: 0 } });
    expect(a.predictions).toEqual([]);
  });
});

describe('analyse v2: shapes and wiring', () => {
  test('feature expansion widths', () => {
    const gr = g(
      [
        input(['x1', 'x2']),
        { id: 'f', type: 'feature', degree: 2, sin: true, cos: false, products: true },
        lin('l', 1),
      ],
      [
        ['in', 'f'],
        ['f', 'l'],
      ],
    );
    const a = analyse(gr, bin);
    expect(a.outWidth['f']).toBe(7);
    expect(a.params['l']).toBe(8);
  });

  test('columns the dataset lacks, duplicate columns, no columns', () => {
    const edit = (cols: string[]) => {
      const gr = spec();
      gr.nodes[0] = input(cols);
      return analyse(gr, reg);
    };
    expect(messages(edit(['x1', 'x3']))).toEqual([
      'Input uses "x3", which this dataset does not have.',
    ]);
    expect(messages(edit(['x1', 'x1']))[0]).toMatch(/more than once/);
    const none = edit([]);
    expect(messages(none)).toEqual(['Input needs at least one data column switched on.']);
    expect(none.outWidth['l1']).toBeUndefined();
    expect(none.params['l1']).toBeUndefined();
  });

  test('add and subtract need equal shapes, with the red wire on b', () => {
    const gr = spec();
    gr.nodes[3] = lin('l2', 2);
    const a = analyse(gr, reg);
    expect(a.issues.find((i) => i.at.node === 'diff')).toEqual({
      at: { node: 'diff', edge: 4 },
      message: 'Subtract needs both inputs to have the same width: 2 and 1.',
    });
    expect(a.outWidth['diff']).toBeUndefined();
    expect(a.outWidth['err']).toBeUndefined();
    // downstream of the mismatch: no issue of its own
    expect(a.issues.filter((i) => i.at.node === 'err' || i.at.node === 'loss')).toEqual([]);
  });

  test('add of a scalar and a matrix: no broadcasting', () => {
    const gr = g(
      [
        input(),
        lin('l', 1),
        { id: 'w', type: 'weights', of: 'l' },
        { id: 'm', type: 'mean' },
        { id: 'add', type: 'add' },
      ],
      [
        ['in', 'l'],
        ['l', 'm'],
        ['m', 'add.a'],
        ['w', 'add.b'],
      ],
    );
    expect(messages(analyse(gr, reg))[0]).toBe(
      'Add needs both inputs to have the same shape: a single number and a 1 by 1 weight matrix.',
    );
  });

  test('mean of a scalar, norm of examples', () => {
    const gr = spec();
    gr.nodes.push({ id: 'm2', type: 'mean' }, { id: 'n', type: 'norm', kind: 'l1' });
    gr.edges.push(['loss', 'm2'], ['a1', 'n']);
    expect(messages(analyse(gr, reg))).toEqual([
      'A norm needs a weight matrix input, but this one is 4 per example.',
      'Mean needs a per-example input, but this one is a single number.',
    ]);
  });

  test('a cycle', () => {
    const gr = g(
      [input(), { id: 'p', type: 'add' }, lin('q', 4)],
      [
        ['in', 'p.a'],
        ['p', 'q'],
        ['q', 'p.b'],
      ],
      ['q'],
      null,
    );
    const a = analyse(gr, reg);
    expect(a.issues.find((i) => i.message.includes('loop'))).toEqual({
      at: { node: 'p' },
      message: 'The wires form a loop through "p", "q"; remove one wire to break it.',
    });
    expect(a.shape['p']).toBeUndefined();
    expect(a.edgeShape[1]).toBeNull();
    expect(a.order).toEqual(['in']);
  });

  test('a self loop', () => {
    const gr = g(
      [input(), act('s')],
      [
        ['s', 's'],
        ['in', 's'],
      ],
    );
    expect(analyse(gr, reg).issues.some((i) => i.message.includes('wired to itself'))).toBe(true);
  });

  test('dangling inputs', () => {
    const gr = spec();
    gr.edges.splice(2, 1); // a1 -> l2
    const a = analyse(gr, reg);
    expect(a.issues).toEqual([{ at: { node: 'l2' }, message: 'Linear needs an input.' }]);
    expect(a.edgeWidth).toEqual([1, 4, null, 1, null, null]);
    const b = spec();
    b.edges.pop(); // err -> loss
    expect(messages(analyse(b, reg))[0]).toBe('Mean needs an input.');
  });

  test('a doubly fed port keeps the first wire', () => {
    const gr = spec();
    gr.nodes.push(lin('extra', 4));
    gr.edges.push(['in', 'extra'], ['extra', 'l2']);
    const a = analyse(gr, reg);
    expect(a.issues).toEqual([
      {
        at: { edge: 8, node: 'l2' },
        message: '"l2" has two wires into its in input; each input takes only one.',
      },
    ]);
    expect(a.edgeWidth[8]).toBeNull();
    expect(a.order).not.toContain('extra');
  });

  test('a bad edge is an issue on that edge', () => {
    const gr = spec();
    gr.edges.push(['ghost', 'l1']);
    expect(analyse(gr, reg).issues[0]?.at).toEqual({ edge: 7 });
  });

  test('duplicate ids and an empty graph do not throw', () => {
    const gr = spec();
    gr.nodes.push(lin('l1', 2));
    expect(messages(analyse(gr, reg))).toEqual(['Two nodes are both called "l1".']);
    const a = analyse(g([], []), reg);
    expect(a.trainable).toBe(false);
    expect(a.order).toEqual([]);
  });

  test('ids that look like Object.prototype keys are safe', () => {
    const a = analyse(g([{ id: 'constructor', type: 'input', columns: ['x1'] }], []), reg);
    expect(a.outWidth['constructor']).toBe(1);
    expect(a.outWidth['toString']).toBeUndefined();
  });
});

describe('analyse v2: Weights taps and the penalty', () => {
  /** spec plus an L2 penalty on l1 and l2. */
  function penalised(): Graph {
    const gr = spec();
    let n = 0;
    const sub = weightPenaltySubgraph({
      kind: 'l2',
      lambda: 0.01,
      linears: ['l1', 'l2'],
      name: (s) => `pen_${s}_${n++}`,
      finalId: 'total',
    });
    gr.nodes.push(...sub.nodes);
    gr.edges.push(...sub.edges, ['loss', 'total.a']);
    gr.loss = 'total';
    return gr;
  }

  test('regions, parameter wires, scalar shapes', () => {
    const a = analyse(penalised(), reg);
    expect(a.issues).toEqual([]);
    expect(a.shape['pen_w_l1_0']).toEqual({ kind: 'param', rows: 1, cols: 4 });
    expect(a.shape['total']).toEqual({ kind: 'scalar' });
    expect(a.paramWires).toHaveLength(2);
    expect(a.paramWires[0]).toEqual({ linear: 'l1', weights: 'pen_w_l1_0' });
    expect(a.region['pen_w_l1_0']).toBe('objective');
    expect(a.region['total']).toBe('objective');
    expect(a.modelOrder).toEqual(['in', 'l1', 'a1', 'l2']);
    expect(a.totalParams).toBe(13);
  });

  test('Weights without a data wire: the Linear is live but its inputs are not in order', () => {
    const gr = g(
      [
        input(),
        lin('l1', 2),
        { id: 'w', type: 'weights', of: 'l1' },
        { id: 'n', type: 'norm', kind: 'l1' },
      ],
      [
        ['in', 'l1'],
        ['w', 'n'],
      ],
      [],
      'n',
    );
    const a = analyse(gr, reg);
    expect(a.issues).toEqual([]);
    expect(a.order).toEqual(['w', 'n']);
    expect(a.paramNodes).toEqual(['l1']);
    expect(a.paramShapes['l1']).toEqual({ W: [1, 2], b: [2] });
    expect(a.totalParams).toBe(4);
    expect(a.paramWires).toEqual([{ linear: 'l1', weights: 'w' }]);
  });

  test('Weights earlier in node order than its Linear still resolves', () => {
    const gr = penalised();
    gr.nodes.reverse();
    expect(analyse(gr, reg).issues).toEqual([]);
  });

  test('weights.of naming no node, or a non-Linear', () => {
    const gr = penalised();
    const w = gr.nodes.find((n) => n.id === 'pen_w_l1_0');
    if (w?.type !== 'weights') throw new Error('setup');
    w.of = 'ghost';
    const a = analyse(gr, reg);
    expect(a.issues).toEqual([
      {
        at: { node: 'pen_w_l1_0' },
        message: 'Weights reads "ghost", which is not a piece in this graph.',
      },
    ]);
    expect(a.paramWires.map((p) => p.weights)).not.toContain('pen_w_l1_0');
    w.of = 'a1';
    expect(messages(analyse(gr, reg))).toEqual([
      'Weights must read a Linear piece, but "a1" is not one.',
    ]);
  });

  test('a deleted Linear leaves a Weights tap with an issue, not a crash', () => {
    const gr = penalised();
    gr.nodes = gr.nodes.filter((n) => n.id !== 'l2');
    gr.edges = gr.edges.filter((e) => !e.some((r) => v2.parsePort(r).node === 'l2'));
    const a = analyse(gr, reg);
    expect(a.trainable).toBe(false);
    expect(messages(a).some((m) => m.includes('which is not a piece'))).toBe(true);
  });

  test('a Weights tap whose Linear has no input has its root cause at the Linear', () => {
    const gr = penalised();
    gr.edges = gr.edges.filter((e) => e[1] !== 'l1');
    const a = analyse(gr, reg);
    expect(a.issues.map((i) => i.at.node)).toContain('l1');
    expect(messages(a).filter((m) => m.includes('Weights'))).toEqual([]);
  });

  test('Linears read only through taps count towards the parameters', () => {
    const gr = g(
      [
        input(),
        lin('big', 64),
        lin('big2', 64),
        act('a'),
        { id: 'w', type: 'weights', of: 'big2' },
        { id: 'n', type: 'norm', kind: 'l2' },
        lin('huge', 64),
        { id: 'w2', type: 'weights', of: 'huge' },
        { id: 'n2', type: 'norm', kind: 'l2' },
        { id: 'sum', type: 'add' },
      ],
      [
        ['in', 'big'],
        ['big', 'a'],
        ['a', 'big2'],
        ['in', 'huge'],
        ['w', 'n'],
        ['w2', 'n2'],
        ['n', 'sum.a'],
        ['n2', 'sum.b'],
      ],
      [],
      'sum',
    );
    const a = analyse(gr, reg);
    expect(a.issues).toEqual([]);
    // big is not live: only big2's tap is in order, not its input.
    expect(a.paramNodes).toEqual(['big2', 'huge']);
    expect(a.totalParams).toBe(64 * 64 + 64 + 64 + 64);
  });

  test('the parameter limit is an issue', () => {
    const gr = g(
      [input(), lin('a', 64), lin('b', 64), lin('c', 64), lin('d', 64), lin('e', 1)],
      [
        ['in', 'a'],
        ['a', 'b'],
        ['b', 'c'],
        ['c', 'd'],
        ['d', 'e'],
      ],
      ['e'],
      null,
    );
    const a = analyse(gr, reg);
    expect(a.totalParams).toBe(128 + 3 * (64 * 64 + 64) + 65);
    expect(messages(a)).toContain('This model has 12,673 parameters; the limit is 10,000.');
  });
});

describe('analyse v2: Weights loops', () => {
  const loop = (): Graph => {
    const gr = spec();
    gr.nodes = gr.nodes.filter((n) => n.id !== 'l1' && n.id !== 'a1');
    gr.nodes.push(
      lin('l1', 4),
      { id: 'add', type: 'add' },
      { id: 'w1', type: 'weights', of: 'l1' },
      { id: 'n1', type: 'norm', kind: 'l2' },
    );
    gr.edges = [
      ['in', 'add.a'],
      ['w1', 'n1'],
      ['n1', 'add.b'],
      ['add', 'l1'],
      ['l1', 'l2'],
      ['l2', 'diff.a'],
      ['y', 'diff.b'],
      ['diff', 'err'],
      ['err', 'loss'],
    ];
    return gr;
  };

  test('a tap that depends on itself is an issue, not silence', () => {
    const a = analyse(loop(), reg);
    expect(a.trainable).toBe(false);
    expect(a.issues).toContainEqual({
      at: { node: 'w1' },
      message: '"w1" reads the weights of "l1", which depends on "w1" itself.',
    });
  });

  test('the safety net: no missing shape with no issue', () => {
    for (const gr of [loop(), spec()]) {
      const a = analyse(gr, reg);
      if (a.trainable) for (const id of a.order) expect(a.shape[id]).toBeDefined();
    }
  });
});

describe('analyse v2: loss pieces take probabilities', () => {
  function build(
    link: 'sigmoid' | 'softmax' | null,
    width = 1,
    kind: 'log_loss' | 'cross_entropy' = 'log_loss',
  ): Graph {
    const nodes: Node[] = [
      input(['x1', 'x2']),
      lin('l', width),
      tgt,
      { id: 'e', type: kind },
      { id: 'loss', type: 'mean' },
    ];
    const edges: Edge[] = [
      ['in', 'l'],
      ['y', 'e.target'],
      ['e', 'loss'],
    ];
    if (link) {
      nodes.push({ id: 'p', type: 'link', fn: link });
      edges.push(['l', 'p'], ['p', 'e.pred']);
    } else edges.push(['l', 'e.pred']);
    return g(nodes, edges, [link ? 'p' : 'l'], 'loss');
  }

  test('sigmoid then log loss is fusable', () => {
    const a = analyse(build('sigmoid'), bin);
    expect(a.issues).toEqual([]);
    expect(a.fusable).toEqual([{ link: 'p', loss: 'e' }]);
  });

  test('softmax then cross-entropy is fusable', () => {
    const a = analyse(build('softmax', 2, 'cross_entropy'), bin);
    expect(a.issues).toEqual([]);
    expect(a.fusable).toEqual([{ link: 'p', loss: 'e' }]);
  });

  test('a missing or wrong link, wrong width, regression data', () => {
    expect(messages(analyse(build(null), bin))).toEqual([
      'Log loss takes probabilities, so wire a Sigmoid link straight into its pred input.',
    ]);
    expect(messages(analyse(build('softmax', 1), bin))).toEqual([
      'This dataset has 2 classes, so the softmax prediction needs width 2, but "p" has width 1.',
      'Log loss takes probabilities, so wire a Sigmoid link straight into its pred input.',
    ]);
    expect(messages(analyse(build('sigmoid', 2), bin))).toContain(
      'Log loss needs a prediction of width 1, but this one is 2.',
    );
    expect(messages(analyse(build('softmax', 3, 'cross_entropy'), bin))).toContain(
      'Cross-entropy needs one probability per class (2), but its pred input has width 3.',
    );
    expect(messages(analyse(build('sigmoid'), { ...reg, columns: ['x1', 'x2'] }))).toEqual([
      'Log loss needs a dataset with two classes, but this dataset is regression.',
    ]);
  });
});

// ---------------------------------------------------------------- v1 parity

const fixturesDir = fileURLToPath(new URL('../../engine/test/fixtures/', import.meta.url));
const presetsDir = fileURLToPath(new URL('../../web/src/presets/', import.meta.url));

function sources(): { name: string; json: unknown }[] {
  const out: { name: string; json: unknown }[] = [];
  for (const dir of [fixturesDir, presetsDir]) {
    for (const f of readdirSync(dir).filter((x) => x.endsWith('.json'))) {
      const j = JSON.parse(readFileSync(dir + f, 'utf8')) as { graph: { version?: number } };
      if (j.graph.version === 2) continue; // prim_* fixtures are already version 2
      out.push({ name: f, json: j.graph });
    }
  }
  return out;
}

describe('analyse v2 against v1 analyse, through migration', () => {
  const all = sources();
  test('there are graphs to check', () => expect(all.length).toBeGreaterThan(30));
  for (const { name, json } of all) {
    test(name, () => {
      const v1 = parseGraphV1(json);
      const parsed = parseGraph(json);
      if (!v1.ok || !parsed.ok) throw new Error('parse');
      const info = sample(v1.graph.data).info;
      const a1 = analyseV1(v1.graph, info);
      const a2 = analyse(parsed.graph, info);
      expect(a2.issues).toEqual([]);
      expect(a2.trainable).toBe(a1.trainable);
      expect(a2.totalParams).toBe(a1.totalParams);
      for (const [id, w] of Object.entries(a1.outWidth)) {
        const mine = a2.outWidth[id];
        if (mine !== undefined) expect(mine, `${id} width`).toBe(w);
      }
      for (const [id, p] of Object.entries(a1.params)) {
        const mine = a2.params[id];
        if (mine !== undefined) expect(mine, `${id} params`).toBe(p);
      }
      for (const id of a2.order) {
        expect(a2.shape[id], `${id} shape`).toBeDefined();
      }
      // roles: L is a single number, every ŷ is in the model region
      expect(a2.lossNode).not.toBeNull();
      expect(a2.shape[a2.lossNode ?? '']).toEqual({ kind: 'scalar' });
      for (const p of a2.predictions) expect(a2.region[p]).toBe('model');
    });
  }
});
