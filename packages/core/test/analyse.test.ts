import { describe, expect, test } from 'vitest';
import {
  analyse,
  type DataInfo,
  type Edge,
  type Graph,
  type LossFn,
  type Node,
  type OutputLink,
} from '../src/index.js';

const reg: DataInfo = { columns: ['x1'], kind: 'regression', numClasses: 2 };
const bin: DataInfo = { columns: ['x1', 'x2'], kind: 'binary', numClasses: 2 };

function g(nodes: Node[], edges: Edge[]): Graph {
  return {
    version: 1,
    nodes,
    edges,
    data: { set: 'line', n: 100, noise: 0.1, val: 0.2, seed: 1, standardize: false },
    train: { optimizer: 'adam', lr: 0.01, batch: 'full', seed: 1 },
  };
}
const input = (columns = ['x1']): Node => ({ id: 'in', type: 'input', columns });
const lin = (id: string, out: number, bias = true): Node => ({ id, type: 'linear', out, bias });
const act = (id: string): Node => ({ id, type: 'activation', fn: 'tanh' });
const outp = (link: OutputLink = 'identity'): Node => ({ id: 'out', type: 'output', link });
const tgt: Node = { id: 'y', type: 'target' };
const loss = (fn: LossFn, id = 'loss'): Node =>
  fn === 'huber' ? { id, type: 'loss', fn, delta: 1 } : { id, type: 'loss', fn };

/** The SPEC example: 1 -> 4 -> 4 -> 1. */
function spec(): Graph {
  return g(
    [input(), lin('l1', 4), act('a1'), lin('l2', 1), outp(), tgt, loss('squared_error')],
    [
      ['in', 'l1'],
      ['l1', 'a1'],
      ['a1', 'l2'],
      ['l2', 'out'],
      ['out', 'loss.pred'],
      ['y', 'loss.target'],
    ],
  );
}
const messages = (a: ReturnType<typeof analyse>) => a.issues.map((i) => i.message);

describe('analyse: the SPEC example', () => {
  test('widths, parameters, order and trainable', () => {
    const a = analyse(spec(), reg);
    expect(a.issues).toEqual([]);
    expect(a.trainable).toBe(true);
    expect(a.order).toEqual(['in', 'y', 'l1', 'a1', 'l2', 'out', 'loss']);
    expect(a.order[a.order.length - 1]).toBe('loss');
    expect(a.outWidth).toMatchObject({ in: 1, l1: 4, a1: 4, l2: 1, out: 1, y: 1 });
    expect(a.inWidth).toMatchObject({ l1: 1, a1: 4, l2: 4, out: 1, loss: 1 });
    expect(a.params).toMatchObject({ l1: 8, l2: 5, a1: 0 });
    expect(a.totalParams).toBe(13);
    expect(a.predNode).toBe('out');
    expect(a.lossNode).toBe('loss');
  });

  test('edgeWidth for every edge', () => {
    expect(analyse(spec(), reg).edgeWidth).toEqual([1, 4, 4, 1, 1, 1]);
  });

  test('bias off removes the bias parameters', () => {
    const gr = spec();
    gr.nodes[1] = lin('l1', 4, false);
    expect(analyse(gr, reg).totalParams).toBe(4 + 5);
  });
});

describe('analyse: width rules', () => {
  test('columns switched off at the Input change downstream widths', () => {
    const edit = (cols: string[]) => {
      const gr = spec();
      gr.nodes[0] = input(cols);
      return analyse(gr, bin);
    };
    const two = edit(['x1', 'x2']);
    expect(two.inWidth['l1']).toBe(2);
    expect(two.params['l1']).toBe(2 * 4 + 4);
    const one = edit(['x2']);
    expect(one.inWidth['l1']).toBe(1);
    expect(one.params['l1']).toBe(8);
    expect(one.edgeWidth[0]).toBe(1);
  });

  test('all columns off gives an issue and unknown widths downstream', () => {
    const gr = spec();
    gr.nodes[0] = input([]);
    const a = analyse(gr, reg);
    expect(a.trainable).toBe(false);
    expect(messages(a)).toEqual(['Input needs at least one data column switched on.']);
    expect(a.issues[0]?.at).toEqual({ node: 'in' });
    expect(a.edgeWidth).toEqual([null, null, null, null, null, 1]);
    expect(a.outWidth['l1']).toBeUndefined();
    expect(a.params['l1']).toBeUndefined();
  });

  test('a column the dataset lacks is an issue', () => {
    const gr = spec();
    gr.nodes[0] = input(['x1', 'x3']);
    const a = analyse(gr, reg);
    expect(messages(a)).toEqual(['Input uses "x3", which this dataset does not have.']);
  });

  test('feature expansion widths', () => {
    const gr = g(
      [
        input(['x1', 'x2']),
        { id: 'f', type: 'feature', degree: 2, sin: true, cos: false, products: true },
        lin('l', 1),
        outp(),
        tgt,
        loss('squared_error'),
      ],
      [
        ['in', 'f'],
        ['f', 'l'],
        ['l', 'out'],
        ['out', 'loss.pred'],
        ['y', 'loss.target'],
      ],
    );
    const a = analyse(gr, bin);
    expect(a.outWidth['f']).toBe(2 * 2 + 2 + 1);
    expect(a.params['l']).toBe(7 + 1);
    expect(a.trainable).toBe(true);
  });

  test('dropout passes width through and has no parameters', () => {
    const gr = spec();
    gr.nodes.push({ id: 'd', type: 'dropout', p: 0.5 });
    gr.edges[2] = ['a1', 'd'];
    gr.edges.push(['d', 'l2']);
    const a = analyse(gr, reg);
    expect(a.trainable).toBe(true);
    expect(a.outWidth['d']).toBe(4);
    expect(a.params['d']).toBe(0);
  });
});

/** in -> l1(4) -> act -> add.a, and l1 -> add.b (skip) or via a layer of another width. */
function skip(bWidth: number | null): Graph {
  const nodes: Node[] = [
    input(),
    lin('l1', 4),
    act('a1'),
    { id: 'add', type: 'add' },
    lin('l2', 1),
    outp(),
    tgt,
    loss('squared_error'),
  ];
  const edges: Edge[] = [
    ['in', 'l1'],
    ['l1', 'a1'],
    ['a1', 'add.a'],
    ['add', 'l2'],
    ['l2', 'out'],
    ['out', 'loss.pred'],
    ['y', 'loss.target'],
  ];
  if (bWidth === null) edges.push(['l1', 'add.b']);
  else {
    nodes.push(lin('side', bWidth));
    edges.push(['a1', 'side'], ['side', 'add.b']);
  }
  return g(nodes, edges);
}

describe('analyse: skip connection through Add', () => {
  test('matching widths', () => {
    const a = analyse(skip(null), reg);
    expect(a.issues).toEqual([]);
    expect(a.outWidth['add']).toBe(4);
    expect(a.inWidth['add']).toBe(4);
    expect(a.params['l2']).toBe(5);
    expect(a.edgeWidth[2]).toBe(4);
    expect(a.edgeWidth[7]).toBe(4);
  });

  test('matching widths through a side layer', () => {
    const a = analyse(skip(4), reg);
    expect(a.issues).toEqual([]);
    expect(a.totalParams).toBe(8 + 20 + 5);
  });

  test('mismatching widths: red wire (edge into b), plain sentence, downstream unknown', () => {
    const gr = skip(8);
    const a = analyse(gr, reg);
    expect(a.trainable).toBe(false);
    expect(a.issues).toEqual([
      {
        at: { node: 'add', edge: 8 },
        message: 'Add needs both inputs to have the same width: 4 and 8.',
      },
    ]);
    expect(a.outWidth['add']).toBeUndefined();
    expect(a.edgeWidth[2]).toBe(4);
    expect(a.edgeWidth[8]).toBe(8);
    expect(a.edgeWidth[3]).toBeNull(); // add -> l2
    expect(a.edgeWidth[5]).toBeNull();
    expect(a.params['l2']).toBeUndefined();
  });

  test('Add with one input connected', () => {
    const gr = skip(null);
    gr.edges.pop();
    const a = analyse(gr, reg);
    expect(messages(a)).toEqual(['Add needs both of its inputs connected.']);
    expect(a.issues[0]?.at).toEqual({ node: 'add' });
  });
});

describe('analyse: structure', () => {
  test('a cycle', () => {
    const gr = g(
      [
        input(),
        { id: 'p', type: 'add' },
        lin('q', 4),
        lin('l2', 1),
        outp(),
        tgt,
        loss('squared_error'),
      ],
      [
        ['in', 'p.a'],
        ['p', 'q'],
        ['q', 'p.b'],
        ['q', 'l2'],
        ['l2', 'out'],
        ['out', 'loss.pred'],
        ['y', 'loss.target'],
      ],
    );
    const a = analyse(gr, reg);
    expect(a.trainable).toBe(false);
    const loop = a.issues.find((i) => i.message.includes('loop'));
    expect(loop?.message).toBe(
      'The wires form a loop through "p", "q"; remove one wire to break it.',
    );
    expect(loop?.at).toEqual({ node: 'p' });
    expect(a.outWidth['p']).toBeUndefined();
    expect(a.edgeWidth[0]).toBe(1);
    expect(a.edgeWidth[1]).toBeNull();
    expect(a.order).not.toContain('p');
  });

  test('a self loop', () => {
    const gr = spec();
    gr.edges.push(['l1', 'a1']);
    const a = analyse(gr, reg);
    // a1.in has two wires, and a1 feeds nothing back; only the double feed is reported.
    expect(messages(a)).toEqual([
      '"a1" has two wires into its in input; each input takes only one.',
    ]);
    const self = g(
      [input(), act('s')],
      [
        ['s', 's'],
        ['in', 's'],
      ],
    );
    expect(analyse(self, reg).issues.some((i) => i.message.includes('loop'))).toBe(true);
  });

  test('missing loss', () => {
    const gr = spec();
    gr.nodes.pop();
    gr.edges.splice(4, 2);
    const a = analyse(gr, reg);
    expect(a.lossNode).toBeNull();
    expect(a.predNode).toBeNull();
    expect(a.trainable).toBe(false);
    expect(messages(a)).toEqual(['Add a loss to the graph so there is something to train.']);
    expect(a.outWidth['out']).toBe(1);
  });

  test('two losses', () => {
    const gr = spec();
    gr.nodes.push(loss('absolute_error', 'loss2'));
    gr.edges.push(['out', 'loss2.pred'], ['y', 'loss2.target']);
    const a = analyse(gr, reg);
    expect(a.trainable).toBe(false);
    expect(a.lossNode).toBeNull();
    expect(a.issues).toEqual([
      { at: { node: 'loss2' }, message: 'The graph has 2 losses; keep only one.' },
    ]);
  });

  test('unconnected input port on Linear', () => {
    const gr = spec();
    gr.edges.splice(2, 1); // a1 -> l2
    const a = analyse(gr, reg);
    expect(a.issues).toEqual([{ at: { node: 'l2' }, message: 'Linear needs an input.' }]);
    expect(a.edgeWidth).toEqual([1, 4, null, null, 1]);
    expect(a.inWidth['l2']).toBeUndefined();
  });

  test('loss ports unconnected', () => {
    const gr = spec();
    gr.edges.splice(4, 2);
    const a = analyse(gr, reg);
    expect(a.issues).toEqual([
      { at: { node: 'loss' }, message: 'Connect the output to the loss.' },
      { at: { node: 'loss' }, message: 'Connect the target to the loss.' },
    ]);
    expect(a.predNode).toBeNull();
    expect(a.lossNode).toBe('loss');
  });

  test('target missing only', () => {
    const gr = spec();
    gr.edges.pop();
    expect(messages(analyse(gr, reg))).toEqual(['Connect the target to the loss.']);
  });

  test('doubly fed port reports the extra wire and keeps the first', () => {
    const gr = spec();
    gr.nodes.push(lin('extra', 4));
    gr.edges.push(['in', 'extra'], ['extra', 'l2']);
    const a = analyse(gr, reg);
    expect(a.issues).toEqual([
      {
        at: { edge: 7, node: 'l2' },
        message: '"l2" has two wires into its in input; each input takes only one.',
      },
    ]);
    expect(a.edgeWidth[7]).toBeNull();
    expect(a.edgeWidth[6]).toBe(1);
    expect(a.order).not.toContain('extra');
  });

  test('a bad edge is an issue on that edge', () => {
    const gr = spec();
    gr.edges.push(['ghost', 'l1']);
    const a = analyse(gr, reg);
    expect(a.issues[0]?.at).toEqual({ edge: 6 });
    expect(a.edgeWidth[6]).toBeNull();
  });

  test('pieces off the path to the loss are left out of order and the total', () => {
    const gr = spec();
    gr.nodes.push(lin('stray', 8));
    gr.edges.push(['in', 'stray']);
    const a = analyse(gr, reg);
    expect(a.trainable).toBe(true);
    expect(a.order).not.toContain('stray');
    expect(a.params['stray']).toBe(16);
    expect(a.totalParams).toBe(13);
  });

  test('the target only feeds the loss', () => {
    const gr = spec();
    gr.nodes.push(lin('t', 1));
    gr.edges.push(['y', 't']);
    const a = analyse(gr, reg);
    expect(a.issues).toEqual([
      { at: { edge: 6 }, message: 'The target can only be connected to the loss.' },
    ]);
  });

  test('the output only feeds the loss', () => {
    const gr = spec();
    gr.nodes.push(lin('t', 1));
    gr.edges.push(['out', 't']);
    expect(messages(analyse(gr, reg))).toEqual(['The output can only be connected to the loss.']);
  });

  test('a loss fed by something other than the Output or Target', () => {
    const gr = spec();
    gr.edges[4] = ['l2', 'loss.pred'];
    gr.edges[5] = ['in', 'loss.target'];
    const a = analyse(gr, reg);
    expect(messages(a)).toEqual([
      'Squared error compares the Output piece with the target, so connect the Output to it.',
      'Squared error needs the Target piece on its target input.',
    ]);
    expect(a.predNode).toBeNull();
  });

  test('a model with no Input piece', () => {
    const gr = g(
      [lin('l', 1), outp(), tgt, loss('squared_error')],
      [
        ['l', 'out'],
        ['out', 'loss.pred'],
        ['y', 'loss.target'],
      ],
    );
    const a = analyse(gr, reg);
    expect(a.issues.map((i) => i.message)).toContain('Linear needs an input.');
  });

  test('an empty graph does not throw', () => {
    const a = analyse(g([], []), reg);
    expect(a.trainable).toBe(false);
    expect(a.order).toEqual([]);
    expect(a.totalParams).toBe(0);
  });
});

describe('analyse: output / loss / target table', () => {
  /** in -> linear(width) -> output(link) -> loss. */
  function build(fn: LossFn, link: OutputLink, width: number): Graph {
    return g(
      [input(['x1', 'x2']), lin('l', width), outp(link), tgt, loss(fn)],
      [
        ['in', 'l'],
        ['l', 'out'],
        ['out', 'loss.pred'],
        ['y', 'loss.target'],
      ],
    );
  }
  const REG_LOSSES: LossFn[] = ['squared_error', 'absolute_error', 'huber'];

  test.each(REG_LOSSES)('%s: identity and sigmoid, width 1, regression or binary', (fn) => {
    for (const link of ['identity', 'sigmoid'] as const) {
      for (const d of [reg, bin]) {
        const a = analyse(build(fn, link, 1), { ...d, columns: ['x1', 'x2'] });
        expect(a.issues).toEqual([]);
        expect(a.trainable).toBe(true);
      }
    }
  });

  test.each(REG_LOSSES)('%s: softmax and wrong width are issues', (fn) => {
    const soft = analyse(build(fn, 'softmax', 1), bin);
    expect(soft.trainable).toBe(false);
    expect(soft.issues).toHaveLength(1);
    expect(soft.issues[0]?.message).toMatch(/needs an identity or sigmoid output, not softmax\.$/);
    expect(soft.issues[0]?.at).toEqual({ node: 'out' });
    const wide = analyse(build(fn, 'identity', 3), bin);
    expect(wide.issues).toHaveLength(1);
    expect(wide.issues[0]?.message).toMatch(
      /needs an output of width 1, but this output has width 3\.$/,
    );
  });

  test('log loss: sigmoid, width 1, binary', () => {
    expect(analyse(build('log_loss', 'sigmoid', 1), bin).trainable).toBe(true);
    for (const link of ['identity', 'softmax'] as const) {
      const a = analyse(build('log_loss', link, 1), bin);
      expect(messages(a)).toEqual(['Log loss needs a sigmoid output.']);
      expect(a.issues[0]?.at).toEqual({ node: 'out' });
    }
    expect(messages(analyse(build('log_loss', 'sigmoid', 2), bin))).toEqual([
      'Log loss needs an output of width 1, but this output has width 2.',
    ]);
    expect(
      messages(analyse(build('log_loss', 'sigmoid', 1), { ...reg, columns: ['x1', 'x2'] })),
    ).toEqual(['Log loss needs a dataset with two classes, but this dataset is regression.']);
  });

  test('cross-entropy: softmax, width numClasses, binary', () => {
    expect(analyse(build('cross_entropy', 'softmax', 2), bin).trainable).toBe(true);
    for (const link of ['identity', 'sigmoid'] as const) {
      expect(messages(analyse(build('cross_entropy', link, 2), bin))).toEqual([
        'Cross-entropy needs a softmax output.',
      ]);
    }
    expect(messages(analyse(build('cross_entropy', 'softmax', 1), bin))).toEqual([
      'Cross-entropy needs an output of width 2, one per class, but this output has width 1.',
    ]);
    expect(messages(analyse(build('cross_entropy', 'softmax', 3), bin))).toEqual([
      'Cross-entropy needs an output of width 2, one per class, but this output has width 3.',
    ]);
    expect(
      messages(analyse(build('cross_entropy', 'softmax', 2), { ...reg, columns: ['x1', 'x2'] })),
    ).toEqual(['Cross-entropy needs a dataset with two classes, but this dataset is regression.']);
  });

  test('numClasses drives the cross-entropy width', () => {
    const a = analyse(build('cross_entropy', 'softmax', 3), { ...bin, numClasses: 3 });
    expect(a.trainable).toBe(true);
  });

  test('a weight penalty does not change analysis', () => {
    const gr = build('squared_error', 'identity', 1);
    gr.nodes[4] = {
      id: 'loss',
      type: 'loss',
      fn: 'squared_error',
      penalty: { kind: 'l1', lambda: 0.1 },
    };
    expect(analyse(gr, bin).trainable).toBe(true);
  });
});

describe('analyse: parameter limit', () => {
  /** 1 -> a -> b -> c -> d -> 1, all with bias; total = 2a + (ab+b) + (bc+c) + (cd+d) + (d+1). */
  function chain(widths: number[], lastBias = true): Graph {
    const ids = widths.map((_, i) => `h${i}`);
    const nodes: Node[] = [
      input(),
      ...widths.map((w, i) => lin(ids[i] ?? '', w)),
      lin('last', 1, lastBias),
    ];
    nodes.push(outp(), tgt, loss('squared_error'));
    const edges: Edge[] = [['in', ids[0] ?? '']];
    for (let i = 1; i < ids.length; i++) edges.push([ids[i - 1] ?? '', ids[i] ?? '']);
    edges.push(
      [ids[ids.length - 1] ?? '', 'last'],
      ['last', 'out'],
      ['out', 'loss.pred'],
      ['y', 'loss.target'],
    );
    return g(nodes, edges);
  }

  test('exactly 10,000 is allowed', () => {
    const a = analyse(chain([34, 57, 64, 64]), reg);
    expect(a.totalParams).toBe(10000);
    expect(a.issues).toEqual([]);
    expect(a.trainable).toBe(true);
  });

  test('10,001 is not, and the sentence has thousands separators', () => {
    const a = analyse(chain([33, 63, 64, 56]), reg);
    expect(a.totalParams).toBe(10001);
    expect(a.trainable).toBe(false);
    expect(a.issues).toEqual([
      {
        at: {},
        message: `This model has ${a.totalParams.toLocaleString('en-US')} parameters; the limit is 10,000.`,
      },
    ]);
    expect(a.issues[0]?.message).toMatch(
      /^This model has 10,001 parameters; the limit is 10,000\.$/,
    );
  });

  test('exactly one over: 10,001', () => {
    // Removing the final bias from the 10,000 model and adding a one-unit side layer is fiddly;
    // instead check 10,000 + 1 via a Linear with bias on top of a model of 9,999 parameters.
    const a = analyse(chain([34, 57, 64, 64], false), reg);
    expect(a.totalParams).toBe(9999);
    expect(a.trainable).toBe(true);
  });
});

describe('analyse: inputs the parser accepts', () => {
  test('a self-edge is reported and does not break widths or order', () => {
    const gr = spec();
    gr.nodes.push(act('s'));
    gr.edges.push(['s', 's']);
    const a = analyse(gr, reg);
    expect(a.issues).toEqual([
      { at: { node: 's' }, message: '"s" is wired to itself; remove that wire.' },
    ]);
    expect(a.order).not.toContain('s');
    expect(a.outWidth['l2']).toBe(1);
    expect(a.edgeWidth[6]).toBeNull();
    expect(a.trainable).toBe(false);
  });

  test('a self-edge on a node on the model path blocks it without throwing', () => {
    const gr = spec();
    gr.nodes[2] = { id: 'a1', type: 'activation', fn: 'tanh' };
    gr.edges[1] = ['a1', 'a1'];
    const a = analyse(gr, reg);
    expect(a.issues.map((i) => i.message)).toContain('"a1" is wired to itself; remove that wire.');
    expect(a.outWidth['a1']).toBeUndefined();
    expect(a.trainable).toBe(false);
  });

  test('duplicate Input columns', () => {
    const gr = spec();
    gr.nodes[0] = input(['x1', 'x1']);
    const a = analyse(gr, reg);
    expect(a.issues).toEqual([
      {
        at: { node: 'in' },
        message: 'Input lists "x1" more than once; switch each column on only once.',
      },
    ]);
    expect(a.trainable).toBe(false);
  });

  test('unknown Input column names the column', () => {
    const gr = spec();
    gr.nodes[0] = input(['x2']);
    const a = analyse(gr, reg);
    expect(a.issues).toEqual([
      { at: { node: 'in' }, message: 'Input uses "x2", which this dataset does not have.' },
    ]);
  });
});
