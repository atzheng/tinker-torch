import { describe, expect, test } from 'vitest';
import {
  analyse,
  addPrediction,
  connect,
  deleteEdge,
  deleteNode,
  insertOnWire,
  movePrediction,
  parseGraph,
  removePrediction,
  setLoss,
  rewireEdge,
  serializeGraph,
  setColumns,
  setLinearOut,
  insertMacro,
  setSetting,
  swapPiece,
  PALETTE,
  type EditResult,
  type Graph,
  type NodeBody,
} from '../src/index.js';

const example = {
  version: 2,
  nodes: [
    { id: 'x', type: 'input', columns: ['x1'] },
    { id: 'lin1', type: 'linear', out: 4, bias: true },
    { id: 'act1', type: 'activation', fn: 'relu' },
    { id: 'lin2', type: 'linear', out: 1, bias: true },
    { id: 'y', type: 'target' },
    { id: 'diff', type: 'subtract' },
    { id: 'err', type: 'square' },
    { id: 'loss', type: 'mean' },
  ],
  edges: [
    ['x', 'lin1'],
    ['lin1', 'act1'],
    ['act1', 'lin2'],
    ['lin2', 'diff.a'],
    ['y', 'diff.b'],
    ['diff', 'err'],
    ['err', 'loss'],
  ],
  predictions: ['lin2'],
  loss: 'loss',
  data: { set: 'sine', n: 40, noise: 0.2, val: 0.3, seed: 7 },
  train: { optimizer: 'sgd', lr: 0.03, batch: 'full', seed: 11 },
};

function deepFreeze<T>(o: T): T {
  if (typeof o === 'object' && o !== null && !Object.isFrozen(o)) {
    Object.freeze(o);
    for (const v of Object.values(o)) deepFreeze(v);
  }
  return o;
}

function parseOk(json: unknown): Graph {
  const r = parseGraph(json);
  if (!r.ok) throw new Error(r.error);
  return r.graph;
}

// Test input is loosely typed JSON on purpose.
type Loose = Record<string, any>; // eslint-disable-line @typescript-eslint/no-explicit-any -- see above

/** A fresh, deeply frozen graph so any mutation throws. */
function base(patch?: (j: Loose) => void): Graph {
  const j = JSON.parse(JSON.stringify(example)) as Loose;
  patch?.(j);
  return deepFreeze(parseOk(j));
}

function ok(r: EditResult): Graph {
  if (!r.ok) throw new Error(`expected ok, got: ${r.error}`);
  // Every ok result re-parses to itself.
  const again = parseGraph(JSON.parse(serializeGraph(r.graph)));
  expect(again.ok).toBe(true);
  if (again.ok) expect(again.graph).toEqual(r.graph);
  return r.graph;
}

function err(r: EditResult): string {
  if (r.ok) throw new Error('expected an error');
  expect(r.error.length).toBeGreaterThan(0);
  return r.error;
}

const entry = (key: string): NodeBody => {
  const e = PALETTE.find((p) => p.key === key);
  if (e?.kind !== 'piece') throw new Error(key);
  return e.make();
};

const info = { columns: ['x1', 'x2'], kind: 'regression' as const, numClasses: 2 };
const node = (g: Graph, id: string) => g.nodes.find((n) => n.id === id);

describe('insertOnWire', () => {
  test('inserts an activation on a wire and selects it', () => {
    const g = base();
    const r = insertOnWire(g, 3, entry('activation'));
    const out = ok(r);
    expect(r.ok && r.select).toBe('activation1');
    expect(node(out, 'activation1')).toEqual({ id: 'activation1', type: 'activation', fn: 'relu' });
    expect(out.edges.slice(2, 6)).toEqual([
      ['act1', 'lin2'],
      ['lin2', 'activation1'],
      ['activation1', 'diff.a'],
      ['y', 'diff.b'],
    ]);
    expect(analyse(out, info).edgeWidth[3]).toBe(1);
  });

  test('new id is the smallest free k', () => {
    let g = base();
    g = ok(insertOnWire(g, 0, entry('activation')));
    g = ok(insertOnWire(g, 0, entry('activation')));
    expect(g.nodes.filter((n) => n.type === 'activation').map((n) => n.id)).toEqual([
      'act1',
      'activation1',
      'activation2',
    ]);
    // free a gap and it is reused
    g = ok(deleteNode(g, 'activation1'));
    g = ok(insertOnWire(g, 0, entry('activation')));
    expect(node(g, 'activation1')).toBeDefined();
  });

  test('an Add goes in on input a', () => {
    const out = ok(insertOnWire(base(), 1, entry('add')));
    expect(out.edges[1]).toEqual(['lin1', 'add1.a']);
    expect(out.edges[2]).toEqual(['add1', 'act1']);
  });

  test('refuses pieces without an input, and bad indexes', () => {
    const g = base();
    err(insertOnWire(g, 0, entry('input')));
    err(insertOnWire(g, 0, entry('target')));
    err(insertOnWire(g, 0, entry('weights')));
    ok(insertOnWire(g, 0, entry('huber')));
    err(insertOnWire(g, 99, entry('linear')));
    err(insertOnWire(g, -1, entry('linear')));
  });

  test('refuses a wire into a locked node', () => {
    const g = base();
    expect(err(insertOnWire(g, 1, entry('activation'), { locked: ['act1'] }))).toBe(
      'This piece is locked.',
    );
    ok(insertOnWire(g, 1, entry('activation'), { locked: ['lin1'] }));
  });
});

describe('swapPiece', () => {
  test('relu -> tanh', () => {
    const out = ok(swapPiece(base(), 'act1', { type: 'activation', fn: 'tanh' }));
    expect(node(out, 'act1')).toEqual({ id: 'act1', type: 'activation', fn: 'tanh' });
    expect(out.edges).toEqual(base().edges);
  });

  test('square -> huber defaults delta; huber -> abs drops it', () => {
    const huber = ok(swapPiece(base(), 'err', { type: 'huber', delta: 1 }));
    expect(node(huber, 'err')).toEqual({ id: 'err', type: 'huber', delta: 1 });
    expect(huber.edges).toEqual(base().edges);
    const abs = ok(swapPiece(huber, 'err', entry('abs')));
    expect(node(abs, 'err')).toEqual({ id: 'err', type: 'abs' });
  });

  test('log_loss -> cross_entropy keeps both wires', () => {
    const g = base((j) => {
      j['nodes'] = j['nodes'].filter((n: Loose) => n.id !== 'diff' && n.id !== 'err');
      j['nodes'].push({ id: 'err', type: 'log_loss' });
      j['edges'] = [
        ['x', 'lin1'],
        ['lin1', 'act1'],
        ['act1', 'lin2'],
        ['lin2', 'err.pred'],
        ['y', 'err.target'],
        ['err', 'loss'],
      ];
    });
    const out = ok(swapPiece(g, 'err', entry('cross_entropy')));
    expect(node(out, 'err')).toEqual({ id: 'err', type: 'cross_entropy' });
    expect(out.edges).toEqual(g.edges);
  });

  test('sigmoid <-> softmax is a link swap; the function is not carried over', () => {
    const g = ok(insertOnWire(base(), 3, entry('sigmoid')));
    const out = ok(swapPiece(g, 'link1', entry('softmax')));
    expect(node(out, 'link1')).toEqual({ id: 'link1', type: 'link', fn: 'softmax' });
    const l1 = ok(insertOnWire(base(), 0, entry('l1')));
    expect(node(ok(swapPiece(l1, 'norm1', entry('l2'))), 'norm1')).toMatchObject({ kind: 'l2' });
  });

  test('refuses activation -> loss and other cross-family swaps', () => {
    const g = base();
    err(swapPiece(g, 'act1', entry('log_loss')));
    err(swapPiece(g, 'lin1', entry('activation')));
    err(swapPiece(g, 'err', entry('mean')));
    err(swapPiece(g, 'loss', entry('square')));
  });

  test('a pin stays on a swapped node', () => {
    const out = ok(swapPiece(base(), 'lin2', { type: 'linear', out: 1, bias: false }));
    expect(out.predictions).toEqual(['lin2']);
  });

  test('refuses unknown and locked nodes', () => {
    const g = base();
    err(swapPiece(g, 'nope', entry('activation')));
    expect(err(swapPiece(g, 'act1', entry('activation'), { locked: ['act1'] }))).toBe(
      'This piece is locked.',
    );
  });

  test('linear keeps its width and bias', () => {
    const g = base((j) => {
      j['nodes'][1] = { id: 'lin1', type: 'linear', out: 9, bias: false };
    });
    const out = ok(swapPiece(g, 'lin1', entry('linear')));
    expect(node(out, 'lin1')).toEqual({ id: 'lin1', type: 'linear', out: 9, bias: false });
  });

  test('pieces with different input ports never swap (dropout and add, weights and norm)', () => {
    const g = ok(insertOnWire(base(), 1, entry('dropout')));
    err(swapPiece(g, 'dropout1', entry('add')));
    const pen = ok(insertMacro(base(), 'weight_penalty'));
    err(swapPiece(pen, 'w_lin1', entry('l2')));
    err(swapPiece(pen, 'n_lin1', entry('weights')));
    err(swapPiece(pen, 'pen', entry('subtract')));
    err(swapPiece(pen, 'loss', entry('scale')));
  });

  test('Input and Target never swap for each other', () => {
    err(swapPiece(base(), 'y', entry('input')));
    err(swapPiece(base(), 'x', entry('target')));
  });

  test('every swap the palette allows on a valid graph gives a valid graph', () => {
    const graphs = [base(), ok(insertMacro(base(), 'weight_penalty'))];
    const open = (g: Graph) =>
      analyse(g, info).issues.filter((i) => /input/.test(i.message)).length;
    for (const g of graphs) {
      for (const n of g.nodes) {
        for (const p of PALETTE) {
          if (p.kind !== 'piece') continue;
          const r = swapPiece(g, n.id, p.make());
          if (!r.ok) continue;
          ok(r);
          expect(r.graph.edges).toHaveLength(g.edges.length);
          expect(open(r.graph), `${n.id} -> ${p.key}`).toBe(open(g));
        }
      }
    }
  });

  test('huber delta carries over huber -> huber', () => {
    const g = base((j) => {
      j['nodes'][6] = { id: 'err', type: 'huber', delta: 3 };
    });
    const out = ok(swapPiece(g, 'err', entry('huber')));
    expect(node(out, 'err')).toMatchObject({ type: 'huber', delta: 3 });
  });
});

describe('insertOnWire pins', () => {
  test('a ŷ pin moves to the inserted node when it was the only wire out of the source', () => {
    const g = base(); // lin2 carries the ŷ pin; its only wire out is edge 3
    expect(ok(insertOnWire(g, 3, entry('sigmoid'))).predictions).toEqual(['link1']);
    // a wire that does not leave the pinned node leaves the pin alone
    expect(ok(insertOnWire(g, 5, entry('abs'))).predictions).toEqual(['lin2']);
  });

  test('an L pin moves too; a pin stays when the source has other wires out', () => {
    const withL = base((j) => {
      j['loss'] = 'diff';
    });
    expect(ok(insertOnWire(withL, 5, entry('scale'))).loss).toBe('scale1');
    const fan = ok(connect(base(), 'lin2', 'loss')); // lin2 now also feeds the mean
    const i = fan.edges.findIndex((e) => e[1] === 'diff.a');
    expect(ok(insertOnWire(fan, i, entry('abs'))).predictions).toEqual(['lin2']);
  });
});

describe('connect', () => {
  test('a skip into add.b', () => {
    let g = ok(insertOnWire(base(), 2, entry('add'))); // act1 -> add1.a -> lin2
    expect(g.edges[2]).toEqual(['act1', 'add1.a']);
    g = ok(connect(g, 'lin1', 'add1.b'));
    expect(g.edges.at(-1)).toEqual(['lin1', 'add1.b']);
    const a = analyse(g, info);
    expect(a.trainable).toBe(true);
  });

  test('replaces whatever fed the port and uses the short form', () => {
    const g = ok(connect(base(), 'x.out', 'act1.in'));
    expect(g.edges).toContainEqual(['x', 'act1']);
    expect(g.edges).not.toContainEqual(['lin1', 'act1']);
    expect(g.edges).toHaveLength(base().edges.length);
  });

  test('refuses cycles, self wires and bad ports', () => {
    const g = base();
    err(connect(g, 'lin2', 'lin1'));
    err(connect(g, 'act1', 'act1'));
    err(connect(g, 'act1', 'x'));
    err(connect(g, 'nope', 'lin1'));
    err(connect(g, 'loss', 'lin1'));
    err(connect(g, 'x', 'diff')); // port not named
  });

  test('refuses an indirect cycle through add', () => {
    const g = ok(insertOnWire(base(), 2, entry('add')));
    err(connect(g, 'lin2', 'add1.b'));
  });

  test('refuses the same wire again, and a locked destination', () => {
    const g = base();
    err(connect(g, 'lin1', 'act1'));
    err(connect(g, 'x', 'act1', { locked: ['act1'] }));
    ok(connect(g, 'x', 'act1', { locked: ['x'] }));
  });

  test('an output may fan out', () => {
    const g = ok(connect(base(), 'lin1', 'lin2'));
    expect(g.edges.filter(([f]) => f === 'lin1')).toHaveLength(2);
  });
});

describe('rewireEdge', () => {
  test('moves the destination end: x -> lin1 becomes x -> act1', () => {
    const g0 = base();
    const g = ok(rewireEdge(g0, 0, 'to', 'act1'));
    expect(g.edges).toContainEqual(['x', 'act1']);
    expect(g.edges).not.toContainEqual(['x', 'lin1']);
    expect(g.edges).not.toContainEqual(['lin1', 'act1']); // act1 had one feeder: replaced
    expect(g.edges).toHaveLength(g0.edges.length - 1);
  });

  test('moves the source end', () => {
    const g = ok(rewireEdge(base(), 1, 'from', 'x')); // lin1 -> act1 becomes x -> act1
    expect(g.edges).toContainEqual(['x', 'act1']);
    expect(g.edges).not.toContainEqual(['lin1', 'act1']);
  });

  test('keeps the other wires', () => {
    const g = ok(rewireEdge(base(), 0, 'to', 'act1'));
    expect(g.edges).toContainEqual(['act1', 'lin2']);
  });

  test('refuses cycles, bad ports and unknown wires', () => {
    err(rewireEdge(base(), 2, 'to', 'lin1')); // act1 -> lin2 becomes act1 -> lin1: a loop
    err(rewireEdge(base(), 1, 'to', 'lin1')); // lin1 -> lin1
    err(rewireEdge(base(), 2, 'to', 'nope'));
    err(rewireEdge(base(), 99, 'to', 'act1'));
  });

  test('refuses when the old or the new destination is locked', () => {
    expect(err(rewireEdge(base(), 0, 'to', 'act1', { locked: ['lin1'] }))).toBe(
      'This piece is locked.',
    );
    expect(err(rewireEdge(base(), 0, 'to', 'act1', { locked: ['act1'] }))).toBe(
      'This piece is locked.',
    );
    ok(rewireEdge(base(), 0, 'to', 'act1', { locked: ['x', 'lin2'] }));
    err(rewireEdge(base(), 1, 'from', 'x', { locked: ['act1'] })); // the source end: old destination
  });

  test('does not mutate its input', () => {
    const g = base();
    const before = serializeGraph(g);
    rewireEdge(g, 0, 'to', 'act1');
    expect(serializeGraph(g)).toBe(before);
  });
});

describe('deleteNode', () => {
  test('a middle activation closes the wire', () => {
    const out = ok(deleteNode(base(), 'act1'));
    expect(out.nodes.some((n) => n.id === 'act1')).toBe(false);
    expect(out.edges).toEqual([
      ['x', 'lin1'],
      ['lin1', 'lin2'],
      ['lin2', 'diff.a'],
      ['y', 'diff.b'],
      ['diff', 'err'],
      ['err', 'loss'],
    ]);
    expect(analyse(out, info).trainable).toBe(true);
  });

  test('a Linear whose widths differ leaves a gap', () => {
    const out = ok(deleteNode(base(), 'lin1')); // 1 -> 4
    expect(out.edges).toEqual([
      ['act1', 'lin2'],
      ['lin2', 'diff.a'],
      ['y', 'diff.b'],
      ['diff', 'err'],
      ['err', 'loss'],
    ]);
  });

  test('a Linear whose widths match closes over', () => {
    const g = base((j) => {
      j['nodes'][1].out = 1;
    });
    const out = ok(deleteNode(g, 'lin1'));
    expect(out.edges).toContainEqual(['x', 'act1']);
  });

  test('a fan-out node closes every outgoing wire', () => {
    let g = ok(insertOnWire(base(), 2, entry('add')));
    g = ok(connect(g, 'act1', 'add1.b'));
    const out = ok(deleteNode(g, 'act1'));
    expect(out.edges.filter(([f]) => f === 'lin1')).toHaveLength(2);
  });

  test('Add, Input, Target, loss: no closing, no crash', () => {
    const g = ok(insertOnWire(base(), 2, entry('add')));
    const noAdd = ok(deleteNode(g, 'add1'));
    expect(noAdd.edges.some(([f, t]) => f === 'act1' && t === 'lin2')).toBe(false);
    ok(deleteNode(base(), 'x'));
    ok(deleteNode(base(), 'y'));
    const noLoss = ok(deleteNode(base(), 'loss'));
    expect(noLoss.edges.some(([f, t]) => f === 'loss' || t === 'loss')).toBe(false);
  });

  test('a ŷ or L pin moves to what fed the node when the shape is kept, else it is dropped', () => {
    const g = base((j) => {
      j['predictions'] = ['act1', 'lin2'];
    });
    const out = ok(deleteNode(g, 'act1')); // 4 -> 4: closes, the pin moves up to lin1
    expect(out.predictions).toEqual(['lin1', 'lin2']);
    const moved = ok(deleteNode(base(), 'err')); // example(1) -> example(1): closes
    expect(moved.edges).toContainEqual(['diff', 'loss']);
    const noPred = ok(deleteNode(base(), 'lin2')); // 4 -> 1: gap, pin dropped
    expect(noPred.predictions).toEqual([]);
    const noL = ok(deleteNode(base(), 'loss')); // example -> scalar: gap, pin dropped
    expect(noL.loss).toBeNull();
    // pinning the same node twice is not possible, so the second pin that lands there is dropped
    const dup = base((j) => {
      j['predictions'] = ['lin1', 'act1'];
    });
    expect(ok(deleteNode(dup, 'act1')).predictions).toEqual(['lin1']);
  });

  test('a Linear takes its Weights taps along', () => {
    const g = ok(insertMacro(base(), 'weight_penalty'));
    expect(g.nodes.some((n) => n.type === 'weights' && n.of === 'lin1')).toBe(true);
    const out = ok(deleteNode(g, 'lin1'));
    expect(out.nodes.some((n) => n.type === 'weights' && n.of === 'lin1')).toBe(false);
    expect(out.nodes.some((n) => n.id === 'w_lin2')).toBe(true);
    expect(out.edges.some(([f, t]) => f === 'w_lin1' || t === 'w_lin1')).toBe(false);
    err(deleteNode(g, 'lin1', { locked: ['w_lin1'] }));
  });

  test('unknown node and locks', () => {
    const g = base();
    err(deleteNode(g, 'nope'));
    expect(err(deleteNode(g, 'act1', { locked: ['act1'] }))).toBe('This piece is locked.');
    // lin2 is fed by act1, so deleting act1 changes lin2
    err(deleteNode(g, 'act1', { locked: ['lin2'] }));
    ok(deleteNode(g, 'act1', { locked: ['x'] }));
  });

  test('a self-edge node does not close over itself', () => {
    const g = base((j) => {
      j['edges'].push(['act1', 'act1']);
    });
    ok(deleteNode(g, 'act1'));
  });
});

describe('deleteEdge', () => {
  test('removes one wire', () => {
    const out = ok(deleteEdge(base(), 1));
    expect(out.edges).toHaveLength(6);
    expect(out.edges).not.toContainEqual(['lin1', 'act1']);
  });
  test('bad index and locked end', () => {
    const g = base();
    err(deleteEdge(g, 7));
    err(deleteEdge(g, -1));
    err(deleteEdge(g, 1, { locked: ['act1'] }));
  });
});

describe('setSetting', () => {
  test('valid values', () => {
    expect(node(ok(setSetting(base(), 'lin1', 'bias', false)), 'lin1')).toMatchObject({
      bias: false,
    });
    expect(node(ok(setSetting(base(), 'act1', 'fn', 'gelu')), 'act1')).toMatchObject({
      fn: 'gelu',
    });
  });
  test('schema violations give a sentence', () => {
    const g = base();
    err(setSetting(g, 'lin1', 'out', 65));
    err(setSetting(g, 'lin1', 'out', 2.5));
    err(setSetting(g, 'lin1', 'out', 'big'));
    err(setSetting(g, 'lin1', 'out', Number.NaN));
    err(setSetting(g, 'lin1', 'colour', 'red'));
    err(setSetting(g, 'lin1', '__proto__', {}));
    err(setSetting(g, 'lin1', 'id', 'other'));
    err(setSetting(g, 'lin1', 'type', 'add'));
    err(setSetting(g, 'act1', 'fn', 'swish'));
    err(setSetting(g, 'err', 'delta', 2)); // only huber
    err(setSetting(g, 'loss', 'penalty', 1));
    err(setSetting(g, 'nope', 'out', 2));
  });
  test('huber delta, scale c, link fn, norm kind', () => {
    const g = ok(insertOnWire(base(), 5, entry('huber')));
    const tuned = ok(setSetting(g, 'huber1', 'delta', 0.3));
    expect(node(tuned, 'huber1')).toMatchObject({ delta: 0.3 });
    err(setSetting(tuned, 'huber1', 'delta', 0));
    const sc = ok(setSetting(ok(insertOnWire(base(), 5, entry('scale'))), 'scale1', 'c', -2));
    expect(node(sc, 'scale1')).toMatchObject({ c: -2 });
    const lk = ok(
      setSetting(ok(insertOnWire(base(), 3, entry('sigmoid'))), 'link1', 'fn', 'softmax'),
    );
    expect(node(lk, 'link1')).toMatchObject({ fn: 'softmax' });
    const nm = ok(setSetting(ok(insertOnWire(base(), 0, entry('l1'))), 'norm1', 'kind', 'l2'));
    expect(node(nm, 'norm1')).toMatchObject({ kind: 'l2' });
  });
  test('undefined removes a setting, which then takes its default', () => {
    const out = ok(setSetting(base(), 'lin1', 'bias', undefined));
    expect(node(out, 'lin1')).toMatchObject({ bias: true });
  });
  test('locked', () => {
    err(setSetting(base(), 'lin1', 'out', 3, { locked: ['lin1'] }));
  });
});

describe('setLinearOut', () => {
  test('bounds', () => {
    const g = base();
    expect(node(ok(setLinearOut(g, 'lin1', 1)), 'lin1')).toMatchObject({ out: 1 });
    expect(node(ok(setLinearOut(g, 'lin1', 64)), 'lin1')).toMatchObject({ out: 64 });
    err(setLinearOut(g, 'lin1', 0));
    err(setLinearOut(g, 'lin1', 65));
    err(setLinearOut(g, 'lin1', 2.5));
    err(setLinearOut(g, 'lin1', Number.NaN));
    err(setLinearOut(g, 'lin1', Infinity));
  });
  test('wrong node, unknown, locked', () => {
    const g = base();
    err(setLinearOut(g, 'act1', 3));
    err(setLinearOut(g, 'nope', 3));
    err(setLinearOut(g, 'lin1', 3, { locked: ['lin1'] }));
  });
});

/** The example without its loss group: a network with a ŷ pin and a target, no L. */
function noLoss(): Graph {
  return base((j) => {
    j['nodes'] = j['nodes'].filter((n: Loose) => !['diff', 'err', 'loss', 'y'].includes(n.id));
    j['edges'] = j['edges'].slice(0, 3);
    j['loss'] = null;
  });
}

describe('insertMacro', () => {
  test('a loss group goes in free-standing, wired to ŷ and a new Target, and takes L', () => {
    const r = insertMacro(noLoss(), 'loss_squared_error');
    const out = ok(r);
    expect(r.ok && r.select).toBe('loss');
    expect(out.nodes.map((n) => n.id)).toEqual([
      'x',
      'lin1',
      'act1',
      'lin2',
      'diff',
      'err',
      'loss',
      'target',
    ]);
    expect(out.edges).toEqual([
      ['x', 'lin1'],
      ['lin1', 'act1'],
      ['act1', 'lin2'],
      ['lin2', 'diff.a'],
      ['target', 'diff.b'],
      ['diff', 'err'],
      ['err', 'loss'],
    ]);
    expect(out.loss).toBe('loss');
    expect(analyse(out, info).trainable).toBe(true);
  });

  test('every loss macro makes a graph that parses; log loss needs a link to be trainable', () => {
    for (const key of PALETTE.flatMap((p) =>
      p.kind === 'macro' && p.key.startsWith('loss_') ? [p.key] : [],
    )) {
      ok(insertMacro(noLoss(), key));
    }
    const withLink = ok(insertOnWire(noLoss(), 2, entry('sigmoid')));
    const out = ok(insertMacro(withLink, 'loss_log_loss'));
    expect(out.nodes.some((n) => n.type === 'log_loss')).toBe(true);
  });

  test('uses the existing Target and keeps an existing L pin', () => {
    const out = ok(insertMacro(base(), 'loss_absolute_error'));
    expect(out.nodes.filter((n) => n.type === 'target')).toHaveLength(1);
    expect(out.loss).toBe('loss');
    expect(node(out, 'loss_2')).toEqual({ id: 'loss_2', type: 'mean' });
    expect(out.edges).toContainEqual(['y', 'diff_2.b']);
    expect(out.edges).toContainEqual(['lin2', 'diff_2.a']);
  });

  test('dropped on a wire, that wire feeds the group and stays', () => {
    const g = noLoss();
    const out = ok(insertMacro(g, 'loss_squared_error', { edge: 1 }));
    expect(out.edges).toContainEqual(['lin1', 'diff.a']);
    expect(out.edges).toContainEqual(['lin1', 'act1']);
    expect(err(insertMacro(g, 'loss_squared_error', { edge: 9 }))).toMatch(/wire/);
  });

  test('the weight penalty reads every Linear, takes the loss in and the L pin over', () => {
    const out = ok(insertMacro(base(), 'weight_penalty'));
    expect(out.nodes.map((n) => n.id).slice(8)).toEqual([
      'w_lin1',
      'n_lin1',
      'w_lin2',
      'n_lin2',
      'sum1',
      'pen',
      'total',
    ]);
    expect(out.edges).toContainEqual(['loss', 'total.a']);
    expect(out.loss).toBe('total');
    const a = analyse(out, info);
    expect(a.trainable).toBe(true);
    expect(a.totalParams).toBe(analyse(base(), info).totalParams);
  });

  test('refusals', () => {
    err(insertMacro(noLoss(), 'nope'));
    err(insertMacro(noLoss(), 'square')); // a single piece, not a group
    const noLinear = base((j) => {
      j['nodes'] = j['nodes'].filter((n: Loose) => n.type !== 'linear');
      j['edges'] = [];
      j['predictions'] = [];
      j['loss'] = null;
    });
    err(insertMacro(noLinear, 'weight_penalty'));
  });
});

describe('setColumns', () => {
  test('sets columns and widths follow', () => {
    const out = ok(setColumns(base(), 'x', ['x1', 'x2']));
    expect(analyse(out, info).outWidth['x']).toBe(2);
  });
  test('empty is allowed (analyse reports it); repeats are refused', () => {
    const out = ok(setColumns(base(), 'x', []));
    expect(analyse(out, info).trainable).toBe(false);
    err(setColumns(base(), 'x', ['x1', 'x1']));
  });
  test('wrong node, locked', () => {
    err(setColumns(base(), 'lin1', ['x1']));
    err(setColumns(base(), 'nope', ['x1']));
    err(setColumns(base(), 'x', ['x1'], { locked: ['x'] }));
  });
});

describe('purity', () => {
  test('a sequence of edits on a frozen graph never throws or changes it', () => {
    const g = base();
    const before = serializeGraph(g);
    let cur: Graph = g;
    cur = ok(insertOnWire(cur, 1, entry('dropout')));
    cur = ok(swapPiece(cur, 'act1', { type: 'activation', fn: 'sigmoid' }));
    cur = ok(setLinearOut(cur, 'lin1', 8));
    cur = ok(insertMacro(cur, 'weight_penalty'));
    cur = ok(setColumns(cur, 'x', ['x1']));
    cur = ok(deleteNode(cur, 'dropout1'));
    cur = ok(deleteEdge(cur, 0));
    cur = ok(connect(cur, 'x', 'lin1'));
    expect(serializeGraph(g)).toBe(before);
    expect(cur).not.toBe(g);
  });

  test('deterministic', () => {
    const g = base();
    expect(insertOnWire(g, 1, entry('activation'))).toEqual(
      insertOnWire(g, 1, entry('activation')),
    );
  });

  test('does not alias the piece argument', () => {
    const piece = entry('activation');
    const out = ok(insertOnWire(base(), 1, piece));
    expect(node(out, 'activation1')).not.toBe(piece);
  });

  test('every palette piece swaps or inserts to a re-parseable graph', () => {
    for (const p of PALETTE) {
      if (p.kind !== 'piece') continue;
      const body = p.make();
      for (let i = 0; i < 6; i++) {
        const r = insertOnWire(base(), i, body);
        if (r.ok) ok(r);
      }
      for (const n of base().nodes) {
        const r = swapPiece(base(), n.id, body);
        if (r.ok) ok(r);
      }
    }
  });
});

describe('pin edits', () => {
  const pins = (g: Graph) => ({ p: g.predictions, l: g.loss });

  test('setLoss moves the L pin and leaves the rest alone', () => {
    const g = base();
    const out = ok(setLoss(g, 'err'));
    expect(pins(out)).toEqual({ p: ['lin2'], l: 'err' });
    expect(out.nodes).toEqual(g.nodes);
    expect(out.edges).toEqual(g.edges);
    expect(pins(ok(setLoss(g, null)))).toEqual({ p: ['lin2'], l: null });
    expect(pins(ok(setLoss(g, 'err.out')))).toEqual({ p: ['lin2'], l: 'err' });
  });

  test('setLoss refuses the same node, unknown nodes and bad ports', () => {
    expect(err(setLoss(base(), 'loss'))).toBe('That is already the loss.');
    expect(err(setLoss(base(), 'nope'))).toMatch(/not a node/);
    expect(err(setLoss(base(), 'err.in'))).toMatch(/out/);
  });

  test('addPrediction appends; the same node is refused', () => {
    const g = base();
    expect(pins(ok(addPrediction(g, 'act1')))).toEqual({ p: ['lin2', 'act1'], l: 'loss' });
    expect(err(addPrediction(g, 'lin2'))).toBe('That piece is already a prediction.');
    expect(err(addPrediction(g, 'nope'))).toMatch(/not a node/);
  });

  test('movePrediction keeps the order; removePrediction drops one', () => {
    const g = ok(addPrediction(base(), 'act1'));
    expect(pins(ok(movePrediction(g, 0, 'lin1'))).p).toEqual(['lin1', 'act1']);
    expect(err(movePrediction(g, 0, 'act1'))).toBe('That piece is already a prediction.');
    expect(err(movePrediction(g, 0, 'lin2'))).toBe('That is already the prediction.');
    expect(err(movePrediction(g, 5, 'lin1'))).toBe('That prediction does not exist.');
    expect(pins(ok(removePrediction(g, 0))).p).toEqual(['act1']);
    expect(pins(ok(removePrediction(base(), 0))).p).toEqual([]);
    expect(err(removePrediction(base(), 1))).toBe('That prediction does not exist.');
  });

  test('a pin on a locked piece is fixed; pinning a free piece is fine', () => {
    const g = base();
    const locked = { locked: ['lin2'] };
    expect(err(movePrediction(g, 0, 'act1', locked))).toBe('This piece is locked.');
    expect(err(removePrediction(g, 0, locked))).toBe('This piece is locked.');
    expect(err(addPrediction(g, 'lin2', locked))).toBe('This piece is locked.');
    expect(err(setLoss(g, 'lin2', { locked: ['loss'] }))).toBe('This piece is locked.');
    expect(err(setLoss(g, 'err', { locked: ['err'] }))).toBe('This piece is locked.');
    ok(addPrediction(g, 'act1', locked));
    ok(setLoss(g, 'err', locked));
  });

  test('inputs are not changed', () => {
    const g = base();
    ok(setLoss(g, 'err'));
    ok(addPrediction(g, 'act1'));
  });
});
