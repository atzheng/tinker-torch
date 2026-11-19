import { describe, expect, test } from 'vitest';
import {
  analyse,
  connect,
  deleteEdge,
  deleteNode,
  insertOnWire,
  parseGraph,
  rewireEdge,
  serializeGraph,
  setColumns,
  setLinearOut,
  setPenalty,
  setSetting,
  swapPiece,
  PALETTE,
  type EditResult,
  type Graph,
  type NodeBody,
} from '../src/index.js';

const example = {
  version: 1,
  nodes: [
    { id: 'x', type: 'input', columns: ['x1'] },
    { id: 'lin1', type: 'linear', out: 4, bias: true },
    { id: 'act1', type: 'activation', fn: 'relu' },
    { id: 'lin2', type: 'linear', out: 1, bias: true },
    { id: 'yhat', type: 'output', link: 'identity' },
    { id: 'y', type: 'target' },
    { id: 'loss', type: 'loss', fn: 'squared_error', penalty: { kind: 'l2', lambda: 0.0 } },
  ],
  edges: [
    ['x', 'lin1'],
    ['lin1', 'act1'],
    ['act1', 'lin2'],
    ['lin2', 'yhat'],
    ['yhat', 'loss.pred'],
    ['y', 'loss.target'],
  ],
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
  if (!e) throw new Error(key);
  return e.make() as NodeBody;
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
      ['activation1', 'yhat'],
      ['yhat', 'loss.pred'],
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

  test('refuses pieces without an input or output, and bad indexes', () => {
    const g = base();
    err(insertOnWire(g, 0, entry('input')));
    err(insertOnWire(g, 0, entry('target')));
    err(insertOnWire(g, 0, entry('huber')));
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

  test('squared_error -> huber defaults delta and keeps the penalty', () => {
    const out = ok(swapPiece(base(), 'loss', entry('absolute_error')));
    const huber = ok(swapPiece(out, 'loss', { type: 'loss', fn: 'huber' }));
    expect(node(huber, 'loss')).toEqual({
      id: 'loss',
      type: 'loss',
      fn: 'huber',
      delta: 1,
      penalty: { kind: 'l2', lambda: 0 },
    });
  });

  test('huber -> other loss drops delta', () => {
    const g = base((j) => {
      j['nodes'][6] = { id: 'loss', type: 'loss', fn: 'huber', delta: 0.5 };
    });
    const out = ok(swapPiece(g, 'loss', entry('squared_error')));
    expect(node(out, 'loss')).toEqual({ id: 'loss', type: 'loss', fn: 'squared_error' });
  });

  test('log_loss -> cross_entropy', () => {
    const g = base((j) => {
      j['nodes'][6] = { id: 'loss', type: 'loss', fn: 'log_loss' };
    });
    const out = ok(swapPiece(g, 'loss', entry('cross_entropy')));
    expect(node(out, 'loss')).toMatchObject({ fn: 'cross_entropy' });
  });

  test('refuses activation -> loss and other cross-family swaps', () => {
    const g = base();
    err(swapPiece(g, 'act1', entry('squared_error')));
    err(swapPiece(g, 'lin1', entry('activation')));
    err(swapPiece(g, 'loss', entry('output')));
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

  test('dropout <-> add moves wires by port position', () => {
    const g = ok(insertOnWire(base(), 1, entry('dropout')));
    const toAdd = ok(swapPiece(g, 'dropout1', entry('add')));
    expect(toAdd.edges[1]).toEqual(['lin1', 'dropout1.a']);
    expect(node(toAdd, 'dropout1')).toEqual({ id: 'dropout1', type: 'add' });
    const back = ok(swapPiece(toAdd, 'dropout1', entry('dropout')));
    expect(back.edges[1]).toEqual(['lin1', 'dropout1']);
    // an Add with both inputs is refused as a dropout rather than silently losing b
    const two = ok(connect(toAdd, 'x', 'dropout1.b'));
    err(swapPiece(two, 'dropout1', entry('dropout')));
  });

  test('Output and Target never swap for each other', () => {
    const g = base();
    err(swapPiece(g, 'y', entry('output')));
    err(swapPiece(g, 'yhat', entry('target')));
    ok(swapPiece(g, 'yhat', { type: 'output', link: 'sigmoid' }));
  });

  test('huber delta carries over huber -> huber', () => {
    const g = base((j) => {
      j['nodes'][6] = { id: 'loss', type: 'loss', fn: 'huber', delta: 3 };
    });
    const out = ok(swapPiece(g, 'loss', entry('huber')));
    expect(node(out, 'loss')).toMatchObject({ fn: 'huber', delta: 3 });
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
    err(connect(g, 'x', 'loss')); // port not named
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
      ['lin2', 'yhat'],
      ['yhat', 'loss.pred'],
      ['y', 'loss.target'],
    ]);
    expect(analyse(out, info).trainable).toBe(true);
  });

  test('a Linear whose widths differ leaves a gap', () => {
    const out = ok(deleteNode(base(), 'lin1')); // 1 -> 4
    expect(out.edges).toEqual([
      ['act1', 'lin2'],
      ['lin2', 'yhat'],
      ['yhat', 'loss.pred'],
      ['y', 'loss.target'],
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
    expect(noLoss.edges.some(([, t]) => t.startsWith('loss'))).toBe(false);
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
    expect(out.edges).toHaveLength(5);
    expect(out.edges).not.toContainEqual(['lin1', 'act1']);
  });
  test('bad index and locked end', () => {
    const g = base();
    err(deleteEdge(g, 6));
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
    expect(node(ok(setSetting(base(), 'yhat', 'link', 'sigmoid')), 'yhat')).toMatchObject({
      link: 'sigmoid',
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
    err(setSetting(g, 'loss', 'delta', 2)); // only huber
    err(setSetting(g, 'nope', 'out', 2));
  });
  test('loss fn and delta', () => {
    const toHuber = ok(setSetting(base(), 'loss', 'fn', 'huber'));
    expect(node(toHuber, 'loss')).toMatchObject({ fn: 'huber', delta: 1 });
    const tuned = ok(setSetting(toHuber, 'loss', 'delta', 0.3));
    expect(node(tuned, 'loss')).toMatchObject({ delta: 0.3 });
    err(setSetting(tuned, 'loss', 'delta', 0));
    const away = ok(setSetting(tuned, 'loss', 'fn', 'absolute_error'));
    expect(node(away, 'loss')).not.toHaveProperty('delta');
  });
  test('undefined removes an optional setting', () => {
    const out = ok(setSetting(base(), 'loss', 'penalty', undefined));
    expect(node(out, 'loss')).not.toHaveProperty('penalty');
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

describe('setPenalty', () => {
  test('set, replace, remove', () => {
    const g = base();
    const l1 = ok(setPenalty(g, 'loss', { kind: 'l1', lambda: 0.5 }));
    expect(node(l1, 'loss')).toMatchObject({ penalty: { kind: 'l1', lambda: 0.5 } });
    const none = ok(setPenalty(l1, 'loss', null));
    expect(node(none, 'loss')).not.toHaveProperty('penalty');
    const l2 = ok(setPenalty(none, 'loss', entry('weight_penalty') as never));
    expect(node(l2, 'loss')).toMatchObject({ penalty: { kind: 'l2', lambda: 0.01 } });
  });
  test('refusals', () => {
    const g = base();
    err(setPenalty(g, 'loss', { kind: 'l2', lambda: -1 }));
    err(setPenalty(g, 'loss', { kind: 'l3' as 'l2', lambda: 1 }));
    err(setPenalty(g, 'lin1', { kind: 'l2', lambda: 1 }));
    err(setPenalty(g, 'nope', null));
    err(setPenalty(g, 'loss', null, { locked: ['loss'] }));
  });
  test('does not alias the argument', () => {
    const p = { kind: 'l2' as const, lambda: 1 };
    const out = ok(setPenalty(base(), 'loss', p));
    const n = node(out, 'loss');
    expect(n?.type === 'loss' && n.penalty !== p).toBe(true);
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
    cur = ok(setPenalty(cur, 'loss', null));
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
      if (p.family === 'penalties') continue;
      const body = p.make() as NodeBody;
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
