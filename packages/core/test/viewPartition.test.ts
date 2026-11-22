import { describe, expect, test } from 'vitest';
import {
  PALETTE,
  analyse,
  effectiveTab,
  layoutLayers,
  paletteTabs,
  pinRolesOf,
  tabIsWhole,
  parseFile,
  partitionNodes,
  shownTabs,
  tabsSplit,
  viewOfTab,
  type DataInfo,
  type Edge,
  type Graph,
  type Node,
} from '../src/index.js';

const info: DataInfo = { columns: ['x1'], kind: 'regression', numClasses: 2 };

function g(nodes: Node[], edges: Edge[], predictions: string[], loss: string | null): Graph {
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

// x -> lin1 -> act1 -> lin2 (yhat) -> diff.a ; y -> diff.b ; diff -> err -> loss (L)
const tail: Node[] = [
  { id: 'y', type: 'target' },
  { id: 'diff', type: 'subtract' },
  { id: 'err', type: 'square' },
  { id: 'loss', type: 'mean' },
];
function mlp(): Graph {
  return g(
    [
      { id: 'x', type: 'input', columns: ['x1'] },
      { id: 'lin1', type: 'linear', out: 3, bias: true },
      { id: 'act1', type: 'activation', fn: 'relu' },
      { id: 'lin2', type: 'linear', out: 1, bias: true },
      ...tail,
    ],
    [
      ['x', 'lin1'],
      ['lin1', 'act1'],
      ['act1', 'lin2'],
      ['lin2', 'diff.a'],
      ['y', 'diff.b'],
      ['diff', 'err'],
      ['err', 'loss'],
    ],
    ['lin2'],
    'loss',
  );
}
// plus an L2 penalty reading lin1, added to the loss
function withPenalty(): Graph {
  const base = mlp();
  return {
    ...base,
    nodes: [
      ...base.nodes,
      { id: 'w1', type: 'weights', of: 'lin1' },
      { id: 'n1', type: 'norm', kind: 'l2' },
      { id: 'sc', type: 'scale', c: 0.01 },
      { id: 'tot', type: 'add' },
    ],
    edges: [...base.edges, ['w1', 'n1'], ['n1', 'sc'], ['loss', 'tot.a'], ['sc', 'tot.b']],
    loss: 'tot',
  };
}
const ids = (v: Graph) => v.nodes.map((n) => n.id);

describe('partitionNodes', () => {
  test('model region is the model tab, the rest of the loss path is the loss tab', () => {
    const graph = withPenalty();
    const p = partitionNodes(graph, analyse(graph, info));
    expect(p['x']).toBe('model');
    expect(p['lin2']).toBe('model');
    for (const id of ['y', 'diff', 'err', 'loss', 'w1', 'n1', 'sc', 'tot']) {
      expect(p[id], id).toBe('loss');
    }
  });

  test('a piece no pin reaches goes by its kind', () => {
    const graph = mlp();
    graph.nodes.push({ id: 'free-lin', type: 'linear', out: 2, bias: true });
    graph.nodes.push({ id: 'free-huber', type: 'huber', delta: 1 });
    graph.nodes.push({ id: 'free-add', type: 'add' });
    const p = partitionNodes(graph, analyse(graph, info));
    expect(p['free-lin']).toBe('model');
    expect(p['free-add']).toBe('model');
    expect(p['free-huber']).toBe('loss');
  });

  test('with no prediction pin everything goes by kind', () => {
    const graph = { ...mlp(), predictions: [] };
    const p = partitionNodes(graph, analyse(graph, info));
    expect(p['lin1']).toBe('model');
    expect(p['diff']).toBe('loss');
    expect(p['y']).toBe('loss');
  });
});

describe('viewOfTab', () => {
  test('model tab: only the model pieces, no loss stubs, no wire into a loss piece', () => {
    const graph = mlp();
    const v = viewOfTab(graph, analyse(graph, info), 'model');
    expect(ids(v.graph)).toEqual(['x', 'lin1', 'act1', 'lin2']);
    expect(v.ghosts).toEqual({});
    // same edge indices: the wire into the loss (edge 3) and the loss-side ones are blanked
    expect(v.graph.edges).toHaveLength(graph.edges.length);
    expect(v.graph.edges[3]).toEqual(['', '']);
    expect(v.graph.edges[4]).toEqual(['', '']);
    expect(v.graph.edges[0]).toEqual(['x', 'lin1']);
    expect(v.graph.edges[2]).toEqual(['act1', 'lin2']);
  });

  test('full tab: the whole graph, no stubs, whatever the preset shows', () => {
    const graph = withPenalty();
    const a = analyse(graph, info);
    const v = viewOfTab(graph, a, 'full');
    expect(v.graph).toBe(graph);
    expect(v.ghosts).toEqual({});
    expect(tabIsWhole('full', undefined)).toBe(true);
    expect(tabIsWhole('model', undefined)).toBe(false);
    expect(tabIsWhole('model', ['tab-model'])).toBe(true);
  });

  test('pin roles: ŷ on the model tab, L on the loss tab, both on the full tab or a whole view', () => {
    expect(pinRolesOf('model', false)).toEqual(['pred']);
    expect(pinRolesOf('loss', false)).toEqual(['loss']);
    expect(pinRolesOf('full', false)).toEqual(['pred', 'loss']);
    expect(pinRolesOf('loss', true)).toEqual(['pred', 'loss']);
  });

  test('loss tab: the loss pieces, plus the prediction as a stub', () => {
    const graph = mlp();
    const v = viewOfTab(graph, analyse(graph, info), 'loss');
    expect(ids(v.graph)).toEqual(['lin2', 'y', 'diff', 'err', 'loss']);
    expect(v.ghosts).toEqual({ lin2: 'model' });
    expect(v.graph.edges[3]).toEqual(['lin2', 'diff.a']);
    expect(v.graph.edges[4]).toEqual(['y', 'diff.b']);
    expect(v.graph.edges[2]).toEqual(['', '']);
  });

  test('a Weights tap makes its Linear a stub on the loss tab, not the model tab', () => {
    const graph = withPenalty();
    const a = analyse(graph, info);
    const loss = viewOfTab(graph, a, 'loss');
    expect(loss.ghosts).toEqual({ lin2: 'model', lin1: 'model' });
    const model = viewOfTab(graph, a, 'model');
    expect(ids(model.graph)).not.toContain('w1');
    expect(model.ghosts).toEqual({});
  });

  test('a wire into a loss piece is not drawn on the model tab', () => {
    const graph = mlp();
    graph.edges.push(['lin2', 'err']);
    const v = viewOfTab(graph, analyse(graph, info), 'model');
    const l = viewOfTab(graph, analyse(graph, info), 'loss');
    // the wire lin2 -> err: the model tab never draws wires into loss pieces
    expect(v.graph.edges[graph.edges.length - 1]).toEqual(['', '']);
    expect(l.ghosts['lin2']).toBe('model');
    // a wire between two stubs on the loss tab is blanked (lin1 -> act1 never reaches it)
    expect(l.graph.edges[0]).toEqual(['', '']);
  });

  test('positions keep only the tab`s own pieces', () => {
    const graph = {
      ...mlp(),
      positions: { lin1: [10, 10], diff: [200, 40] } as Graph['positions'],
    };
    const a = analyse(graph, info);
    expect(viewOfTab(graph, a, 'model').graph.positions).toEqual({ lin1: [10, 10] });
    expect(viewOfTab(graph, a, 'loss').graph.positions).toEqual({ diff: [200, 40] });
    const only = { ...graph, positions: { lin1: [8, 8] } as Graph['positions'] };
    expect(viewOfTab(only, a, 'loss').graph.positions).toBeUndefined();
  });

  test('layout of a tab places every piece it draws, including the stubs', () => {
    const graph = withPenalty();
    const a = analyse(graph, info);
    for (const tab of ['model', 'loss'] as const) {
      const v = viewOfTab(graph, a, tab);
      const l = layoutLayers(v.graph, a);
      for (const n of v.graph.nodes) expect(l.nodes[n.id], `${tab}:${n.id}`).toBeDefined();
      for (const w of l.wires) expect(v.graph.edges[w.edge]).not.toEqual(['', '']);
    }
  });

  test('one tab shown: that tab draws the whole graph with no stubs', () => {
    const graph = mlp();
    const v = viewOfTab(graph, analyse(graph, info), 'loss', true);
    expect(v.graph).toBe(graph);
    expect(v.ghosts).toEqual({});
    expect(tabsSplit(['tab-loss'])).toBe(false);
    expect(tabsSplit(undefined)).toBe(true);
  });

  test('an unwired Add is drawn on both tabs', () => {
    const graph = mlp();
    graph.nodes.push({ id: 'free-add', type: 'add' });
    const a = analyse(graph, info);
    expect(ids(viewOfTab(graph, a, 'model').graph)).toContain('free-add');
    expect(ids(viewOfTab(graph, a, 'loss').graph)).toContain('free-add');
    expect(viewOfTab(graph, a, 'loss').ghosts['free-add']).toBeUndefined();
  });

  test('the two tabs together cover every node and every wire', () => {
    const graph = withPenalty();
    const a = analyse(graph, info);
    const m = viewOfTab(graph, a, 'model');
    const l = viewOfTab(graph, a, 'loss');
    expect([...new Set([...ids(m.graph), ...ids(l.graph)])].sort()).toEqual(ids(graph).sort());
    graph.edges.forEach((e, i) => {
      const seen = m.graph.edges[i]?.[0] === e[0] || l.graph.edges[i]?.[0] === e[0];
      expect(seen, `edge ${i}`).toBe(true);
    });
  });
});

describe('tabs in a preset', () => {
  test('no tab id listed: all three tabs; some listed: only those', () => {
    expect(shownTabs(undefined)).toEqual(['model', 'loss', 'full']);
    expect(shownTabs(['panel-fit'])).toEqual(['model', 'loss', 'full']);
    expect(shownTabs(['tab-loss', 'tab-full'])).toEqual(['loss', 'full']);
    expect(shownTabs(['tab-full'])).toEqual(['full']);
    expect(shownTabs(['tab-loss', 'panel-loss'])).toEqual(['loss']);
    expect(shownTabs(['tab-model', 'tab-loss'])).toEqual(['model', 'loss']);
  });

  test('the effective tab never is a hidden one', () => {
    expect(effectiveTab('model', ['tab-loss'])).toBe('loss');
    expect(effectiveTab('model', undefined)).toBe('model');
    expect(effectiveTab('loss', ['tab-model'])).toBe('model');
    expect(effectiveTab('full', ['tab-model', 'tab-loss'])).toBe('model');
    expect(effectiveTab('full', undefined)).toBe('full');
  });

  const preset = (show: unknown) =>
    parseFile({
      kind: 'tinker',
      version: 2,
      graph: JSON.parse(JSON.stringify(mlp())),
      title: 'T',
      show,
    });
  test('show.tab picks the opening tab; show.ui may hide one', () => {
    const r = preset({ ui: ['tab-loss', 'panel-loss'], tab: 'loss' });
    expect(r.ok && 'show' in r.file ? r.file.show : null).toEqual({
      ui: ['tab-loss', 'panel-loss'],
      tab: 'loss',
    });
    expect(preset({ tab: 'loss' }).ok).toBe(true);
    expect(preset({ tab: 'full' }).ok).toBe(true);
    expect(preset({ ui: ['tab-model', 'tab-loss'], tab: 'full' }).ok).toBe(false);
    expect(preset({ ui: ['tab-full'] }).ok).toBe(true);
  });

  test('plain sentences for a bad tab', () => {
    const bad = preset({ tab: 'both' });
    expect(!bad.ok && bad.error).toBe('The preset "show.tab" must be "model", "loss" or "full".');
    const hidden = preset({ ui: ['tab-model'], tab: 'loss' });
    expect(!hidden.ok && hidden.error).toBe(
      'The preset opens on the loss tab, but its "show.ui" hides that tab.',
    );
  });
});

describe('paletteTabs', () => {
  const tabs = (key: string) => {
    const e = PALETTE.find((x) => x.key === key);
    if (!e) throw new Error(key);
    return paletteTabs(e);
  };
  test('model pieces on the model tab, loss and penalty pieces on the loss tab', () => {
    expect(tabs('linear')).toEqual(['model', 'full']);
    expect(tabs('input')).toEqual(['model', 'full']);
    expect(tabs('sigmoid')).toEqual(['model', 'full']);
    expect(tabs('target')).toEqual(['loss', 'full']);
    expect(tabs('square')).toEqual(['loss', 'full']);
    expect(tabs('l2')).toEqual(['loss', 'full']);
    expect(tabs('weights')).toEqual(['loss', 'full']);
    expect(tabs('loss_squared_error')).toEqual(['loss', 'full']);
    expect(tabs('weight_penalty')).toEqual(['loss', 'full']);
    expect(tabs('add')).toEqual(['model', 'loss', 'full']);
  });

  test('every palette entry is on at least one tab', () => {
    for (const e of PALETTE) expect(paletteTabs(e).length).toBeGreaterThan(0);
  });
});
