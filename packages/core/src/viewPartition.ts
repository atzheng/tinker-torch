/**
 * The model tab and the loss tab (CONTRACTS.md section 9, "Tabs"): two views of the one graph.
 * Pure; nothing here is stored. `partitionNodes` says which tab a node lives on, `viewOfTab` gives
 * the graph a tab draws: its own nodes plus a stub ("ghost") for every piece on the other tab that
 * a wire or a Weights tap crosses to.
 */
import type { Analysis } from './analyse.js';
import type { Graph, Node, NodeId, PieceType } from './graph.js';
import { resolveEdge } from './graph.js';
import type { PaletteEntry } from './registry.js';

export type ViewTab = 'model' | 'loss';
export const VIEW_TABS: readonly ViewTab[] = ['model', 'loss'];
/** The `UI_IDS` id of each tab. */
export const TAB_UI_IDS: Record<ViewTab, string> = { model: 'tab-model', loss: 'tab-loss' };

/**
 * The tabs a preset leaves visible. `show.ui` lists neither tab id: both are shown (presets written
 * before tabs, and presets that say nothing about them). Otherwise exactly the listed tabs.
 */
export function shownTabs(ui: readonly string[] | undefined): ViewTab[] {
  if (ui === undefined) return [...VIEW_TABS];
  const listed = VIEW_TABS.filter((t) => ui.includes(TAB_UI_IDS[t]));
  return listed.length === 0 ? [...VIEW_TABS] : listed;
}

/** The tab to draw: `want` when it is shown, else the first shown one. */
export function effectiveTab(want: ViewTab, ui: readonly string[] | undefined): ViewTab {
  const shown = shownTabs(ui);
  return shown.includes(want) ? want : (shown[0] ?? 'model');
}

/**
 * Pieces that belong to the loss tab by kind: the error and loss pieces, the Target, and the
 * penalty pieces (Weights, Norm, Scale). Everything else is model. Kind only decides for a piece
 * no role reaches (and for every piece while no ŷ pin resolves).
 */
const LOSS_TYPES: ReadonlySet<PieceType> = new Set<PieceType>([
  'target',
  'subtract',
  'square',
  'abs',
  'huber',
  'log_loss',
  'cross_entropy',
  'mean',
  'scale',
  'weights',
  'norm',
]);

/** The tab a piece of this type belongs to when nothing else decides. */
export function pieceTab(type: PieceType): ViewTab {
  return LOSS_TYPES.has(type) ? 'loss' : 'model';
}

/**
 * The tab each node lives on: the model region (ancestors of the ŷ pins) is the model tab, the
 * rest of what the L pin needs is the loss tab, and a piece no pin reaches goes by its kind. With
 * no resolving ŷ pin everything goes by kind.
 */
export function partitionNodes(graph: Graph, analysis: Analysis): Record<NodeId, ViewTab> {
  const out = Object.create(null) as Record<NodeId, ViewTab>;
  const byRegion = analysis.predictions.length > 0;
  for (const n of graph.nodes) {
    if (out[n.id] !== undefined) continue;
    const region = byRegion ? analysis.region[n.id] : undefined;
    out[n.id] = region === 'model' ? 'model' : region === 'objective' ? 'loss' : pieceTab(n.type);
  }
  return out;
}

/** An edge no tab draws: it never resolves, so layout and drawing skip it and indices stay put. */
const HIDDEN_EDGE: [string, string] = ['', ''];

export type TabView = {
  /**
   * The graph this tab draws: own nodes and ghosts only, `edges` the same length as the real
   * ones (so edge indices mean the same) with the wires this tab does not draw blanked out. Draw
   * and hit-test with this; compute every edit from the real graph.
   */
  graph: Graph;
  /** The stubs: node id to the tab the piece really lives on. */
  ghosts: Record<NodeId, ViewTab>;
};

/**
 * What `tab` draws. A piece on the other tab is a ghost when a wire joins it to a piece here, or
 * (loss tab) when it is the Linear a Weights tap here reads. A wire is drawn when at least one end
 * is a piece of this tab and both ends are on screen.
 */
export function viewOfTab(graph: Graph, analysis: Analysis, tab: ViewTab): TabView {
  const part = partitionNodes(graph, analysis);
  const ghosts = Object.create(null) as Record<NodeId, ViewTab>;
  const resolved = graph.edges.map((e) => resolveEdge(graph.nodes, e));
  const mine = (id: NodeId) => part[id] === tab;
  for (const r of resolved) {
    if (!r.ok) continue;
    const a = r.from.node;
    const b = r.to.node;
    const pa = part[a];
    const pb = part[b];
    if (mine(a) && pb !== undefined && pb !== tab) ghosts[b] = pb;
    if (mine(b) && pa !== undefined && pa !== tab) ghosts[a] = pa;
  }
  for (const n of graph.nodes) {
    if (n.type !== 'weights' || !mine(n.id)) continue;
    const of = part[n.of];
    if (of !== undefined && of !== tab) ghosts[n.of] = of;
  }
  const keep = (n: Node) => mine(n.id) || ghosts[n.id] !== undefined;
  const nodes = graph.nodes.filter(keep);
  const ids = new Set(nodes.map((n) => n.id));
  const edges = graph.edges.map((e, i) => {
    const r = resolved[i];
    if (!r?.ok) return e;
    const a = r.from.node;
    const b = r.to.node;
    return ids.has(a) && ids.has(b) && (mine(a) || mine(b)) ? e : HIDDEN_EDGE;
  });
  const positions: Record<NodeId, [number, number]> = {};
  for (const [id, xy] of Object.entries(graph.positions ?? {})) if (mine(id)) positions[id] = xy;
  const view: Graph = { ...graph, nodes, edges };
  if (Object.keys(positions).length > 0) view.positions = positions;
  else delete view.positions;
  return { graph: view, ghosts };
}

/** The tabs whose palette offers this entry: pieces by kind, every ready-made group on the loss tab, and Add on both (it joins a loss and its penalty). */
export function paletteTabs(entry: PaletteEntry): ViewTab[] {
  if (entry.kind === 'macro') return ['loss'];
  const type = entry.make().type;
  return type === 'add' ? ['model', 'loss'] : [pieceTab(type)];
}
