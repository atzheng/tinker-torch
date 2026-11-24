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

/** The two halves a node can live on. */
export type SplitTab = 'model' | 'loss';
/**
 * The tabs, in tab order: the model half, the loss half, the whole graph ("Full graph") and the
 * generated PyTorch script ("Code"). `code` is not a canvas view: `viewOfTab` is never called with it.
 */
export type ViewTab = SplitTab | 'full' | 'code';
export const VIEW_TABS: readonly ViewTab[] = ['model', 'loss', 'full', 'code'];
/** The `UI_IDS` id of each tab. */
export const TAB_UI_IDS: Record<ViewTab, string> = {
  model: 'tab-model',
  loss: 'tab-loss',
  full: 'tab-full',
  code: 'tab-code',
};

/**
 * The tabs a preset leaves visible. `show.ui` lists no tab id: all four are shown (presets written
 * before tabs, and presets that say nothing about them). Otherwise exactly the listed tabs, so a
 * preset that lists `tab-model` and `tab-loss` stays a two-tab preset.
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
export function pieceTab(type: PieceType): SplitTab {
  return LOSS_TYPES.has(type) ? 'loss' : 'model';
}

/**
 * The tab each node lives on: the model region (ancestors of the ŷ pins) is the model tab, the
 * rest of what the L pin needs is the loss tab, and a piece no pin reaches goes by its kind. With
 * no resolving ŷ pin everything goes by kind.
 */
export function partitionNodes(graph: Graph, analysis: Analysis): Record<NodeId, SplitTab> {
  const out = Object.create(null) as Record<NodeId, SplitTab>;
  const byRegion = analysis.predictions.length > 0;
  for (const n of graph.nodes) {
    if (out[n.id] !== undefined) continue;
    const region = byRegion ? analysis.region[n.id] : undefined;
    out[n.id] = region === 'model' ? 'model' : region === 'objective' ? 'loss' : pieceTab(n.type);
  }
  return out;
}

/** Whether the preset leaves more than one tab (with one, that tab draws the whole graph). */
export function tabsSplit(ui: readonly string[] | undefined): boolean {
  return shownTabs(ui).filter((t) => t !== 'code').length >= 2;
}

/** Whether `tab` draws the whole graph: the full tab, or the only tab the preset leaves. */
export function tabIsWhole(tab: ViewTab, ui: readonly string[] | undefined): boolean {
  return tab === 'full' || !tabsSplit(ui);
}

/**
 * The pin roles that may be set, moved or dropped on `tab` (CONTRACTS.md section 10, "Role pins"):
 * only ŷ on the model tab, only L on the loss tab, both on the full tab and on a whole-graph tab.
 * An existing pin of the other role is still drawn; it just cannot be moved or removed there.
 */
export function pinRolesOf(tab: ViewTab, whole: boolean): ('pred' | 'loss')[] {
  if (whole || tab === 'full') return ['pred', 'loss'];
  return tab === 'model' ? ['pred'] : ['loss'];
}

/**
 * Pieces no role reaches whose palette entry is offered on both tabs (an unwired Add): drawn as
 * their own piece on both tabs, so one dropped on either tab stays in view.
 */
function onBothTabs(graph: Graph, analysis: Analysis): Set<NodeId> {
  const out = new Set<NodeId>();
  for (const n of graph.nodes) {
    if (n.type === 'add' && analysis.region[n.id] === undefined) out.add(n.id);
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
  /** The stubs: node id to the tab the piece really lives on. Only the loss tab has any. */
  ghosts: Record<NodeId, SplitTab>;
};

/**
 * What `tab` draws. The full tab (and `whole`) draws everything, no stubs. The model tab draws only
 * its own pieces and the wires between them: no stubs of loss pieces, no wires into them. The loss
 * tab also draws a stub for each ŷ-pinned node and each model piece a wire joins to a piece here, or the Linear a Weights
 * tap here reads. A wire is drawn when at least one end is a piece of this tab and both ends are on
 * screen.
 */
export function viewOfTab(
  graph: Graph,
  analysis: Analysis,
  tab: ViewTab,
  /** One tab shown (`!tabsSplit(show.ui)`): that tab draws the whole graph, no stubs. */
  whole = false,
): TabView {
  if (whole || tab === 'full') {
    return { graph, ghosts: Object.create(null) as Record<NodeId, SplitTab> };
  }
  const part = partitionNodes(graph, analysis);
  const both = onBothTabs(graph, analysis);
  const ghosts = Object.create(null) as Record<NodeId, SplitTab>;
  const resolved = graph.edges.map((e) => resolveEdge(graph.nodes, e));
  const mine = (id: NodeId) => part[id] === tab || both.has(id);
  // The model tab has no stubs: loss pieces are edited on the loss or full tab.
  for (const r of tab === 'model' ? [] : resolved) {
    if (!r.ok) continue;
    const a = r.from.node;
    const b = r.to.node;
    const pa = part[a];
    const pb = part[b];
    if (mine(a) && pb !== undefined && !mine(b)) ghosts[b] = pb;
    if (mine(b) && pa !== undefined && !mine(a)) ghosts[a] = pa;
  }
  // Every node carrying a ŷ pin is a stub on the loss tab, wired or not, so a wire from it into a
  // loss piece can be started there.
  if (tab === 'loss') {
    for (const id of analysis.predictions) {
      const pid = part[id];
      if (pid !== undefined && !mine(id)) ghosts[id] = pid;
    }
  }
  for (const n of graph.nodes) {
    if (tab === 'model' || n.type !== 'weights' || !mine(n.id)) continue;
    const of = part[n.of];
    if (of !== undefined && !mine(n.of)) ghosts[n.of] = of;
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

/** The tabs whose palette offers this entry: pieces by kind, every ready-made group on the loss tab, Add on both (it joins a loss and its penalty), and everything on the full tab. */
export function paletteTabs(entry: PaletteEntry): ViewTab[] {
  if (entry.kind === 'macro') return ['loss', 'full'];
  const type = entry.make().type;
  return type === 'add' ? ['model', 'loss', 'full'] : [pieceTab(type), 'full'];
}
