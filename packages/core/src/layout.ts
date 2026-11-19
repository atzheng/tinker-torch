/**
 * Layout geometry (CONTRACTS.md section 9): pure, deterministic, nothing stored. Works on the whole
 * node set (not `analysis.order`), so invalid graphs lay out too: every node gets a position.
 */
import type { Graph, Node, NodeId } from './legacy/graph.js';
import { inputPortsOf, resolveEdge } from './legacy/graph.js';
import type { Analysis } from './analyse.js';

export type Box = { x: number; y: number; w: number; h: number };
export type LayerLayout = {
  nodes: Record<NodeId, Box & { col: number }>;
  /** `labelAt`: where the width label sits, the midpoint of the longest horizontal segment (first on ties). */
  wires: { edge: number; points: [number, number][]; labelAt: [number, number] }[];
  size: [number, number];
};
export type NeuronUnitKind = 'in' | 'sum' | 'act' | 'out' | 'target';
export type NeuronLayout = {
  units: { node: NodeId; index: number; kind: NeuronUnitKind; x: number; y: number }[];
  edges: {
    node: NodeId;
    i: number;
    j: number;
    from: [number, number];
    to: [number, number];
  }[];
  collapsed: NodeId[];
  boxes: LayerLayout['nodes'];
  wires: LayerLayout['wires'];
  size: [number, number];
};

/** Abstract units. The web canvas scales the whole drawing to fit. */
export const LAYOUT = {
  boxW: 112,
  boxH: 56,
  /** Space between columns in layer view (wire labels sit here). */
  layerGap: 72,
  /** Space between columns in neuron view (weight edges cross it). */
  neuronGap: 150,
  /** Width of a column of unit circles, and the vertical pitch of the circles. */
  unitW: 36,
  unitPitch: 44,
  /** Height reserved above a unit column for the piece's name (neuron view). */
  unitLabelH: 20,
  /** Space between rows. */
  rowGap: 24,
  /** Distance between skip-wire lanes, and outer margin. */
  lane: 18,
  pad: 24,
  /** A Linear expands only while both sides have at most this many units. */
  maxUnits: 8,
} as const;

type Plan = { w: number; h: number; unit: NeuronUnitKind | null; count: number };

type Core = {
  boxes: LayerLayout['nodes'];
  wires: LayerLayout['wires'];
  size: [number, number];
};

function nullRecord<T>(): Record<string, T> {
  return Object.create(null) as Record<string, T>;
}

/** Nodes with duplicates dropped (first wins) and the resolved, endpoint-valid edges. */
function prepare(g: Graph) {
  const nodes: Node[] = [];
  const byId = nullRecord<Node>();
  for (const n of g.nodes) {
    if (byId[n.id] === undefined) {
      byId[n.id] = n;
      nodes.push(n);
    }
  }
  const edges: { index: number; from: NodeId; to: NodeId; port: string }[] = [];
  g.edges.forEach((e, index) => {
    const r = resolveEdge(nodes, e);
    if (r.ok) edges.push({ index, from: r.from.node, to: r.to.node, port: r.to.port });
  });
  return { nodes, byId, edges };
}

/**
 * Column per node: longest path from the inputs. The edge into a loss from a target does not count
 * (the target is then placed one column left of its loss). A cycle is broken at its first node in
 * graph order; nodes with no wires at all (other than Input and Target) go in the last column.
 */
function assignColumns(
  nodes: Node[],
  byId: Record<string, Node>,
  edges: { from: NodeId; to: NodeId }[],
): Record<NodeId, number> {
  const col = nullRecord<number>();
  const counted = edges.filter(
    (e) => e.from !== e.to && !(byId[e.from]?.type === 'target' && byId[e.to]?.type === 'loss'),
  );
  const indeg = nullRecord<number>();
  const preds = nullRecord<NodeId[]>();
  const succs = nullRecord<NodeId[]>();
  for (const n of nodes) {
    indeg[n.id] = 0;
    preds[n.id] = [];
    succs[n.id] = [];
  }
  for (const e of counted) {
    indeg[e.to] = (indeg[e.to] ?? 0) + 1;
    preds[e.to]?.push(e.from);
    succs[e.from]?.push(e.to);
  }
  const done = nullRecord<boolean>();
  const place = (id: NodeId) => {
    let c = 0;
    for (const p of preds[id] ?? []) {
      const pc = col[p];
      if (pc !== undefined) c = Math.max(c, pc + 1);
    }
    col[id] = c;
    done[id] = true;
    for (const s of succs[id] ?? []) {
      if (done[s]) continue;
      indeg[s] = (indeg[s] ?? 1) - 1;
    }
  };
  for (;;) {
    const ready = nodes.find((n) => !done[n.id] && indeg[n.id] === 0);
    if (ready) {
      place(ready.id);
      continue;
    }
    const stuck = nodes.find((n) => !done[n.id]);
    if (!stuck) break;
    place(stuck.id);
  }
  let last = 0;
  for (const n of nodes) {
    if ((preds[n.id]?.length ?? 0) + (succs[n.id]?.length ?? 0) > 0) {
      last = Math.max(last, col[n.id] ?? 0);
    }
  }
  const touched = nullRecord<boolean>();
  for (const e of edges) {
    touched[e.from] = true;
    touched[e.to] = true;
  }
  for (const n of nodes) {
    if (!touched[n.id] && n.type !== 'input' && n.type !== 'target') col[n.id] = last;
  }
  for (const e of edges) {
    if (byId[e.from]?.type === 'target' && byId[e.to]?.type === 'loss') {
      col[e.from] = Math.max(0, (col[e.to] ?? 0) - 1);
    }
  }
  return col;
}

/** Row per node: follow the first-listed source, move down past taken slots; targets sit below the prediction. */
function assignRows(
  nodes: Node[],
  byId: Record<string, Node>,
  edges: { index: number; from: NodeId; to: NodeId; port: string }[],
  col: Record<NodeId, number>,
): Record<NodeId, number> {
  const row = nullRecord<number>();
  const wired = new Set(edges.flatMap((e) => [e.from, e.to]));
  const columns = [...new Set(nodes.map((n) => col[n.id] ?? 0))].sort((a, b) => a - b);
  for (const c of columns) {
    const here = nodes.filter((n) => (col[n.id] ?? 0) === c);
    const ordered = [
      ...here.filter((n) => wired.has(n.id) && n.type !== 'target'),
      ...here.filter((n) => wired.has(n.id) && n.type === 'target'),
      ...here.filter((n) => !wired.has(n.id)),
    ];
    const set = new Set<number>();
    for (const n of ordered) {
      let want: number;
      if (n.type === 'target') {
        const toLoss = edges.find((e) => e.from === n.id && byId[e.to]?.type === 'loss');
        const pred = toLoss
          ? edges.find((e) => e.to === toLoss.to && e.port === 'pred')
          : undefined;
        const pr = pred ? row[pred.from] : undefined;
        want = pr === undefined ? 1 : pr + 1;
      } else {
        const src = edges.find(
          (e) => e.to === n.id && e.from !== n.id && row[e.from] !== undefined,
        );
        want = src ? (row[src.from] ?? 0) : 0;
      }
      let r = want;
      while (set.has(r)) r++;
      set.add(r);
      row[n.id] = r;
    }
  }
  return row;
}

function labelAnchor(points: [number, number][]): [number, number] {
  let best: [number, number] = points[0] ?? [0, 0];
  let len = -1;
  for (let k = 0; k + 1 < points.length; k++) {
    const a = points[k];
    const b = points[k + 1];
    if (!a || !b || a[1] !== b[1]) continue;
    const l = Math.abs(b[0] - a[0]);
    if (l > len) {
      len = l;
      best = [(a[0] + b[0]) / 2, a[1]];
    }
  }
  return best;
}

function layoutCore(g: Graph, plan: (n: Node) => Plan, gap: number): Core {
  const { nodes, byId, edges } = prepare(g);
  const col = assignColumns(nodes, byId, edges);
  const row = assignRows(nodes, byId, edges, col);
  const plans = nullRecord<Plan>();
  for (const n of nodes) plans[n.id] = plan(n);

  const maxCol = nodes.reduce((m, n) => Math.max(m, col[n.id] ?? 0), 0);
  const maxRow = nodes.reduce((m, n) => Math.max(m, row[n.id] ?? 0), 0);
  const colW: number[] = Array.from({ length: maxCol + 1 }, () => 0);
  const rowH: number[] = Array.from({ length: maxRow + 1 }, () => 0);
  for (const n of nodes) {
    const p = plans[n.id];
    if (!p) continue;
    const c = col[n.id] ?? 0;
    const r = row[n.id] ?? 0;
    colW[c] = Math.max(colW[c] ?? 0, p.w);
    rowH[r] = Math.max(rowH[r] ?? 0, p.h);
  }

  // Skip wires (and back wires) get a lane above everything.
  const isSkip = (e: { from: NodeId; to: NodeId }) => (col[e.to] ?? 0) - (col[e.from] ?? 0) !== 1;
  const skips = edges
    .filter(isSkip)
    .sort(
      (a, b) =>
        Math.abs((col[a.to] ?? 0) - (col[a.from] ?? 0)) -
          Math.abs((col[b.to] ?? 0) - (col[b.from] ?? 0)) || a.index - b.index,
    );
  const laneOf = new Map<number, number>();
  skips.forEach((e, k) => laneOf.set(e.index, k));
  const top = LAYOUT.pad + (skips.length > 0 ? skips.length * LAYOUT.lane + LAYOUT.lane : 0);

  const colX: number[] = [];
  let x = LAYOUT.pad;
  for (let c = 0; c <= maxCol; c++) {
    colX[c] = x;
    x += (colW[c] ?? 0) + gap;
  }
  const width = x - gap + LAYOUT.pad;
  const rowY: number[] = [];
  let y = top;
  for (let r = 0; r <= maxRow; r++) {
    rowY[r] = y;
    y += (rowH[r] ?? 0) + LAYOUT.rowGap;
  }
  const height = y - LAYOUT.rowGap + LAYOUT.pad;

  const boxes = nullRecord<Box & { col: number }>() as LayerLayout['nodes'];
  for (const n of nodes) {
    const p = plans[n.id];
    if (!p) continue;
    const c = col[n.id] ?? 0;
    const r = row[n.id] ?? 0;
    boxes[n.id] = {
      x: (colX[c] ?? 0) + ((colW[c] ?? 0) - p.w) / 2,
      y: (rowY[r] ?? 0) + ((rowH[r] ?? 0) - p.h) / 2,
      w: p.w,
      h: p.h,
      col: c,
    };
  }

  const wires: LayerLayout['wires'] = [];
  for (const e of edges) {
    const a = boxes[e.from];
    const b = boxes[e.to];
    const dest = byId[e.to];
    if (!a || !b || !dest) continue;
    const ports = inputPortsOf(dest.type);
    const k = Math.max(0, ports.indexOf(e.port));
    const x0 = a.x + a.w;
    const y0 = a.y + a.h / 2;
    const x1 = b.x;
    const y1 = b.y + (b.h * (k + 1)) / (ports.length + 1);
    const ca = col[e.from] ?? 0;
    const cb = col[e.to] ?? 0;
    const gapR = (colX[ca] ?? 0) + (colW[ca] ?? 0) + gap / 2;
    const points: [number, number][] = [];
    if (cb - ca === 1) {
      points.push([x0, y0], [gapR, y0], [gapR, y1], [x1, y1]);
    } else {
      const gapL = (colX[cb] ?? 0) - gap / 2;
      const laneY = top - LAYOUT.lane * (1 + (laneOf.get(e.index) ?? 0));
      points.push([x0, y0], [gapR, y0], [gapR, laneY], [gapL, laneY], [gapL, y1], [x1, y1]);
    }
    wires.push({ edge: e.index, points, labelAt: labelAnchor(points) });
  }
  return { boxes, wires, size: [width, height] };
}

const boxPlan = (): Plan => ({ w: LAYOUT.boxW, h: LAYOUT.boxH, unit: null, count: 0 });

/** Columns by longest path; the target sits beside the prediction, below it; skip wires run above everything. */
export function layoutLayers(g: Graph, _a: Analysis): LayerLayout {
  const core = layoutCore(g, boxPlan, LAYOUT.layerGap);
  return { nodes: core.boxes, wires: core.wires, size: core.size };
}

/** Same columns as layer view; a Linear with at most 8 units on each side opens into circles and one edge per weight. */
export function layoutNeurons(g: Graph, a: Analysis): NeuronLayout {
  const { nodes, byId, edges } = prepare(g);
  const firstSource = (id: NodeId): NodeId | undefined => edges.find((e) => e.to === id)?.from;
  const expanded = nullRecord<boolean>();
  const expandable = (n: Node): boolean => {
    if (n.type !== 'linear') return false;
    const m = a.inWidth[n.id];
    return (
      m !== undefined && m >= 1 && m <= LAYOUT.maxUnits && n.out >= 1 && n.out <= LAYOUT.maxUnits
    );
  };
  for (const n of nodes) if (expandable(n)) expanded[n.id] = true;

  const unitPlan = (kind: NeuronUnitKind, count: number): Plan => ({
    w: LAYOUT.unitW,
    h: LAYOUT.unitLabelH + count * LAYOUT.unitPitch,
    unit: kind,
    count,
  });
  const small = (w: number | undefined): w is number =>
    w !== undefined && w >= 1 && w <= LAYOUT.maxUnits;
  const plan = (n: Node): Plan => {
    switch (n.type) {
      case 'input': {
        const w = a.outWidth[n.id];
        return small(w) ? unitPlan('in', w) : boxPlan();
      }
      case 'linear':
        return expanded[n.id] ? unitPlan('sum', n.out) : boxPlan();
      case 'activation': {
        const src = firstSource(n.id);
        const s = src === undefined ? undefined : byId[src];
        return s?.type === 'linear' && expanded[s.id] ? unitPlan('act', s.out) : boxPlan();
      }
      case 'output': {
        const w = a.outWidth[n.id];
        return small(w) ? unitPlan('out', w) : boxPlan();
      }
      case 'target':
        return unitPlan('target', 1);
      default:
        return boxPlan();
    }
  };

  const core = layoutCore(g, plan, LAYOUT.neuronGap);
  const units: NeuronLayout['units'] = [];
  const unitsOf = nullRecord<[number, number][]>();
  for (const n of nodes) {
    const p = plan(n);
    const b = core.boxes[n.id];
    if (!b || p.unit === null) continue;
    const pts: [number, number][] = [];
    for (let i = 0; i < p.count; i++) {
      const pt: [number, number] = [
        b.x + b.w / 2,
        b.y + LAYOUT.unitLabelH + (i + 0.5) * LAYOUT.unitPitch,
      ];
      pts.push(pt);
      units.push({ node: n.id, index: i, kind: p.unit, x: pt[0], y: pt[1] });
    }
    unitsOf[n.id] = pts;
  }

  const wEdges: NeuronLayout['edges'] = [];
  const collapsed: NodeId[] = [];
  for (const n of nodes) {
    if (n.type !== 'linear') continue;
    if (!expanded[n.id]) {
      collapsed.push(n.id);
      continue;
    }
    const m = a.inWidth[n.id] ?? 0;
    const src = firstSource(n.id);
    const upUnits = src === undefined ? undefined : unitsOf[src];
    const upBox = src === undefined ? undefined : core.boxes[src];
    const to = unitsOf[n.id] ?? [];
    for (let i = 0; i < m; i++) {
      let from: [number, number] | undefined = upUnits?.length === m ? upUnits[i] : undefined;
      if (!from && upBox) from = [upBox.x + upBox.w, upBox.y + ((i + 0.5) / m) * upBox.h];
      if (!from) continue;
      for (let j = 0; j < n.out; j++) {
        const t = to[j];
        if (t) wEdges.push({ node: n.id, i, j, from, to: t });
      }
    }
  }
  return {
    units,
    edges: wEdges,
    collapsed,
    boxes: core.boxes,
    wires: core.wires,
    size: core.size,
  };
}
