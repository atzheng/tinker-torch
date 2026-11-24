/**
 * Layout geometry (CONTRACTS.md section 9): pure, deterministic, nothing stored. Works on the whole
 * node set (not `analysis.order`), so invalid graphs lay out too: every node gets a position.
 */
import type { Graph, Node, NodeId } from './graph.js';
import { inputPortsOf, isVariadicPort, resolveEdge } from './graph.js';
import type { Analysis } from './analyse.js';

export type Box = { x: number; y: number; w: number; h: number };
export type LayerLayout = {
  nodes: Record<NodeId, Box & { col: number }>;
  /** `labelAt`: where the width label sits, the midpoint of the longest horizontal segment (first on ties). */
  wires: {
    edge: number;
    points: [number, number][];
    labelAt: [number, number];
  }[];
  size: [number, number];
};
export type NeuronUnitKind = 'in' | 'sum' | 'act' | 'out' | 'target' | 'cat';
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
  /** Widest a unit-column name may be: it reaches into the gaps either side, clear of the wires at mid-gap. */
  unitNameW: 132,
  /** Space between rows. */
  rowGap: 24,
  /** Distance between skip-wire lanes, and outer margin. */
  lane: 18,
  /** Offset between wires of different sources that would share a vertical run in one column gap. */
  laneStep: 5,
  /** Two such runs closer than this (end to end, or overlapping) are separated into lanes. */
  laneClear: 14,
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

type LayoutEdge = {
  index: number;
  from: NodeId;
  to: NodeId;
  port: string;
};

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
  const edges: LayoutEdge[] = [];
  g.edges.forEach((e, index) => {
    const r = resolveEdge(g.nodes, e);
    if (r.ok) edges.push({ index, from: r.from.node, to: r.to.node, port: r.to.port });
  });
  return { nodes, byId, edges };
}

/**
 * Column per node: longest path from the inputs. A Weights tap counts as one column after its Linear
 * (the tap is a parameter read, not a wire). The edges out of a Target do not count: the Target is
 * placed one column left of the first node it feeds, so it sits beside the prediction. A cycle is
 * broken at its first node in graph order; nodes with no wires at all (other than Input and Target)
 * go in the last column.
 */
function assignColumns(
  nodes: Node[],
  byId: Record<string, Node>,
  edges: { from: NodeId; to: NodeId }[],
): Record<NodeId, number> {
  const col = nullRecord<number>();
  const taps = nodes.flatMap((n) =>
    n.type === 'weights' && byId[n.of] !== undefined ? [{ from: n.of, to: n.id }] : [],
  );
  const counted = [
    ...edges.filter((e) => e.from !== e.to && byId[e.from]?.type !== 'target'),
    ...taps,
  ];
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
  for (const e of [...edges, ...taps]) {
    touched[e.from] = true;
    touched[e.to] = true;
  }
  for (const n of nodes) {
    if (!touched[n.id] && n.type !== 'input' && n.type !== 'target') col[n.id] = last;
  }
  const placed = new Set<NodeId>();
  for (const e of edges) {
    if (byId[e.from]?.type === 'target' && !placed.has(e.from)) {
      placed.add(e.from);
      col[e.from] = Math.max(0, (col[e.to] ?? 0) - 1);
    }
  }
  return col;
}

/** Row per node: follow the first-listed source, move down past taken slots; targets sit below the prediction. */
function assignRows(
  nodes: Node[],
  byId: Record<string, Node>,
  edges: LayoutEdge[],
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
        // Below whatever else feeds the first node the target feeds (the prediction).
        const toNode = edges.find((e) => e.from === n.id);
        const pred = toNode
          ? edges.find((e) => e.to === toNode.to && e.from !== n.id && row[e.from] !== undefined)
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

type Rect = { x: number; y: number; w: number; h: number };

const overlaps = (a: Rect, b: Rect, m: number): boolean =>
  a.x < b.x + b.w + m && b.x < a.x + a.w + m && a.y < b.y + b.h + m && b.y < a.y + a.h + m;

/** Whether the segment p-q (axis-parallel) passes through the box inflated by `m`. */
function segmentHits(p: [number, number], q: [number, number], b: Rect, m: number): boolean {
  const x0 = Math.min(p[0], q[0]);
  const x1 = Math.max(p[0], q[0]);
  const y0 = Math.min(p[1], q[1]);
  const y1 = Math.max(p[1], q[1]);
  return x0 < b.x + b.w + m && x1 > b.x - m && y0 < b.y + b.h + m && y1 > b.y - m;
}

/**
 * Route of a wire that touches a manually placed (or displaced) piece, where the column gaps and
 * lanes no longer apply. Forward (destination right of the source by more than two stubs): out,
 * one vertical, in, with the vertical at the first of a few candidate x positions (mid-gap first)
 * where no other piece is crossed. Otherwise (backwards, or too close): a stub right of the source,
 * a run above or below both pieces, a stub left of the destination. Not a full path search: when
 * nothing is free the mid-gap route is used.
 */
function manualRoute(a: Rect, b: Rect, y1: number, obstacles: Rect[]): [number, number][] {
  const x0 = a.x + a.w;
  const y0 = a.y + a.h / 2;
  const x1 = b.x;
  const stub = LAYOUT.lane;
  const free = (pts: [number, number][]) => {
    for (let k = 0; k + 1 < pts.length; k++) {
      const p = pts[k];
      const q = pts[k + 1];
      if (p && q && obstacles.some((o) => segmentHits(p, q, o, 2))) return false;
    }
    return true;
  };
  if (x1 - x0 >= 2 * stub) {
    const xs = [0.5, 0.25, 0.75, 0.1, 0.9].map((f) => x0 + (x1 - x0) * f);
    const routes = xs.map((xm): [number, number][] => [
      [x0, y0],
      [xm, y0],
      [xm, y1],
      [x1, y1],
    ]);
    return routes.find(free) ?? routes[0] ?? [];
  }
  const above = Math.min(a.y, b.y) - LAYOUT.lane;
  const below = Math.max(a.y + a.h, b.y + b.h) + LAYOUT.lane;
  const mid = (y0 + y1) / 2;
  const prefer = Math.abs(mid - above) <= Math.abs(mid - below) ? [above, below] : [below, above];
  const routes = prefer.map((yv): [number, number][] => [
    [x0, y0],
    [x0 + stub, y0],
    [x0 + stub, yv],
    [x1 - stub, yv],
    [x1 - stub, y1],
    [x1, y1],
  ]);
  return routes.find(free) ?? routes[0] ?? [];
}

function layoutCore(
  g: Graph,
  plan: (n: Node) => Plan,
  gap: number,
  fixed?: Record<NodeId, [number, number]>,
): Core {
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

  // Manual positions (layer view only): a placed piece goes exactly where it was put. A piece with
  // no position stays where the columns put it, unless that overlaps a placed piece or an earlier
  // displaced one: it then moves down in steps of one row until it is clear. Graph order, so the
  // result is deterministic. Wires touching a placed or displaced piece are routed by `manualRoute`.
  const manual = new Set<NodeId>();
  if (fixed) {
    for (const n of nodes) {
      const xy = fixed[n.id];
      const b = boxes[n.id];
      if (xy && b) {
        b.x = Math.max(LAYOUT.pad, xy[0]);
        b.y = Math.max(LAYOUT.pad, xy[1]);
        manual.add(n.id);
      }
    }
    if (manual.size > 0) {
      const settled: Rect[] = [...manual].flatMap((id) => (boxes[id] ? [boxes[id]] : []));
      for (const n of nodes) {
        const b = boxes[n.id];
        if (!b || manual.has(n.id)) continue;
        const start = b.y;
        let guard = 0;
        while (settled.some((o) => overlaps(b, o, LAYOUT.rowGap / 2)) && guard++ < 200) {
          b.y += b.h + LAYOUT.rowGap;
        }
        if (b.y !== start) manual.add(n.id);
        // Only displaced pieces are obstacles for later ones; untouched ones never overlap each other.
        if (b.y !== start) settled.push(b);
      }
    }
  }

  // Wires into a variadic port enter at different heights along that port's band, ordered by where
  // they come from (top source first, ties by edge index), so k wires neither overlap nor cross.
  const slotOf = new Map<number, [slot: number, count: number]>();
  const groups = new Map<string, typeof edges>();
  for (const e of edges) {
    const dest = byId[e.to];
    if (!dest || !isVariadicPort(dest.type, e.port) || !boxes[e.from] || !boxes[e.to]) continue;
    const key = `${e.to}\0${e.port}`;
    const list = groups.get(key);
    if (list) list.push(e);
    else groups.set(key, [e]);
  }
  for (const list of groups.values()) {
    const cy = (id: NodeId) => (boxes[id]?.y ?? 0) + (boxes[id]?.h ?? 0) / 2;
    list.sort((p, q) => cy(p.from) - cy(q.from) || p.index - q.index);
    list.forEach((e, j) => slotOf.set(e.index, [j, list.length]));
  }

  // An output with several wires: they share the first horizontal run and the elbow, so a label
  // at its midpoint would sit under every sibling's. Wires out of such an output
  // put their label on the last run instead, beside the destination, where each wire has its own row.
  const outCount = new Map<NodeId, number>();
  for (const e of edges) outCount.set(e.from, (outCount.get(e.from) ?? 0) + 1);

  const wires: LayerLayout['wires'] = [];
  const autoWires: { wire: number; from: NodeId; relabel: boolean }[] = [];
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
    // One slot per port; a variadic port's slot is a band (half a slot either side of its centre)
    // shared by its wires, and a lone wire sits at the centre as on any other port.
    const slot = slotOf.get(e.index);
    const y1 = variadicSlotY(b, k, ports.length, slot?.[0] ?? 0, slot?.[1] ?? 1);
    const ca = col[e.from] ?? 0;
    const cb = col[e.to] ?? 0;
    const gapR = (colX[ca] ?? 0) + (colW[ca] ?? 0) + gap / 2;
    let points: [number, number][] = [];
    if (manual.has(e.from) || manual.has(e.to)) {
      const obstacles = nodes.flatMap((n) =>
        n.id === e.from || n.id === e.to || !boxes[n.id] ? [] : [boxes[n.id] as Rect],
      );
      points = manualRoute(a, b, y1, obstacles);
    } else if (cb - ca === 1) {
      points.push([x0, y0], [gapR, y0], [gapR, y1], [x1, y1]);
    } else {
      const gapL = (colX[cb] ?? 0) - gap / 2;
      const laneY = top - LAYOUT.lane * (1 + (laneOf.get(e.index) ?? 0));
      points.push([x0, y0], [gapR, y0], [gapR, laneY], [gapL, laneY], [gapL, y1], [x1, y1]);
    }
    const isManual = manual.has(e.from) || manual.has(e.to);
    const fanned = (outCount.get(e.from) ?? 0) > 1;
    let labelAt: [number, number];
    if (fanned && (isManual || cb - ca === 1)) {
      const last = points[points.length - 1] ?? [x1, y1];
      const prev = points[points.length - 2] ?? last;
      labelAt =
        !isManual || prev[1] !== last[1] ? [(gapR + x1) / 2, y1] : [(prev[0] + last[0]) / 2, y1];
    } else labelAt = labelAnchor(points);
    wires.push({
      edge: e.index,
      points,
      labelAt,
    });
    if (!isManual) autoWires.push({ wire: wires.length - 1, from: e.from, relabel: !(fanned && cb - ca === 1) });
  }
  separateLanes(wires, autoWires);
  if (manual.size === 0) return { boxes, wires, size: [width, height] };
  // With manual positions the drawing is as large as what is actually there (pieces and routes).
  let right = 0;
  let bottom = 0;
  for (const b of Object.values(boxes)) {
    right = Math.max(right, b.x + b.w);
    bottom = Math.max(bottom, b.y + b.h);
  }
  for (const w of wires) {
    for (const [px, py] of w.points) {
      right = Math.max(right, px);
      bottom = Math.max(bottom, py);
    }
  }
  return { boxes, wires, size: [right + LAYOUT.pad, bottom + LAYOUT.pad] };
}

/**
 * Lanes in a column gap (tinker-fsv). The column routing puts every wire's vertical run on the one
 * x of its gap, so two wires of different sources whose runs overlap, or nearly touch, read as one
 * line. Per gap, the runs of one source (a fan-out tree) form one item; items that overlap or lie
 * within `LAYOUT.laneClear` of each other get different lanes, `laneStep` apart (the first stays
 * on the gap's x, later ones go right and left alternately). Items without such a neighbour do not
 * move, so layouts without overlaps are unchanged. Only automatically routed wires take part.
 */
function separateLanes(
  wires: LayerLayout['wires'],
  auto: { wire: number; from: NodeId; relabel: boolean }[],
): void {
  type Run = { w: number; i: number; lo: number; hi: number };
  type Item = { lo: number; hi: number; runs: Run[]; order: number; lane: number };
  const byGap = new Map<number, Map<NodeId, Run[]>>();
  const order = new Map<NodeId, number>();
  for (const { wire, from } of auto) {
    const pts = wires[wire]?.points ?? [];
    if (!order.has(from)) order.set(from, order.size);
    for (let i = 0; i + 1 < pts.length; i++) {
      const p = pts[i] as [number, number];
      const q = pts[i + 1] as [number, number];
      if (p[0] !== q[0] || Math.abs(p[1] - q[1]) < 1e-6) continue;
      let sources = byGap.get(p[0]);
      if (!sources) byGap.set(p[0], (sources = new Map()));
      const list = sources.get(from);
      const run = { w: wire, i, lo: Math.min(p[1], q[1]), hi: Math.max(p[1], q[1]) };
      if (list) list.push(run);
      else sources.set(from, [run]);
    }
  }
  const moved = new Set<number>();
  for (const sources of byGap.values()) {
    const items: Item[] = [];
    for (const [from, runs] of sources) {
      // Runs of one source that touch or overlap are one item (they share a trunk).
      const sorted = [...runs].sort((a, b) => a.lo - b.lo);
      let cur: Item | null = null;
      for (const r of sorted) {
        if (cur && r.lo <= cur.hi) {
          cur.hi = Math.max(cur.hi, r.hi);
          cur.runs.push(r);
        } else {
          cur = { lo: r.lo, hi: r.hi, runs: [r], order: order.get(from) ?? 0, lane: 0 };
          items.push(cur);
        }
      }
    }
    items.sort((a, b) => a.lo - b.lo || a.order - b.order);
    const clash = (a: Item, b: Item) =>
      a.lo < b.hi + LAYOUT.laneClear && b.lo < a.hi + LAYOUT.laneClear;
    items.forEach((it, n) => {
      const used = new Set<number>();
      for (let m = 0; m < n; m++) {
        const o = items[m] as Item;
        if (clash(it, o)) used.add(o.lane);
      }
      while (used.has(it.lane)) it.lane++;
    });
    for (const it of items) {
      if (it.lane === 0) continue;
      const step = Math.ceil(it.lane / 2) * LAYOUT.laneStep;
      const dx = it.lane % 2 === 1 ? step : -step;
      for (const r of it.runs) {
        const pts = wires[r.w]?.points;
        const p = pts?.[r.i];
        const q = pts?.[r.i + 1];
        if (!p || !q) continue;
        p[0] += dx;
        q[0] += dx;
        moved.add(r.w);
      }
    }
  }
  for (const { wire, relabel } of auto) {
    const w = wires[wire];
    if (w && relabel && moved.has(wire)) w.labelAt = labelAnchor(w.points);
  }
}

/**
 * Height at which wire `slot` (0-based, in wire order) of `count` wires enters input port `k` of
 * `ports` on `box`. One wire (the default) enters at the port's centre, as on any port.
 */
export function variadicSlotY(
  box: { y: number; h: number },
  k: number,
  ports: number,
  slot = 0,
  count = 1,
): number {
  return box.y + (box.h * (k + 0.5 + (slot + 1) / (count + 1))) / (ports + 1);
}

const boxPlan = (): Plan => ({ w: LAYOUT.boxW, h: LAYOUT.boxH, unit: null, count: 0 });

/** Columns by longest path; the target sits beside the prediction, below it; skip wires run above everything. */
export function layoutLayers(g: Graph, _a: Analysis): LayerLayout {
  const core = layoutCore(g, boxPlan, LAYOUT.layerGap, g.positions);
  return { nodes: core.boxes, wires: core.wires, size: core.size };
}

/** Category count a Lookup reads (the shape of its `in` wire), undefined while unknown. */
export function lookupCategories(a: Analysis, id: NodeId): number | undefined {
  const from = a.inputs[id]?.['in']?.[0]?.from;
  const s = from === undefined ? undefined : a.shape[from];
  return s?.kind === 'example' ? s.category?.count : undefined;
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
      case 'link': {
        const w = a.outWidth[n.id];
        return small(w) ? unitPlan('out', w) : boxPlan();
      }
      case 'target':
        return unitPlan('target', 1);
      case 'tree': {
        // one dot per leaf of the fitted tree, a column like the units of a layer
        const w = a.outWidth[n.id];
        return small(w) ? unitPlan('in', w) : boxPlan();
      }
      case 'onehot': {
        const w = a.outWidth[n.id];
        return small(w) ? unitPlan('cat', w) : boxPlan();
      }
      case 'lookup': {
        // One dial per category, only for a scalar value per category (width 1).
        const k = lookupCategories(a, n.id);
        return n.width === 1 && small(k) ? unitPlan('cat', k) : boxPlan();
      }
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
