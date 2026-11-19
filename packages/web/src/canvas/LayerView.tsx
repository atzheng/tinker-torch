import { useRef, useState, type ComponentType } from 'react';
import type { Analysis, Graph, HostParams, LayerLayout, Node } from '@tinker/core';
import { WEB_PIECES } from './drawings/index.js';
import type { LayerProps } from './drawings/types.js';
import { dnd, inputPortAt, outputPortAt, wireSource, type DndState } from './dnd.js';
import {
  edgeIndexOf,
  layerHitShapes,
  pinHitRect,
  pinKey,
  pinsOf,
  placePins,
  pointsToPath,
  type PinRole,
} from './pure.js';
import { PinMark } from './PinMark.js';
import { lockedIds } from '../state/locks.js';
import { addPrediction, inputPortsOf, resolveEdge, setLoss } from '@tinker/core';
import { pointerDownWith } from './tap.js';
import { Wire, WireLabel } from './Wire.js';
import type { Selection } from '../state/viewSlice.js';
import { useStore } from '../state/store.js';
import { roleOfNode, NODE_RADIUS, PORT_R } from './drawings/NodeShell.js';
import { LockBadge } from './drawings/LockBadge.js';

export type LayerViewProps = {
  graph: Graph;
  analysis: Analysis;
  layout: LayerLayout;
  /** Weights to draw; entries may be missing or stale, see `useWeights`. */
  weights: HostParams | undefined;
  /** Edge indices and node ids to draw in the error colour. */
  errorEdges: ReadonlySet<number>;
  errorNodes: ReadonlySet<string>;
  /** Pins (`pinKey`) to draw in the error colour. */
  errorPins?: ReadonlySet<string>;
  selection: Selection | null;
  select: (s: Selection | null) => void;
  /** Screen pixels per abstract unit. */
  scale: number;
  coarse: boolean;
  /** The palette or wire drag in progress, for drop highlights and port handles. */
  drag?: DndState | null;
  /** Wires and pieces a dragged or armed palette entry (or a wire drag) could land on. */
  candidates?: { edges: ReadonlySet<number>; nodes: ReadonlySet<string> } | null;
  /**
   * Moving a piece (CONTRACTS.md section 10, "Moving pieces"): a press on a piece body that goes
   * past the slop is a move, not a tap. `start` at the slop, `move` with the pointer's movement
   * since the press in client pixels, `end` on release (commit), `cancel` on pointercancel.
   * Left out, pieces only select.
   */
  mover?: {
    start: (id: string) => void;
    move: (id: string, dx: number, dy: number) => void;
    end: (id: string, dx: number, dy: number) => void;
    cancel: () => void;
  };
  /** The piece being moved, drawn lifted. */
  moving?: string | null;
};

/** Props for a piece's layer drawing, from the analysis and the weights. */
export function layerProps(
  node: Node,
  box: { w: number; h: number },
  analysis: Analysis,
  weights: HostParams | undefined,
  scale?: number,
): LayerProps<typeof node.type> {
  const inWidth = analysis.inWidth[node.id];
  const outWidth = analysis.outWidth[node.id];
  const base = { w: box.w, h: box.h, paramCount: analysis.params[node.id] ?? 0 };
  const props: Record<string, unknown> = { ...base, node };
  if (scale !== undefined) props['scale'] = scale;
  if (inWidth !== undefined) props['inWidth'] = inWidth;
  if (outWidth !== undefined) props['outWidth'] = outWidth;
  if (node.type === 'linear' && inWidth !== undefined) {
    const entry = weights?.[node.id]; // undefined for a Linear off the loss path
    if (entry?.W && entry.W.length === inWidth * node.out) {
      props['weights'] = { W: entry.W, rows: inWidth, cols: node.out };
    }
  }
  return props as LayerProps<typeof node.type>;
}

/** Padding of a region's shading around a piece, and its stroke width along a wire (abstract units). */
const REGION_PAD = 10;
const REGION_WIRE = 30;

/**
 * Subtle shading behind the pieces for the model and objective regions (`analysis.region`): a
 * padded rounded rect per piece plus a fat round-cap stroke along every wire inside one region, in
 * one group per region so overlaps do not darken. Not a layout zone: it follows whatever the pins
 * and wires say, and a piece in neither region has none.
 */
function Regions({
  graph,
  analysis,
  layout,
}: Pick<LayerViewProps, 'graph' | 'analysis' | 'layout'>) {
  return (
    <g className="tk-regions" data-testid="regions" pointerEvents="none">
      {(['model', 'objective'] as const).map((region) => {
        const ids = graph.nodes.filter(
          (n) => analysis.region[n.id] === region && layout.nodes[n.id],
        );
        if (ids.length === 0) return null;
        const wires = layout.wires.filter((w) => {
          const edge = graph.edges[w.edge];
          const r = edge ? resolveEdge(graph.nodes, edge) : undefined;
          return (
            r?.ok === true &&
            analysis.region[r.from.node] === region &&
            analysis.region[r.to.node] === region
          );
        });
        return (
          <g
            key={region}
            className="tk-region"
            data-region={region}
            data-testid={`region-${region}`}
          >
            {wires.map((w) => (
              <path key={w.edge} d={pointsToPath(w.points)} strokeWidth={REGION_WIRE} />
            ))}
            {ids.map((n) => {
              const b = layout.nodes[n.id];
              if (!b) return null;
              return (
                <rect
                  key={n.id}
                  x={b.x - REGION_PAD}
                  y={b.y - REGION_PAD}
                  width={b.w + 2 * REGION_PAD}
                  height={b.h + 2 * REGION_PAD}
                  rx={NODE_RADIUS + REGION_PAD}
                />
              );
            })}
          </g>
        );
      })}
    </g>
  );
}

/**
 * The layer view: wires first (so pieces sit on top and win presses), then the pieces. Each piece
 * is a `<g data-testid="node-<id>">` with a transparent hit rectangle at least 32 screen px
 * across; each wire has a fat transparent stroke. A press-and-release selects.
 */
export function LayerView(p: LayerViewProps) {
  const { graph, analysis, layout, scale } = p;
  const hitPx = p.coarse ? 32 : 20;
  const outputs = new Set(graph.nodes.map((n) => n.id));
  const shapes = layerHitShapes(layout, scale, analysis.edgeWidth, outputs);
  const target = p.drag?.target ?? null;
  const dropEdge = target?.kind === 'wire' ? target.edge : -1;
  const dropNode = target?.kind === 'node' ? target.id : target?.kind === 'port' ? target.node : '';
  const lockedList = useStore((s) => s.preset?.locked);
  const locked = new Set(lockedList ?? []);
  const cand = p.candidates;
  const [hover, setHover] = useState(-1);
  const hoverTimer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);
  // Moving from a wire onto its end handle leaves the wire first: wait a moment before hiding.
  const hoverWire = (edge: number) => {
    clearTimeout(hoverTimer.current);
    if (edge >= 0) setHover(edge);
    else hoverTimer.current = setTimeout(() => setHover(-1), 200);
  };
  // The wire a drop would replace: the one already feeding the target port.
  const replaced =
    target?.kind === 'port' && p.drag && p.drag.payload.kind !== 'palette'
      ? graph.edges.findIndex(
          (e, i) =>
            e[1] === target.port &&
            !(p.drag?.payload.kind === 'rewire' && p.drag.payload.edge === i),
        )
      : -1;
  const src = p.drag ? wireSource(p.drag.payload) : undefined;
  const lifted = p.drag?.payload.kind === 'rewire' ? p.drag.payload.edge : -1;
  const sel = p.selection;
  // Role pins: the real ones, and while a piece is selected the dashed "add" form of each role it
  // does not carry (tap to pin ŷ or L there). Hidden during any drag.
  const pinDrag = p.drag?.payload.kind === 'pin' ? p.drag.payload : null;
  const pinMarks: { role: PinRole; node: string; ghost: boolean }[] = pinsOf(graph).map((x) => ({
    ...x,
    ghost: false,
  }));
  if (sel?.kind === 'node' && !p.drag && layout.nodes[sel.id]) {
    const have = new Set(pinMarks.map((x) => pinKey(x.role, x.node)));
    for (const role of ['pred', 'loss'] as const) {
      if (!have.has(pinKey(role, sel.id))) pinMarks.push({ role, node: sel.id, ghost: true });
    }
  }
  const spots = placePins(layout.nodes, pinMarks, graph.positions !== undefined);
  const selectedEdge = sel?.kind === 'wire' ? edgeIndexOf(graph, sel.from, sel.to) : -1;
  return (
    <g data-testid="layer-view">
      <defs>
        <marker id="tk-arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">
          <path d="M0 0L8 4L0 8z" className="tk-arrow" />
        </marker>
        <marker
          id="tk-arrow-error"
          markerWidth="8"
          markerHeight="8"
          refX="7"
          refY="4"
          orient="auto"
        >
          <path d="M0 0L8 4L0 8z" className="tk-arrow-error" />
        </marker>
      </defs>
      <Regions graph={graph} analysis={analysis} layout={layout} />
      {analysis.paramWires.map((pw) => {
        // A Weights tap reads the parameters of its Linear: a dashed wire, not a data wire.
        const a = layout.nodes[pw.linear];
        const b = layout.nodes[pw.weights];
        if (!a || !b) return null;
        const sx = a.x + a.w / 2;
        const sy = a.y + a.h;
        const ex = b.x;
        const ey = b.y + b.h / 2;
        return (
          <path
            key={`pw-${pw.weights}`}
            data-testid={`param-wire-${pw.weights}`}
            className="tk-param-wire"
            d={`M${sx} ${sy}C${sx} ${sy + 36} ${ex - 40} ${ey} ${ex} ${ey}`}
            markerEnd="url(#tk-arrow)"
          />
        );
      })}
      {layout.wires.map((w) => {
        const edge = graph.edges[w.edge];
        if (!edge) return null;
        return (
          <Wire
            key={w.edge}
            edge={w.edge}
            points={w.points}
            labelAt={w.labelAt}
            width={analysis.edgeWidth[w.edge] ?? null}
            error={p.errorEdges.has(w.edge)}
            selected={selectedEdge === w.edge}
            drop={dropEdge === w.edge}
            valid={cand?.edges.has(w.edge)}
            lifted={lifted === w.edge}
            replace={replaced === w.edge}
            onHover={hoverWire}
            scale={scale}
            hitPx={hitPx}
            onSelect={() => p.select({ kind: 'wire', from: edge[0], to: edge[1] })}
          />
        );
      })}
      {graph.nodes.map((node) => {
        const box = layout.nodes[node.id];
        if (!box) return null;
        const selected = p.selection?.kind === 'node' && p.selection.id === node.id;
        const found = shapes.find((s) => s.kind === 'node' && s.id === node.id);
        const hit = found ?? { x: box.x, y: box.y, w: box.w, h: box.h };
        const Layer = WEB_PIECES[node.type].Layer as ComponentType<LayerProps<typeof node.type>>;
        return (
          <g
            key={node.id}
            transform={`translate(${box.x} ${box.y})`}
            data-testid={`node-${node.id}`}
            data-type={node.type}
            data-role={roleOfNode(node, graph)}
            data-locked={locked.has(node.id)}
            data-selected={selected}
            data-error={p.errorNodes.has(node.id)}
            data-drop={dropNode === node.id}
            data-valid={cand?.nodes.has(node.id) === true}
            className="tk-node"
            role="button"
            tabIndex={0}
            aria-label={`${node.type} ${node.id}`}
            aria-pressed={selected}
            data-moving={p.moving === node.id}
            onPointerDown={pointerDownWith({
              onTap: () => p.select({ kind: 'node', id: node.id }),
              ...(p.mover
                ? {
                    onStart: () => p.mover?.start(node.id),
                    onMove: (pt) => p.mover?.move(node.id, pt.dx, pt.dy),
                    onEnd: (pt) => p.mover?.end(node.id, pt.dx, pt.dy),
                    onCancel: () => p.mover?.cancel(),
                  }
                : {}),
            })}
            onKeyDown={(e) => {
              if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                p.select({ kind: 'node', id: node.id });
              }
            }}
          >
            <rect
              className="tk-hit"
              x={hit.x - box.x}
              y={hit.y - box.y}
              width={hit.w}
              height={hit.h}
            />
            <rect
              className="tk-ring"
              x={-4}
              y={-4}
              width={box.w + 8}
              height={box.h + 8}
              rx={NODE_RADIUS + 4}
            />
            <Layer {...layerProps(node, box, analysis, p.weights, scale)} />
            {locked.has(node.id) && <LockBadge x={box.w - 2} y={2} />}
          </g>
        );
      })}
      {graph.nodes.map((node) => {
        const at = outputPortAt(layout, node.id);
        const shape = shapes.find((s) => s.kind === 'port' && s.node === node.id);
        if (!at || !shape) return null;
        return (
          <g
            key={`out-${node.id}`}
            data-testid={`port-out-${node.id}`}
            className="tk-port"
            onPointerDown={pointerDownWith({
              onTap: () => p.select({ kind: 'node', id: node.id }),
              onStart: (pt) => dnd.begin({ kind: 'wire', from: node.id }, pt.x, pt.y),
              onMove: (pt) => dnd.move(pt.x, pt.y),
              onEnd: (pt) => dnd.end(pt.x, pt.y),
              onCancel: () => dnd.cancel(),
            })}
          >
            <rect className="tk-hit" x={shape.x} y={shape.y} width={shape.w} height={shape.h} />
            <circle className="tk-port-dot" cx={at[0]} cy={at[1]} r={PORT_R} />
            <title>Drag to another piece to connect</title>
          </g>
        );
      })}
      {pinMarks.map(({ role, node, ghost }) => {
        const box = layout.nodes[node];
        if (!box) return null;
        const lifted = pinDrag?.role === role && pinDrag.node === node;
        const key = pinKey(role, node);
        const activate = () => {
          if (!ghost) return p.select({ kind: 'pin', role, node });
          const g = useStore.getState().graph;
          const r =
            role === 'pred'
              ? addPrediction(g, node, { locked: lockedIds() })
              : setLoss(g, node, { locked: lockedIds() });
          if (useStore.getState().apply(r).ok) p.select({ kind: 'pin', role, node });
        };
        return (
          <PinMark
            key={`${ghost ? 'add' : 'pin'}-${key}`}
            role={role}
            node={node}
            box={box}
            at={spots.get(key)}
            hit={pinHitRect(box, role, scale, spots.get(key))}
            ghost={ghost}
            selected={!ghost && sel?.kind === 'pin' && sel.role === role && sel.node === node}
            error={!ghost && p.errorPins?.has(key) === true}
            lifted={lifted}
            title={
              ghost
                ? role === 'pred'
                  ? 'Add a prediction (\u0177) pin here'
                  : 'Move the loss (L) pin here'
                : role === 'pred'
                  ? 'Prediction pin: drag to another piece to move it'
                  : 'Loss pin: drag to another piece to move it'
            }
            onActivate={activate}
            onPointerDown={pointerDownWith(
              ghost
                ? { onTap: activate }
                : {
                    onTap: () => p.select({ kind: 'pin', role, node }),
                    onStart: (pt) => dnd.begin({ kind: 'pin', role, node }, pt.x, pt.y),
                    onMove: (pt) => dnd.move(pt.x, pt.y),
                    onEnd: (pt) => dnd.end(pt.x, pt.y),
                    onCancel: () => dnd.cancel(),
                  },
            )}
          />
        );
      })}
      {src !== undefined &&
        graph.nodes.map((node) => {
          const ports = inputPortsOf(node.type);
          if (node.id === src) return null;
          return ports.map((port, k) => {
            const at = inputPortAt(layout, node.id, k, ports.length);
            if (!at) return null;
            const on =
              target?.kind === 'port' &&
              target.node === node.id &&
              target.port === (ports.length === 1 ? node.id : `${node.id}.${port}`);
            return (
              <circle
                key={`in-${node.id}-${port}`}
                data-testid={`port-in-${node.id}-${port}`}
                data-drop={on}
                className="tk-port-in"
                cx={at[0]}
                cy={at[1]}
                r={on ? 8 : PORT_R + 1}
              />
            );
          });
        })}
      {layout.wires.map((w) => {
        const edge = graph.edges[w.edge];
        const hit = shapes.find((s) => s.kind === 'wire' && s.edge === w.edge);
        if (!edge || !hit) return null;
        return (
          <WireLabel
            key={w.edge}
            edge={w.edge}
            points={w.points}
            labelAt={w.labelAt}
            width={analysis.edgeWidth[w.edge] ?? null}
            error={p.errorEdges.has(w.edge)}
            selected={selectedEdge === w.edge}
            drop={dropEdge === w.edge}
            valid={cand?.edges.has(w.edge)}
            lifted={lifted === w.edge}
            replace={replaced === w.edge}
            onHover={hoverWire}
            scale={scale}
            hitPx={hitPx}
            hit={hit}
            onSelect={() => p.select({ kind: 'wire', from: edge[0], to: edge[1] })}
          />
        );
      })}
      {layout.wires.map((w) => {
        // The destination end of a wire is a handle, shown only while the wire is hovered (mouse)
        // or selected (touch: tap the wire, then drag its end). It sits just outside the
        // destination piece, so the piece body always wins a press (CONTRACTS section 10).
        // The handle being dragged stays mounted (it holds the pointer capture).
        if (lifted !== w.edge && (p.drag || (hover !== w.edge && selectedEdge !== w.edge))) {
          return null;
        }
        const edge = graph.edges[w.edge];
        const end = w.points[w.points.length - 1];
        const r = edge ? resolveEdge(graph.nodes, edge) : undefined;
        if (!edge || !end || !r?.ok) return null;
        // A selected (touch) handle is always at least 32 px; a hover (mouse) one is hitPx.
        const hw = (selectedEdge === w.edge ? Math.max(hitPx, 32) : hitPx) / scale;
        return (
          <g
            key={`end-${w.edge}`}
            data-testid={`wire-end-${w.edge}`}
            data-selected={selectedEdge === w.edge}
            className="tk-wire-end"
            onPointerEnter={(e) => e.pointerType === 'mouse' && hoverWire(w.edge)}
            onPointerLeave={(e) => e.pointerType === 'mouse' && hoverWire(-1)}
            onPointerDown={pointerDownWith({
              onTap: () => p.select({ kind: 'wire', from: edge[0], to: edge[1] }),
              onStart: (pt) =>
                dnd.begin(
                  { kind: 'rewire', edge: w.edge, from: r.from.node, to: edge[1] },
                  pt.x,
                  pt.y,
                ),
              onMove: (pt) => dnd.move(pt.x, pt.y),
              onEnd: (pt) => dnd.end(pt.x, pt.y),
              onCancel: () => dnd.cancel(),
            })}
          >
            <rect className="tk-hit" x={end[0] - hw} y={end[1] - hw / 2} width={hw} height={hw} />
            <circle className="tk-wire-end-dot" cx={end[0] - 6} cy={end[1]} r={PORT_R + 2} />
            <title>Drag to another input to rewire</title>
          </g>
        );
      })}
    </g>
  );
}
