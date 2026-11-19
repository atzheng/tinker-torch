import { memo, useLayoutEffect, useMemo, useRef, useState, type ComponentType } from 'react';
import {
  LAYOUT,
  resolveEdge,
  setLinearOut,
  type Analysis,
  type Graph,
  type HostParams,
  type NeuronLayout,
  type Node,
} from '@tinker/core';
import { ACTIVATION_SYMBOL } from './drawings/activation.js';
import { WEB_PIECES } from './drawings/index.js';
import { NODE_RADIUS, NodeShell, roleOf } from './drawings/NodeShell.js';
import { LockBadge } from './drawings/LockBadge.js';
import type { LayerProps } from './drawings/types.js';
import { layerProps } from './LayerView.js';
import {
  edgeStyle,
  neuronHitShapes,
  nearestEdge,
  trimSegment,
  UNIT_R,
  weightAt,
  type NeuronEdge,
} from './neuronPure.js';
import type { DragHandlers } from '../gestures/drag.js';
import { pointerDownWith } from './tap.js';
import { weightDragHandlers } from './weightEdit.js';
import { hostShape } from '../state/hostShape.js';
import { lockedIds } from '../state/locks.js';
import { useStore } from '../state/store.js';
import type { Selection } from '../state/viewSlice.js';

export type NeuronViewProps = {
  graph: Graph;
  analysis: Analysis;
  layout: NeuronLayout;
  /** Step-0 weights, drawn until the train slice has a `host`. Edge colours follow `host` themselves. */
  fallback: HostParams | undefined;
  /** Edge indices and node ids to draw in the error colour. */
  errorEdges: ReadonlySet<number>;
  errorNodes: ReadonlySet<string>;
  selection: Selection | null;
  select: (s: Selection | null) => void;
  /** Screen pixels per abstract unit. */
  scale: number;
  coarse: boolean;
  /** Client coordinates to abstract units (the canvas knows its own transform). */
  toAbstract: (clientX: number, clientY: number) => { x: number; y: number };
};

type Dims = Record<string, { inW: number; out: number } | undefined>;

/**
 * One line per weight. Mounted once per layout: colour and thickness are written straight onto
 * the `<line>` elements whenever the train slice publishes a new `host` (at most once per frame),
 * so a frame costs a few hundred attribute writes and no React render.
 */
const EdgeLayer = memo(function EdgeLayer(p: {
  edges: readonly NeuronEdge[];
  dims: Dims;
  fallback: HostParams | undefined;
  hitPx: number;
  scale: number;
  onPress: (k: number, e: PointerEvent | React.PointerEvent) => void;
  /** Drag handlers that set the weight of the edge nearest to the press (own edge `k` as fallback). */
  dragOf: (k: number) => DragHandlers;
  onHover: (k: number | null, e: React.PointerEvent) => void;
}) {
  const lines = useRef<(SVGLineElement | null)[]>([]);
  useLayoutEffect(() => {
    const paint = (host: HostParams | undefined) => {
      p.edges.forEach((e, k) => {
        const el = lines.current[k];
        const d = p.dims[e.node];
        if (!el || !d) return;
        const w = weightAt(
          host,
          e.node,
          e.i,
          e.j,
          d.inW,
          d.out,
          hostShape(host, e.node, useStore.getState().dataInfo),
        );
        if (w === undefined) {
          el.setAttribute('data-sign', 'none');
          el.setAttribute('stroke-width', '0.8');
          return;
        }
        const st = edgeStyle(w);
        el.setAttribute('data-sign', st.sign);
        el.setAttribute('stroke-width', String(st.width));
      });
    };
    paint(useStore.getState().host ?? p.fallback);
    return useStore.subscribe((s, prev) => {
      if (s.host !== prev.host) paint(s.host ?? p.fallback);
    });
  }, [p.edges, p.dims, p.fallback]);
  return (
    <g data-testid="edge-layer">
      {p.edges.map((e, k) => {
        const [x1, y1, x2, y2] = trimSegment(e.from, e.to, UNIT_R, UNIT_R);
        return (
          <g
            key={`${e.node}:${e.i}:${e.j}`}
            data-testid={`edge-${e.node}-${e.i}-${e.j}`}
            className="tk-edge"
            onPointerDown={pointerDownWith({
              ...p.dragOf(k),
              onTap: (pt) => p.onPress(k, pt.event),
            })}
            onPointerMove={(ev) => p.onHover(k, ev)}
            onPointerLeave={(ev) => p.onHover(null, ev)}
          >
            <line
              className="tk-hit"
              x1={x1}
              y1={y1}
              x2={x2}
              y2={y2}
              strokeWidth={p.hitPx / p.scale}
            />
            <line
              ref={(el) => {
                lines.current[k] = el;
              }}
              className="tk-edge-line"
              data-sign="none"
              x1={x1}
              y1={y1}
              x2={x2}
              y2={y2}
            />
          </g>
        );
      })}
    </g>
  );
});

function UnitCircle(p: { kind: NeuronLayout['units'][number]['kind']; node: Node }) {
  if (p.kind === 'target') return <circle className="tk-dot" r={UNIT_R * 0.55} />;
  return (
    <>
      <circle className="tk-shape" r={UNIT_R} />
      {p.kind === 'sum' && (
        <text className="tk-text" textAnchor="middle" dominantBaseline="central">
          +
        </text>
      )}
      {p.kind === 'act' && p.node.type === 'activation' && (
        <text
          className="tk-text tk-small"
          style={{ fontSize: ACTIVATION_SYMBOL[p.node.fn].length > 2 ? 7.5 : 12 }}
          textAnchor="middle"
          dominantBaseline="central"
        >
          {ACTIVATION_SYMBOL[p.node.fn]}
        </text>
      )}
      {p.kind === 'out' && <circle className="tk-dot" r={UNIT_R * 0.35} />}
    </>
  );
}

/**
 * The neuron view: the same graph drawn as unit circles with one edge per weight (SPEC "Grain").
 * Expanded Linears show input units, weight edges (blue when positive, orange when negative,
 * thicker when larger), plus circles and activation circles; wider ones stay boxes labelled with
 * their edge count. Nothing here is stored: it is drawn from `layoutNeurons` and the weights.
 *
 * Presses: a unit selects `{ kind: 'unit' }`, an edge `{ kind: 'edge' }` (the edge whose line is
 * nearest to the press wins, see `nearestEdge`), anything else on a piece selects the piece.
 * Mouse hover over an edge shows its value; on touch the inspector shows it.
 */
export function NeuronView(p: NeuronViewProps) {
  const { graph, analysis, layout, scale } = p;
  const hitPx = p.coarse ? 32 : 20;
  const lockedList = useStore((s) => s.preset?.locked);
  const lockedSet = new Set(lockedList ?? []);
  const [hover, setHover] = useState<{ k: number; x: number; y: number } | null>(null);

  const dims = useMemo(() => {
    const d: Dims = Object.create(null) as Dims;
    for (const n of graph.nodes) {
      const inW = analysis.inWidth[n.id];
      if (n.type === 'linear' && inW !== undefined && d[n.id] === undefined) {
        d[n.id] = { inW, out: n.out };
      }
    }
    return d;
  }, [graph, analysis]);

  const unitsByNode = useMemo(() => {
    const m = new Map<string, NeuronLayout['units']>();
    for (const u of layout.units) {
      const list = m.get(u.node);
      if (list) list.push(u);
      else m.set(u.node, [u]);
    }
    return m;
  }, [layout]);

  const shapes = useMemo(
    () => neuronHitShapes(layout, scale, LAYOUT.unitPitch, LAYOUT.unitW),
    [layout, scale],
  );

  // Plain links: wires that are not weight edges. A wire into an expanded Linear is its edges.
  const links = useMemo(() => {
    const out: { key: string; d: string; error: boolean }[] = [];
    for (const w of layout.wires) {
      const e = graph.edges[w.edge];
      if (!e) continue;
      const r = resolveEdge(graph.nodes, e);
      if ('error' in r) continue;
      const src = unitsByNode.get(r.from.node);
      const dst = unitsByNode.get(r.to.node);
      const error = p.errorEdges.has(w.edge);
      if (dst?.[0]?.kind === 'sum') continue;
      if (src && dst && src.length === dst.length) {
        src.forEach((a, k) => {
          const b = dst[k];
          if (!b) return;
          const [x1, y1, x2, y2] = trimSegment([a.x, a.y], [b.x, b.y], UNIT_R, UNIT_R);
          out.push({ key: `${w.edge}-${k}`, d: `M${x1} ${y1} L${x2} ${y2}`, error });
        });
      } else {
        const pts = w.points.map(([x, y], i) => `${i === 0 ? 'M' : 'L'}${x} ${y}`).join(' ');
        out.push({ key: `${w.edge}`, d: pts, error });
      }
    }
    return out;
  }, [layout, graph, unitsByNode, p.errorEdges]);

  const pressEdge = (own: number, ev: PointerEvent | React.PointerEvent) => {
    const at = p.toAbstract(ev.clientX, ev.clientY);
    const k = nearestEdge(layout.edges, at.x, at.y, hitPx / 2 / scale) ?? own;
    const e = layout.edges[k];
    if (e) p.select({ kind: 'edge', node: e.node, i: e.i, j: e.j });
  };
  const hoverEdge = (own: number | null, ev: React.PointerEvent) => {
    if (ev.pointerType !== 'mouse') return;
    if (own === null) return setHover(null);
    const at = p.toAbstract(ev.clientX, ev.clientY);
    const k = nearestEdge(layout.edges, at.x, at.y, hitPx / 2 / scale) ?? own;
    setHover((h) => (h && h.k === k && h.x === at.x && h.y === at.y ? h : { k, x: at.x, y: at.y }));
  };

  const dragOfEdge = (own: number): DragHandlers =>
    weightDragHandlers(
      (pt) => {
        const at = p.toAbstract(pt.startX, pt.startY);
        const k = nearestEdge(layout.edges, at.x, at.y, hitPx / 2 / scale) ?? own;
        const e = layout.edges[k];
        return e ? { kind: 'W', node: e.node, i: e.i, j: e.j } : null;
      },
      (t) => t.kind === 'W' && p.select({ kind: 'edge', node: t.node, i: t.i, j: t.j }),
    );

  const sel = p.selection;
  const selEdge =
    sel?.kind === 'edge'
      ? layout.edges.find((e) => e.node === sel.node && e.i === sel.i && e.j === sel.j)
      : undefined;

  return (
    <g data-testid="neuron-view">
      <g className="tk-links">
        {links.map((l) => (
          <path key={l.key} d={l.d} className="tk-link" data-error={l.error} />
        ))}
      </g>
      <EdgeLayer
        edges={layout.edges}
        dims={dims}
        fallback={p.fallback}
        hitPx={hitPx}
        scale={scale}
        onPress={pressEdge}
        dragOf={dragOfEdge}
        onHover={hoverEdge}
      />
      {selEdge && (
        <line
          className="tk-edge-halo"
          data-testid="edge-selected"
          {...(() => {
            const [x1, y1, x2, y2] = trimSegment(selEdge.from, selEdge.to, UNIT_R, UNIT_R);
            return { x1, y1, x2, y2 };
          })()}
        />
      )}
      {graph.nodes.map((node) => {
        const box = layout.boxes[node.id];
        if (!box) return null;
        const units = unitsByNode.get(node.id);
        const nodeSelected = sel?.kind === 'node' && sel.id === node.id;
        const hit = shapes.find((s) => s.kind === 'node' && s.id === node.id);
        return (
          <g
            key={node.id}
            transform={`translate(${box.x} ${box.y})`}
            data-testid={`node-${node.id}`}
            data-type={node.type}
            data-role={roleOf(node.type)}
            data-locked={lockedSet.has(node.id)}
            data-selected={nodeSelected}
            data-error={p.errorNodes.has(node.id)}
            className="tk-node"
            role="button"
            tabIndex={0}
            aria-label={`${node.type} ${node.id}`}
            aria-pressed={nodeSelected}
            onPointerDown={pointerDownWith({
              onTap: () => p.select({ kind: 'node', id: node.id }),
            })}
            onKeyDown={(e) => {
              if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                p.select({ kind: 'node', id: node.id });
              }
            }}
          >
            {hit && (
              <rect
                className="tk-hit"
                x={hit.x - box.x}
                y={hit.y - box.y}
                width={hit.w}
                height={hit.h}
              />
            )}
            <rect
              className="tk-ring"
              x={-4}
              y={-4}
              width={box.w + 8}
              height={box.h + 8}
              rx={NODE_RADIUS + 4}
            />
            {units ? null : <BoxDrawing node={node} box={box} p={p} />}
            {lockedSet.has(node.id) && <LockBadge x={box.w - 2} y={2} />}
          </g>
        );
      })}
      {layout.units.map((u) => {
        const node = graph.nodes.find((n) => n.id === u.node);
        if (!node) return null;
        const selected = sel?.kind === 'unit' && sel.node === u.node && sel.index === u.index;
        const hit = shapes.find(
          (s) => s.kind === 'unit' && s.node === u.node && s.index === u.index,
        );
        return (
          <g
            key={`${u.node}:${u.index}`}
            transform={`translate(${u.x} ${u.y})`}
            data-testid={`unit-${u.node}-${u.index}`}
            data-kind={u.kind}
            data-role={
              u.kind === 'in' || u.kind === 'target'
                ? 'input'
                : u.kind === 'out'
                  ? 'pred'
                  : undefined
            }
            data-locked={lockedSet.has(u.node)}
            data-selected={selected}
            data-error={p.errorNodes.has(u.node)}
            className="tk-node tk-unit"
            onPointerDown={pointerDownWith({
              // A drag on a plus circle sets that unit's bias; other units only select.
              // Any other unit cell hands a drag to the edge nearest to where it began.
              ...(u.kind === 'sum'
                ? weightDragHandlers(
                    () => ({ kind: 'b', node: u.node, j: u.index }),
                    () => p.select({ kind: 'unit', node: u.node, index: u.index }),
                  )
                : dragOfEdge(-1)),
              onTap: () => p.select({ kind: 'unit', node: u.node, index: u.index }),
            })}
          >
            {hit && (
              <rect
                className="tk-hit"
                x={hit.x - u.x}
                y={hit.y - u.y}
                width={hit.w}
                height={hit.h}
              />
            )}
            <circle className="tk-ring" r={UNIT_R + 4} />
            <UnitCircle kind={u.kind} node={node} />
          </g>
        );
      })}
      {graph.nodes.map((node) => {
        const box = layout.boxes[node.id];
        if (node.type !== 'linear' || !box) return null;
        return (
          <UnitButtons
            key={node.id}
            node={node}
            box={box}
            scale={scale}
            y={layout.size[1] - LAYOUT.pad + 4}
          />
        );
      })}
      {hover && (
        <EdgeTip layout={layout} hover={hover} dims={dims} fallback={p.fallback} scale={scale} />
      )}
    </g>
  );
}

/** A piece that stays a box: collapsed Linears say how many edges they hide. */
function BoxDrawing({
  node,
  box,
  p,
}: {
  node: Node;
  box: { w: number; h: number };
  p: NeuronViewProps;
}) {
  if (node.type === 'linear') {
    const inW = p.analysis.inWidth[node.id];
    return (
      <NodeShell
        piece="linear"
        w={box.w}
        h={box.h}
        name="Linear"
        secondary={[
          `${inW ?? '?'} to ${node.out}`,
          inW === undefined ? 'no input' : `${inW * node.out} edges`,
        ]}
        secondaryTestId={`edges-${node.id}`}
        inPorts={1}
        outPort
      />
    );
  }
  return <OtherBox node={node} box={box} p={p} />;
}

/** Any other piece that is not drawn as units: its layer drawing. */
function OtherBox({
  node,
  box,
  p,
}: {
  node: Node;
  box: { w: number; h: number };
  p: NeuronViewProps;
}) {
  const Layer = WEB_PIECES[node.type].Layer as ComponentType<LayerProps<typeof node.type>>;
  return <Layer {...layerProps(node, box, p.analysis, undefined)} />;
}

/** The value of the hovered edge, drawn beside the pointer in screen-sized text. */
function EdgeTip(p: {
  layout: NeuronLayout;
  hover: { k: number; x: number; y: number };
  dims: Dims;
  fallback: HostParams | undefined;
  scale: number;
}) {
  const host = useStore((s) => s.host) ?? p.fallback;
  const e = p.layout.edges[p.hover.k];
  const d = e && p.dims[e.node];
  if (!e || !d) return null;
  const w = weightAt(
    host,
    e.node,
    e.i,
    e.j,
    d.inW,
    d.out,
    hostShape(host, e.node, useStore.getState().dataInfo),
  );
  if (w === undefined) return null;
  const text = w.toFixed(3);
  const fs = 13 / p.scale;
  const tw = text.length * 0.62 * fs + fs;
  return (
    <g
      data-testid="edge-tip"
      transform={`translate(${p.hover.x + 10 / p.scale} ${p.hover.y - 24 / p.scale})`}
      pointerEvents="none"
    >
      <rect className="tk-label-bg" width={tw} height={1.7 * fs} rx={0.4 * fs} />
      <text
        className="tk-label-text"
        x={tw / 2}
        y={0.85 * fs}
        fontSize={fs}
        textAnchor="middle"
        dominantBaseline="central"
      >
        {text}
      </text>
    </g>
  );
}

export const UNIT_BTN_W = 28;
export const UNIT_BTN_H = 24;
/** Extra abstract height the canvas reserves under the neuron drawing for the buttons. */
export const UNIT_BTN_ROOM = 16;

/**
 * The minus and plus buttons under a Linear: one more or one fewer unit (`setLinearOut`, one undo
 * step, 1 to 64). They sit under collapsed Linears too, so a column that grew past 8 can shrink
 * back. A locked piece refuses through the toast. The hit area is at least 32 screen px.
 */
function UnitButtons(p: {
  node: Extract<Node, { type: 'linear' }>;
  box: { x: number; y: number; w: number; h: number };
  scale: number;
  /** Top of the button row: under every unit and edge, which never reach below the last box. */
  y: number;
}) {
  const { node, box, scale } = p;
  const change = (delta: number) => {
    const s = useStore.getState();
    s.apply(setLinearOut(s.graph, node.id, node.out + delta, { locked: lockedIds() }));
  };
  // The two hit areas share the column centre and never overlap, so at small scales the buttons
  // move apart rather than the hit areas shrinking below 32 px.
  const hit = Math.max(UNIT_BTN_W + 6, 32 / scale);
  const hitH = Math.max(UNIT_BTN_H, 32 / scale);
  const btn = (sign: -1 | 1) => {
    const disabled = sign < 0 ? node.out <= 1 : node.out >= 64;
    const x = sign * (hit / 2) - UNIT_BTN_W / 2;
    const label = sign < 0 ? 'Remove a unit' : 'Add a unit';
    return (
      <g
        data-testid={`unit-${sign < 0 ? 'minus' : 'plus'}-${node.id}`}
        className="tk-unitbtn"
        data-disabled={disabled}
        role="button"
        tabIndex={0}
        aria-label={label}
        aria-disabled={disabled}
        onPointerDown={pointerDownWith({ onTap: () => !disabled && change(sign) })}
        onKeyDown={(e) => {
          if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            if (!disabled) change(sign);
          }
        }}
      >
        <rect
          className="tk-hit"
          x={x + UNIT_BTN_W / 2 - hit / 2}
          y={UNIT_BTN_H / 2 - hitH / 2}
          width={hit}
          height={hitH}
        />
        <rect className="tk-shape" x={x} y={0} width={UNIT_BTN_W} height={UNIT_BTN_H} rx={6} />
        <text
          className="tk-text"
          x={x + UNIT_BTN_W / 2}
          y={UNIT_BTN_H / 2}
          textAnchor="middle"
          dominantBaseline="central"
        >
          {sign < 0 ? '−' : '+'}
        </text>
      </g>
    );
  };
  return (
    <g transform={`translate(${box.x + box.w / 2} ${p.y})`} data-testid={`unitbtns-${node.id}`}>
      {btn(-1)}
      {btn(1)}
    </g>
  );
}
