import type { ComponentType } from 'react';
import type { Analysis, Graph, HostParams, LayerLayout, Node } from '@tinker/core';
import { WEB_PIECES } from './drawings/index.js';
import type { LayerProps } from './drawings/types.js';
import { dnd, hasOutputPort, inputPortAt, outputPortAt, type DndState } from './dnd.js';
import { edgeIndexOf, layerHitShapes } from './pure.js';
import { inputPortsOf } from '@tinker/core';
import { pointerDownWith } from './tap.js';
import { Wire, WireLabel } from './Wire.js';
import type { Selection } from '../state/viewSlice.js';
import { useStore } from '../state/store.js';
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
  selection: Selection | null;
  select: (s: Selection | null) => void;
  /** Screen pixels per abstract unit. */
  scale: number;
  coarse: boolean;
  /** The palette or wire drag in progress, for drop highlights and port handles. */
  drag?: DndState | null;
};

/** Props for a piece's layer drawing, from the analysis and the weights. */
export function layerProps(
  node: Node,
  box: { w: number; h: number },
  analysis: Analysis,
  weights: HostParams | undefined,
): LayerProps<typeof node.type> {
  const inWidth = analysis.inWidth[node.id];
  const outWidth = analysis.outWidth[node.id];
  const base = { w: box.w, h: box.h, paramCount: analysis.params[node.id] ?? 0 };
  const props: Record<string, unknown> = { ...base, node };
  if (inWidth !== undefined) props['inWidth'] = inWidth;
  if (outWidth !== undefined) props['outWidth'] = outWidth;
  if (node.type === 'linear' && inWidth !== undefined) {
    const entry = weights?.[node.id]; // undefined for a Linear off the loss path
    if (entry && entry.W.length === inWidth * node.out) {
      props['weights'] = { W: entry.W, rows: inWidth, cols: node.out };
    }
  }
  return props as LayerProps<typeof node.type>;
}

/**
 * The layer view: wires first (so pieces sit on top and win presses), then the pieces. Each piece
 * is a `<g data-testid="node-<id>">` with a transparent hit rectangle at least 32 screen px
 * across; each wire has a fat transparent stroke. A press-and-release selects.
 */
export function LayerView(p: LayerViewProps) {
  const { graph, analysis, layout, scale } = p;
  const hitPx = p.coarse ? 32 : 20;
  const outputs = new Set(graph.nodes.filter((n) => hasOutputPort(n.type)).map((n) => n.id));
  const shapes = layerHitShapes(layout, scale, analysis.edgeWidth, outputs);
  const target = p.drag?.target ?? null;
  const dropEdge = target?.kind === 'wire' ? target.edge : -1;
  const dropNode = target?.kind === 'node' ? target.id : target?.kind === 'port' ? target.node : '';
  const lockedList = useStore((s) => s.preset?.locked);
  const locked = new Set(lockedList ?? []);
  const sel = p.selection;
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
            data-selected={selected}
            data-error={p.errorNodes.has(node.id)}
            data-drop={dropNode === node.id}
            className="tk-node"
            role="button"
            tabIndex={0}
            aria-label={`${node.type} ${node.id}`}
            aria-pressed={selected}
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
            <rect
              className="tk-hit"
              x={hit.x - box.x}
              y={hit.y - box.y}
              width={hit.w}
              height={hit.h}
            />
            <rect className="tk-ring" x={-4} y={-4} width={box.w + 8} height={box.h + 8} rx={12} />
            <Layer {...layerProps(node, box, analysis, p.weights)} />
            {locked.has(node.id) && <LockBadge x={box.w - 8} y={8} />}
          </g>
        );
      })}
      {graph.nodes.map((node) => {
        const at = hasOutputPort(node.type) ? outputPortAt(layout, node.id) : undefined;
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
            <circle className="tk-port-dot" cx={shape.x + shape.w / 2} cy={at[1]} r={5} />
          </g>
        );
      })}
      {p.drag?.payload.kind === 'wire' &&
        graph.nodes.map((node) => {
          const ports = inputPortsOf(node.type);
          if (node.id === (p.drag?.payload.kind === 'wire' ? p.drag.payload.from : '')) return null;
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
                r={on ? 8 : 5.5}
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
            scale={scale}
            hitPx={hitPx}
            hit={hit}
            onSelect={() => p.select({ kind: 'wire', from: edge[0], to: edge[1] })}
          />
        );
      })}
    </g>
  );
}
