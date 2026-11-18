import type { ComponentType } from 'react';
import type { Analysis, Graph, HostParams, LayerLayout, Node } from '@tinker/core';
import { WEB_PIECES } from './drawings/index.js';
import type { LayerProps } from './drawings/types.js';
import { edgeIndexOf, hitPad } from './pure.js';
import { pointerDownWith } from './tap.js';
import { Wire } from './Wire.js';
import type { Selection } from '../state/viewSlice.js';

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
};

/** Props for a piece's layer drawing, from the analysis and the weights. */
function layerProps(
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
        const padX = hitPad(box.w, scale);
        const padY = hitPad(box.h, scale);
        const Layer = WEB_PIECES[node.type].Layer as ComponentType<LayerProps<typeof node.type>>;
        return (
          <g
            key={node.id}
            transform={`translate(${box.x} ${box.y})`}
            data-testid={`node-${node.id}`}
            data-type={node.type}
            data-selected={selected}
            data-error={p.errorNodes.has(node.id)}
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
              x={-padX}
              y={-padY}
              width={box.w + 2 * padX}
              height={box.h + 2 * padY}
            />
            <rect className="tk-ring" x={-4} y={-4} width={box.w + 8} height={box.h + 8} rx={12} />
            <Layer {...layerProps(node, box, analysis, p.weights)} />
          </g>
        );
      })}
    </g>
  );
}
