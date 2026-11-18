import { pointerDownWith } from './tap.js';
import { labelSize, pointsToPath } from './pure.js';

export type WireProps = {
  /** Index into graph.edges for this render only; never kept across edits. */
  edge: number;
  points: [number, number][];
  labelAt: [number, number];
  /** Width on the wire, null when unknown. */
  width: number | null;
  error: boolean;
  selected: boolean;
  /** Screen pixels per abstract unit: labels and the hit stroke are sized in pixels. */
  scale: number;
  /** Width in screen pixels of the transparent stroke that receives presses. */
  hitPx: number;
  onSelect: () => void;
};

/** One wire in layer view: the line, an arrowhead and a fat invisible hit stroke (a secondary target; the label is the primary one). */
export function Wire(p: WireProps) {
  const d = pointsToPath(p.points);
  return (
    <g
      data-testid={`wire-${p.edge}`}
      data-error={p.error}
      data-selected={p.selected}
      data-width={p.width ?? ''}
      className="tk-wire"
      onPointerDown={pointerDownWith({ onTap: p.onSelect })}
    >
      <path d={d} className="tk-hit" strokeWidth={p.hitPx / p.scale} />
      {p.selected && <path d={d} className="tk-wire-halo" strokeWidth={7} />}
      <path
        d={d}
        className="tk-wire-line"
        strokeWidth={p.error ? 2.5 : 2}
        markerEnd={p.error ? 'url(#tk-arrow-error)' : 'url(#tk-arrow)'}
      />
    </g>
  );
}

/**
 * The width label of a wire, a sibling of the wire (not nested in it) painted above the pieces so
 * it is always the topmost, at least 32 px target for its wire. `hit` comes from `layerHitShapes`.
 */
export function WireLabel(p: WireProps & { hit: { x: number; y: number; w: number; h: number } }) {
  if (p.width === null) return null;
  const text = String(p.width);
  const { fs, lw, lh } = labelSize(text, p.scale);
  return (
    <g
      data-testid={`wire-label-${p.edge}`}
      data-error={p.error}
      data-selected={p.selected}
      className="tk-wire tk-wire-label"
      onPointerDown={pointerDownWith({ onTap: p.onSelect })}
    >
      <rect className="tk-hit" x={p.hit.x} y={p.hit.y} width={p.hit.w} height={p.hit.h} />
      <rect
        className="tk-label-bg"
        x={p.labelAt[0] - lw / 2}
        y={p.labelAt[1] - lh / 2}
        width={lw}
        height={lh}
        rx={lh / 3}
      />
      <text
        className="tk-label-text"
        x={p.labelAt[0]}
        y={p.labelAt[1]}
        fontSize={fs}
        textAnchor="middle"
        dominantBaseline="central"
      >
        {text}
      </text>
    </g>
  );
}
