import { pointerDownWith } from './tap.js';
import { pointsToPath } from './pure.js';

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

/** One wire in layer view: the line, an arrowhead, the width label, and a fat invisible hit stroke. */
export function Wire(p: WireProps) {
  const d = pointsToPath(p.points);
  const fs = 12 / p.scale;
  const text = p.width === null ? '' : String(p.width);
  const lw = text.length * 0.62 * fs + 0.8 * fs;
  const lh = 1.4 * fs;
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
      {text && (
        <g data-testid={`wire-label-${p.edge}`}>
          <rect
            className="tk-hit"
            x={p.labelAt[0] - lw / 2 - 4 / p.scale}
            y={p.labelAt[1] - Math.max(lh, 32 / p.scale) / 2}
            width={lw + 8 / p.scale}
            height={Math.max(lh, 32 / p.scale)}
          />
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
      )}
    </g>
  );
}
