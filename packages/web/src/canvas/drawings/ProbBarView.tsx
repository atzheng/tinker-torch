import type { ProbBar } from '../probBarPure.js';

/** Bar geometry in node-local units: under the name, left of the glyph slot (x from 60). */
const X = 12;
const W = 36;
const BAR_H = 8;

/** Two significant decimals, e.g. 0.27; never "1.00" or "0.00" for a value that is not exactly so. */
export function probText(v: number): string {
  if (v >= 0.995 && v < 1) return '>0.99';
  if (v > 0 && v < 0.005) return '<0.01';
  return v.toFixed(2);
}

/**
 * Diagnostic read-out (tinker-t6o): a flat, outline-free track filled to the probability, with the
 * number in muted monospace beneath it. A sigmoid is one fill; a softmax is stacked class shares in
 * the `--class-*` colours. When the value differs between examples it is a mean over the training
 * examples and says so ("mean" in the number line and the tooltip). Never interactive.
 */
export function ProbBarView({ bar, h }: { bar: ProbBar; h: number }) {
  const y = h - 25;
  let at = 0;
  const single = bar.values.length === 1;
  // a stacked softmax bar has no room for the shares as numbers: they are in the tooltip
  const label = single ? probText(bar.values[0] ?? 0) : '';
  const title = `${bar.mean ? 'Mean probability over the training examples' : 'Probability'}: ${bar.values
    .map((v) => v.toFixed(3))
    .join(', ')}`;
  return (
    <g className="tk-probbar" data-testid="prob-bar" data-mean={bar.mean}>
      <title>{title}</title>
      <rect className="tk-probbar-track" x={X} y={y} width={W} height={BAR_H} rx={2} />
      {bar.values.map((v, i) => {
        const x = X + at * W;
        at += v;
        return (
          <rect
            key={i}
            className="tk-probbar-fill"
            data-testid="prob-bar-fill"
            data-value={v}
            x={x}
            y={y}
            width={Math.max(0, v * W)}
            height={BAR_H}
            rx={single ? 2 : 0}
            style={single ? undefined : { fill: `var(--class-${i % 6})` }}
          />
        );
      })}
      <text className="tk-probbar-text" data-testid="prob-bar-text" x={X} y={h - 8}>
        {bar.mean ? `mean ${label}`.trim() : label}
      </text>
    </g>
  );
}
