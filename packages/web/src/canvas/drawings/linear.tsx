import { heatMap } from '../pure.js';
import { CheckField, NumberField, editOut, editSetting } from './fields.js';
import type { WebPiece } from './types.js';

/** Layer drawing: a box "d to m", a small heat map of W, and the parameter count. */
export const linearPiece: WebPiece<'linear'> = {
  Layer: ({ node, w, h, inWidth, paramCount, weights }) => {
    const heat = weights ? heatMap(weights.W, weights.rows, weights.cols) : null;
    const size = 28;
    const hx = w - size - 6;
    const hy = (h - size) / 2;
    const cw = heat ? size / heat.cols : 0;
    const ch = heat ? size / heat.rows : 0;
    return (
      <g data-piece="linear">
        <rect className="tk-shape" x={0} y={0} width={w} height={h} rx={8} />
        <text className="tk-text" x={10} y={h * 0.36} dominantBaseline="middle">
          {inWidth ?? '?'} to {node.out}
        </text>
        <text className="tk-text tk-small tk-muted" x={10} y={h * 0.74} dominantBaseline="middle">
          {inWidth === undefined ? 'no input' : `${paramCount} params`}
        </text>
        {heat && (
          <g data-testid={`heat-${node.id}`} aria-label="Weights heat map">
            <rect x={hx - 1} y={hy - 1} width={size + 2} height={size + 2} className="tk-heat-bg" />
            {heat.values.map((v, k) => {
              const r = Math.floor(k / heat.cols);
              const c = k % heat.cols;
              const a = heat.maxAbs > 0 ? Math.min(1, Math.abs(v) / heat.maxAbs) : 0;
              return (
                <rect
                  key={k}
                  x={hx + c * cw}
                  y={hy + r * ch}
                  width={cw}
                  height={ch}
                  fill={v >= 0 ? 'var(--weight-pos)' : 'var(--weight-neg)'}
                  fillOpacity={0.15 + 0.85 * a}
                />
              );
            })}
          </g>
        )}
      </g>
    );
  },
  Neuron: () => <g data-placeholder="linear-neuron" />,
  Fields: ({ node }) => (
    <>
      <NumberField
        label="Output width"
        testid="field-out"
        value={node.out}
        min={1}
        max={64}
        step={1}
        onCommit={(v) => editOut(node.id, v)}
      />
      <CheckField
        label="Bias"
        testid="field-bias"
        checked={node.bias}
        onChange={(v) => editSetting(node.id, 'bias', v)}
      />
    </>
  ),
};
