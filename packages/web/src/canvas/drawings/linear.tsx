import { heatMap } from '../pure.js';
import { NodeShell } from './NodeShell.js';
import { CheckField, NumberField, editOut, editSetting } from './fields.js';
import type { WebPiece } from './types.js';

/** Layer drawing: the shared node; "d to m" and the parameter count are its secondary lines, a heat map of W its glyph. */
export const linearPiece: WebPiece<'linear'> = {
  Layer: ({ node, w, h, inWidth, paramCount, weights, scale }) => {
    const heat = weights ? heatMap(weights.W, weights.rows, weights.cols) : null;
    const size = 26;
    const hx = w - size - 12;
    const hy = h / 2 - size / 2 + 4;
    const cw = heat ? size / heat.cols : 0;
    const ch = heat ? size / heat.rows : 0;
    return (
      <NodeShell
        piece="linear"
        w={w}
        h={h}
        scale={scale}
        name="Linear"
        secondary={[
          `${inWidth ?? '?'} to ${node.out}`,
          inWidth === undefined ? 'no input' : `${paramCount} params`,
        ]}
        inPorts={1}
        outPort
        slot={
          heat ? (
            <g data-testid={`heat-${node.id}`} aria-label="Weights heat map">
              <rect
                x={hx - 1}
                y={hy - 1}
                width={size + 2}
                height={size + 2}
                className="tk-heat-bg"
              />
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
          ) : undefined
        }
      />
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
