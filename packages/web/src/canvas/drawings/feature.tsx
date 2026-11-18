import { CheckField, NumberField, editSetting } from './fields.js';
import type { WebPiece } from './types.js';

/** Layer drawing: a small box with a short formula. (Minimal: the swapping epic may refine it.) */
export const featurePiece: WebPiece<'feature'> = {
  Layer: ({ node, w, h }) => {
    const parts = [`x^1..${node.degree}`];
    if (node.sin) parts.push('sin');
    if (node.cos) parts.push('cos');
    if (node.products) parts.push('xy');
    return (
      <g data-piece="feature">
        <rect className="tk-shape" x={0} y={h * 0.2} width={w} height={h * 0.6} rx={4} />
        <text
          className="tk-text tk-small"
          x={w / 2}
          y={h / 2}
          textAnchor="middle"
          dominantBaseline="middle"
        >
          {parts.join(', ')}
        </text>
      </g>
    );
  },
  Neuron: () => <g data-placeholder="feature-neuron" />,
  Fields: ({ node }) => (
    <>
      <NumberField
        label="Highest power"
        testid="field-degree"
        value={node.degree}
        min={1}
        max={5}
        step={1}
        onCommit={(v) => editSetting(node.id, 'degree', v)}
      />
      <CheckField
        label="sin"
        testid="field-sin"
        checked={node.sin}
        onChange={(v) => editSetting(node.id, 'sin', v)}
      />
      <CheckField
        label="cos"
        testid="field-cos"
        checked={node.cos}
        onChange={(v) => editSetting(node.id, 'cos', v)}
      />
      <CheckField
        label="Products of columns"
        testid="field-products"
        checked={node.products}
        onChange={(v) => editSetting(node.id, 'products', v)}
      />
    </>
  ),
};
