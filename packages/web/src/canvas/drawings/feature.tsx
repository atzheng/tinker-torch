import { useAnalysis } from '../../state/derived.js';
import { CheckField, NumberField, editSetting } from './fields.js';
import type { NodeOf, WebPiece } from './types.js';

/** Layer drawing: a small box with a short formula. (Minimal: the swapping epic may refine it.) */
export const featurePiece: WebPiece<'feature'> = {
  Layer: ({ node, w, h, outWidth }) => {
    const extra = [node.sin ? 'sin' : '', node.cos ? 'cos' : '', node.products ? 'xy' : '']
      .filter(Boolean)
      .join(' ');
    return (
      <g data-piece="feature">
        <rect className="tk-shape" x={0} y={0} width={w} height={h} rx={4} />
        <text className="tk-text tk-small" x={w / 2} y={h * 0.3} textAnchor="middle">
          {node.degree === 1 ? 'x' : `x to x^${node.degree}`}
        </text>
        <text
          className="tk-text tk-small tk-muted"
          x={w / 2}
          y={h * 0.55}
          textAnchor="middle"
          data-testid={`feature-extra-${node.id}`}
        >
          {extra}
        </text>
        <text className="tk-text tk-small" x={w / 2} y={h * 0.84} textAnchor="middle">
          {outWidth === undefined ? '? columns' : `${outWidth} columns`}
        </text>
      </g>
    );
  },
  Neuron: () => <g data-placeholder="feature-neuron" />,
  Fields: ({ node }) => <FeatureFields node={node} />,
};

function FeatureFields({ node }: { node: NodeOf<'feature'> }) {
  const out = useAnalysis().outWidth[node.id];
  return (
    <>
      <p className="tk-muted" data-testid="feature-width">
        {out === undefined
          ? 'Columns produced: unknown until the input is connected.'
          : `Produces ${out} columns`}
      </p>
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
  );
}
