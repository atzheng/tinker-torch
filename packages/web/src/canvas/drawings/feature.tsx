import { useAnalysis } from '../../state/derived.js';
import { NodeShell } from './NodeShell.js';
import { CheckField, NumberField, editSetting } from './fields.js';
import type { NodeOf, WebPiece } from './types.js';

/** Layer drawing: the shared node; the formula and the columns produced are its secondary lines. */
export const featurePiece: WebPiece<'feature'> = {
  Layer: ({ node, w, h, outWidth, scale }) => {
    const extra = [node.sin ? '\\sin' : '', node.cos ? '\\cos' : '', node.products ? 'xy' : '']
      .filter(Boolean)
      .join('\\ ');
    const main = node.degree === 1 ? '$x$' : `$x, \\dots, x^{${node.degree}}$`;
    const cols = outWidth === undefined ? '? columns' : `${outWidth} columns`;
    return (
      <NodeShell
        piece="feature"
        w={w}
        h={h}
        scale={scale}
        name="Feature"
        secondary={[main, cols]}
        tag={extra}
        inPorts={1}
        outPort
      />
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
