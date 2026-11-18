import { NumberField, editSetting } from './fields.js';
import type { WebPiece } from './types.js';

/** Layer drawing: a dashed circle with p. (Minimal: the swapping epic may refine it.) */
export const dropoutPiece: WebPiece<'dropout'> = {
  Layer: ({ node, w, h }) => (
    <g data-piece="dropout">
      <circle
        className="tk-shape"
        strokeDasharray="4 3"
        cx={w / 2}
        cy={h / 2}
        r={Math.min(w, h) / 2 - 4}
      />
      <text
        className="tk-text tk-small"
        x={w / 2}
        y={h / 2}
        textAnchor="middle"
        dominantBaseline="middle"
      >
        p={node.p}
      </text>
    </g>
  ),
  Neuron: () => <g data-placeholder="dropout-neuron" />,
  Fields: ({ node }) => (
    <NumberField
      label="Drop probability p"
      testid="field-p"
      value={node.p}
      min={0}
      max={0.99}
      step={0.05}
      onCommit={(v) => editSetting(node.id, 'p', v)}
    />
  ),
};
