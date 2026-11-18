import { ACTIVATION_FNS, type ActivationFn } from '@tinker/core';
import { SelectField, editSetting } from './fields.js';
import type { WebPiece } from './types.js';

export const ACTIVATION_SYMBOL: Record<ActivationFn, string> = {
  sigmoid: 'σ',
  tanh: 'tanh',
  relu: 'ReLU',
  gelu: 'GELU',
};
const LABEL: Record<ActivationFn, string> = {
  sigmoid: 'Sigmoid',
  tanh: 'Tanh',
  relu: 'ReLU',
  gelu: 'GELU',
};

/** Layer drawing: a circle that applies a function, with its symbol. */
export const activationPiece: WebPiece<'activation'> = {
  Layer: ({ node, w, h }) => (
    <g data-piece="activation">
      <circle className="tk-shape" cx={w / 2} cy={h / 2} r={Math.min(w, h) / 2 - 4} />
      <text className="tk-text" x={w / 2} y={h / 2} textAnchor="middle" dominantBaseline="middle">
        {ACTIVATION_SYMBOL[node.fn]}
      </text>
    </g>
  ),
  Neuron: () => <g data-placeholder="activation-neuron" />,
  Fields: ({ node }) => (
    <SelectField
      label="Function"
      testid="field-fn"
      value={node.fn}
      options={ACTIVATION_FNS.map((f) => [f, LABEL[f]] as const)}
      onChange={(v) => editSetting(node.id, 'fn', v)}
    />
  ),
};
