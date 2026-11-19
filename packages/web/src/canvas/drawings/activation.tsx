import { ACTIVATION_FNS, type ActivationFn } from '@tinker/core';
import { NodeShell } from './NodeShell.js';
import { SelectField, editSetting } from './fields.js';
import type { WebPiece } from './types.js';

export const ACTIVATION_SYMBOL: Record<ActivationFn, string> = {
  sigmoid: 'σ',
  tanh: 'tanh',
  relu: 'ReLU',
  gelu: 'GELU',
};
export const ACTIVATION_LABEL: Record<ActivationFn, string> = {
  sigmoid: 'Sigmoid',
  tanh: 'Tanh',
  relu: 'ReLU',
  gelu: 'GELU',
};

/** The operation in the layer drawing's glyph slot; the name beside it is the function's name. */
const ACTIVATION_GLYPH: Record<ActivationFn, string> = {
  sigmoid: 'σ(x)',
  tanh: 'tanh(x)',
  relu: 'max(0,x)',
  gelu: 'x·Φ(x)',
};

/** Layer drawing: the shared node with the function's symbol as its glyph. */
export const activationPiece: WebPiece<'activation'> = {
  Layer: ({ node, w, h, scale }) => (
    <NodeShell
      piece="activation"
      w={w}
      h={h}
      scale={scale}
      name={ACTIVATION_LABEL[node.fn]}
      glyph={ACTIVATION_GLYPH[node.fn]}
      inPorts={1}
      outPort
    />
  ),
  Neuron: () => <g data-placeholder="activation-neuron" />,
  Fields: ({ node }) => (
    <SelectField
      label="Function"
      testid="field-fn"
      value={node.fn}
      options={ACTIVATION_FNS.map((f) => [f, ACTIVATION_LABEL[f]] as const)}
      onChange={(v) => editSetting(node.id, 'fn', v)}
    />
  ),
};
