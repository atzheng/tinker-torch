import type { OutputLink } from '@tinker/core';
import { SelectField, editSetting } from './fields.js';
import { NodeShell } from './NodeShell.js';
import type { WebPiece } from './types.js';

const LINKS: readonly (readonly [OutputLink, string])[] = [
  ['identity', 'Raw value'],
  ['sigmoid', 'Sigmoid probability'],
  ['softmax', 'Softmax probabilities'],
];

/** Layer drawing: the shared node in the prediction role, badge y-hat, the link name as secondary info. */
export const outputPiece: WebPiece<'output'> = {
  Layer: ({ node, w, h, scale }) => (
    <NodeShell
      piece="output"
      role="pred"
      badge
      w={w}
      h={h}
      scale={scale}
      name="Output"
      glyph={'ŷ'}
      secondary={node.link === 'identity' ? [] : [node.link]}
      inPorts={1}
      outPort
    />
  ),
  Neuron: () => <g data-placeholder="output-neuron" />,
  Fields: ({ node }) => (
    <SelectField
      label="Output"
      testid="field-link"
      value={node.link}
      options={LINKS}
      onChange={(v) => editSetting(node.id, 'link', v)}
    />
  ),
};
