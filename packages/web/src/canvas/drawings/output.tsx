import type { OutputLink } from '@tinker/core';
import { SelectField, editSetting } from './fields.js';
import { legibleAt } from '../pure.js';
import type { WebPiece } from './types.js';

const LINKS: readonly (readonly [OutputLink, string])[] = [
  ['identity', 'Raw value'],
  ['sigmoid', 'Sigmoid probability'],
  ['softmax', 'Softmax probabilities'],
];

/** Layer drawing: a dot labelled y-hat (with the link name when it is not the raw value). */
export const outputPiece: WebPiece<'output'> = {
  Layer: ({ node, w, h, scale }) => (
    <g data-piece="output">
      <circle className="tk-dot" cx={w * 0.3} cy={h / 2} r={8} />
      <text
        className="tk-text"
        x={w * 0.3 + 16}
        y={node.link === 'identity' ? h / 2 : h / 2 - 8}
        dominantBaseline="middle"
      >
        {'ŷ'}
      </text>
      {node.link !== 'identity' && legibleAt(scale, 12.5) && (
        <text
          className="tk-text tk-small tk-muted"
          x={w * 0.3 + 16}
          y={h / 2 + 10}
          dominantBaseline="middle"
        >
          {node.link}
        </text>
      )}
    </g>
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
