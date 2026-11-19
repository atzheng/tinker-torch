import { NodeShell } from './NodeShell.js';
import { InputFields } from './InputFields.js';
import type { WebPiece } from './types.js';

const MAX_SHOWN = 3;

/** Layer drawing: the shared node in the input role, badge x, the selected columns as its secondary line. */
export const inputPiece: WebPiece<'input'> = {
  Layer: ({ node, w, h, scale }) => {
    const shown = node.columns.slice(0, MAX_SHOWN);
    const more = node.columns.length - shown.length;
    return (
      <NodeShell
        piece="input"
        role="input"
        badge
        w={w}
        h={h}
        scale={scale}
        name="Input"
        glyph="x"
        secondary={[shown.join(' ') + (more > 0 ? ` +${more}` : '')]}
        inPorts={0}
        outPort
      />
    );
  },
  Neuron: () => <g data-placeholder="input-neuron" />,
  Fields: InputFields,
};
