import { NodeShell } from './NodeShell.js';
import { useStore } from '../../state/store.js';
import { inputLine } from '../../state/names.js';
import { InputFields } from './InputFields.js';
import type { WebPiece } from './types.js';

/** Layer drawing: the shared node in the input role, badge x, the selected columns as its secondary line. */
export const inputPiece: WebPiece<'input'> = {
  Layer: function InputLayer({ node, w, h, scale }) {
    const info = useStore((s) => s.dataInfo);
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
        secondary={[inputLine(info, node.columns, w - 48)]}
        inPorts={0}
        outPort
      />
    );
  },
  Neuron: () => <g data-placeholder="input-neuron" />,
  Fields: InputFields,
};
