import { NodeShell } from './NodeShell.js';
import { NumberField, editSetting } from './fields.js';
import type { WebPiece } from './types.js';

/** Layer drawing: the shared node with the drop probability as its glyph. */
export const dropoutPiece: WebPiece<'dropout'> = {
  Layer: ({ node, w, h, scale }) => (
    <NodeShell
      piece="dropout"
      w={w}
      h={h}
      scale={scale}
      name="Dropout"
      glyph={`p=${node.p}`}
      inPorts={1}
      outPort
    />
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
