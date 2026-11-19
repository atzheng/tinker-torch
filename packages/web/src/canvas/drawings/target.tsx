import { NodeShell } from './NodeShell.js';
import type { WebPiece } from './types.js';

/** Layer drawing: the shared node in the input role (the target y is data entering), badge y. */
export const targetPiece: WebPiece<'target'> = {
  Layer: ({ w, h, scale }) => (
    <NodeShell
      piece="target"
      role="input"
      badge
      w={w}
      h={h}
      scale={scale}
      name="Target"
      glyph="y"
      inPorts={0}
      outPort
    />
  ),
  Neuron: () => <g data-placeholder="target-neuron" />,
  Fields: () => <p className="tk-muted">No settings.</p>,
};
