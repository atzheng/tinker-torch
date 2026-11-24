import { NodeShell } from './NodeShell.js';
import type { WebPiece } from './types.js';

/** Stub (tinker-6h1.2): a plain box. The real drawing, palette wiring and inspector are tinker-6h1.4. */
export const treePiece: WebPiece<'tree'> = {
  Layer: function TreeLayer({ w, h, scale, outWidth }) {
    return (
      <NodeShell
        piece="tree"
        w={w}
        h={h}
        scale={scale}
        name="Tree"
        glyph="leaf"
        secondary={[outWidth === undefined ? '? leaves' : `${outWidth} leaves`]}
        inPorts={1}
        outPort
      />
    );
  },
  Neuron: () => <g data-placeholder="tree-neuron" />,
  Fields: () => null,
};
