import type { WebPiece } from './types.js';

/** Placeholder: the drawings and inspector fields for 'target' arrive with the canvas beads. */
export const targetPiece: WebPiece<'target'> = {
  Layer: () => <g data-placeholder="target-layer" />,
  Neuron: () => <g data-placeholder="target-neuron" />,
  Fields: () => <div>target</div>,
};
