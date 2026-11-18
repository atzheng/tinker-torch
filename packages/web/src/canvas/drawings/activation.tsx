import type { WebPiece } from './types.js';

/** Placeholder: the drawings and inspector fields for 'activation' arrive with the canvas beads. */
export const activationPiece: WebPiece<'activation'> = {
  Layer: () => <g data-placeholder="activation-layer" />,
  Neuron: () => <g data-placeholder="activation-neuron" />,
  Fields: () => <div>activation</div>,
};
