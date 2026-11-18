import type { WebPiece } from './types.js';

/** Placeholder: the drawings and inspector fields for 'loss' arrive with the canvas beads. */
export const lossPiece: WebPiece<'loss'> = {
  Layer: () => <g data-placeholder="loss-layer" />,
  Neuron: () => <g data-placeholder="loss-neuron" />,
  Fields: () => <div>loss</div>,
};
