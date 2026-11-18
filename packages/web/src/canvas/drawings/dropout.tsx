import type { WebPiece } from './types.js';

/** Placeholder: the drawings and inspector fields for 'dropout' arrive with the canvas beads. */
export const dropoutPiece: WebPiece<'dropout'> = {
  Layer: () => <g data-placeholder="dropout-layer" />,
  Neuron: () => <g data-placeholder="dropout-neuron" />,
  Fields: () => <div>dropout</div>,
};
