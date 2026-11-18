import type { WebPiece } from './types.js';

/** Placeholder: the drawings and inspector fields for 'linear' arrive with the canvas beads. */
export const linearPiece: WebPiece<'linear'> = {
  Layer: () => <g data-placeholder="linear-layer" />,
  Neuron: () => <g data-placeholder="linear-neuron" />,
  Fields: () => <div>linear</div>,
};
