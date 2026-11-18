import type { WebPiece } from './types.js';

/** Placeholder: the drawings and inspector fields for 'input' arrive with the canvas beads. */
export const inputPiece: WebPiece<'input'> = {
  Layer: () => <g data-placeholder="input-layer" />,
  Neuron: () => <g data-placeholder="input-neuron" />,
  Fields: () => <div>input</div>,
};
