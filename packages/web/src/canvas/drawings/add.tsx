import type { WebPiece } from './types.js';

/** Placeholder: the drawings and inspector fields for 'add' arrive with the canvas beads. */
export const addPiece: WebPiece<'add'> = {
  Layer: () => <g data-placeholder="add-layer" />,
  Neuron: () => <g data-placeholder="add-neuron" />,
  Fields: () => <div>add</div>,
};
