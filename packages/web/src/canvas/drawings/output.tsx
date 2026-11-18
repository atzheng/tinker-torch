import type { WebPiece } from './types.js';

/** Placeholder: the drawings and inspector fields for 'output' arrive with the canvas beads. */
export const outputPiece: WebPiece<'output'> = {
  Layer: () => <g data-placeholder="output-layer" />,
  Neuron: () => <g data-placeholder="output-neuron" />,
  Fields: () => <div>output</div>,
};
