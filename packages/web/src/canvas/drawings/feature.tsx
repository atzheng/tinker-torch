import type { WebPiece } from './types.js';

/** Placeholder: the drawings and inspector fields for 'feature' arrive with the canvas beads. */
export const featurePiece: WebPiece<'feature'> = {
  Layer: () => <g data-placeholder="feature-layer" />,
  Neuron: () => <g data-placeholder="feature-neuron" />,
  Fields: () => <div>feature</div>,
};
