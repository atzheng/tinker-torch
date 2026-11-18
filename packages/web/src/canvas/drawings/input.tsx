import { InputFields } from './InputFields.js';
import type { WebPiece } from './types.js';

/** Placeholder: the drawings for 'input' arrive with the canvas beads; Fields is tinker-27m.5. */
export const inputPiece: WebPiece<'input'> = {
  Layer: () => <g data-placeholder="input-layer" />,
  Neuron: () => <g data-placeholder="input-neuron" />,
  Fields: InputFields,
};
