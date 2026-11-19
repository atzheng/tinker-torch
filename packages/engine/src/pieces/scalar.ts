import type { NodeOf } from '@tinker/core';
import type { EnginePiece } from './types.js';

const todo = (name: string) => (): never => {
  throw new Error(`The ${name} piece is not available in the engine yet.`);
};

// TODO(9io.4): real forwards, params and fixtures for the scalar pieces.
export const column: EnginePiece<NodeOf<'column'>> = { forward: todo('Column') };
export const scalew: EnginePiece<NodeOf<'scalew'>> = { forward: todo('Weight') };
export const bias: EnginePiece<NodeOf<'bias'>> = { forward: todo('Bias') };
export const sum: EnginePiece<NodeOf<'sum'>> = { forward: todo('Sum') };
