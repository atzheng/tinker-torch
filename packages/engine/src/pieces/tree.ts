import type { NodeOf } from '@tinker/core';
import type { EnginePiece } from './types.js';

/** Stub (tinker-6h1.2): the engine Tree piece (host-computed one-hot) is tinker-6h1.3. */
export const tree: EnginePiece<NodeOf<'tree'>> = {
  forward() {
    throw new Error('Tree is not implemented in the engine yet.');
  },
};
