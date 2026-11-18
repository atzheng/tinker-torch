import type { NodeOf } from '@tinker/core';
import { notYet } from './types.js';
import type { EnginePiece } from './types.js';

/** Placeholder: filled by a later engine bead (tinker-hj4.3). */
export const dropout: EnginePiece<NodeOf<'dropout'>> = {
  implemented: false,
  forward() {
    return notYet('Dropout');
  },
};
