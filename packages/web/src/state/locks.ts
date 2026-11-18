import type { NodeId } from '@tinker/core';

/**
 * The pieces the current preset locks (CONTRACTS.md sections 8 and 12). Every web edit call site
 * passes `{ locked: lockedIds() }`; the presets bead (tinker-ysl.2) fills this in from the loaded
 * preset. Until then nothing is locked.
 */
export function lockedIds(): readonly NodeId[] {
  return [];
}
