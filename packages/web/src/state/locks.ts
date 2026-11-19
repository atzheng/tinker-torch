import type { NodeId } from '@tinker/core';
import { useStore } from './store.js';

const NONE: readonly NodeId[] = [];

/**
 * The pieces the current preset locks (CONTRACTS.md sections 8 and 12). Every web edit call site
 * passes `{ locked: lockedIds() }`. Empty when no preset is loaded.
 */
export function lockedIds(): readonly NodeId[] {
  return useStore.getState().preset?.locked ?? NONE;
}
