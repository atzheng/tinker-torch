import type { StateCreator } from 'zustand';
import type { NodeId, PortRef } from '@tinker/core';

/**
 * What the inspector shows. A wire is identified by its endpoints, not its edge index: indices
 * shift after edits (CONTRACTS.md section 8). Resolve it with `edgeIndexOf` (canvas/pure.ts).
 */
export type Selection = { kind: 'node'; id: NodeId } | { kind: 'wire'; from: PortRef; to: PortRef };

/**
 * View slice, owned by the canvas beads (tinker-27m.3 and later). Still to fill in by later beads:
 *   TODO grain: 'layer' | 'neuron'      the grain toggle (tinker-ran.1)
 *   TODO hover: { kind: 'wire' | 'edge' | 'node'; id: string } | null
 *   TODO toggles: Record<string, boolean>   results view toggles (surface, overlay, ...)
 *   TODO pins: Pin[]                    pinned runs
 */
export type ViewSlice = {
  /**
   * The selected piece or wire, or null. It may name something that no longer exists (after undo
   * or delete); read it through `useSelection` (state/derived.ts), which treats that as none.
   */
  selection: Selection | null;
  select: (selection: Selection | null) => void;
  selectNode: (id: NodeId) => void;
};

export const createViewSlice: StateCreator<ViewSlice, [], [], ViewSlice> = (set) => ({
  selection: null,
  select: (selection) => set({ selection }),
  selectNode: (id) => set({ selection: { kind: 'node', id } }),
});
