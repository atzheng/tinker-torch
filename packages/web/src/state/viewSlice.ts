import type { StateCreator } from 'zustand';
import type { NodeId, PortRef } from '@tinker/core';
import { PIN_LIMIT, type Pin } from '../results/pinsModel.js';

/**
 * What the inspector shows. A wire is identified by its endpoints, not its edge index: indices
 * shift after edits (CONTRACTS.md section 8). Resolve it with `edgeIndexOf` (canvas/pure.ts).
 */
export type Selection =
  | { kind: 'node'; id: NodeId }
  | { kind: 'wire'; from: PortRef; to: PortRef }
  /** Neuron view: the weight W[i][j] of a Linear (row-major [in, out], CONTRACTS.md section 5). */
  | { kind: 'edge'; node: NodeId; i: number; j: number }
  /** Neuron view: one circle of a unit column; `index` counts from the top, as in `NeuronLayout.units`. */
  | { kind: 'unit'; node: NodeId; index: number };

/** The two drawings of the same graph (SPEC "Grain"). Never stored in the graph. */
export type Grain = 'layer' | 'neuron';

/**
 * View slice, owned by the canvas beads (tinker-27m.3 and later). Still to fill in by later beads:
 *   TODO hover: { kind: 'wire' | 'edge' | 'node'; id: string } | null
 *   TODO toggles: Record<string, boolean>   results view toggles (surface, overlay, ...)
 */
export type ViewSlice = {
  /**
   * The selected piece or wire, or null. It may name something that no longer exists (after undo
   * or delete); read it through `useSelection` (state/derived.ts), which treats that as none.
   */
  selection: Selection | null;
  select: (selection: Selection | null) => void;
  selectNode: (id: NodeId) => void;
  /** Which drawing the canvas shows. Changing it never touches the graph. */
  grain: Grain;
  /** Switch grain. A selected edge or unit exists only in neuron view, so it is cleared. */
  setGrain: (grain: Grain) => void;
  /**
   * Pinned runs (tinker-iej.2), oldest first: plain data, kept across graph edits, undo and
   * dataset changes (they are comparisons, not part of the graph or its history).
   */
  pins: Pin[];
  /** Add a pin; refused (false) when PIN_LIMIT pins exist. */
  addPin: (pin: Pin) => boolean;
  removePin: (id: string) => void;
};

export const createViewSlice: StateCreator<ViewSlice, [], [], ViewSlice> = (set) => ({
  selection: null,
  select: (selection) => set({ selection }),
  selectNode: (id) => set({ selection: { kind: 'node', id } }),
  grain: 'layer',
  setGrain: (grain) =>
    set((s) => ({
      grain,
      selection:
        grain === 'layer' && (s.selection?.kind === 'edge' || s.selection?.kind === 'unit')
          ? null
          : s.selection,
    })),
  pins: [],
  addPin: (pin) => {
    let ok = false;
    set((s) => {
      if (s.pins.length >= PIN_LIMIT) return s;
      ok = true;
      return { pins: [...s.pins, pin] };
    });
    return ok;
  },
  removePin: (id) => set((s) => ({ pins: s.pins.filter((p) => p.id !== id) })),
});
