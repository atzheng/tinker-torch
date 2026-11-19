import type { StateCreator } from 'zustand';
import type { NodeId, PortRef } from '@tinker/core';
import { PIN_LIMIT, type Pin } from '../results/pinsModel.js';
import {
  DEFAULT_ORDER,
  DEFAULT_WEIGHTS,
  type PanelId,
  type PanelWeights,
} from '../results/panelLayout.js';

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
  | { kind: 'unit'; node: NodeId; index: number }
  /** A role pin (ŷ or L) on a node's output. Identified by node: a node carries each role at most once. */
  | { kind: 'pin'; role: 'pred' | 'loss'; node: NodeId };

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
  /** The node whose ŷ pin the results plots draw; null (or a node that is no longer a ŷ): the first pin. See `plotIndex`. */
  plotPrediction: NodeId | null;
  setPlotPrediction: (id: NodeId | null) => void;
  /**
   * The piece whose weight inspector is open (tinker-fn5), or null. A Linear, Weight or Bias; it
   * may name a piece that no longer exists (the panel then shows nothing). View state only.
   */
  weightsOf: NodeId | null;
  openWeights: (id: NodeId | null) => void;
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
  /**
   * Results panels (tinker-ong): top to bottom, and each one's share of the column. View state of
   * this session only: not in the graph, not in files, not in the undo history.
   */
  panelOrder: PanelId[];
  panelWeights: PanelWeights;
  setPanelOrder: (order: PanelId[]) => void;
  setPanelWeights: (weights: PanelWeights) => void;
  /**
   * Pieces that a Group / Expand just created: they fade in for 300 ms (tinker-9io.7). Cleared by
   * a timer; empty with prefers-reduced-motion. Not part of the graph or its history.
   */
  fresh: ReadonlySet<NodeId>;
  setFresh: (ids: ReadonlySet<NodeId>) => void;
};

export const createViewSlice: StateCreator<ViewSlice, [], [], ViewSlice> = (set) => ({
  selection: null,
  select: (selection) => set({ selection }),
  selectNode: (id) => set({ selection: { kind: 'node', id } }),
  grain: 'layer',
  plotPrediction: null,
  setPlotPrediction: (plotPrediction) => set({ plotPrediction }),
  weightsOf: null,
  openWeights: (weightsOf) => set({ weightsOf }),
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
  panelOrder: [...DEFAULT_ORDER],
  panelWeights: { ...DEFAULT_WEIGHTS },
  setPanelOrder: (panelOrder) => set({ panelOrder }),
  setPanelWeights: (panelWeights) => set({ panelWeights }),
  fresh: new Set(),
  setFresh: (fresh) => set({ fresh }),
});
