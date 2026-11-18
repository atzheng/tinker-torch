import type { StateCreator } from 'zustand';

/**
 * Empty slice, owned by the canvas beads (tinker-27m.3 and later). Shape to fill in:
 *   TODO grain: 'layer' | 'neuron'      the grain toggle
 *   TODO selected: NodeId | null        node shown in the inspector
 *   TODO hover: { kind: 'wire' | 'edge' | 'node'; id: string } | null
 *   TODO toggles: Record<string, boolean>   results view toggles (surface, overlay, ...)
 *   TODO pins: Pin[]                    pinned runs
 */
export type ViewSlice = Record<never, never>;

export const createViewSlice: StateCreator<ViewSlice, [], [], ViewSlice> = () => ({});
