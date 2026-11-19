import type { DragHandlers, DragPoint } from '../gestures/drag.js';
import { lockedIds } from '../state/locks.js';
import { hostShape } from '../state/hostShape.js';
import { useStore } from '../state/store.js';
import { recordWeightEdit, type LayerValues } from '../state/weightHistory.js';
import { biasAt, weightAt } from './neuronPure.js';

/** A weight `W[i][j]` or a bias `b[j]` of a Linear. */
export type WeightTarget =
  { kind: 'W'; node: string; i: number; j: number } | { kind: 'b'; node: string; j: number };

/** Weight change per screen pixel of vertical drag (up raises the value): 100 px is 1.0. */
export const WEIGHT_PER_PX = 0.01;
/** Typed and dragged values are kept within this range. */
export const WEIGHT_LIMIT = 100;

/** Same sentence as the core edits give for a locked piece. */
const LOCKED_MESSAGE = 'This piece is locked.';
export const PAUSE_FIRST = 'Pause training to set a weight by hand.';

export type WeightEdit = {
  /** Value when the edit began. */
  start: number;
  /** Write a value into the live model (previews: no undo step yet). Returns the value written. */
  set: (v: number) => number;
  /** End the edit; a changed value becomes one undo step. */
  commit: () => void;
  /** End the edit and put the old values back. */
  cancel: () => void;
};

const clamp = (v: number): number => Math.max(-WEIGHT_LIMIT, Math.min(WEIGHT_LIMIT, v));

/**
 * Start a hand edit of one weight or bias. Refuses (null, with the toast sentence) while running,
 * for a locked piece, or when the model has no value for it. Hand edits are one undo step per
 * drag or typed value, in the ordinary undo stack (see `state/weightHistory.ts`); the loss
 * history and the step counter are kept; the optimizer state is kept too. They need a paused run
 * because a running loop would overwrite the value on its next step.
 */
export function beginWeightEdit(t: WeightTarget): WeightEdit | null {
  const s = useStore.getState();
  if (s.running) {
    useStore.setState({ error: PAUSE_FIRST });
    return null;
  }
  if (lockedIds().includes(t.node)) {
    useStore.setState({ error: LOCKED_MESSAGE });
    return null;
  }
  const host = s.host ?? undefined;
  const node = s.graph.nodes.find((n) => n.id === t.node);
  const inW = s.analysis.inWidth[t.node];
  const layer = host?.[t.node];
  if (!host || !layer || node?.type !== 'linear' || inW === undefined || !s.canTrain) return null;
  const shape = hostShape(host, t.node, s.dataInfo);
  const start =
    t.kind === 'W'
      ? weightAt(host, t.node, t.i, t.j, inW, node.out, shape)
      : shape && shape[1] === node.out
        ? biasAt(host, t.node, t.j)
        : undefined;
  if (start === undefined) return null;

  const graph = s.graph;
  const before: LayerValues = { W: Float32Array.from(layer.W) };
  if (layer.b) before.b = Float32Array.from(layer.b);
  let last = start;
  const write = (v: number): boolean => {
    if (t.kind === 'W') {
      const W = Float32Array.from(before.W);
      W[t.i * node.out + t.j] = v;
      return useStore.getState().writeParams(t.node, { W });
    }
    if (!before.b) return false;
    const b = Float32Array.from(before.b);
    b[t.j] = v;
    return useStore.getState().writeParams(t.node, { b });
  };
  return {
    start,
    set: (v) => {
      const c = clamp(Number.isFinite(v) ? v : start);
      if (write(c)) last = c;
      return last;
    },
    commit: () => {
      const st = useStore.getState();
      if (last === start) return;
      const cur = st.host?.[t.node];
      if (!cur || st.graph !== graph) return;
      const after: LayerValues = { W: Float32Array.from(cur.W) };
      if (cur.b) after.b = Float32Array.from(cur.b);
      const copy = { ...graph };
      recordWeightEdit(graph, copy, t.node, before, after);
      st.apply({ ok: true, graph: copy });
    },
    cancel: () => {
      if (last !== start) write(start);
      last = start;
    },
  };
}

/**
 * Drag handlers for `pointerDownWith`: a vertical drag from `resolve(p)` (called once, when the
 * drag passes the slop) sets the value, up raising it. The model is written at most once per
 * animation frame; the release commits one undo step. A press that stays inside the slop never
 * reaches these (it is a tap, and selects).
 */
export function weightDragHandlers(
  resolve: (p: DragPoint) => WeightTarget | null,
  onBegin?: (t: WeightTarget) => void,
): DragHandlers {
  let edit: WeightEdit | null = null;
  let pending: number | null = null;
  let frame: number | null = null;
  const flush = () => {
    frame = null;
    if (edit && pending !== null) edit.set(pending);
    pending = null;
  };
  const stop = () => {
    if (frame !== null) cancelAnimationFrame(frame);
    frame = null;
    pending = null;
  };
  return {
    onStart: (p) => {
      const t = resolve(p);
      edit = t ? beginWeightEdit(t) : null;
      if (t && edit) onBegin?.(t);
    },
    onMove: (p) => {
      if (!edit) return;
      pending = edit.start - p.dy * WEIGHT_PER_PX;
      frame ??= requestAnimationFrame(flush);
    },
    onEnd: (p) => {
      if (!edit) return;
      stop();
      edit.set(edit.start - p.dy * WEIGHT_PER_PX);
      edit.commit();
      edit = null;
    },
    onCancel: () => {
      stop();
      edit?.cancel();
      edit = null;
    },
  };
}
