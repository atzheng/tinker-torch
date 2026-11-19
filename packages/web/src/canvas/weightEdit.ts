import { LOCKED_MESSAGE } from '@tinker/core';
import type { DragHandlers, DragPoint } from '../gestures/drag.js';
import { lockedIds } from '../state/locks.js';
import { hostShape } from '../state/hostShape.js';
import { useStore } from '../state/store.js';
import {
  recordWeightEdit,
  weightGeneration,
  whileApplyingEdit,
  type LayerValues,
} from '../state/weightHistory.js';
import { biasAt, weightAt } from './neuronPure.js';

/**
 * What a hand edit can set: a weight `W[i][j]` or a bias `b[j]` of a Linear, the one value of a
 * Weight piece (`scalew`, `W` of shape [1, 1]: the weight on a weight wire) or of a Bias piece
 * (`bias`, `b` of shape [1]).
 */
export type WeightTarget =
  | { kind: 'W'; node: string; i: number; j: number }
  | { kind: 'b'; node: string; j: number }
  | { kind: 'scalew'; node: string }
  | { kind: 'bias'; node: string };

/** Weight change per screen pixel of vertical drag (up raises the value): 100 px is 1.0. */
export const WEIGHT_PER_PX = 0.01;
/** Typed and dragged values are kept within this range. */
export const WEIGHT_LIMIT = 100;

export const PAUSE_FIRST = 'Pause training to set a weight by hand.';
export const DIVERGED_FIRST = 'Reset the run before setting a weight by hand.';

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
  if (s.diverged) {
    useStore.setState({ error: DIVERGED_FIRST });
    return null;
  }
  if (lockedIds().includes(t.node)) {
    useStore.setState({ error: LOCKED_MESSAGE });
    return null;
  }
  const host = s.host ?? undefined;
  const node = s.graph.nodes.find((n) => n.id === t.node);
  const layer = host?.[t.node];
  if (!host || !layer || !node || !s.canTrain) return null;
  // Which array of the layer holds the value, where in it, and what it is now.
  let name: 'W' | 'b';
  let index = 0;
  let start: number | undefined;
  if (t.kind === 'scalew' || t.kind === 'bias') {
    if (node.type !== (t.kind === 'scalew' ? 'scalew' : 'bias')) return null;
    name = t.kind === 'scalew' ? 'W' : 'b';
    start = layer[name]?.length === 1 ? layer[name]?.[0] : undefined;
  } else {
    const inW = s.analysis.inWidth[t.node];
    if (node.type !== 'linear' || inW === undefined || !layer['W']) return null;
    const shape = hostShape(host, t.node, s.dataInfo);
    name = t.kind === 'W' ? 'W' : 'b';
    index = t.kind === 'W' ? t.i * node.out + t.j : t.j;
    start =
      t.kind === 'W'
        ? weightAt(host, t.node, t.i, t.j, inW, node.out, shape)
        : shape && shape[1] === node.out
          ? biasAt(host, t.node, t.j)
          : undefined;
  }
  if (start === undefined) return null;

  const graph = s.graph;
  const gen = weightGeneration();
  const before: LayerValues = {};
  if (layer.W) before.W = Float32Array.from(layer.W);
  if (layer.b) before.b = Float32Array.from(layer.b);
  const size = before[name]?.length;
  let last = start;
  let dead = false;
  /** Current value of the target in the live host, undefined when its shape changed. */
  const current = (): number | undefined => {
    const cur = useStore.getState().host?.[t.node];
    const arr = cur?.[name];
    return arr && arr.length === size ? arr[index] : undefined;
  };
  // Written on top of the live layer, so values another action put there are kept.
  const write = (v: number): boolean => {
    const arr = useStore.getState().host?.[t.node]?.[name];
    if (!arr || arr.length !== size) return false;
    const next = Float32Array.from(arr);
    next[index] = v;
    return useStore.getState().writeParams(t.node, { [name]: next });
  };
  // The graph changed under a drag in progress (Ctrl+Z, a toolbar edit): the preview has no undo
  // step, so take it back and end the edit.
  const unsub = useStore.subscribe(() => {
    if (dead || useStore.getState().graph === graph) return;
    dead = true;
    unsub();
    if (last !== start && current() === Math.fround(last)) write(start);
  });
  const end = () => {
    dead = true;
    unsub();
  };
  return {
    start,
    set: (v) => {
      if (dead) return last;
      const c = clamp(Number.isFinite(v) ? v : start);
      if (write(c)) last = c;
      return last;
    },
    commit: () => {
      const st = useStore.getState();
      const live = !dead && st.graph === graph && weightGeneration() === gen;
      end();
      if (!live || last === start) return;
      const cur = st.host?.[t.node];
      if (!cur) return;
      const after: LayerValues = {};
      if (cur.W) after.W = Float32Array.from(cur.W);
      if (cur.b) after.b = Float32Array.from(cur.b);
      const copy = { ...graph };
      recordWeightEdit(graph, copy, t.node, before, after);
      whileApplyingEdit(() => st.apply({ ok: true, graph: copy }));
    },
    cancel: () => {
      if (!dead && last !== start) write(start);
      last = start;
      end();
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
