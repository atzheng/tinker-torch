/**
 * Pan and zoom of the neuron view (tinker-qvj): pure maths, no React. Transforms are in
 * canvas-element pixels, like `Transform` in pure.ts. Unit tested in test/viewport.test.ts.
 */
import { HIT_MIN_PX, type Transform } from './pure.js';

type Size = readonly [number, number];
type Box = { w: number; h: number };

/** Largest scale the neuron view zooms to. */
export const NEURON_MAX_SCALE = 3;

/**
 * Smallest scale at which a unit cell (the whole column pitch) is at least 32 screen px tall. A
 * touch view starts at no less than this and pans, instead of fitting a wide graph into tiny cells.
 */
export function touchScaleFloor(pitch: number): number {
  return Math.ceil((HIT_MIN_PX / pitch) * 1000) / 1000;
}

/**
 * Keep the drawing reachable. Smaller than the area below `top` on an axis it is centred; larger,
 * it may be dragged until its edge is `margin` px inside the view edge.
 */
export function clampView(t: Transform, size: Size, view: Box, top = 0, margin = 24): Transform {
  const axis = (len: number, avail: number, offset: number, pos: number) => {
    const px = len * t.scale;
    if (px + 2 * margin <= avail) return offset + (avail - px) / 2;
    return Math.min(offset + margin, Math.max(offset + avail - margin - px, pos));
  };
  return {
    scale: t.scale,
    tx: axis(size[0], view.w, 0, t.tx),
    ty: axis(size[1], view.h - top, top, t.ty),
  };
}

/** Scale by `factor` keeping the drawing point under (cx, cy) fixed; the scale stays in [min, max]. */
export function zoomAbout(
  t: Transform,
  factor: number,
  cx: number,
  cy: number,
  min: number,
  max: number,
): Transform {
  const scale = Math.min(max, Math.max(min, t.scale * factor));
  const k = scale / t.scale;
  return { scale, tx: cx - (cx - t.tx) * k, ty: cy - (cy - t.ty) * k };
}

/**
 * Zoom factor of one Ctrl/Cmd+wheel event: exponential in the pixel delta (line and page deltas
 * are converted), capped at 1.25 either way so one mouse notch is about a quarter step while a
 * trackpad pinch, which sends many small deltas, stays smooth.
 */
export function wheelZoomFactor(deltaY: number, deltaMode = 0): number {
  const px = deltaY * (deltaMode === 1 ? 16 : deltaMode === 2 ? 400 : 1);
  return Math.min(1.25, Math.max(1 / 1.25, Math.exp(-px * 0.01)));
}

type Pt = readonly [number, number];

/** The transform of a two-finger gesture: pinch distance zooms, the midpoint's movement pans. */
export function pinchView(
  start: Transform,
  a0: Pt,
  b0: Pt,
  a1: Pt,
  b1: Pt,
  min: number,
  max: number,
): Transform {
  const d0 = Math.max(1, Math.hypot(b0[0] - a0[0], b0[1] - a0[1]));
  const d1 = Math.hypot(b1[0] - a1[0], b1[1] - a1[1]);
  const m0 = [(a0[0] + b0[0]) / 2, (a0[1] + b0[1]) / 2] as const;
  const m1 = [(a1[0] + b1[0]) / 2, (a1[1] + b1[1]) / 2] as const;
  const z = zoomAbout(start, d1 / d0, m0[0], m0[1], min, max);
  return { ...z, tx: z.tx + m1[0] - m0[0], ty: z.ty + m1[1] - m0[1] };
}

/**
 * Where the neuron view starts: the fit transform, but on a coarse pointer never below the scale
 * that makes unit cells 32 px, left-aligned and vertically centred so the user pans to the rest.
 */
export function neuronBaseView(
  fit: Transform,
  size: Size,
  view: Box,
  top: number,
  coarse: boolean,
  pitch: number,
): Transform {
  const floor = touchScaleFloor(pitch);
  if (!coarse || fit.scale >= floor) return fit;
  return clampView({ scale: floor, tx: 8, ty: fit.ty }, size, view, top);
}
