import { useEffect, useRef, type RefObject } from 'react';
import { readColors, type PlotColors } from './color.js';
import { THEME_EVENT } from '../themePref.js';

export type DrawFn = (ctx: CanvasRenderingContext2D, w: number, h: number, c: PlotColors) => void;

/**
 * Resizes a canvas backing store to its CSS size times devicePixelRatio. Only touches the canvas
 * when the size actually changed (assigning width clears it).
 */
export function fitCanvas(
  canvas: HTMLCanvasElement,
  cssW: number,
  cssH: number,
  dpr: number,
): void {
  const w = Math.max(1, Math.round(cssW * dpr));
  const h = Math.max(1, Math.round(cssH * dpr));
  if (canvas.width !== w) canvas.width = w;
  if (canvas.height !== h) canvas.height = h;
}

/**
 * A canvas that redraws at most once per animation frame. `draw` always sees the latest closure;
 * `deps` say when the picture changed. Size comes from a ResizeObserver, colours are re-read when
 * the colour scheme changes. Returns the ref for the <canvas>, which must have a CSS size.
 */
export function usePlotCanvas(
  draw: DrawFn,
  deps: readonly unknown[],
): RefObject<HTMLCanvasElement | null> {
  const ref = useRef<HTMLCanvasElement | null>(null);
  const drawRef = useRef<DrawFn>(draw);
  drawRef.current = draw;
  const state = useRef({ raf: 0, colors: null as PlotColors | null, w: 0, h: 0 });
  const requestRef = useRef<() => void>(() => {});

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const st = state.current;
    const paint = () => {
      st.raf = 0;
      const ctx = canvas.getContext('2d');
      if (!ctx || st.w === 0 || st.h === 0) return;
      fitCanvas(canvas, st.w, st.h, window.devicePixelRatio || 1);
      st.colors ??= readColors(canvas);
      ctx.setTransform(canvas.width / st.w, 0, 0, canvas.height / st.h, 0, 0);
      ctx.clearRect(0, 0, st.w, st.h);
      drawRef.current(ctx, st.w, st.h, st.colors);
    };
    const request = () => {
      if (st.raf === 0) st.raf = requestAnimationFrame(paint);
    };
    requestRef.current = request;
    const ro = new ResizeObserver((entries) => {
      const r = entries[0]?.contentRect;
      if (!r) return;
      st.w = r.width;
      st.h = r.height;
      request();
    });
    ro.observe(canvas);
    const mq = window.matchMedia('(prefers-color-scheme: dark)');
    const onTheme = () => {
      st.colors = null;
      request();
    };
    mq.addEventListener('change', onTheme);
    window.addEventListener(THEME_EVENT, onTheme);
    return () => {
      ro.disconnect();
      mq.removeEventListener('change', onTheme);
      window.removeEventListener(THEME_EVENT, onTheme);
      if (st.raf) cancelAnimationFrame(st.raf);
      st.raf = 0;
      requestRef.current = () => {};
    };
  }, []);

  useEffect(() => {
    requestRef.current();
  }, deps);

  return ref;
}
