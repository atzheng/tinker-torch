import { useEffect, useMemo, useRef, useState } from 'react';
import { linearScale, niceTicks } from '../plot/axes.js';
import { usePlotCanvas, type DrawFn } from '../plot/canvas.js';
import { AxisLabels } from '../plot/AxisLabels.js';
import { MARGINS, drawAxes, drawLegend } from '../plot/draw.js';
import {
  SURFACE_N,
  appendPath,
  axisLabels,
  contourLevels,
  contourSegments,
  heat,
  linspace,
  lossScale,
  nextRange,
  pairLabels,
  resolvePair,
  heldKey,
  shownSurfaceParams,
  surfaceDue,
  SURFACE_EVERY_MS,
  type Pt,
  type SurfaceAxis,
  type SurfaceRange,
} from '../plot/lossSurface.js';
import { useUiLabel } from '../state/disclosure.js';
import { useStore } from '../state/store.js';
import type { CurveParam } from '../plot/lossCurve1d.js';
import styles from './Results.module.css';

/**
 * Loss surface: contours of the training loss over the model's two parameters, with the path of
 * the parameters so far. Rendered when the model has two or more parameters (SPEC "Result
 * views"); otherwise nothing, and the results region keeps its two panels.
 */
export function LossSurface() {
  const host = useStore((s) => s.host);
  const order = useStore((s) => s.analysis.order);
  const ui = useStore((s) => s.show?.ui);
  const params = useMemo(
    () => (host ? shownSurfaceParams(host, order, ui) : null),
    [host, order, ui],
  );
  if (!host || !params) return null;
  return <Surface params={params} hostKey={host} />;
}

/** Any CSS colour to [r, g, b]: draw one pixel and read it back. */
function rgbOf(color: string): [number, number, number] {
  const cv = document.createElement('canvas');
  cv.width = 1;
  cv.height = 1;
  const g = cv.getContext('2d', { willReadFrequently: true });
  if (!g) return [0, 0, 0];
  g.fillStyle = color;
  g.fillRect(0, 0, 1, 1);
  const d = g.getImageData(0, 0, 1, 1).data;
  return [d[0] ?? 0, d[1] ?? 0, d[2] ?? 0];
}

type PathState = { key: object | null; pair: string; path: Pt[]; range: SurfaceRange | null };

function Surface({ params, hostKey }: { params: CurveParam[]; hostKey: object }) {
  // Two pickers choose the pair when the model has more than two parameters (tinker-qzd); the
  // others are held at their current values by the engine. With exactly two there is no picker.
  const [pick, setPick] = useState<[string, string] | null>(null);
  const pair = resolvePair(params, pick) as [CurveParam, CurveParam]; // params has two or more
  const axes: [SurfaceAxis, SurfaceAxis] = [pair[0].axis, pair[1].axis];
  const point: [number, number] = [pair[0].value, pair[1].value];
  const pairKey = `${pair[0].key}|${pair[1].key}`;
  const showPickers = params.length > 2;
  const dataset = useStore((s) => s.dataset);
  const graph = useStore((s) => s.graph);
  const step = useStore((s) => s.step);
  const canTrain = useStore((s) => s.canTrain);
  const diverged = useStore((s) => s.diverged);
  const lossSurface = useStore((s) => s.lossSurface);
  const running = useStore((s) => s.running);

  // The held parameters (all but the pair) move while training or when edited by hand: the grid
  // is re-evaluated when they change, at once when not running and at most every
  // SURFACE_EVERY_MS while running (with a trailing update so it settles).
  const held = showPickers ? heldKey(params, [pair[0].key, pair[1].key]) : '';
  const ev = useRef({ key: '', time: 0 });
  const [, setTick] = useState(0);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  if (ev.current.key !== held) {
    const now = performance.now();
    if (ev.current.time === 0 || surfaceDue(now, ev.current.time, running)) {
      ev.current = { key: held, time: now };
    }
  }
  const evalKey = ev.current.key;
  useEffect(() => {
    if (evalKey === held || timer.current) return;
    timer.current = setTimeout(() => {
      timer.current = null;
      setTick((t) => t + 1);
    }, SURFACE_EVERY_MS);
  });
  useEffect(
    () => () => {
      if (timer.current) clearTimeout(timer.current);
    },
    [],
  );

  // The path and the range live in a ref; they change once per published host (not per redraw).
  // A new run (step 0: Reset, structural edit, hand-set weights) starts a new path and range.
  const st = useRef<PathState>({ key: null, pair: '', path: [], range: null });
  if (st.current.key !== hostKey || st.current.pair !== pairKey) {
    const p = st.current;
    p.key = hostKey;
    const newPair = p.pair !== pairKey;
    p.pair = pairKey;
    if (step === 0 || newPair) {
      p.path = [point];
      p.range = null;
    } else {
      p.path = appendPath(p.path, point);
    }
    p.range = nextRange(p.range, p.path);
  }
  const { path, range } = st.current;

  // The grid is recomputed when the graph, the data or the range changes, never per frame.
  const surface = useMemo(() => {
    if (!canTrain || !range) return null;
    const a = linspace(range.a[0], range.a[1], SURFACE_N);
    const b = linspace(range.b[0], range.b[1], SURFACE_N);
    const grid = lossSurface(axes, a, b);
    if (!grid) return null;
    const scale = lossScale(grid);
    if (!scale) return null;
    const contours = contourLevels(scale).map((lv) =>
      contourSegments(grid, SURFACE_N, SURFACE_N, lv),
    );
    return { grid, scale, contours };
    // `axes` only changes with the graph; the array is rebuilt on every host, so it is not a dep
  }, [range, pairKey, evalKey, dataset, graph, canTrain, lossSurface]);

  const [labelA, labelB] = showPickers ? pairLabels(pair) : axisLabels(axes);

  const draw: DrawFn = (ctx, w, h, c) => {
    if (!range) return;
    const m = MARGINS;
    const xs = linearScale(range.a[0], range.a[1], m.l, w - m.r);
    const ys = linearScale(range.b[0], range.b[1], h - m.b, m.t);
    const n = SURFACE_N;
    ctx.save();
    ctx.beginPath();
    ctx.rect(m.l, m.t, w - m.l - m.r, h - m.t - m.b);
    ctx.clip();
    if (surface) {
      const { grid, scale, contours } = surface;
      const cw = (w - m.l - m.r) / (n - 1);
      const ch = (h - m.t - m.b) / (n - 1);
      // one pixel per grid point in an offscreen canvas, drawn scaled up (smooth, no seams)
      const rgb = rgbOf(c.accent);
      const img = new ImageData(n, n);
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
          const t = heat(grid[i * n + j] ?? Number.NaN, scale);
          if (Number.isNaN(t)) continue;
          const o = ((n - 1 - j) * n + i) * 4;
          img.data[o] = rgb[0] ?? 0;
          img.data[o + 1] = rgb[1] ?? 0;
          img.data[o + 2] = rgb[2] ?? 0;
          img.data[o + 3] = Math.round(255 * (0.08 + 0.6 * (1 - t)));
        }
      }
      const off = document.createElement('canvas');
      off.width = n;
      off.height = n;
      off.getContext('2d')?.putImageData(img, 0, 0);
      ctx.imageSmoothingEnabled = true;
      ctx.drawImage(off, m.l - cw / 2, m.t - ch / 2, n * cw, n * ch);
      ctx.globalAlpha = 0.9;
      ctx.strokeStyle = c.muted;
      ctx.lineWidth = 1;
      ctx.beginPath();
      const px = (i: number) => m.l + i * cw;
      const py = (j: number) => m.t + (n - 1 - j) * ch;
      for (const seg of contours) {
        for (let k = 0; k + 3 < seg.length; k += 4) {
          ctx.moveTo(px(seg[k] ?? 0), py(seg[k + 1] ?? 0));
          ctx.lineTo(px(seg[k + 2] ?? 0), py(seg[k + 3] ?? 0));
        }
      }
      ctx.stroke();
      ctx.globalAlpha = 1;
    }
    // path and current point
    ctx.lineJoin = 'round';
    ctx.lineWidth = 2;
    ctx.strokeStyle = c.val;
    ctx.beginPath();
    path.forEach((p, i) => {
      if (i === 0) ctx.moveTo(xs.px(p[0]), ys.px(p[1]));
      else ctx.lineTo(xs.px(p[0]), ys.px(p[1]));
    });
    ctx.stroke();
    const first = path[0];
    if (first && path.length > 1) {
      ctx.fillStyle = c.bg;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.arc(xs.px(first[0]), ys.px(first[1]), 3.5, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
    }
    ctx.fillStyle = c.val;
    ctx.strokeStyle = c.bg;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(xs.px(point[0]), ys.px(point[1]), 5, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
    ctx.restore();
    drawAxes(
      ctx,
      c,
      w,
      h,
      m,
      xs,
      ys,
      niceTicks(range.a[0], range.a[1], 5),
      niceTicks(range.b[0], range.b[1], 4),
    );
    drawLegend(ctx, c, m, [
      { label: 'path', color: c.val, kind: 'line' },
      { label: 'now', color: c.val, kind: 'dot' },
    ]);
  };

  const ref = usePlotCanvas(draw, [surface, path, range, hostKey]);
  const panelTitle = useUiLabel('panel-surface', 'Loss surface');
  return (
    <div className={styles.panel} data-testid="loss-surface">
      <div className={styles.head} data-panel-head="">
        <span className={styles.title}>{panelTitle}</span>
        {diverged ? <span data-testid="surface-diverged">Diverged</span> : null}
        {showPickers && (
          <>
            {([0, 1] as const).map((k) => (
              <select
                key={k}
                className={styles.picker}
                value={pair[k].key}
                onChange={(e) => {
                  const next: [string, string] = [pair[0].key, pair[1].key];
                  next[k] = e.target.value;
                  setPick(next);
                }}
                data-testid={k === 0 ? 'surface-axis-x' : 'surface-axis-y'}
                aria-label={
                  k === 0 ? 'Parameter on the horizontal axis' : 'Parameter on the vertical axis'
                }
              >
                {params.map((p) => (
                  <option key={p.key} value={p.key}>
                    {p.label.replace(/\$/g, '').replace(/[{}]/g, '')}
                  </option>
                ))}
              </select>
            ))}
          </>
        )}
      </div>
      <div className={styles.plot}>
        <canvas ref={ref} aria-label="Loss surface of the two parameters" />
        <AxisLabels x={labelA} y={labelB} />
      </div>
    </div>
  );
}
