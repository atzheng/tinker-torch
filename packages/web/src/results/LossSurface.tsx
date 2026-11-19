import { useMemo, useRef } from 'react';
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
  twoParams,
  type Pt,
  type SurfaceAxis,
  type SurfaceRange,
} from '../plot/lossSurface.js';
import { useStore } from '../state/store.js';
import styles from './Results.module.css';

/**
 * Loss surface: contours of the training loss over the model's two parameters, with the path of
 * the parameters so far. Rendered only when the model has exactly two parameters (SPEC "Result
 * views"); otherwise nothing, and the results region keeps its two panels.
 */
export function LossSurface() {
  const host = useStore((s) => s.host);
  const order = useStore((s) => s.analysis.order);
  const info = host ? twoParams(host, order) : null;
  if (!host || !info) return null;
  return <Surface axes={info.axes} point={info.values} hostKey={host} />;
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

type PathState = { key: object | null; path: Pt[]; range: SurfaceRange | null };

function Surface({
  axes,
  point,
  hostKey,
}: {
  axes: [SurfaceAxis, SurfaceAxis];
  point: [number, number];
  hostKey: object;
}) {
  const dataset = useStore((s) => s.dataset);
  const graph = useStore((s) => s.graph);
  const step = useStore((s) => s.step);
  const canTrain = useStore((s) => s.canTrain);
  const diverged = useStore((s) => s.diverged);
  const lossSurface = useStore((s) => s.lossSurface);

  // The path and the range live in a ref; they change once per published host (not per redraw).
  // A new run (step 0: Reset, structural edit, hand-set weights) starts a new path and range.
  const st = useRef<PathState>({ key: null, path: [], range: null });
  if (st.current.key !== hostKey) {
    const p = st.current;
    p.key = hostKey;
    if (step === 0) {
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
  }, [range, dataset, graph, canTrain, lossSurface]);

  const [labelA, labelB] = axisLabels(axes);

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
  return (
    <div className={styles.panel} data-testid="loss-surface">
      <div className={styles.head} data-panel-head="">
        <span className={styles.title}>Loss surface</span>
        {diverged ? <span data-testid="surface-diverged">Diverged</span> : null}
      </div>
      <div className={styles.plot}>
        <canvas ref={ref} aria-label="Loss surface of the two parameters" />
        <AxisLabels x={labelA} y={labelB} />
      </div>
    </div>
  );
}
