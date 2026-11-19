import { useMemo, useState } from 'react';
import { grid2d, truth, truthMargin, unstandardizeRows } from '@tinker/core';
import { CLASS_COUNT, classColor, type PlotColors } from '../plot/color.js';
import { formatTick, linearScale, niceTicks, paddedRange } from '../plot/axes.js';
import { usePlotCanvas, type DrawFn } from '../plot/canvas.js';
import { AxisLabels } from '../plot/AxisLabels.js';
import { MARGINS, drawAxes } from '../plot/draw.js';
import {
  column,
  contourSegments,
  fillGridImage,
  finiteRange,
  indexToDomain,
  makeRamp,
  parseColor,
  rampCss,
  unit,
} from '../plot/surface.js';
import { useStore } from '../state/store.js';
import { ResampleButton, ResampleReadout, useOverlay2D } from './ResampleOverlay.js';
import { PredictionPicker } from './PredictionPicker.js';
import styles from './Results.module.css';

/** Grid points per side of the surface. Fixed: every new n makes the engine retrace. */
export const GRID_2D = 64;

/**
 * Paints the decision regions of a multi-class model: each pixel takes the colour of its most
 * probable class, more opaque where the model is more confident. `probs` is row-major [m*m, k].
 * Same pixel layout as `fillGridImage`.
 */
export function fillClassImage(
  rgba: Uint8ClampedArray,
  probs: ArrayLike<number>,
  m: number,
  k: number,
  c: PlotColors,
): void {
  const rgb = Array.from({ length: k }, (_, i) => parseColor(classColor(c, i)));
  for (let iy = 0; iy < m; iy++) {
    for (let ix = 0; ix < m; ix++) {
      const g = iy * m + ix;
      const o = ((m - 1 - iy) * m + ix) * 4;
      let best = 0;
      let bp = -Infinity;
      for (let j = 0; j < k; j++) {
        const p = probs[g * k + j] ?? Number.NaN;
        if (p > bp) {
          bp = p;
          best = j;
        }
      }
      if (!Number.isFinite(bp)) {
        rgba[o + 3] = 0;
        continue;
      }
      const conf = Math.min(1, Math.max(0, (bp - 1 / k) / (1 - 1 / k)));
      const col = rgb[best] ?? [128, 128, 128];
      rgba[o] = col[0];
      rgba[o + 1] = col[1];
      rgba[o + 2] = col[2];
      rgba[o + 3] = Math.round(255 * (0.18 + 0.5 * conf));
    }
  }
}

/** Fit plot for 2-D data: decision surface (classification) or prediction heat map (regression). */
export function FitPlot2D() {
  const dataset = useStore((s) => s.dataset);
  const set = useStore((s) => s.graph.data.set);
  const host = useStore((s) => s.host);
  const predictGrid = useStore((s) => s.predictGrid);
  const noPrediction = useStore((s) => s.graph.predictions.length === 0);
  const pick = useStore((s) => s.plotPrediction);
  const predictions = useStore((s) => s.graph.predictions);
  const [showTruth, setShowTruth] = useState(false);
  const classification = dataset.info.kind !== 'regression';
  const multiclass = dataset.info.kind === 'multiclass';
  const numClasses = dataset.info.numClasses;
  const classNames =
    dataset.info.classNames ?? Array.from({ length: numClasses }, (_, i) => `class ${i}`);

  // Per dataset: raw-unit points, the grid, the true function on it, the colour range, buffers.
  const fixed = useMemo(() => {
    const st = dataset.standardization;
    const xt = st ? unstandardizeRows(dataset.xTrain, 2, st) : dataset.xTrain;
    const xv = st ? unstandardizeRows(dataset.xVal, 2, st) : dataset.xVal;
    const grid = grid2d(dataset.domain, GRID_2D, st);
    const f = truth(set, grid.xRaw);
    // smooth signed field for the classification boundary (zero level), not the 0/1 truth
    const margin = truthMargin(set, grid.xRaw);
    const range = classification
      ? ([0, 1] as [number, number])
      : (() => {
          const r = finiteRange(dataset.yTrain, dataset.yVal) ?? ([-1, 1] as [number, number]);
          return paddedRange(r[0], r[1], 0.05);
        })();
    const off = typeof document === 'undefined' ? null : document.createElement('canvas');
    if (off) {
      off.width = GRID_2D;
      off.height = GRID_2D;
    }
    return {
      xt,
      xv,
      grid,
      f,
      margin,
      range,
      off,
      img: new ImageData(GRID_2D, GRID_2D),
      values: new Float32Array(GRID_2D * GRID_2D),
    };
  }, [dataset, set, classification]);

  // The model on the grid, recomputed when the parameters (host) or the dataset change.
  const surface = useMemo(() => {
    if (!host) return null;
    const out = predictGrid(fixed.grid.xStd, fixed.grid.n);
    if (!out) return null;
    // multi-class models: keep all k class probabilities, row-major [n, k]
    if (multiclass) return out.length === fixed.grid.n * numClasses ? out : null;
    // binary models with two outputs: take P(class 1)
    return column(fixed.values, out, fixed.grid.n, 1);
  }, [host, predictGrid, fixed, pick, predictions, multiclass, numClasses]);

  // With the overlay on, the surface drawn is the average of the copies.
  const overlay = useOverlay2D();
  const shown = overlay?.mean ?? surface;

  // Multi-class: per-class margin p_c - max of the others (zero on the region boundaries).
  const margins = useMemo(() => {
    if (!multiclass || !shown) return null;
    const n = fixed.grid.n;
    return Array.from({ length: numClasses }, (_, c) => {
      const m = new Float32Array(n);
      for (let i = 0; i < n; i++) {
        let other = -Infinity;
        for (let j = 0; j < numClasses; j++)
          if (j !== c) other = Math.max(other, shown[i * numClasses + j] ?? Number.NaN);
        m[i] = (shown[i * numClasses + c] ?? Number.NaN) - other;
      }
      return m;
    });
  }, [multiclass, shown, fixed, numClasses]);

  // The true class regions as one 0/1 field per class (boundary at 0.5).
  const truthRegions = useMemo(
    () =>
      multiclass
        ? Array.from({ length: numClasses }, (_, c) => fixed.f.map((v) => (v === c ? 1 : 0)))
        : null,
    [multiclass, fixed, numClasses],
  );

  const mapped = useMemo(
    () => (classification ? null : finiteRange(fixed.f)),
    [classification, fixed],
  );

  const draw: DrawFn = (ctx, w, h, c) => {
    const m = MARGINS;
    const [x1lo, x1hi] = dataset.domain[0] ?? [0, 1];
    const [x2lo, x2hi] = dataset.domain[1] ?? [0, 1];
    const xs = linearScale(x1lo, x1hi, m.l, w - m.r);
    const ys = linearScale(x2lo, x2hi, h - m.b, m.t);
    drawAxes(ctx, c, w, h, m, xs, ys, niceTicks(x1lo, x1hi, 6), niceTicks(x2lo, x2hi, 4));
    const lut = classification ? makeRamp([c.divNeg, c.divMid, c.divPos]) : makeRamp(c.seq);
    const [rlo, rhi] = fixed.range;

    ctx.save();
    ctx.beginPath();
    ctx.rect(m.l, m.t, w - m.l - m.r, h - m.t - m.b);
    ctx.clip();

    // surface: one pixel per grid point, smoothed up to the plot size
    if (shown && fixed.off) {
      const octx = fixed.off.getContext('2d');
      if (octx) {
        if (multiclass) fillClassImage(fixed.img.data, shown, GRID_2D, numClasses, c);
        else fillGridImage(fixed.img.data, shown, GRID_2D, lut, rlo, rhi);
        octx.putImageData(fixed.img, 0, 0);
        const cw = (xs.px(x1hi) - xs.px(x1lo)) / (GRID_2D - 1);
        const ch = (ys.px(x2lo) - ys.px(x2hi)) / (GRID_2D - 1);
        ctx.imageSmoothingEnabled = true;
        ctx.drawImage(
          fixed.off,
          xs.px(x1lo) - cw / 2,
          ys.px(x2hi) - ch / 2,
          cw * GRID_2D,
          ch * GRID_2D,
        );
      }
    }

    const toPx = (gx: number, gy: number): [number, number] => [
      xs.px(indexToDomain(gx, GRID_2D, x1lo, x1hi)),
      ys.px(indexToDomain(gy, GRID_2D, x2lo, x2hi)),
    ];
    const lines = (values: ArrayLike<number>, level: number) => {
      const seg = contourSegments(values, GRID_2D, level);
      ctx.beginPath();
      for (let i = 0; i < seg.length; i += 4) {
        const [ax, ay] = toPx(seg[i] ?? 0, seg[i + 1] ?? 0);
        const [bx, by] = toPx(seg[i + 2] ?? 0, seg[i + 3] ?? 0);
        ctx.moveTo(ax, ay);
        ctx.lineTo(bx, by);
      }
      ctx.stroke();
    };
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';
    // the model's 0.5 boundary: halo then line, readable on both ends of the ramp
    if (multiclass && margins) {
      ctx.setLineDash([]);
      ctx.strokeStyle = c.bg;
      ctx.lineWidth = 4;
      for (const m of margins) lines(m, 0);
      ctx.strokeStyle = c.text;
      ctx.lineWidth = 2;
      for (const m of margins) lines(m, 0);
    } else if (classification && shown) {
      ctx.setLineDash([]);
      ctx.strokeStyle = c.bg;
      ctx.lineWidth = 4;
      lines(shown, 0.5);
      ctx.strokeStyle = c.text;
      ctx.lineWidth = 2;
      lines(shown, 0.5);
    }
    if (showTruth) {
      ctx.setLineDash([6, 4]);
      if (multiclass && truthRegions) {
        ctx.strokeStyle = c.bg;
        ctx.lineWidth = 3.5;
        ctx.setLineDash([]);
        for (const f of truthRegions) lines(f, 0.5);
        ctx.setLineDash([6, 4]);
        ctx.strokeStyle = c.truth;
        ctx.lineWidth = 2;
        for (const f of truthRegions) lines(f, 0.5);
      } else if (classification) {
        ctx.strokeStyle = c.bg;
        ctx.lineWidth = 3.5;
        ctx.setLineDash([]);
        lines(fixed.margin, 0);
        ctx.setLineDash([6, 4]);
        ctx.strokeStyle = c.truth;
        ctx.lineWidth = 2;
        lines(fixed.margin, 0);
      } else if (mapped) {
        const [flo, fhi] = mapped;
        for (const level of niceTicks(flo, fhi, 6)) {
          ctx.strokeStyle = c.bg;
          ctx.lineWidth = 2.5;
          ctx.setLineDash([]);
          lines(fixed.f, level);
          ctx.setLineDash([5, 4]);
          ctx.strokeStyle = c.text;
          ctx.lineWidth = 1.2;
          lines(fixed.f, level);
        }
      }
      ctx.setLineDash([]);
    }

    // points: filled = training, hollow = validation; same colours as the surface
    const colourOf = (y: number) =>
      multiclass
        ? classColor(c, Math.round(y))
        : classification
          ? y >= 0.5
            ? c.class1
            : c.class0
          : rampCss(lut, unit(y, rlo, rhi));
    const dots = (x: Float32Array, y: Float32Array, n: number, hollow: boolean) => {
      for (let i = 0; i < n; i++) {
        const px = xs.px(x[i * 2] ?? 0);
        const py = ys.px(x[i * 2 + 1] ?? 0);
        const col = colourOf(y[i] ?? 0);
        ctx.beginPath();
        ctx.arc(px, py, hollow ? 4 : 3.5, 0, Math.PI * 2);
        if (!classification) {
          // halo so the point reads on a region of its own colour
          ctx.strokeStyle = c.halo;
          ctx.lineWidth = hollow ? 5.5 : 4.5;
          ctx.stroke();
        }
        if (hollow) {
          ctx.strokeStyle = c.bg;
          ctx.lineWidth = 4;
          ctx.stroke();
          ctx.strokeStyle = classification ? col : c.text;
          ctx.lineWidth = 1.8;
          ctx.stroke();
        } else {
          ctx.fillStyle = col;
          ctx.fill();
          ctx.strokeStyle = classification ? c.bg : c.text;
          ctx.lineWidth = classification ? 1.3 : 1.6;
          ctx.stroke();
        }
      }
    };
    dots(fixed.xv, dataset.yVal, dataset.nVal, true);
    dots(fixed.xt, dataset.yTrain, dataset.nTrain, false);
    ctx.restore();
    ctx.save();
    ctx.strokeStyle = c.border;
    ctx.lineWidth = 1;
    ctx.strokeRect(m.l + 0.5, m.t + 0.5, w - m.l - m.r, h - m.t - m.b);
    ctx.restore();
  };

  // `surface` is a reused buffer, so `host` (a new object each frame) says that it changed.
  const ref = usePlotCanvas(draw, [
    dataset,
    fixed,
    shown,
    host,
    showTruth,
    mapped,
    margins,
    truthRegions,
  ]);
  return (
    <div className={styles.panel} data-testid="fit-plot">
      <div className={`${styles.head} ${styles.headWrap}`} data-panel-head="">
        <span className={styles.title}>Fit</span>
        {multiclass ? (
          <span className={`${styles.key} tk-diag-label`} data-testid="fit-key">
            {classNames.map((name, i) => (
              <span key={name + i} className={styles.key} data-testid="fit-key-class">
                <i
                  className={styles.swatch}
                  style={{ background: `var(--class-${i % CLASS_COUNT})` }}
                />
                {name}
              </span>
            ))}{' '}
            <span aria-hidden>
              {'\u25CF'} train {'\u25CB'} val
            </span>
          </span>
        ) : classification ? (
          <span className={`${styles.key} tk-diag-label`} data-testid="fit-key">
            <i className={styles.swatch} style={{ background: 'var(--weight-neg)' }} />
            class 0
            <i className={styles.swatch} style={{ background: 'var(--weight-pos)' }} />
            class 1{' '}
            <span aria-hidden>
              {'\u25CF'} train {'\u25CB'} val
            </span>
          </span>
        ) : (
          <span className={`${styles.key} tk-diag-label`} data-testid="fit-key">
            {formatTick(fixed.range[0])}
            <i
              className={styles.bar}
              style={{
                background: 'linear-gradient(90deg, var(--seq-0), var(--seq-1), var(--seq-2))',
              }}
            />
            {formatTick(fixed.range[1])}{' '}
            <span aria-hidden>
              {'\u25CF'} train {'\u25CB'} val
            </span>
          </span>
        )}
        <label className={styles.toggle}>
          <input
            type="checkbox"
            checked={showTruth}
            onChange={(e) => setShowTruth(e.target.checked)}
            data-testid="truth-toggle"
          />
          {classification ? 'True boundary' : 'True contours'}
        </label>
        <PredictionPicker />
        <ResampleButton />
      </div>
      <ResampleReadout />
      {noPrediction && (
        <p className={`${styles.readout} tk-diag-label`} data-testid="no-prediction">
          No prediction selected.
        </p>
      )}
      <div className={styles.plot}>
        <canvas ref={ref} aria-label="Fit plot" />
        <AxisLabels x="$x_1$" y="$x_2$" />
      </div>
    </div>
  );
}
