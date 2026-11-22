import { useUiShown } from '../state/disclosure.js';
import { useMemo, useState } from 'react';
import { grid1d, hasTruth, truth, unstandardizeRows } from '@tinker/core';
import { fitYRange, linearScale, niceTicks } from '../plot/axes.js';
import { usePlotCanvas, type DrawFn } from '../plot/canvas.js';
import { AxisLabels } from '../plot/AxisLabels.js';
import { MARGINS, drawAxes, drawLegend, type LegendItem } from '../plot/draw.js';
import { classColor } from '../plot/color.js';
import { useStore } from '../state/store.js';
import { axisLabel, featureName } from '../state/names.js';
import { FitPlot2D } from './FitPlot2D.js';
import {
  ResampleButton,
  ResampleReadout,
  drawOverlay1D,
  overlayLegend,
  useOverlay1D,
} from './ResampleOverlay.js';
import { PredictionPicker } from './PredictionPicker.js';
import styles from './Results.module.css';

/** Number of grid points for the model curve. Fixed: every new n makes the engine retrace. */
export const GRID_N = 200;

/** Fit plot: 1-D data with the model curve. 2-D data goes to FitPlot2D. */
export function FitPlot() {
  const d = useStore((s) => s.dataset.info.columns.length);
  return d === 1 ? <FitPlot1D /> : <FitPlot2D />;
}

function FitPlot1D() {
  const dataset = useStore((s) => s.dataset);
  const set = useStore((s) => s.graph.data.set);
  const host = useStore((s) => s.host);
  const predictGrid = useStore((s) => s.predictGrid);
  const noPrediction = useStore((s) => s.graph.predictions.length === 0);
  const pick = useStore((s) => s.plotPrediction);
  const predictions = useStore((s) => s.graph.predictions);
  const [showTruth, setShowTruth] = useState(false);
  const overlay = useOverlay1D();
  const pins = useStore((s) => s.pins);

  // Per dataset: raw-unit points, the grid and the true function on it.
  const fixed = useMemo(() => {
    const st = dataset.standardization;
    const xt = st ? unstandardizeRows(dataset.xTrain, 1, st) : dataset.xTrain;
    const xv = st ? unstandardizeRows(dataset.xVal, 1, st) : dataset.xVal;
    const grid = grid1d(dataset.domain, GRID_N, st);
    return { xt, xv, grid, f: hasTruth(set) ? truth(set, grid.xRaw) : new Float32Array(grid.n) };
  }, [dataset, set]);

  // The model on the grid; recomputed when the parameters (host) or the dataset change.
  // Classification (1 input): binary shows P(class 1); multi-class one probability curve per class.
  const kind = dataset.info.kind;
  const numClasses = dataset.info.numClasses;
  const classNames = dataset.info.classNames ?? [];
  const classCurves = useMemo(() => {
    if (!host || kind !== 'multiclass') return null;
    const out = predictGrid(fixed.grid.xStd, fixed.grid.n);
    if (!out || out.length !== fixed.grid.n * numClasses) return null;
    return Array.from({ length: numClasses }, (_, c) => {
      const y = new Float32Array(fixed.grid.n);
      for (let i = 0; i < y.length; i++) y[i] = out[i * numClasses + c] ?? Number.NaN;
      return y;
    });
  }, [host, predictGrid, fixed, pick, predictions, kind, numClasses]);
  const curve = useMemo(() => {
    if (!host || kind === 'multiclass') return null;
    const out = predictGrid(fixed.grid.xStd, fixed.grid.n);
    if (!out) return null;
    const width = Math.max(1, Math.round(out.length / fixed.grid.n));
    // a binary model with two outputs: P(class 1) is the second
    const col = kind === 'binary' && width === 2 ? 1 : 0;
    const y = new Float32Array(fixed.grid.n);
    for (let i = 0; i < y.length; i++) y[i] = out[i * width + col] ?? Number.NaN;
    return y;
  }, [host, predictGrid, fixed, pick, predictions, kind]);
  // Multi-class points have no height of their own: they sit in a rug along the bottom, one row per class.
  const pointY = (y: number) => (kind === 'multiclass' ? 0.03 + 0.05 * Math.round(y) : y);

  const draw: DrawFn = (ctx, w, h, c) => {
    const m = MARGINS;
    const [xlo, xhi] = dataset.domain[0] ?? [0, 1];
    // y range from the points (and the truth when shown); the curve may extend it a little
    let lo = Infinity;
    let hi = -Infinity;
    const see = (v: number) => {
      if (!Number.isFinite(v)) return;
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    };
    for (let i = 0; i < dataset.nTrain; i++) see(pointY(dataset.yTrain[i] ?? Number.NaN));
    for (let i = 0; i < dataset.nVal; i++) see(pointY(dataset.yVal[i] ?? Number.NaN));
    if (showTruth) for (let i = 0; i < fixed.f.length; i++) see(fixed.f[i] ?? Number.NaN);
    if (overlay) for (let i = 0; i < overlay.mean.length; i++) see(overlay.mean[i] ?? Number.NaN);
    if (!(hi >= lo)) [lo, hi] = [-1, 1];
    const [ylo, yhi] =
      kind === 'regression'
        ? fitYRange(lo, hi, curve)
        : ([Math.min(-0.05, lo - 0.03), Math.max(1.05, hi + 0.03)] as [number, number]);
    const xs = linearScale(xlo, xhi, m.l, w - m.r);
    const ys = linearScale(ylo, yhi, h - m.b, m.t);
    drawAxes(ctx, c, w, h, m, xs, ys, niceTicks(xlo, xhi, 6), niceTicks(ylo, yhi, 4));
    const legend: LegendItem[] = [
      { label: 'train', color: c.train, kind: 'dot' },
      { label: 'val', color: c.val, kind: 'ring' },
      { label: 'model', color: c.text, kind: 'line' },
    ];
    if (classCurves) {
      legend.length = 0;
      classCurves.forEach((_, i) =>
        legend.push({
          label: classNames[i] ?? `class ${i}`,
          color: classColor(c, i),
          kind: 'line',
        }),
      );
    }
    if (showTruth) legend.push({ label: 'truth', color: c.truth, kind: 'dash' });
    if (overlay) legend.push(...overlayLegend(c));
    drawLegend(ctx, c, m, legend);

    ctx.save();
    ctx.beginPath();
    ctx.rect(m.l, m.t, w - m.l - m.r, h - m.t - m.b);
    ctx.clip();
    const line = (y: Float32Array) => {
      ctx.beginPath();
      let started = false;
      for (let i = 0; i < y.length; i++) {
        const v = y[i];
        if (v === undefined || !Number.isFinite(v)) {
          started = false;
          continue;
        }
        const px = xs.px(fixed.grid.xRaw[i] ?? 0);
        if (started) ctx.lineTo(px, ys.px(v));
        else ctx.moveTo(px, ys.px(v));
        started = true;
      }
      ctx.stroke();
    };
    if (showTruth) {
      ctx.strokeStyle = c.truth;
      ctx.lineWidth = 2;
      ctx.setLineDash([5, 4]);
      line(fixed.f);
      ctx.setLineDash([]);
    }
    // Pinned fits (tinker-iej.2) under the live curve, in the pin's colour.
    for (const p of pins) {
      if (!p.fit) continue;
      ctx.strokeStyle = c.pins[p.slot] ?? c.accent;
      ctx.lineWidth = 1.25;
      ctx.lineJoin = 'round';
      ctx.beginPath();
      let started = false;
      for (let i = 0; i < p.fit.y.length; i++) {
        const v = p.fit.y[i];
        if (v === undefined || !Number.isFinite(v)) {
          started = false;
          continue;
        }
        const px = xs.px(p.fit.x[i] ?? 0);
        if (started) ctx.lineTo(px, ys.px(v));
        else ctx.moveTo(px, ys.px(v));
        started = true;
      }
      ctx.stroke();
    }
    if (curve) {
      ctx.strokeStyle = c.text;
      ctx.lineWidth = 2;
      ctx.lineJoin = 'round';
      line(curve);
    }
    if (classCurves) {
      ctx.lineWidth = 2;
      ctx.lineJoin = 'round';
      classCurves.forEach((y, i) => {
        ctx.strokeStyle = classColor(c, i);
        line(y);
      });
    }
    // validation: hollow rings; training: filled dots
    ctx.strokeStyle = c.val;
    ctx.lineWidth = 1.8;
    ctx.beginPath();
    for (let i = 0; i < dataset.nVal; i++) {
      const px = xs.px(fixed.xv[i] ?? 0);
      const py = ys.px(pointY(dataset.yVal[i] ?? 0));
      if (kind === 'multiclass') {
        ctx.beginPath();
        ctx.strokeStyle = classColor(c, Math.round(dataset.yVal[i] ?? 0));
        ctx.arc(px, py, 4, 0, Math.PI * 2);
        ctx.stroke();
        continue;
      }
      ctx.moveTo(px + 4, py);
      ctx.arc(px, py, 4, 0, Math.PI * 2);
    }
    ctx.stroke();
    ctx.fillStyle = c.train;
    ctx.beginPath();
    for (let i = 0; i < dataset.nTrain; i++) {
      const px = xs.px(fixed.xt[i] ?? 0);
      const py = ys.px(pointY(dataset.yTrain[i] ?? 0));
      if (kind === 'multiclass') {
        ctx.fillStyle = classColor(c, Math.round(dataset.yTrain[i] ?? 0));
        ctx.beginPath();
        ctx.arc(px, py, 3.5, 0, Math.PI * 2);
        ctx.fill();
        continue;
      }
      ctx.moveTo(px + 3.5, py);
      ctx.arc(px, py, 3.5, 0, Math.PI * 2);
    }
    ctx.fill();
    // over the points: at large n they would hide the copies
    if (overlay) drawOverlay1D(ctx, c, overlay, fixed.grid.xRaw, xs, ys);
    ctx.restore();
  };

  const ref = usePlotCanvas(draw, [dataset, fixed, curve, classCurves, showTruth, overlay, pins]);
  const showTruthToggle = useUiShown('truth-toggle');
  const showResample = useUiShown('resample-run');
  const showPicker = useUiShown('prediction-picker');
  return (
    <div className={styles.panel} data-testid="fit-plot">
      <div className={styles.head} data-panel-head="">
        <span className={styles.title}>Fit</span>
        {showTruthToggle && hasTruth(set) && (
          <label className={styles.toggle}>
            <input
              type="checkbox"
              checked={showTruth}
              onChange={(e) => setShowTruth(e.target.checked)}
              data-testid="truth-toggle"
            />
            True function
          </label>
        )}
        {showPicker && <PredictionPicker />}
        {showResample && <ResampleButton />}
      </div>
      {showResample && <ResampleReadout />}
      {noPrediction && (
        <p className={`${styles.readout} tk-diag-label`} data-testid="no-prediction">
          No prediction selected.
        </p>
      )}
      <div className={styles.plot}>
        <canvas ref={ref} aria-label="Fit plot" />
        <AxisLabels
          x={dataset.info.featureNames ? axisLabel(featureName(dataset.info, 'x1')) : '$x$'}
          y={
            kind !== 'regression'
              ? '$P$'
              : dataset.info.targetName && dataset.info.targetName !== 'y'
                ? axisLabel(dataset.info.targetName)
                : '$y$'
          }
        />
      </div>
    </div>
  );
}
