import { useUiLabel, useUiOption, useUiShown } from '../state/disclosure.js';
import { useMemo } from 'react';
import {
  axisCategoryCount,
  categoryJitter,
  categoryNames,
  categorySeries,
  fitView,
  grid1dSlice,
  gridCategories,
  hasTruth,
  heldValues,
  truth,
  unstandardizeRows,
  usedColumns,
} from '@tinker/core';
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

/**
 * Fit plot. Two or more continuous columns the model reads: FitPlot2D (its axes are continuous columns only).
 * Otherwise the model curve over the one continuous column, or, when every column is categorical,
 * the model's value at each category as a mark over the data (tinker-hxq; `fitView`).
 */
export function FitPlot() {
  const info = useStore((s) => s.dataset.info);
  const graph = useStore((s) => s.graph);
  // only the columns the model's Input / Column pieces read count (tinker-oef)
  const used = useMemo(() => usedColumns(graph, info), [graph, info]);
  const view = useMemo(() => fitView(info, used), [info, used]);
  return view.type === 'surface' ? (
    <FitPlot2D used={used} />
  ) : (
    <FitPlot1D col={view.col} categorical={view.type === 'categories'} />
  );
}

function FitPlot1D({ col, categorical }: { col: number; categorical: boolean }) {
  const dataset = useStore((s) => s.dataset);
  const set = useStore((s) => s.graph.data.set);
  const host = useStore((s) => s.host);
  const predictGrid = useStore((s) => s.predictGrid);
  const noPrediction = useStore((s) => s.graph.predictions.length === 0);
  const pick = useStore((s) => s.plotPrediction);
  const predictions = useStore((s) => s.graph.predictions);
  const [showTruthOpt, setShowTruth] = useUiOption('truth-toggle');
  const showTruth = showTruthOpt && !categorical;
  const overlay0 = useOverlay1D();
  const overlay = categorical ? null : overlay0;
  const pins = useStore((s) => s.pins);

  // Per dataset: raw-unit points, the grid and the true function on it.
  const fixed = useMemo(() => {
    const st = dataset.standardization;
    const d = dataset.info.columns.length;
    const raw = (x: Float32Array, n: number) => {
      const all = st ? unstandardizeRows(x, d, st) : x;
      return Float32Array.from({ length: n }, (_, i) => all[i * d + col] ?? 0);
    };
    const xt = raw(dataset.xTrain, dataset.nTrain);
    const xv = raw(dataset.xVal, dataset.nVal);
    const held = heldValues(dataset);
    // a categorical axis: one grid row per category, not a continuous grid over [0, K-1]
    let maxIndex = 0;
    for (const v of xt) maxIndex = Math.max(maxIndex, v);
    for (const v of xv) maxIndex = Math.max(maxIndex, v);
    const k = categorical ? axisCategoryCount(dataset.info, col, maxIndex) : 0;
    const grid = categorical
      ? gridCategories(held, col, k, st)
      : grid1dSlice(dataset.domain, held, col, GRID_N, st);
    const f = hasTruth(set) && !categorical ? truth(set, grid.xRaw) : new Float32Array(grid.n);
    return { xt, xv, grid, f, k };
  }, [dataset, set, col, categorical]);
  const levels = categoryNames(dataset.info, col);
  const catLabel = (t: number) => levels[Math.round(t)] ?? String(Math.round(t));

  // The model on the grid; recomputed when the parameters (host) or the dataset change.
  // Classification (1 input): binary shows P(class 1); multi-class one probability curve per class.
  const kind = dataset.info.kind;
  const numClasses = dataset.info.numClasses;
  const classNames = dataset.info.classNames ?? [];
  const classCurves = useMemo(() => {
    if (!host || kind !== 'multiclass') return null;
    const out = predictGrid(fixed.grid.xStd, fixed.grid.n);
    if (!out) return null;
    return categorySeries(out, fixed.grid.n, 'multiclass', numClasses);
  }, [host, predictGrid, fixed, pick, predictions, kind, numClasses]);
  const curve = useMemo(() => {
    if (!host || kind === 'multiclass') return null;
    const out = predictGrid(fixed.grid.xStd, fixed.grid.n);
    if (!out) return null;
    // a binary model with two outputs: P(class 1) is the second
    return categorySeries(out, fixed.grid.n, kind, numClasses)?.[0] ?? null;
  }, [host, predictGrid, fixed, pick, predictions, kind]);
  // Multi-class points have no height of their own: they sit in a rug along the bottom, one row per class.
  const pointY = (y: number) => (kind === 'multiclass' ? 0.03 + 0.05 * Math.round(y) : y);

  const draw: DrawFn = (ctx, w, h, c) => {
    const m = MARGINS;
    const [xlo, xhi] = categorical
      ? ([-0.5, fixed.k - 0.5] as [number, number])
      : (dataset.domain[col] ?? [0, 1]);
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
    drawAxes(
      ctx,
      c,
      w,
      h,
      m,
      xs,
      ys,
      categorical ? Array.from({ length: fixed.k }, (_, i) => i) : niceTicks(xlo, xhi, 6),
      niceTicks(ylo, yhi, 4),
      categorical ? catLabel : undefined,
      categorical,
    );
    // category index plus a fixed per-point offset (the points of a category spread sideways)
    const at = (v: number, i: number) => xs.px(categorical ? v + categoryJitter(i) : v);
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
    // a categorical axis: a short bar per category instead of a curve
    const line = (y: Float32Array) => {
      if (categorical) {
        ctx.lineWidth = 3;
        ctx.lineCap = 'round';
        ctx.beginPath();
        for (let i = 0; i < y.length; i++) {
          const v = y[i];
          if (v === undefined || !Number.isFinite(v)) continue;
          ctx.moveTo(xs.px(i - 0.35), ys.px(v));
          ctx.lineTo(xs.px(i + 0.35), ys.px(v));
        }
        ctx.stroke();
        ctx.lineCap = 'butt';
        return;
      }
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
      if (!p.fit || categorical) continue;
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
      const px = at(fixed.xv[i] ?? 0, i);
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
      const px = at(fixed.xt[i] ?? 0, i + dataset.nVal);
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

  const ref = usePlotCanvas(draw, [
    dataset,
    fixed,
    curve,
    classCurves,
    showTruth,
    overlay,
    pins,
    col,
    categorical,
  ]);
  const showTruthToggle = useUiShown('truth-toggle');
  const showResample = useUiShown('resample-run');
  const showPicker = useUiShown('prediction-picker');
  const lblTruth = useUiLabel('truth-toggle', 'True function');
  const panelTitle = useUiLabel('panel-fit', 'Fit');
  return (
    <div className={styles.panel} data-testid="fit-plot">
      <div className={styles.head} data-panel-head="">
        <span className={styles.title}>{panelTitle}</span>
        {showTruthToggle && hasTruth(set) && !categorical && (
          <label className={styles.toggle}>
            <input
              type="checkbox"
              checked={showTruth}
              onChange={(e) => setShowTruth(e.target.checked)}
              data-testid="truth-toggle"
              aria-label={lblTruth}
            />
            {lblTruth}
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
          x={
            dataset.info.featureNames || col > 0
              ? axisLabel(featureName(dataset.info, dataset.info.columns[col] ?? 'x1'))
              : '$x$'
          }
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
