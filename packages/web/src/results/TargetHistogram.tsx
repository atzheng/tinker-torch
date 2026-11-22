import { useMemo, useRef, useState, type PointerEvent as ReactPointerEvent } from 'react';
import { startDrag } from '../gestures/drag.js';
import { beginWeightEdit, type WeightEdit } from '../canvas/weightEdit.js';
import { AxisLabels } from '../plot/AxisLabels.js';
import { linearScale, niceTicks } from '../plot/axes.js';
import { usePlotCanvas, type DrawFn } from '../plot/canvas.js';
import { MARGINS, drawAxes, fontOf } from '../plot/draw.js';
import {
  P_MAX,
  P_MIN,
  axisFor,
  histogramBins,
  lineDomain,
  predictionChain,
  squareSides,
  valueToParam,
} from '../plot/histogramPure.js';
import { useUiShown } from '../state/disclosure.js';
import { plotIndex } from '../state/plotPrediction.js';
import { useStore } from '../state/store.js';
import { axisLabel } from '../state/names.js';
import { GRID_N } from './FitPlot.js';
import styles from './Results.module.css';

/**
 * Does the histogram panel apply: the plotted prediction has no Input upstream (one number, not a
 * curve), the target is a number or a 0/1 label, and the model is valid enough to have a host.
 * Shared by the panel list and the disclosure region check.
 */
export function useHistogramApplies(): boolean {
  return useStore((s) => {
    const pred = s.graph.predictions[plotIndex(s.graph, s.plotPrediction)];
    if (pred === undefined || s.dataset.info.kind === 'multiclass') return false;
    const node = pred.split('.')[0] ?? pred;
    return !predictionChain(s.graph, node).hasInput;
  });
}

const fmt = (v: number): string => (Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(3));

/**
 * Target histogram with a draggable prediction line, for a model with no input (CONTRACTS.md
 * section 10, "Results panels"). The line is the model's one number: drag it anywhere in the plot
 * (mouse or touch) and the Bias piece takes that value; for sigmoid(b) the drag is in probability
 * units and b = logit(p), with p kept inside (0.005, 0.995). Like any hand edit, a drag is refused
 * while running and is one undo step.
 *
 * Squares: one square per example with side |y - yhat|. The squares share the histogram's
 * horizontal axis, so their size is true to scale: the plot uses equal pixels per unit in both
 * directions, sized so the tallest square possible (data at one end of the range, line at the
 * other) fits. Squares stand on the baseline and overlap with transparency, largest first; the
 * total area (and the mean) is printed in the header.
 */
export function TargetHistogram() {
  const dataset = useStore((s) => s.dataset);
  const graph = useStore((s) => s.graph);
  const host = useStore((s) => s.host);
  const pick = useStore((s) => s.plotPrediction);
  const predictGrid = useStore((s) => s.predictGrid);
  const running = useStore((s) => s.running);
  const [squares, setSquares] = useState(false);
  const showSquaresToggle = useUiShown('histogram-squares');
  const boxRef = useRef<HTMLDivElement>(null);
  const editRef = useRef<WeightEdit | null>(null);

  const binary = dataset.info.kind === 'binary';
  const ys = dataset.yTrain;
  const cols = dataset.info.columns.length;
  const predNode = (graph.predictions[plotIndex(graph, pick)] ?? '').split('.')[0] ?? '';
  const chain = useMemo(() => predictionChain(graph, predNode), [graph, predNode]);
  const sig = chain.param?.sigmoid ?? false;
  const locked = useStore((s) => s.preset?.locked?.includes(chain.param?.node ?? '') ?? false);
  const frozen = running || locked; // a drag would be refused: show it before the press

  const bins = useMemo(() => histogramBins(ys, binary), [ys, binary]);
  const dom = useMemo(() => lineDomain(ys, binary), [ys, binary]);
  // The line: the model's prediction on a dummy input (it has no input, so any x gives the same).
  const yhat = useMemo(() => {
    if (!host) return null;
    const out = predictGrid(new Float32Array(GRID_N * cols), GRID_N);
    const v = out?.[0];
    return v !== undefined && Number.isFinite(v) ? v : null;
  }, [host, predictGrid, graph.predictions, pick, cols]);
  const sides = useMemo(
    () => (squares && yhat !== null ? squareSides(ys, yhat) : []),
    [squares, ys, yhat],
  );
  const area = sides.reduce((a, s) => a + s * s, 0);

  const geometry = (w: number, h: number) => {
    const m = MARGINS;
    const pw = w - m.l - m.r;
    const ph = h - m.t - m.b;
    const ax = axisFor(dom, pw, ph, squares);
    return { m, pw, ph, ...ax };
  };

  const draw: DrawFn = (ctx, w, h, c) => {
    const g = geometry(w, h);
    const { m } = g;
    const xs = linearScale(g.xlo, g.xhi, m.l, w - m.r);
    const base = h - m.b;
    // The vertical axis is a count for the bars; with squares it is also "length", 1 unit = g.unit px.
    const yc = linearScale(0, bins.max * 1.1, base, m.t);
    drawAxes(ctx, c, w, h, m, xs, yc, niceTicks(g.xlo, g.xhi, 6), niceTicks(0, bins.max, 3));
    ctx.save();
    ctx.beginPath();
    ctx.rect(m.l, m.t, g.pw, g.ph);
    ctx.clip();
    // bars
    ctx.fillStyle = c.train;
    ctx.globalAlpha = squares ? 0.25 : 0.55;
    const gap = bins.counts.length > 2 ? 1 : 6;
    bins.counts.forEach((n, i) => {
      const x0 = xs.px(bins.lo + i * bins.width);
      const x1 = xs.px(bins.lo + (i + 1) * bins.width);
      ctx.fillRect(x0 + gap / 2, yc.px(n), Math.max(1, x1 - x0 - gap), base - yc.px(n));
    });
    ctx.globalAlpha = 1;
    if (yhat !== null) {
      if (squares) {
        // square for example y: one edge is the segment from y to yhat on the baseline
        const vals = Array.from(ys).filter(Number.isFinite);
        vals.sort((a, b) => Math.abs(b - yhat) - Math.abs(a - yhat));
        ctx.fillStyle = c.accent;
        ctx.strokeStyle = c.accent;
        ctx.lineWidth = 1;
        for (const v of vals) {
          const side = Math.abs(v - yhat) * g.unit;
          if (side < 0.5) continue;
          const x0 = Math.min(xs.px(v), xs.px(yhat));
          ctx.globalAlpha = 0.1;
          ctx.fillRect(x0, base - side, side, side);
          ctx.globalAlpha = 0.55;
          ctx.strokeRect(x0 + 0.5, base - side + 0.5, side, side);
        }
        ctx.globalAlpha = 1;
      }
      const lx = Math.round(xs.px(yhat)) + 0.5;
      ctx.strokeStyle = c.text;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(lx, m.t);
      ctx.lineTo(lx, base);
      ctx.stroke();
      // a handle so it reads as draggable
      ctx.fillStyle = c.text;
      ctx.beginPath();
      ctx.arc(lx, m.t + 10, 6, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = c.bg;
      ctx.font = fontOf(c, 9);
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText('↔', lx, m.t + 10);
    }
    ctx.restore();
  };
  const ref = usePlotCanvas(draw, [bins, dom, yhat, squares, ys]);

  /** Pointer x (client) to the line's value, within the plot's range. */
  const valueAt = (clientX: number): number | null => {
    const r = boxRef.current?.getBoundingClientRect();
    if (!r || r.width <= 0) return null;
    const g = geometry(r.width, r.height);
    const v = g.xlo + ((clientX - r.left - g.m.l) / g.pw) * (g.xhi - g.xlo);
    const lo = sig ? P_MIN : dom.dlo;
    const hi = sig ? P_MAX : dom.dhi;
    return Math.min(hi, Math.max(lo, v));
  };

  const onDown = (e: ReactPointerEvent<HTMLElement>) => {
    if (e.pointerType === 'mouse' && e.button !== 0) return;
    e.stopPropagation(); // the panel's header drag is separate; this is the plot
    const param = chain.param;
    if (!param) return;
    const set = (clientX: number) => {
      const v = valueAt(clientX);
      if (v !== null) editRef.current?.set(valueToParam(v, param.sigmoid));
    };
    const begin = (clientX: number): boolean => {
      editRef.current = beginWeightEdit({ kind: 'bias', node: param.node });
      if (!editRef.current) return false;
      set(clientX);
      return true;
    };
    // Touch must not scroll the page while dragging: the plot has touch-action: none.
    startDrag(e, {
      onStart: (p) => {
        begin(p.startX);
        set(p.x);
      },
      onMove: (p) => set(p.x),
      onEnd: (p) => {
        set(p.x);
        editRef.current?.commit();
        editRef.current = null;
      },
      onTap: (p) => {
        if (begin(p.x)) editRef.current?.commit();
        editRef.current = null;
      },
      onCancel: () => {
        editRef.current?.cancel();
        editRef.current = null;
      },
    });
  };

  const target =
    dataset.info.targetName && dataset.info.targetName !== 'y' ? dataset.info.targetName : null;
  return (
    <div className={styles.panel} data-testid="target-histogram">
      <div className={`${styles.head} ${styles.headWrap}`} data-panel-head="">
        <span className={styles.title}>Target</span>
        {yhat !== null && (
          <span className={styles.readout} data-testid="histogram-yhat">
            {sig ? 'p' : 'ŷ'} = {fmt(yhat)}
          </span>
        )}
        {squares && yhat !== null && (
          <span className={styles.readout} data-testid="histogram-area">
            area {fmt(area)}, mean {fmt(area / Math.max(1, sides.length))}
          </span>
        )}
        {showSquaresToggle && (
          <label className={styles.toggle}>
            <input
              type="checkbox"
              checked={squares}
              onChange={(e) => setSquares(e.target.checked)}
              data-testid="histogram-squares"
            />
            Squares
          </label>
        )}
      </div>
      <div
        className={styles.plot}
        ref={boxRef}
        style={{ touchAction: 'none', cursor: chain.param && !frozen ? 'ew-resize' : 'default' }}
        onPointerDown={onDown}
        data-testid="histogram-plot"
        data-draggable={chain.param !== null && !frozen}
        data-sigmoid={sig}
      >
        <canvas ref={ref} aria-label="Target histogram" />
        <AxisLabels
          x={binary ? '$y$' : target ? axisLabel(target) : '$y$'}
          y={squares ? 'length' : 'count'}
        />
      </div>
    </div>
  );
}
