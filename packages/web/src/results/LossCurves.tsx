import { useState } from 'react';
import { decimatePath, linearScale, logScale, logTicks, niceTicks } from '../plot/axes.js';
import { usePlotCanvas, type DrawFn } from '../plot/canvas.js';
import { MARGINS, drawAxes, drawLegend } from '../plot/draw.js';
import { lossRange } from '../plot/loss.js';
import { useStore } from '../state/store.js';
import styles from './Results.module.css';

/** Train and validation loss against step. Plotted by `step` (the history may be decimated). */
export function LossCurves() {
  const history = useStore((s) => s.history);
  const best = useStore((s) => s.best);
  const restoreBest = useStore((s) => s.restoreBest);
  const canRestore = useStore((s) => s.canTrain);
  const miniBatch = useStore((s) => s.graph.train.batch !== 'full');
  const [log, setLog] = useState(false);

  const draw: DrawFn = (ctx, w, h, c) => {
    const m = MARGINS;
    const r = lossRange(history, log, best?.val);
    const xs = linearScale(0, r.stepMax, m.l, w - m.r);
    const ys = log ? logScale(r.lo, r.hi, h - m.b, m.t) : linearScale(r.lo, r.hi, h - m.b, m.t);
    const yTicks = log ? logTicks(r.lo, r.hi) : niceTicks(r.lo, r.hi, 4);
    drawAxes(ctx, c, w, h, m, xs, ys, niceTicks(0, r.stepMax, 4), yTicks, 'step', 'loss');
    drawLegend(ctx, c, m, [
      { label: miniBatch ? 'train (batch)' : 'train', color: c.train, kind: 'line' },
      { label: 'val', color: c.val, kind: 'line' },
      ...(best ? [{ label: 'best', color: c.val, kind: 'ring' as const }] : []),
    ]);

    ctx.save();
    ctx.beginPath();
    ctx.rect(m.l, m.t, w - m.l - m.r, h - m.t - m.b);
    ctx.clip();
    ctx.lineWidth = 1.5;
    ctx.lineJoin = 'round';
    const series = (pick: (i: number) => number, color: string) => {
      const y = log ? (i: number) => (pick(i) > 0 ? pick(i) : Number.NaN) : pick;
      ctx.strokeStyle = color;
      ctx.beginPath();
      let started = false;
      decimatePath(
        history.length,
        (i) => history[i]?.step ?? 0,
        y,
        xs.px,
        ys.px,
        (px, py) => {
          if (started) ctx.lineTo(px, py);
          else ctx.moveTo(px, py);
          started = true;
        },
      );
      ctx.stroke();
    };
    ctx.globalAlpha = 0.75;
    series((i) => history[i]?.train ?? Number.NaN, c.train);
    ctx.globalAlpha = 1;
    series((i) => history[i]?.val ?? Number.NaN, c.val);

    if (best && (!log || best.val > 0)) {
      const bx = xs.px(best.step);
      const by = ys.px(best.val);
      ctx.strokeStyle = c.val;
      ctx.lineWidth = 1;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(bx + 0.5, m.t);
      ctx.lineTo(bx + 0.5, h - m.b);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = c.bg;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(bx, by, 4.5, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
    }
    ctx.restore();
  };

  const ref = usePlotCanvas(draw, [history, best, log, miniBatch]);
  return (
    <div className={styles.panel} data-testid="loss-curves">
      <div className={styles.head}>
        <span className={styles.title}>Loss</span>
        <label className={styles.toggle}>
          <input
            type="checkbox"
            checked={log}
            onChange={(e) => setLog(e.target.checked)}
            data-testid="loss-log-toggle"
          />
          Log scale
        </label>
        <button
          type="button"
          onClick={restoreBest}
          disabled={!best || !canRestore}
          data-testid="loss-restore-best"
        >
          Restore best
        </button>
      </div>
      <div className={styles.plot}>
        <canvas ref={ref} aria-label="Loss curves" />
      </div>
    </div>
  );
}
