import { useUiLabel, useUiOption, useUiShown } from '../state/disclosure.js';
import { decimatePath, linearScale, logScale, logTicks, niceTicks } from '../plot/axes.js';
import { usePlotCanvas, type DrawFn } from '../plot/canvas.js';
import { AxisLabels } from '../plot/AxisLabels.js';
import { MARGINS, drawAxes, drawLegend } from '../plot/draw.js';
import { lossRange } from '../plot/loss.js';
import { LossReadout } from './LossReadout.js';
import { lossChartShown } from './panelLayout.js';
import { Pins } from './Pins.js';
import { mergeBounds } from './pinsModel.js';
import { isMiniBatch } from '../state/trainSlice.js';
import { useStore } from '../state/store.js';
import styles from './Results.module.css';

/** Train and validation loss against step. Plotted by `step` (the history may be decimated). */
export function LossCurves() {
  const chart = useStore((s) => lossChartShown(s.show?.ui));
  return chart ? <LossChartPanel /> : <ScorePanel />;
}

/** The Loss panel with only its score readout (`panel-score` without `panel-loss`): no chart, log toggle, Restore best or pins. */
function ScorePanel() {
  const showReadout = useUiShown('loss-readout');
  const panelTitle = useUiLabel('panel-score', 'Loss');
  return (
    <div className={`${styles.panel} ${styles.lossPanel}`} data-testid="loss-curves">
      <div className={styles.head} data-panel-head="">
        <span className={styles.title}>{panelTitle}</span>
      </div>
      {showReadout && <LossReadout />}
    </div>
  );
}

function LossChartPanel() {
  const history = useStore((s) => s.history);
  const best = useStore((s) => s.best);
  const restoreBest = useStore((s) => s.restoreBest);
  const canRestore = useStore((s) => s.canTrain);
  const miniBatch = useStore((s) => isMiniBatch(s.graph.train.batch, s.dataset.nTrain));
  const pins = useStore((s) => s.pins);
  const [log, setLog] = useUiOption('loss-log-toggle');

  const draw: DrawFn = (ctx, w, h, c) => {
    const m = MARGINS;
    let r = lossRange(history, log, best?.val);
    // Pinned runs (tinker-iej.2) widen the axes so their curves are not clipped.
    for (const p of pins) r = mergeBounds(r, lossRange(p.history, log), log);
    const xs = linearScale(0, r.stepMax, m.l, w - m.r);
    const ys = log ? logScale(r.lo, r.hi, h - m.b, m.t) : linearScale(r.lo, r.hi, h - m.b, m.t);
    const yTicks = log ? logTicks(r.lo, r.hi) : niceTicks(r.lo, r.hi, 4);
    drawAxes(ctx, c, w, h, m, xs, ys, niceTicks(0, r.stepMax, 4), yTicks);
    drawLegend(ctx, c, m, [
      { label: 'train', color: c.train, kind: 'line' },
      ...(miniBatch ? [{ label: 'batch', color: c.train, kind: 'line' as const, alpha: 0.3 }] : []),
      { label: 'val', color: c.val, kind: 'line' },
      ...(best ? [{ label: 'best', color: c.val, kind: 'ring' as const }] : []),
    ]);

    ctx.save();
    ctx.beginPath();
    ctx.rect(m.l, m.t, w - m.l - m.r, h - m.t - m.b);
    ctx.clip();
    ctx.lineWidth = 1.5;
    ctx.lineJoin = 'round';
    const series = (
      pick: (i: number) => number,
      color: string,
      hist: readonly { step: number }[] = history,
    ) => {
      const y = log ? (i: number) => (pick(i) > 0 ? pick(i) : Number.NaN) : pick;
      ctx.strokeStyle = color;
      ctx.beginPath();
      let started = false;
      decimatePath(
        hist.length,
        (i) => hist[i]?.step ?? 0,
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
    // Pinned runs under the current one: training faint, validation stronger, in the pin's colour.
    for (const p of pins) {
      const col = c.pins[p.slot] ?? c.accent;
      ctx.globalAlpha = 0.3;
      series((i) => p.history[i]?.train ?? Number.NaN, col, p.history);
      ctx.globalAlpha = 0.7;
      series((i) => p.history[i]?.val ?? Number.NaN, col, p.history);
    }
    ctx.globalAlpha = miniBatch ? 0.3 : 0.75;
    series((i) => history[i]?.train ?? Number.NaN, c.train);
    ctx.globalAlpha = 1;
    if (miniBatch) series((i) => history[i]?.trainFull ?? Number.NaN, c.train);
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

  const ref = usePlotCanvas(draw, [history, best, log, miniBatch, pins]);
  const showLog = useUiShown('loss-log-toggle');
  const showRestore = useUiShown('loss-restore-best');
  const showReadout = useUiShown('loss-readout');
  const showPins = useUiShown('pins');
  const lblLog = useUiLabel('loss-log-toggle', 'Log scale');
  const lblRestore = useUiLabel('loss-restore-best', 'Restore best');
  const panelTitle = useUiLabel('panel-loss', 'Loss');
  return (
    <div className={`${styles.panel} ${styles.lossPanel}`} data-testid="loss-curves">
      <div className={styles.head} data-panel-head="">
        <span className={styles.title}>{panelTitle}</span>
        {showLog && (
          <label className={styles.toggle}>
            <input
              type="checkbox"
              checked={log}
              onChange={(e) => setLog(e.target.checked)}
              data-testid="loss-log-toggle"
              aria-label={lblLog}
            />
            {lblLog}
          </label>
        )}
        {showRestore && (
          <button
            type="button"
            onClick={restoreBest}
            disabled={!best || !canRestore}
            data-testid="loss-restore-best"
            aria-label={lblRestore}
          >
            {lblRestore}
          </button>
        )}
      </div>
      {showReadout && <LossReadout />}
      {showPins && <Pins />}
      <div className={styles.plot}>
        <canvas ref={ref} aria-label="Loss curves" />
        <AxisLabels x="step" y="loss" />
      </div>
    </div>
  );
}
