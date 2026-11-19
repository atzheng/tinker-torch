import type { PlotColors } from '../plot/color.js';
import type { Scale } from '../plot/axes.js';
import { useStore } from '../state/store.js';
import { TexText } from '../tex/Tex.js';
import styles from './Results.module.css';
import {
  COPIES,
  clearOverlay,
  runOverlay,
  useOverlay,
  type OverlayResult,
} from './resampleStore.js';

/** Colour of the copies and their average (distinct from the model line and the truth). */
const copyColor = (c: PlotColors) => c.accent;

/** The result for 1-D data (curves), or null. */
export function useOverlay1D(): OverlayResult | null {
  const r = useOverlay((s) => s.result);
  return r && r.d === 1 ? r : null;
}

/** The result for 2-D data (mean surface), or null. */
export function useOverlay2D(): OverlayResult | null {
  const r = useOverlay((s) => s.result);
  return r && r.d === 2 ? r : null;
}

/**
 * Draws the copies faintly and their average in bold over the 1-D fit plot. `xRaw` is the fixed
 * grid in raw units (the same grid every copy was predicted on). Call it inside the plot clip.
 */
export function drawOverlay1D(
  ctx: CanvasRenderingContext2D,
  c: PlotColors,
  r: OverlayResult,
  xRaw: Float32Array,
  xs: Scale,
  ys: Scale,
): void {
  const line = (y: Float32Array) => {
    ctx.beginPath();
    for (let i = 0; i < y.length; i++) {
      const px = xs.px(xRaw[i] ?? 0);
      if (i === 0) ctx.moveTo(px, ys.px(y[i] ?? 0));
      else ctx.lineTo(px, ys.px(y[i] ?? 0));
    }
    ctx.stroke();
  };
  ctx.save();
  ctx.strokeStyle = copyColor(c);
  ctx.lineJoin = 'round';
  ctx.globalAlpha = 0.22;
  ctx.lineWidth = 1.2;
  for (const y of r.curves) line(y);
  ctx.globalAlpha = 1;
  // halo, so the average stays readable over the points
  ctx.strokeStyle = c.bg;
  ctx.lineWidth = 6;
  line(r.mean);
  ctx.strokeStyle = copyColor(c);
  ctx.lineWidth = 3;
  line(r.mean);
  ctx.restore();
}

/** Legend entries for the overlay. */
export const overlayLegend = (c: PlotColors) => [
  { label: 'copies', color: copyColor(c), kind: 'line' as const },
  { label: 'mean', color: copyColor(c), kind: 'line' as const },
];

const fmt = (v: number) => (Number.isFinite(v) ? v.toPrecision(3) : 'n/a');

/** The button for the results header. Disabled while a run is going or the graph cannot train. */
export function ResampleButton() {
  const status = useOverlay((s) => s.status);
  const canTrain = useStore((s) => s.canTrain);
  const busy = status === 'busy';
  return (
    <>
      {status === 'done' && (
        <button type="button" onClick={clearOverlay} data-testid="resample-clear">
          Clear
        </button>
      )}
      <button
        type="button"
        onClick={runOverlay}
        disabled={busy || !canTrain}
        aria-busy={busy}
        data-testid="resample-run"
      >
        {busy ? `Training ${COPIES}…` : `Resample × ${COPIES}`}
      </button>
    </>
  );
}

/** One line under the header: bias and variance, or the busy, error and fallback notes. */
export function ResampleReadout() {
  const { status, result, message } = useOverlay();
  if (status === 'idle') return null;
  let text: string;
  if (status === 'busy') text = `Training ${COPIES} copies in the background…`;
  else if (status === 'error' || !result) text = message ?? 'The copies could not be trained.';
  else {
    const notes = [
      `${result.used} copies, ${result.steps} steps${result.fullBatchNote ? ', full batch' : ''}`,
    ];
    if (result.fallbackSteps) notes.push('the run has 0 steps, so 500 are used');
    if (result.diverged > 0) notes.push(`${result.diverged} diverged and left out`);
    if (result.d === 2) notes.push('mean of the copies shown');
    text = `$\\mathrm{bias}^2$ ${fmt(result.bias2)}  variance ${fmt(result.variance)}  (${notes.join('; ')})`;
  }
  return (
    <div className={`${styles.readout} tk-diag`} data-testid="resample-readout" role="status">
      <TexText>{text}</TexText>
    </div>
  );
}
