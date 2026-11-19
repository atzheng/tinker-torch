import { usePlotCanvas, type DrawFn } from '../plot/canvas.js';
import { useStore } from '../state/store.js';
import { PIN_LIMIT, makePin, type Pin } from './pinsModel.js';
import styles from './Pins.module.css';

/** A small picture of a pinned run's loss curve (validation when there is one, else training). */
function Thumb({ pin }: { pin: Pin }) {
  const draw: DrawFn = (ctx, w, h, c) => {
    const pts = pin.history;
    const useVal = pts.some((p) => p.val !== null);
    const get = (i: number) => (useVal ? pts[i]?.val : pts[i]?.train) ?? Number.NaN;
    let hi = 0;
    let stepMax = 1;
    for (let i = 0; i < pts.length; i++) {
      const v = get(i);
      if (Number.isFinite(v) && v > hi) hi = v;
      stepMax = Math.max(stepMax, pts[i]?.step ?? 0);
    }
    if (hi <= 0) hi = 1;
    ctx.strokeStyle = c.pins[pin.slot] ?? c.accent;
    ctx.lineWidth = 1.5;
    ctx.lineJoin = 'round';
    ctx.beginPath();
    let started = false;
    for (let i = 0; i < pts.length; i++) {
      const v = get(i);
      if (!Number.isFinite(v)) continue;
      const x = 1 + ((pts[i]?.step ?? 0) / stepMax) * (w - 2);
      const y = h - 2 - (v / hi) * (h - 4);
      if (started) ctx.lineTo(x, y);
      else ctx.moveTo(x, y);
      started = true;
    }
    ctx.stroke();
  };
  const ref = usePlotCanvas(draw, [pin]);
  return <canvas ref={ref} className={styles.thumb} aria-hidden="true" />;
}

/**
 * The pin row of the loss panel: a Pin button and one chip per pinned run (curve thumbnail, label,
 * remove). Pressing a chip loads that run's graph (one undo step) and its weights. The chip's
 * top edge has the colour of the run's overlay in the loss curves, so the row is also the legend.
 */
export function Pins() {
  const pins = useStore((s) => s.pins);
  const hasRun = useStore((s) => s.history.length > 0 && s.host !== null);
  const addPin = useStore((s) => s.addPin);
  const removePin = useStore((s) => s.removePin);
  const full = pins.length >= PIN_LIMIT;

  const pin = () => {
    const s = useStore.getState();
    if (!s.host) return;
    const p = makePin(s.graph, s.host, s.history, s.pins);
    if (p) addPin(p);
  };
  const load = (p: Pin) => {
    const s = useStore.getState();
    s.apply({ ok: true, graph: p.graph });
    useStore.getState().loadHost(p.host);
  };

  return (
    <div className={styles.row} data-testid="pins">
      <button
        type="button"
        className={styles.pinBtn}
        onClick={pin}
        disabled={!hasRun || full}
        title={full ? `Up to ${PIN_LIMIT} pins: remove one first` : 'Keep this run for comparison'}
        data-testid="pin-add"
      >
        Pin {pins.length}/{PIN_LIMIT}
      </button>
      {pins.map((p) => (
        <div
          key={p.id}
          className={styles.chip}
          style={{ borderTopColor: `var(--pin-${p.slot + 1})` }}
          data-testid="pin-chip"
        >
          <button
            type="button"
            className={styles.load}
            onClick={() => load(p)}
            title={`Load this run: ${p.label}`}
            data-testid="pin-load"
          >
            <Thumb pin={p} />
            <span className={styles.label}>{p.label}</span>
          </button>
          <button
            type="button"
            className={styles.remove}
            onClick={() => removePin(p.id)}
            aria-label={`Remove pin ${p.label}`}
            data-testid="pin-remove"
          >
            ×
          </button>
        </div>
      ))}
    </div>
  );
}
