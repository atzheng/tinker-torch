import { useMemo, useRef, useState } from 'react';
import { linearScale, niceTicks } from '../plot/axes.js';
import { usePlotCanvas, type DrawFn } from '../plot/canvas.js';
import { AxisLabels } from '../plot/AxisLabels.js';
import { MARGINS, drawAxes, drawLegend } from '../plot/draw.js';
import {
  listParams,
  type CurveParam,
  nextAxisRange,
  sweepValues,
  tangent,
  visit,
  yRange,
  type Visited,
} from '../plot/lossCurve1d.js';
import { useUiLabel, useUiOption, useUiShown } from '../state/disclosure.js';
import { useStore } from '../state/store.js';
import styles from './Results.module.css';
import own from './LossCurve1D.module.css';

type Track = {
  host: object | null;
  key: string;
  graph: object | null;
  step: number;
  visited: Visited | null;
  range: [number, number] | null;
};

/**
 * 1-D loss curve (tinker-6t7.2): the training loss against one parameter while every other
 * parameter stays at its current value. The dot is the current value; it moves live while the
 * weight is dragged or the model trains. Options: which parameter is the axis, paint only the
 * values visited so far ("explored only"), and the tangent line with its slope. Shown whenever
 * the model has a parameter.
 */
export function LossCurve1D() {
  const host = useStore((s) => s.host);
  const order = useStore((s) => s.analysis.order);
  const params = useMemo(() => (host ? listParams(host, order) : []), [host, order]);
  if (!host || params.length === 0) return null;
  return <Panel params={params} hostKey={host} />;
}

function Panel({ params, hostKey }: { params: ReturnType<typeof listParams>; hostKey: object }) {
  const graph = useStore((s) => s.graph);
  const dataset = useStore((s) => s.dataset);
  const step = useStore((s) => s.step);
  const canTrain = useStore((s) => s.canTrain);
  const diverged = useStore((s) => s.diverged);
  const lossCurve = useStore((s) => s.lossCurve);
  const lossSlope = useStore((s) => s.lossSlope);
  const showAxis = useUiShown('loss-1d-axis');
  const showExplored = useUiShown('loss-1d-explored');
  const showTangent = useUiShown('loss-1d-tangent');
  const panelTitle = useUiLabel('panel-loss-1d', 'Loss vs parameter');

  const [pick, setPick] = useState('');
  const [explored, setExplored] = useUiOption('loss-1d-explored');
  const [tan, setTan] = useUiOption('loss-1d-tangent');
  const param = (params.find((p) => p.key === pick) ?? params[0]) as CurveParam; // params is not empty
  const cur = param.value;

  // The visited interval and the range live in a ref and change once per published host (never per
  // redraw). A new axis, a new graph or a step counter that went back (Reset) starts them afresh.
  const tr = useRef<Track>({
    host: null,
    key: '',
    graph: null,
    step: 0,
    visited: null,
    range: null,
  });
  const t = tr.current;
  if (t.host !== hostKey) {
    if (t.key !== param.key || t.graph !== graph || step < t.step) {
      t.visited = null;
      t.range = null;
    }
    t.host = hostKey;
    t.key = param.key;
    t.graph = graph;
    t.step = step;
    t.visited = visit(t.visited, cur);
    if (t.visited) t.range = nextAxisRange(t.range, t.visited, cur);
  }
  const { visited, range } = t;

  // Evaluated once per published host (at most once per frame while running), not per redraw.
  const data = useMemo(() => {
    if (!canTrain || !range || !visited) return null;
    const values = sweepValues(range, visited, explored);
    const all = new Float32Array(values.length + 1);
    all.set(values);
    all[values.length] = cur; // the last entry is the loss at the dot
    const out = lossCurve(param.axis, all);
    if (!out) return null;
    const now = out[values.length] ?? Number.NaN;
    const curve = out.subarray(0, values.length);
    const slope = tan ? lossSlope(param.axis) : null;
    return { values, curve, now, slope, y: yRange(curve, now) };
    // `hostKey` stands for every change of the parameters; range and visited follow it
  }, [hostKey, param.key, explored, tan, canTrain, dataset, graph, lossCurve, lossSlope]);

  const draw: DrawFn = (ctx, w, h, c) => {
    if (!range || !data) return;
    const m = MARGINS;
    const xs = linearScale(range[0], range[1], m.l, w - m.r);
    const ys = linearScale(data.y.lo, data.y.hi, h - m.b, m.t);
    ctx.save();
    ctx.beginPath();
    ctx.rect(m.l, m.t, w - m.l - m.r, h - m.t - m.b);
    ctx.clip();
    ctx.lineJoin = 'round';
    ctx.lineWidth = 2;
    ctx.strokeStyle = c.train;
    ctx.beginPath();
    let pen = false;
    data.values.forEach((v, i) => {
      const y = data.curve[i] ?? Number.NaN;
      if (!Number.isFinite(y)) {
        pen = false;
        return;
      }
      if (pen) ctx.lineTo(xs.px(v), ys.px(y));
      else ctx.moveTo(xs.px(v), ys.px(y));
      pen = true;
    });
    ctx.stroke();
    if (Number.isFinite(data.now)) {
      if (data.slope !== null) {
        const seg = tangent(cur, data.now, data.slope, (range[1] - range[0]) / 5);
        if (seg) {
          ctx.strokeStyle = c.val;
          ctx.lineWidth = 1.5;
          ctx.beginPath();
          ctx.moveTo(xs.px(seg.x1), ys.px(seg.y1));
          ctx.lineTo(xs.px(seg.x2), ys.px(seg.y2));
          ctx.stroke();
        }
      }
      ctx.fillStyle = c.val;
      ctx.strokeStyle = c.bg;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(xs.px(cur), ys.px(data.now), 5, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
    }
    ctx.restore();
    drawAxes(
      ctx,
      c,
      w,
      h,
      m,
      xs,
      ys,
      niceTicks(range[0], range[1], 5),
      niceTicks(data.y.lo, data.y.hi, 4),
    );
    drawLegend(ctx, c, m, [
      { label: 'loss', color: c.train, kind: 'line' },
      { label: 'now', color: c.val, kind: 'dot' },
      ...(data.slope !== null ? [{ label: 'slope', color: c.val, kind: 'line' as const }] : []),
    ]);
  };

  const ref = usePlotCanvas(draw, [data, range, hostKey]);
  const slopeText =
    data?.slope !== null && data?.slope !== undefined && Number.isFinite(data.slope)
      ? `slope ${data.slope.toFixed(3)}`
      : null;
  return (
    <div className={styles.panel} data-testid="loss-curve-1d">
      <div className={styles.head} data-panel-head="">
        <span className={styles.title}>{panelTitle}</span>
        {diverged ? <span data-testid="curve1d-diverged">Diverged</span> : null}
        {slopeText && (
          <span className={own.slope} data-testid="curve1d-slope">
            {slopeText}
          </span>
        )}
      </div>
      {(showAxis || showExplored || showTangent) && (
        <div className={own.opts}>
          {showAxis && params.length > 1 && (
            <select
              className={styles.picker}
              value={param.key}
              onChange={(e) => setPick(e.target.value)}
              data-testid="loss-1d-axis"
              aria-label="Parameter on the axis"
            >
              {params.map((p) => (
                <option key={p.key} value={p.key}>
                  {p.label.replace(/\$/g, '').replace(/[{}]/g, '')}
                </option>
              ))}
            </select>
          )}
          {showExplored && (
            <label className={styles.toggle}>
              <input
                type="checkbox"
                checked={explored}
                onChange={(e) => setExplored(e.target.checked)}
                data-testid="loss-1d-explored"
              />
              Explored only
            </label>
          )}
          {showTangent && (
            <label className={styles.toggle}>
              <input
                type="checkbox"
                checked={tan}
                onChange={(e) => setTan(e.target.checked)}
                data-testid="loss-1d-tangent"
              />
              Tangent
            </label>
          )}
        </div>
      )}
      <div className={styles.plot}>
        <canvas ref={ref} aria-label="Loss against one parameter" />
        <AxisLabels x={param.label} y="loss" />
      </div>
    </div>
  );
}
