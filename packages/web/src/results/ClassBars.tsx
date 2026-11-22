import { useEffect, useRef, useState } from 'react';
import { parsePort } from '@tinker/core';
import { PROB_BAR_EVERY_MS, probBarDue, probBarTargets } from '../canvas/probBarPure.js';
import { useUiLabel } from '../state/disclosure.js';
import { plotIndex } from '../state/plotPrediction.js';
import { useStore } from '../state/store.js';
import { CLASS_BARS_MAX_ROWS, classBars, type ClassBar } from './classBarsModel.js';
import styles from './Results.module.css';
import own from './ClassBars.module.css';

const pct = (v: number): string => `${(v * 100).toFixed(1)}%`;

/**
 * Predicted versus observed class bars (tinker-6t7.16), for a classification model: per class, the
 * model's probability averaged over the training examples next to the share of examples observed in
 * the class. Opt-in (`panel-class-bars`). Predictions come from `predictPins` (plain arrays; no
 * jax array is kept); while running they are re-read at most every PROB_BAR_EVERY_MS.
 */
export function ClassBars() {
  const graph = useStore((s) => s.graph);
  const analysis = useStore((s) => s.analysis);
  const dataset = useStore((s) => s.dataset);
  const host = useStore((s) => s.host);
  const running = useStore((s) => s.running);
  const pick = useStore((s) => s.plotPrediction);
  const predictPins = useStore((s) => s.predictPins);
  const title = useUiLabel('panel-class-bars', 'Classes');
  const [, bump] = useState(0);
  const cache = useRef<{
    t: number;
    host: unknown;
    deps: unknown[];
    bars: ClassBar[] | null;
  } | null>(null);
  const deps = [graph, analysis, dataset, predictPins, pick];
  const c = cache.current;
  const depsSame = c !== null && c.deps.every((d, i) => d === deps[i]);
  const now = performance.now();
  if (!c || !depsSame || (c.host !== host && probBarDue(now, c.t, running, false))) {
    let bars: ClassBar[] | null = null;
    const idx = plotIndex(graph, pick);
    const pin = graph.predictions[idx];
    const d = dataset.info.columns.length;
    const nObs = Math.min(dataset.nTrain, CLASS_BARS_MAX_ROWS);
    // a model with no input predicts the same for every row: one row is enough
    const n = d === 0 ? 1 : nObs;
    if (pin !== undefined && dataset.info.kind !== 'regression' && n >= 1) {
      const node = parsePort(pin).node;
      const width = analysis.outWidth[node];
      const isLink = probBarTargets(graph, analysis.outWidth).some((t) => t.id === node);
      const outs =
        width === undefined || !isLink ? null : predictPins(dataset.xTrain.slice(0, n * d), n);
      if (outs && width !== undefined) {
        bars = classBars(
          outs[idx],
          n,
          width,
          dataset.yTrain,
          dataset.info.numClasses,
          dataset.info.classNames,
          nObs,
        );
      }
    }
    cache.current = { t: now, host, deps, bars };
  }
  const pending = cache.current?.host !== host;
  useEffect(() => {
    if (!pending) return;
    const id = setTimeout(() => bump((k) => k + 1), PROB_BAR_EVERY_MS);
    return () => clearTimeout(id);
  });
  const bars = cache.current?.bars ?? null;

  return (
    <div className={styles.panel} data-testid="class-bars">
      <div className={styles.head} data-panel-head="">
        <span className={styles.title}>{title}</span>
      </div>
      {!bars ? (
        <div className={own.note} data-testid="class-bars-none">
          No class probabilities to show: the plotted prediction must be a sigmoid or softmax of a
          valid, trained-or-initialised model (not diverged).
        </div>
      ) : (
        <div className={own.body}>
          <div className={own.key}>
            <span>
              <span className={own.swatch} />
              predicted (mean)
            </span>
            <span>
              <span className={`${own.swatch} ${own.swatchObs}`} />
              observed
            </span>
          </div>
          {bars.map((b, k) => {
            const color = `var(--class-${k % 6})`;
            return (
              <div key={k} className={own.row} data-testid="class-bars-row">
                <span className={own.name} title={b.name}>
                  {b.name}
                </span>
                <div className={own.bars}>
                  <div className={own.line}>
                    <div className={own.track}>
                      <div
                        className={own.fill}
                        data-testid="class-bar-pred"
                        data-value={b.predicted}
                        style={{ width: `${b.predicted * 100}%`, background: color, opacity: 0.85 }}
                      />
                    </div>
                    <span className={own.val}>{pct(b.predicted)}</span>
                  </div>
                  <div className={own.line}>
                    <div className={own.track}>
                      <div
                        className={own.fill}
                        data-testid="class-bar-obs"
                        data-value={b.observed}
                        style={{ width: `${b.observed * 100}%`, border: `2px solid ${color}` }}
                      />
                    </div>
                    <span className={own.val}>{pct(b.observed)}</span>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
