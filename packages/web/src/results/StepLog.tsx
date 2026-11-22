import { useMemo } from 'react';
import { listParams } from '../plot/lossCurve1d.js';
import { STEP_LOG_MAX_PARAMS } from '../state/trainSlice.js';
import { useUiLabel } from '../state/disclosure.js';
import { useStore } from '../state/store.js';
import styles from './Results.module.css';
import own from './StepLog.module.css';

const num = (v: number | null): string => (v === null || !Number.isFinite(v) ? '' : v.toFixed(4));

/**
 * Step log (tinker-6t7.10): one row per press of Step. Columns: step, each parameter AFTER the
 * step, the prediction (first prediction pin at those parameters; first training example when the
 * model has inputs), the slope of the loss per parameter at the parameters BEFORE the step (the
 * gradient the update used; left out unless plain SGD, full batch, no dropout, where the full-batch slope is
 * not what the step applied) and the loss the step measured BEFORE the update. Opt-in
 * (`panel-step-log`); small models only.
 */
export function StepLog() {
  const host = useStore((s) => s.host);
  const order = useStore((s) => s.analysis.order);
  const rows = useStore((s) => s.stepLog);
  const params = useMemo(() => (host ? listParams(host, order) : []), [host, order]);
  if (!host || params.length === 0) return null;
  const small = params.length <= STEP_LOG_MAX_PARAMS;
  const full = params.map((p) => p.label.replace(/\$/g, '').replace(/[{}]/g, ''));
  const sym = full.map((n) => n.split(' ').slice(-1)[0] ?? n);
  // the bare symbol (b, W1) when it is unique, else the piece name too
  const names = full.map((n, i) =>
    sym.filter((x) => x === sym[i]).length === 1 ? (sym[i] ?? n) : n,
  );
  const hasSlope = rows.some((r) => r.slopes !== null);
  const panelTitle = useUiLabel('panel-step-log', 'Steps');
  return (
    <div className={styles.panel} data-testid="step-log">
      <div className={styles.head} data-panel-head="">
        <span className={styles.title}>{panelTitle}</span>
      </div>
      {!small ? (
        <p className={own.note} data-testid="step-log-large">
          The step log is for models with at most {STEP_LOG_MAX_PARAMS} parameters.
        </p>
      ) : rows.length === 0 ? (
        <p className={own.note} data-testid="step-log-empty">
          Press Step to add a row.
        </p>
      ) : (
        <div className={own.wrap}>
          <p className={own.caption}>
            Parameters and ŷ are after the step; slope and loss are before it.
            {hasSlope ? ' New value = old value - learning rate x slope.' : ''}
          </p>
          <table className={own.table}>
            <thead>
              <tr>
                <th scope="col">step</th>
                {names.map((n) => (
                  <th key={n} scope="col">
                    {n}
                  </th>
                ))}
                <th scope="col">ŷ</th>
                {hasSlope &&
                  names.map((n) => (
                    <th key={`s${n}`} scope="col">
                      slope {n}
                    </th>
                  ))}
                <th scope="col">loss</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.step} data-testid="step-log-row">
                  <td>{r.step}</td>
                  {r.values.map((v, i) => (
                    <td key={i}>{num(v)}</td>
                  ))}
                  <td>{num(r.prediction)}</td>
                  {hasSlope && r.slopes?.map((v, i) => <td key={`s${i}`}>{num(v)}</td>)}
                  <td>{num(r.loss)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
