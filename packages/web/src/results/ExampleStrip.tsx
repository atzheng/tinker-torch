import { useMemo, useState, type CSSProperties } from 'react';
import { exampleSource } from '@tinker/core';
import { useStore } from '../state/store.js';
import { formatCell, NUMBERS_MAX, stripCells } from './exampleStripModel.js';
import styles from './Results.module.css';
import own from './ExampleStrip.module.css';

/**
 * Per-example strip (tinker-6t7.6): one cell per training example, in row order. For log loss and
 * cross-entropy a cell shows p of what happened (fill height; the number up to NUMBERS_MAX
 * examples) in the colour of the observed class; any other loss shows the per-example loss
 * (shade). Beyond NUMBERS_MAX the cells are small and wrap and the panel scrolls; the value of a
 * cell shows in the header on hover or tap. Recomputed once per published host.
 */
export function ExampleStrip() {
  const host = useStore((s) => s.host);
  const graph = useStore((s) => s.graph);
  const analysis = useStore((s) => s.analysis);
  const dataset = useStore((s) => s.dataset);
  const exampleLosses = useStore((s) => s.exampleLosses);
  const [active, setActive] = useState<number | null>(null);

  const data = useMemo(() => {
    const src = exampleSource(graph, analysis);
    const losses = src ? exampleLosses('train') : null;
    if (!src || !losses) return null;
    const mode = src.type === 'log_loss' || src.type === 'cross_entropy' ? 'prob' : 'loss';
    const cells = stripCells(losses, dataset.yTrain, mode, dataset.info.kind !== 'regression');
    return { mode, cells };
    // `host` stands for every change of the parameters
  }, [host, graph, analysis, dataset, exampleLosses]);

  const n = data?.cells.length ?? 0;
  const numbers = n <= NUMBERS_MAX;
  const cur = data && active !== null ? data.cells[active] : undefined;
  const size = numbers ? '34px' : '9px';
  return (
    <div className={styles.panel} data-testid="example-strip">
      <div className={styles.head} data-panel-head="">
        <span className={styles.title}>Examples</span>
        {data && (
          <span className={own.legend}>
            {data.mode === 'prob' ? 'p of what happened' : 'loss per example'}
          </span>
        )}
        {cur && active !== null && (
          <span className={own.read} data-testid="example-read">
            #{active + 1} {formatCell(cur.value)}
          </span>
        )}
      </div>
      {!data || n === 0 ? (
        <div className={own.note} data-testid="example-none">
          The score is not a plain average over the examples, so there is no value per example.
        </div>
      ) : (
        <div className={own.body}>
          <div
            className={own.grid}
            data-testid="example-grid"
            style={{ '--cell': size, '--cell-h': size } as CSSProperties}
            onPointerLeave={(e) => {
              if (e.pointerType === 'mouse') setActive(null); // a tap keeps its value shown
            }}
          >
            {data.cells.map((c, i) => {
              const color = c.cls >= 0 ? `var(--class-${c.cls % 6})` : 'var(--accent)';
              return (
                <div
                  key={i}
                  className={own.cell}
                  data-testid="example-cell"
                  data-value={c.value}
                  title={formatCell(c.value)}
                  onPointerEnter={() => setActive(i)}
                  onPointerDown={() => setActive(i)}
                >
                  <div
                    className={own.fill}
                    style={
                      data.mode === 'prob'
                        ? { height: `${c.frac * 100}%`, background: color, opacity: 0.8 }
                        : { height: '100%', background: color, opacity: 0.1 + 0.8 * c.frac }
                    }
                  />
                  {numbers && <span className={own.num}>{formatCell(c.value)}</span>}
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
