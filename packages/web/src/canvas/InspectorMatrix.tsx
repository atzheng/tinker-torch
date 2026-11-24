import { memo, useMemo, type CSSProperties } from 'react';
import type { Node } from '@tinker/core';
import { TexText, plainOf } from '../tex/Tex.js';
import { useAnalysis, useWeights } from '../state/derived.js';
import { useUiShown } from '../state/disclosure.js';
import { hostShape } from '../state/hostShape.js';
import { useStore } from '../state/store.js';
import { inlineFits, shapeText, unitLabel } from './inspectorMatrixPure.js';
import { LIVE_MS, useThrottledHost } from './WeightInspector.js';
import { cellText, gridOf, rowNames, tintPct } from './weightsPure.js';

/**
 * The weight matrix of the selected Linear inside the piece inspector (tinker-nj1): W (rows =
 * inputs, columns = output units) and the bias row, for layers of at most `INLINE_MAX_ROWS` x
 * `INLINE_MAX_COLS` (inspectorMatrixPure.ts); a larger layer shows its shape only. Read-only: the
 * "Inspect weights" panel (tinker-fn5) is where cells are edited. Follows `host` through the same
 * throttled hook as that panel, so the canvas does not re-render per frame. Disclosure id
 * `inspector-matrix`.
 */
export const InspectorMatrix = memo(function InspectorMatrix({ node }: { node: Node }) {
  const shown = useUiShown('inspector-matrix');
  const analysis = useAnalysis();
  const graph = useStore((s) => s.graph);
  const dataset = useStore((s) => s.dataset);
  const initial = useWeights(analysis, false);
  const live = useThrottledHost(LIVE_MS);
  const host = live ?? initial;
  const inW = analysis.inWidth[node.id];
  const grid = useMemo(
    () =>
      node.type === 'linear' ? gridOf(host, node, inW, hostShape(host, node.id, dataset)) : null,
    [host, node, inW, dataset],
  );
  const names = useMemo(
    () =>
      node.type === 'linear' && inW !== undefined
        ? rowNames(graph, node.id, inW, dataset.info.columns)
        : [],
    [graph, node, inW, dataset.info.columns],
  );
  if (!shown || node.type !== 'linear' || inW === undefined) return null;
  if (!inlineFits(inW, node.out)) {
    return (
      <p className="tk-diag" data-testid="inspector-matrix-large">
        {shapeText(inW, node.out)}
      </p>
    );
  }
  if (!grid || !grid.W) return null;
  const W = grid.W;
  const maxAbs = Math.max(
    0,
    ...Array.from(W, Math.abs),
    ...(grid.b ? Array.from(grid.b, Math.abs) : []),
  );
  const cell = (testid: string, label: string, v: number) => (
    <span
      key={testid}
      className="tk-imcell"
      role="gridcell"
      tabIndex={0}
      data-testid={testid}
      data-value={v}
      aria-label={`${label}: ${v.toFixed(4)}`}
      title={`${label}: ${v.toFixed(4)}`}
      style={
        {
          '--cell-bg': `color-mix(in srgb, var(${v < 0 ? '--weight-neg' : '--weight-pos'}) ${tintPct(v, maxAbs)}%, var(--surface))`,
        } as CSSProperties
      }
    >
      {cellText(v)}
    </span>
  );
  const cols = grid.cols;
  return (
    <div
      className="tk-imatrix"
      data-testid="inspector-matrix"
      role="grid"
      aria-label={`Weights of ${node.id}`}
      style={{ '--icols': cols } as CSSProperties}
    >
      <div className="tk-imrow" role="row">
        <span />
        {Array.from({ length: cols }, (_, j) => (
          <span key={j} className="tk-imlabel tk-imcol" role="columnheader">
            {unitLabel(j)}
          </span>
        ))}
      </div>
      {Array.from({ length: grid.rows }, (_, i) => (
        <div className="tk-imrow" role="row" key={i}>
          <span className="tk-imlabel" role="rowheader" title={plainOf(names[i] ?? '')}>
            <TexText>{names[i] ?? `input ${i}`}</TexText>
          </span>
          {Array.from({ length: cols }, (_, j) =>
            cell(
              `imcell-${i}-${j}`,
              `${plainOf(names[i] ?? `input ${i}`)} to unit ${j}`,
              W[i * cols + j] ?? 0,
            ),
          )}
        </div>
      ))}
      {grid.b && (
        <div className="tk-imrow tk-imbias" role="row">
          <span className="tk-imlabel" role="rowheader">
            bias
          </span>
          {Array.from({ length: cols }, (_, j) =>
            cell(`imcell-b-${j}`, `bias of unit ${j}`, grid.b?.[j] ?? 0),
          )}
        </div>
      )}
    </div>
  );
});
