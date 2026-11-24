import { memo, useEffect, useMemo, useState, type CSSProperties } from 'react';
import type { HostParams, Node } from '@tinker/core';
import { TexText, plainOf } from '../tex/Tex.js';
import { useAnalysis, useWeights } from '../state/derived.js';
import { hostShape } from '../state/hostShape.js';
import { useStore } from '../state/store.js';
import { pointerDownWith } from './tap.js';
import { ValueField } from './ValueField.js';
import { weightDragHandlers, type WeightTarget } from './weightEdit.js';
import {
  cellText,
  gridOf,
  rowNames,
  statsOf,
  tintPct,
  windowOf,
  type WeightGrid,
} from './weightsPure.js';

/**
 * Weight inspector (tinker-fn5): every learned number of one Linear (the grid W, rows = inputs,
 * columns = output units, plus the bias row), a Weight or a Bias, as a labelled grid on a diverging
 * blue / orange scale. Opened from the piece's inspector (`viewSlice.weightsOf`).
 *
 * Cells are controls when the value can be changed now (paused, not diverged, piece not locked):
 * drag a cell up or down, or type in the field below the grid. Both go through `beginWeightEdit`
 * (weightEdit.ts), so locks, the pause-first rule, the clamp and one undo step per edit are those
 * of the neuron view. Otherwise a cell is a diagnostic: flat, with its exact value on hover or tap.
 *
 * Live during a run: the panel follows the train slice's published `host`, but at most every
 * `LIVE_MS` (the canvas and plots redraw each frame; a grid of up to 32 x 32 cells need not). When
 * the run is paused an edit shows at once. A layer over 32 rows or columns shows one page of 32 at
 * a time, with min, max, mean and standard deviation of the whole layer above.
 */
export const LIVE_MS = 250;

/** Counters for the "no per-frame re-render" check; read through `window.__tinker.weights` in dev. */
export const weightStats = { renders: 0 };
if (import.meta.env.DEV && typeof window !== 'undefined') {
  const w = window as unknown as { __tinker?: Record<string, unknown> };
  w.__tinker = { ...w.__tinker, weights: weightStats };
}

/** The store's `host`, republished at most every `ms` while a run is going, at once otherwise. */
export function useThrottledHost(ms: number): HostParams | undefined {
  const [host, setHost] = useState<HostParams | undefined>(
    () => useStore.getState().host ?? undefined,
  );
  useEffect(() => {
    let timer: ReturnType<typeof setTimeout> | null = null;
    const now = () => setHost(useStore.getState().host ?? undefined);
    now();
    const unsub = useStore.subscribe((s, prev) => {
      if (s.host === prev.host) return;
      if (!s.running) {
        if (timer !== null) clearTimeout(timer);
        timer = null;
        now();
      } else if (timer === null) {
        timer = setTimeout(() => {
          timer = null;
          now();
        }, ms);
      }
    });
    return () => {
      unsub();
      if (timer !== null) clearTimeout(timer);
    };
  }, [ms]);
  return host;
}

type Cell = { kind: 'W'; i: number; j: number } | { kind: 'b'; j: number };
const sameCell = (a: Cell | null, b: Cell | null): boolean =>
  !!a && !!b && a.kind === b.kind && a.j === b.j && (a.kind === 'b' || a.i === (b as typeof a).i);

/** What a cell edits, for the node's type. */
function targetOf(node: Node, c: Cell): WeightTarget {
  if (node.type === 'scalew') return { kind: 'scalew', node: node.id };
  if (node.type === 'bias') return { kind: 'bias', node: node.id };
  return c.kind === 'W'
    ? { kind: 'W', node: node.id, i: c.i, j: c.j }
    : { kind: 'b', node: node.id, j: c.j };
}

const CellButton = memo(function CellButton(p: {
  testid: string;
  value: number;
  maxAbs: number;
  editable: boolean;
  selected: boolean;
  label: string;
  showText: boolean;
  target: WeightTarget;
  onSelect: () => void;
  onHover: (on: boolean) => void;
}) {
  const pct = tintPct(p.value, p.maxAbs);
  const style = {
    '--cell-bg': `color-mix(in srgb, var(${p.value < 0 ? '--weight-neg' : '--weight-pos'}) ${pct}%, var(--surface))`,
  } as CSSProperties;
  return (
    <button
      type="button"
      className="tk-wcell"
      data-testid={p.testid}
      data-editable={p.editable}
      data-selected={p.selected}
      data-value={p.value}
      aria-label={`${p.label}: ${p.value.toFixed(4)}`}
      aria-pressed={p.selected}
      title={`${p.label}: ${p.value.toFixed(4)}`}
      style={style}
      onPointerDown={(e) =>
        pointerDownWith({
          onTap: p.onSelect,
          ...weightDragHandlers(
            () => p.target,
            () => p.onSelect(),
          ),
        })(e)
      }
      onPointerEnter={() => p.onHover(true)}
      onPointerLeave={() => p.onHover(false)}
      onKeyDown={(e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          p.onSelect();
        }
      }}
    >
      {p.showText ? cellText(p.value) : ''}
    </button>
  );
});

function fmt(v: number): string {
  return v.toFixed(3).replace('-', '−');
}

function Stats({ label, values }: { label: string; values: ArrayLike<number> }) {
  const s = statsOf(values);
  if (!s) return null;
  return (
    <p className="tk-diag tk-wstats" data-testid={`weights-stats-${label}`}>
      {label}: min {fmt(s.min)} · max {fmt(s.max)} · mean {fmt(s.mean)} · std {fmt(s.std)}
    </p>
  );
}

function Pager(p: {
  label: string;
  total: number;
  page: number;
  onPage: (n: number) => void;
  testid: string;
}) {
  const w = windowOf(p.total, p.page);
  if (w.pages <= 1) return null;
  const cur = Math.min(Math.max(0, p.page), w.pages - 1);
  return (
    <span className="tk-wpager" data-testid={p.testid}>
      <button
        type="button"
        disabled={cur === 0}
        aria-label={`Previous ${p.label}`}
        onClick={() => p.onPage(cur - 1)}
      >
        {'‹'}
      </button>
      <span className="tk-diag">
        {p.label} {w.from}–{w.to - 1} of {p.total}
      </span>
      <button
        type="button"
        disabled={cur >= w.pages - 1}
        aria-label={`Next ${p.label}`}
        onClick={() => p.onPage(cur + 1)}
      >
        {'›'}
      </button>
    </span>
  );
}

const Panel = memo(function Panel({ node }: { node: Node }) {
  weightStats.renders++;
  const analysis = useAnalysis();
  const graph = useStore((s) => s.graph);
  const dataset = useStore((s) => s.dataset);
  const info = dataset.info;
  const running = useStore((s) => s.running);
  const diverged = useStore((s) => s.diverged);
  const canTrain = useStore((s) => s.canTrain);
  const preset = useStore((s) => s.preset);
  const close = useStore((s) => s.openWeights);
  // The paused / initial weights; while a run goes the throttled snapshot replaces them.
  const initial = useWeights(analysis, false);
  const live = useThrottledHost(LIVE_MS);
  const host = live ?? initial;
  const inW = analysis.inWidth[node.id];
  const grid: WeightGrid | null = useMemo(
    () => gridOf(host, node, inW, hostShape(host, node.id, dataset)),
    [host, node, inW, dataset],
  );
  const names = useMemo(
    () =>
      node.type === 'linear' && inW !== undefined
        ? rowNames(graph, node.id, inW, info.columns)
        : [],
    [graph, node, inW, info.columns],
  );
  // Escape closes the panel and hands focus back to the button that opened it (when still there).
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== 'Escape' || e.defaultPrevented) return;
      if ((e.target as Element | null)?.closest?.('input, select, textarea')) return;
      e.stopPropagation(); // this Escape only closes the panel; the piece stays selected
      document.querySelector<HTMLElement>('[data-testid="inspector-open-weights"]')?.focus();
      close(null);
    };
    document.addEventListener('keydown', onKey, true);
    return () => document.removeEventListener('keydown', onKey, true);
  }, [close]);
  const [sel, setSel] = useState<Cell | null>(null);
  const [hover, setHover] = useState<Cell | null>(null);
  const [rowPage, setRowPage] = useState(0);
  const [colPage, setColPage] = useState(0);
  // The same piece with other widths or another piece: start from the first page, nothing selected.
  useEffect(() => {
    setSel(null);
    setHover(null);
    setRowPage(0);
    setColPage(0);
  }, [node.id]);

  const locked = (preset?.locked ?? []).includes(node.id);
  const editable = !running && !diverged && canTrain && !locked;
  const rowW = grid ? windowOf(grid.rows, rowPage) : null;
  const colW = grid ? windowOf(grid.cols, colPage) : null;
  const all = useMemo(() => {
    if (!grid) return null;
    return Math.max(
      0,
      ...(grid.W ? Array.from(grid.W, Math.abs) : []),
      ...(grid.b ? Array.from(grid.b, Math.abs) : []),
    );
  }, [grid]);

  const title = node.type === 'linear' ? 'Weights' : node.type === 'scalew' ? 'Weight' : 'Bias';
  const valueAt = (c: Cell): number | undefined =>
    grid ? (c.kind === 'W' ? grid.W?.[c.i * grid.cols + c.j] : grid.b?.[c.j]) : undefined;
  const nameOf = (c: Cell): string => {
    if (node.type !== 'linear')
      return node.type === 'scalew' ? `Weight ${node.id}` : `Bias ${node.id}`;
    return c.kind === 'W'
      ? `${plainOf(names[c.i] ?? `input ${c.i}`)} to unit ${c.j}`
      : `bias of unit ${c.j}`;
  };
  const shown = hover ?? sel;
  const shownValue = shown ? valueAt(shown) : undefined;
  const selValue = sel ? valueAt(sel) : undefined;
  const showText = !!colW && colW.to - colW.from <= 16;
  const cols = colW ? colW.to - colW.from : 0;

  return (
    <aside className="tk-weights" data-testid="weights-panel" aria-label="Weight inspector">
      <button
        type="button"
        className="tk-inspector-close"
        aria-label="Close weight inspector"
        data-testid="weights-close"
        onClick={() => close(null)}
      >
        {'×'}
      </button>
      <h2 className="tk-inspector-title">
        {title} <code>{node.id}</code>
        {grid && node.type === 'linear' && (
          <span className="tk-muted">
            {' '}
            {grid.rows} to {grid.cols}
          </span>
        )}
      </h2>
      {!grid || !all || !rowW || !colW ? (
        <p className="tk-muted" data-testid="weights-none">
          No values yet.
        </p>
      ) : (
        <>
          {grid.W && grid.W.length > 1 && <Stats label="W" values={grid.W} />}
          {grid.b && grid.cols > 1 && <Stats label="b" values={grid.b} />}
          <div
            className="tk-wscroll"
            data-testid="weights-grid"
            role="grid"
            aria-label={`${title} of ${node.id}`}
            style={{ '--wcols': cols } as CSSProperties}
          >
            <div className="tk-wrow tk-whead" role="row">
              <span className="tk-wcorner" />
              {Array.from({ length: cols }, (_, k) => (
                <span key={k} className="tk-wlabel tk-wcolname">
                  {node.type === 'linear' ? `u${colW.from + k}` : ''}
                </span>
              ))}
            </div>
            {grid.W &&
              Array.from({ length: rowW.to - rowW.from }, (_, r) => {
                const i = rowW.from + r;
                return (
                  <div className="tk-wrow" role="row" key={i}>
                    <span className="tk-wlabel tk-wrowname" title={plainOf(names[i] ?? '')}>
                      {node.type === 'linear' ? <TexText>{names[i] ?? `input ${i}`}</TexText> : 'w'}
                    </span>
                    {Array.from({ length: cols }, (_, k) => {
                      const j = colW.from + k;
                      const c: Cell = { kind: 'W', i, j };
                      const v = grid.W?.[i * grid.cols + j] ?? 0;
                      return (
                        <CellButton
                          key={j}
                          testid={`wcell-${i}-${j}`}
                          value={v}
                          maxAbs={all}
                          editable={editable}
                          selected={sameCell(sel, c)}
                          label={nameOf(c)}
                          showText={showText}
                          target={targetOf(node, c)}
                          onSelect={() => setSel(c)}
                          onHover={(on) => setHover(on ? c : null)}
                        />
                      );
                    })}
                  </div>
                );
              })}
            {grid.b && (
              <div className="tk-wrow tk-wbias" role="row">
                <span className="tk-wlabel tk-wrowname">bias</span>
                {Array.from({ length: cols }, (_, k) => {
                  const j = colW.from + k;
                  const c: Cell = { kind: 'b', j };
                  return (
                    <CellButton
                      key={j}
                      testid={`wcell-b-${j}`}
                      value={grid.b?.[j] ?? 0}
                      maxAbs={all}
                      editable={editable}
                      selected={sameCell(sel, c)}
                      label={nameOf(c)}
                      showText={showText}
                      target={targetOf(node, c)}
                      onSelect={() => setSel(c)}
                      onHover={(on) => setHover(on ? c : null)}
                    />
                  );
                })}
              </div>
            )}
          </div>
          {(rowW.pages > 1 || colW.pages > 1) && (
            <div className="tk-wpagers">
              <Pager
                label="rows"
                total={grid.rows}
                page={rowPage}
                onPage={setRowPage}
                testid="weights-rows-pager"
              />
              <Pager
                label="units"
                total={grid.cols}
                page={colPage}
                onPage={setColPage}
                testid="weights-cols-pager"
              />
            </div>
          )}
          <p className="tk-diag tk-wreadout" data-testid="weights-readout">
            {shown && shownValue !== undefined
              ? `${nameOf(shown)} = ${shownValue.toFixed(4)}`
              : 'Hover or tap a cell for its exact value.'}
          </p>
          {sel && selValue !== undefined && editable && (
            <ValueField
              testid="weights-value-input"
              label="Set value"
              value={selValue}
              target={targetOf(node, sel)}
            />
          )}
          {!editable && (
            <p className="tk-diag" data-testid="weights-readonly">
              {locked ? 'Locked.' : running ? 'Values update while training runs.' : ''}
            </p>
          )}
        </>
      )}
    </aside>
  );
});

/**
 * The panel for `viewSlice.weightsOf`, or nothing. Mounted by the canvas; costs nothing while
 * closed. Closes itself when its piece is deleted.
 */
export function WeightInspector() {
  const id = useStore((s) => s.weightsOf);
  const node = useStore((s) => (id ? s.graph.nodes.find((n) => n.id === id) : undefined));
  const open = !!node && (node.type === 'linear' || node.type === 'scalew' || node.type === 'bias');
  const close = useStore((s) => s.openWeights);
  useEffect(() => {
    if (id && !open) close(null);
  }, [id, open, close]);
  if (!node || !open) return null;
  return <Panel node={node} />;
}
