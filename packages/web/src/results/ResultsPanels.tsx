import { Fragment, useRef, useState, type PointerEvent as ReactPointerEvent } from 'react';
import { startDrag } from '../gestures/drag.js';
import { shownSurfaceParams } from '../plot/lossSurface.js';
import { useStore } from '../state/store.js';
import { FitPlot } from './FitPlot.js';
import { LossCurves } from './LossCurves.js';
import { ExampleStrip } from './ExampleStrip.js';
import { ClassBars } from './ClassBars.js';
import { StepLog } from './StepLog.js';
import { LossCurve1D } from './LossCurve1D.js';
import { LossSurface } from './LossSurface.js';
import { TargetHistogram, useHistogramApplies } from './TargetHistogram.js';
import { useSplitTabs, useViewTab } from '../state/disclosure.js';
import { useLayoutMode } from '../layout/useLayoutMode.js';
import { ResizeHandle } from '../layout/ResizeHandle.js';
import {
  deltaAlong,
  dropIndex,
  extentAlong,
  midAlong,
  movePanel,
  panelOnTab,
  posAlong,
  pairValues,
  resizePair,
  stepPair,
  type Orientation,
  type PanelId,
  panelListed,
} from './panelLayout.js';
import styles from './ResultsPanels.module.css';

const TITLES: Record<PanelId, string> = {
  fit: 'Fit',
  loss: 'Loss',
  surface: 'Loss surface',
  'loss-1d': 'Loss vs parameter',
  histogram: 'Target',
  examples: 'Examples',
  'step-log': 'Steps',
  'class-bars': 'Classes',
};

function Panel({ id }: { id: PanelId }) {
  if (id === 'fit') return <FitPlot />;
  if (id === 'loss') return <LossCurves />;
  if (id === 'loss-1d') return <LossCurve1D />;
  if (id === 'examples') return <ExampleStrip />;
  if (id === 'class-bars') return <ClassBars />;
  if (id === 'step-log') return <StepLog />;
  if (id === 'histogram') return <TargetHistogram />;
  return <LossSurface />;
}

/** Presses on these inside a panel header are the panel's own controls, never a drag of the panel. */
const CONTROLS = 'button, input, select, textarea, label, a, [role="button"]';

/**
 * The results column (CONTRACTS.md section 10, "Results panels"): the fit, loss and loss-surface
 * panels in `viewSlice.panelOrder`, each with its share (`panelWeights`) of the column.
 * Drag a panel's header (its title or empty header space) up or down to reorder; drag the bar
 * between two panels to resize them. The loss surface only has a slot while the model has exactly
 * two parameters. Both gestures are Pointer Events with the usual slop, so they work with touch.
 */
export function ResultsPanels() {
  // Portrait: the panels sit side by side in a band under the canvas; landscape: stacked.
  const orient: Orientation = useLayoutMode() === 'portrait' ? 'row' : 'column';
  const order = useStore((s) => s.panelOrder);
  const weights = useStore((s) => s.panelWeights);
  const setOrder = useStore((s) => s.setPanelOrder);
  const setWeights = useStore((s) => s.setPanelWeights);
  const hasParams = useStore((s) => s.host !== null && s.analysis.paramNodes.length > 0);
  const hasSurface = useStore((s) =>
    s.host ? shownSurfaceParams(s.host, s.analysis.order, s.show?.ui) !== null : false,
  );
  const hasHistogram = useHistogramApplies();
  const classification = useStore((s) => s.dataset.info.kind !== 'regression');
  const columnRef = useRef<HTMLDivElement>(null);
  // The panel being dragged, how far it has moved, and its index among the other visible panels.
  const [lift, setLift] = useState<{ id: PanelId; d: number; to: number } | null>(null);

  const allowed = useStore((s) => s.show?.ui);
  const labels = useStore((s) => s.preset?.labels);
  const tab = useViewTab();
  const split = useSplitTabs();
  const panelTabs = useStore((s) => s.show?.panelTabs);
  const visible = order.filter(
    (p) =>
      panelOnTab(p, tab, split, panelTabs) &&
      panelListed(allowed, p) &&
      (p !== 'surface' || hasSurface) &&
      ((p !== 'loss-1d' && p !== 'step-log') ||
        (hasParams && allowed?.includes(`panel-${p}`) === true)) &&
      (p !== 'examples' || allowed?.includes('panel-examples') === true) &&
      (p !== 'class-bars' || (classification && allowed?.includes('panel-class-bars') === true)) &&
      (p !== 'histogram' || hasHistogram),
  );
  const visibleSet = new Set(visible);

  const slotOf = (id: PanelId): HTMLElement | null =>
    columnRef.current?.querySelector<HTMLElement>(`[data-panel="${id}"]`) ?? null;

  const onHeadDown = (id: PanelId, e: ReactPointerEvent<HTMLElement>) => {
    const target = e.target as Element;
    if (e.pointerType === 'mouse' && e.button !== 0) return;
    if (!target.closest('[data-panel-head]') || target.closest(CONTROLS)) return;
    const others = visible.filter((p) => p !== id);
    const landing = (y: number) =>
      dropIndex(
        others.map((p) => {
          const r = slotOf(p)?.getBoundingClientRect();
          return r ? midAlong(orient, r) : 0;
        }),
        y,
      );
    startDrag(e, {
      onStart: (p) => setLift({ id, d: deltaAlong(orient, p), to: landing(posAlong(orient, p)) }),
      onMove: (p) => setLift({ id, d: deltaAlong(orient, p), to: landing(posAlong(orient, p)) }),
      onEnd: (p) => {
        setLift(null);
        const to = landing(posAlong(orient, p));
        if (to !== visible.indexOf(id)) setOrder(movePanel(order, visibleSet, id, to));
      },
      onCancel: () => setLift(null),
    });
  };

  /** Pointer down on the separator between `a` and `b`: the move function for the drag. */
  const beginSplit = (a: PanelId, b: PanelId) => {
    const start = { ...weights };
    const extent = extentAlong(
      orient,
      columnRef.current?.getBoundingClientRect() ?? { width: 1, height: 1 },
    );
    return (delta: number) => setWeights(resizePair(start, visible, a, b, delta, extent));
  };

  const others = lift ? visible.filter((p) => p !== lift.id) : [];
  const before = lift ? others[lift.to] : undefined; // the panel it lands above; none: the end
  const last = others[others.length - 1];
  return (
    <div
      className={styles.column}
      data-orientation={orient}
      ref={columnRef}
      data-testid="results-panels"
    >
      {visible.map((id, i) => {
        const prev = visible[i - 1];
        const dragged = lift?.id === id;
        return (
          <Fragment key={id}>
            {prev !== undefined && (
              <ResizeHandle
                axis={orient === 'row' ? 'x' : 'y'}
                testId={`splitter-${prev}-${id}`}
                label={`Resize ${labels?.[`panel-${prev}`] ?? TITLES[prev]} and ${labels?.[`panel-${id}`] ?? TITLES[id]}`}
                {...pairValues(weights, visible, prev, id)}
                begin={() => beginSplit(prev, id)}
                onStep={(dir, big) => setWeights(stepPair(weights, visible, prev, id, dir, big))}
              />
            )}
            <div
              className={styles.slot}
              data-panel={id}
              data-lifted={dragged}
              data-drop-before={lift !== null && before === id}
              data-drop-after={lift !== null && before === undefined && last === id}
              style={{
                flexGrow: weights[id],
                ...(dragged && lift
                  ? { transform: `${orient === 'row' ? 'translateX' : 'translateY'}(${lift.d}px)` }
                  : {}),
              }}
              onPointerDown={(e) => onHeadDown(id, e)}
            >
              <Panel id={id} />
            </div>
          </Fragment>
        );
      })}
    </div>
  );
}
