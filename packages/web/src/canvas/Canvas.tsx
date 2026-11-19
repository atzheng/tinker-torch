import { useEffect, useMemo, useRef, useState, useSyncExternalStore } from 'react';
import {
  LAYOUT,
  clearPositions,
  moveNodes,
  inputPortsOf as inputPortsOfType,
  layoutLayers,
  layoutNeurons,
  parsePort,
  type LayerLayout,
} from '@tinker/core';
import './canvas.css';
import './pins.css';
import { useEditShortcuts } from './actions.js';
import {
  dnd,
  dropCandidates,
  editForDrop,
  pinCanLandOn,
  inputPortAt,
  outputPortAt,
  resolveDrop,
  wireSource,
  type DragPayload,
} from './dnd.js';
import { lockedIds } from '../state/locks.js';
import { Inspector } from './Inspector.js';
import { PinDisc } from './PinMark.js';
import { LayerView } from './LayerView.js';
import { NeuronView, UNIT_BTN_ROOM } from './NeuronView.js';
import { Toast } from './Toast.js';
import { HINTS } from '../hints.js';
import { startDrag } from '../gestures/drag.js';
import { useViewGestures } from './useViewGestures.js';
import { clampView, layerBaseView, neuronBaseView, NEURON_MAX_SCALE } from './viewport.js';
import {
  edgeIndexOf,
  errorNodes,
  fitTransform,
  PIN,
  PIN_ROOM_X,
  pinKey,
  pinsOf,
  placePins,
  placeInspector,
  placeIssues,
  type InspectorPlacement,
  type Transform,
} from './pure.js';
import { useAnalysis, useSelection, useWeights } from '../state/derived.js';
import { useStore } from '../state/store.js';

/** Pieces snap to this grid (layout units): the `--sp-2` spacing token. */
const SNAP = 8;
const snap = (v: number): number => Math.round(v / SNAP) * SNAP;

const DEFAULT_SIZE = { w: 800, h: 500 };
const INSPECTOR_W = 248;
/** Height in px of the strip above the drawing (the grain toggle sits in it). */
const HEADER_H = 44;

function useElementSize(ref: React.RefObject<HTMLElement | null>) {
  const [size, setSize] = useState(DEFAULT_SIZE);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const read = () => {
      const r = el.getBoundingClientRect();
      if (r.width > 0 && r.height > 0) setSize({ w: r.width, h: r.height });
    };
    read();
    if (typeof ResizeObserver === 'undefined') return;
    const ro = new ResizeObserver(read);
    ro.observe(el);
    return () => ro.disconnect();
  }, [ref]);
  return size;
}

const COARSE = '(pointer: coarse)';
const hasMatchMedia = () =>
  typeof window !== 'undefined' && typeof window.matchMedia === 'function';

/** Whether the primary pointer is coarse (touch); follows changes of the media query. */
function useCoarse(): boolean {
  return useSyncExternalStore(
    (notify) => {
      if (!hasMatchMedia()) return () => {};
      const mq = window.matchMedia(COARSE);
      mq.addEventListener('change', notify);
      return () => mq.removeEventListener('change', notify);
    },
    () => hasMatchMedia() && window.matchMedia(COARSE).matches,
    () => false,
  );
}

function Callout(props: {
  x: number;
  y: number;
  viewW: number;
  text: string[];
  testid: string;
  above?: boolean;
}) {
  const half = 100;
  const left = Math.min(Math.max(props.x, half + 4), Math.max(half + 4, props.viewW - half - 4));
  return (
    <div
      className="tk-callout"
      data-testid={props.testid}
      style={{
        left,
        top: props.y,
        ...(props.above ? { transform: 'translate(-50%, -100%)' } : {}),
      }}
    >
      {props.text.map((t) => (
        <p key={t}>{t}</p>
      ))}
    </div>
  );
}

/**
 * The canvas region: the layer view drawn to fit, issue callouts, and the inspector beside the
 * selected piece. Selection lives in the view slice (`selection`); tapping the background clears it.
 * `viewSlice.grain` picks the drawing: the layer view or the neuron view (tinker-ran.1).
 */
export function Canvas() {
  const ref = useRef<HTMLDivElement>(null);
  const view = useElementSize(ref);
  const graph = useStore((s) => s.graph);
  const select = useStore((s) => s.select);
  const analysis = useAnalysis();
  const grain = useStore((s) => s.grain);
  const setGrain = useStore((s) => s.setGrain);
  // Neuron view repaints its edges from `host` itself, so the canvas does not follow it per frame.
  const weights = useWeights(analysis, grain === 'layer');
  const selection = useSelection(analysis);
  const coarse = useCoarse();
  const apply = useStore((s) => s.apply);
  const selectNode = useStore((s) => s.selectNode);
  const drag = useSyncExternalStore(dnd.subscribe, dnd.get, () => null);
  const armed = useSyncExternalStore(dnd.subscribe, dnd.armed, () => null);
  const canUndo = useStore((s) => s.canUndo);
  useEditShortcuts();
  useEffect(() => () => dnd.cancel(), []); // unmounted mid-drag: leave no preview
  useEffect(() => () => dnd.arm(null), []);
  // Tap-to-add ends when the ground moves under it: any graph change (edit, undo, redo, preset or
  // file load), a grain switch, or a pan or zoom (`userView`).
  const [edited, setEdited] = useState(false);
  useEffect(() => {
    if (canUndo) setEdited(true);
  }, [canUndo]);

  // A piece being moved: where its box's top-left is now, in layout units (not yet in the graph).
  const [moving, setMoving] = useState<{ id: string; x: number; y: number } | null>(null);
  // The graph as drawn: while a piece is moved it already sits at the pointer, wires follow.
  const shown = useMemo(
    () =>
      moving
        ? {
            ...graph,
            positions: {
              ...graph.positions,
              [moving.id]: [moving.x, moving.y] as [number, number],
            },
          }
        : graph,
    [graph, moving],
  );
  const layerLayout = useMemo(() => layoutLayers(shown, analysis), [shown, analysis]);
  const neuronLayout = useMemo(
    () => (grain === 'neuron' ? layoutNeurons(graph, analysis) : null),
    [grain, graph, analysis],
  );
  // Where the pins sit (real pins only; LayerView adds the add pins of a selected piece) and how
  // far they reach, so fit and pan bounds include them.
  const pinSpots = useMemo(
    () => placePins(layerLayout.nodes, pinsOf(graph), graph.positions !== undefined),
    [layerLayout, graph],
  );
  const pinExtent = useMemo((): [number, number] => {
    let w = 0;
    let h = 0;
    // Without hand-placed pieces the automatic layout already leaves room (PIN_ROOM_X).
    if (graph.positions) {
      for (const [x, y] of pinSpots.values()) {
        w = Math.max(w, x + PIN.r + 2);
        h = Math.max(h, y + PIN.r + 8);
      }
    }
    return [w, h];
  }, [pinSpots, graph.positions]);
  // Boxes and wires of the drawing on screen. The neuron layout has the same shape, so drops,
  // callouts and the inspector anchor work in both grains.
  const layout: LayerLayout = useMemo(
    () =>
      neuronLayout
        ? {
            nodes: neuronLayout.boxes,
            wires: neuronLayout.wires,
            // room under the drawing for the unit +/- buttons
            size: [neuronLayout.size[0], neuronLayout.size[1] + UNIT_BTN_ROOM],
          }
        : {
            ...layerLayout,
            // room for a pin on the last column, and for pins placed further out
            size: [
              Math.max(layerLayout.size[0] + PIN_ROOM_X, pinExtent[0]),
              Math.max(layerLayout.size[1], pinExtent[1]),
            ],
          },
    [neuronLayout, layerLayout, pinExtent],
  );
  // A header strip above the drawing holds the grain toggle, so it never covers a piece or wire.
  const isNeuron = neuronLayout !== null;
  const fit: Transform = useMemo(() => {
    const f = fitTransform(layout.size, { w: view.w, h: Math.max(1, view.h - HEADER_H) });
    return { ...f, ty: f.ty + HEADER_H };
  }, [layout.size, view]);
  // Neuron view only: the base is the fit (never below 32 px unit cells on touch); a pinch, the
  // wheel or a background drag moves `userView` away from it, `Reset view` clears it.
  const base: Transform = useMemo(
    () =>
      isNeuron
        ? neuronBaseView(fit, layout.size, view, HEADER_H, coarse, LAYOUT.unitPitch)
        : layerBaseView(fit, layout.size, view, HEADER_H),
    [isNeuron, fit, layout.size, view, coarse],
  );
  // null: the start view (`base`); 'fit': the whole drawing; a Transform: moved by the user.
  const [userView, setUserViewRaw] = useState<Transform | 'fit' | null>(null);
  const setUserView = (v: Transform | 'fit' | null) => setUserViewRaw(v);
  const loadEpoch = useStore((st) => st.loadEpoch);
  useEffect(() => setUserViewRaw(null), [grain, loadEpoch]);
  useEffect(() => dnd.arm(null), [graph, grain, loadEpoch, userView]);
  const clampT = (x: Transform) => clampView(x, layout.size, view, HEADER_H);
  const t: Transform = useMemo(
    () =>
      userView === 'fit' ? fit : userView ? clampView(userView, layout.size, view, HEADER_H) : base,
    [userView, fit, base, layout.size, view],
  );
  const tRef = useRef(t);
  tRef.current = t;
  const clampRef = useRef(clampT);
  clampRef.current = clampT;
  useViewGestures(ref, {
    enabled: true,
    current: () => tRef.current,
    min: Math.min(fit.scale, base.scale),
    max: Math.max(NEURON_MAX_SCALE, base.scale),
    clamp: (x) => clampRef.current(x),
    set: setUserView,
  });
  // Wires and pieces a dragged or armed palette entry could land on; for a wire drag, the pieces
  // that have an input to connect to.
  const candidates = useMemo(() => {
    const entry = drag?.payload.kind === 'palette' ? drag.payload.entry : armed;
    if (entry) return dropCandidates(entry, graph);
    if (drag?.payload.kind === 'pin') {
      const pin = drag.payload;
      return {
        edges: new Set<number>(),
        nodes: new Set(graph.nodes.filter((n) => pinCanLandOn(graph, pin, n.id)).map((n) => n.id)),
      };
    }
    const src = drag ? wireSource(drag.payload) : undefined;
    if (src === undefined) return null;
    return {
      edges: new Set<number>(),
      nodes: new Set(
        graph.nodes
          .filter((n) => n.id !== src && inputPortsOfType(n.type).length > 0)
          .map((n) => n.id),
      ),
    };
  }, [drag?.payload, armed, graph]);
  // Tap-to-add: while an entry is armed, a tap on the drawing places it (or cancels when nothing fits).
  const onArmedDown = (e: React.PointerEvent<SVGSVGElement>) => {
    if (!dnd.armed()) return;
    if (e.pointerType === 'mouse' && e.button !== 0) return;
    e.stopPropagation();
    startDrag(e, { onTap: (p) => dnd.tapAt(p.x, p.y), onCancel: () => dnd.arm(null) });
  };
  const onBackgroundDown = (e: React.PointerEvent<SVGSVGElement>) => {
    if (e.pointerType === 'mouse' && e.button !== 0) return;
    const t0 = tRef.current;
    startDrag(e, {
      onTap: () => select(null),
      onMove: (p) => setUserView(clampT({ ...t0, tx: t0.tx + p.dx, ty: t0.ty + p.dy })),
    });
  };
  // Moving a piece: positions are layout units, so the pointer's movement is divided by the zoom.
  // Snapped to the 8-unit grid (the `--sp-2` spacing token), kept inside the margin, committed as
  // one edit (one undo step) on release, and only when the piece actually ended up elsewhere.
  const moveStart = useRef<{ id: string; x: number; y: number } | null>(null);
  const moveTo = (id: string, dx: number, dy: number) => {
    const s0 = moveStart.current;
    if (!s0 || s0.id !== id) return null;
    const x = Math.max(LAYOUT.pad, snap(s0.x + dx / tRef.current.scale));
    const y = Math.max(LAYOUT.pad, snap(s0.y + dy / tRef.current.scale));
    return { id, x, y, from: s0 };
  };
  const mover = {
    start: (id: string) => {
      const b = layerLayout.nodes[id];
      if (!b) return;
      moveStart.current = { id, x: b.x, y: b.y };
      // Freeze the view while pieces move: the drawing's size may change, the zoom must not.
      setUserView(tRef.current);
      setMoving({ id, x: b.x, y: b.y });
    },
    move: (id: string, dx: number, dy: number) => {
      const m = moveTo(id, dx, dy);
      if (m) setMoving({ id, x: m.x, y: m.y });
    },
    end: (id: string, dx: number, dy: number) => {
      const m = moveTo(id, dx, dy);
      moveStart.current = null;
      setMoving(null);
      if (!m) return;
      if (m.x !== m.from.x || m.y !== m.from.y) {
        const r = apply(moveNodes(graph, { [id]: [m.x, m.y] }));
        if (!r.ok) return;
      }
      selectNode(id);
    },
    cancel: () => {
      moveStart.current = null;
      setMoving(null);
    },
  };
  // Layer view draws pin issues at the pin; neuron view has no pins, so they stay at the piece.
  const placed = useMemo(
    () =>
      placeIssues(
        analysis.issues,
        new Set(layout.wires.map((w) => w.edge)),
        new Set(graph.nodes.map((n) => n.id)),
        undefined,
        isNeuron
          ? undefined
          : (pin) => {
              const ref = pin.role === 'loss' ? graph.loss : graph.predictions[pin.index];
              return ref === null || ref === undefined ? undefined : parsePort(ref).node;
            },
      ),
    [analysis, layout, graph, isNeuron],
  );
  const errNodes = useMemo(() => errorNodes(analysis.issues, !isNeuron), [analysis, isNeuron]);
  const errPins = useMemo(() => new Set(placed.byPin.keys()), [placed]);
  const errEdges = useMemo(() => new Set(placed.byEdge.keys()), [placed]);

  const toAbstract = useMemo(
    () => (cx: number, cy: number) => {
      const r = ref.current?.getBoundingClientRect();
      return {
        x: (cx - (r?.left ?? 0) - t.tx) / t.scale,
        y: (cy - (r?.top ?? 0) - t.ty) / t.scale,
      };
    },
    [t],
  );

  // Drag and drop: the palette and the port handles ask the canvas what is under the pointer.
  useEffect(() => {
    return dnd.setHandler({
      resolve: (payload: DragPayload, cx, cy) => {
        const target = resolveDrop(
          payload,
          graph,
          layout,
          analysis.edgeWidth,
          t.scale,
          toAbstract(cx, cy),
          coarse ? 32 : 20,
        );
        if (target?.kind !== 'free') return target;
        // "Free" (a ready-made group on empty canvas) only counts inside the drawing.
        const r = ref.current?.getBoundingClientRect();
        const inside = r && cx >= r.left && cx <= r.right && cy >= r.top && cy <= r.bottom;
        return inside ? target : null;
      },
      commit: (payload, target) => {
        const edit = editForDrop(graph, payload, target, { locked: lockedIds() });
        if (!edit) return;
        const result = apply(edit);
        if (result.ok && result.select !== undefined) selectNode(result.select);
      },
    });
  }, [graph, layout, analysis, t, coarse, apply, selectNode, toAbstract]);

  const px = (x: number) => x * t.scale + t.tx;
  const py = (y: number) => y * t.scale + t.ty;

  // The inspector goes below the piece (or wire label), above when there is no room below, so it
  // does not cover the wires beside the piece.
  let anchor: InspectorPlacement | null = null;
  if (selection?.kind === 'node') {
    const box = layout.nodes[selection.id];
    if (box) {
      anchor = placeInspector(
        // clear of the pins (and the add pins) above and below the piece
        { cx: px(box.x + box.w / 2), top: py(box.y - 30), bottom: py(box.y + box.h + 30) },
        view,
        INSPECTOR_W,
      );
    }
  } else if (selection?.kind === 'pin') {
    const box = layout.nodes[selection.node];
    if (box) {
      const [cx, cy] = pinSpots.get(pinKey(selection.role, selection.node)) ?? [0, 0];
      // A ŷ pin sits over the Target and the Subtract inputs below it: open the inspector above.
      const above = selection.role === 'pred' && py(cy) - 20 >= 230;
      anchor = placeInspector(
        above
          ? { cx: px(cx), top: py(cy) - 20, bottom: view.h + 1000 }
          : { cx: px(cx), top: py(cy) - 16, bottom: py(cy) + 16 },
        view,
        INSPECTOR_W,
      );
    }
  } else if (selection?.kind === 'edge' || selection?.kind === 'unit') {
    let at: [number, number] | undefined;
    if (selection.kind === 'unit') {
      const u = neuronLayout?.units.find(
        (x) => x.node === selection.node && x.index === selection.index,
      );
      if (u) at = [u.x, u.y];
    } else {
      const e = neuronLayout?.edges.find(
        (x) => x.node === selection.node && x.i === selection.i && x.j === selection.j,
      );
      if (e) at = [(e.from[0] + e.to[0]) / 2, (e.from[1] + e.to[1]) / 2];
    }
    if (at) {
      anchor = placeInspector(
        { cx: px(at[0]), top: py(at[1]) - 14, bottom: py(at[1]) + 14 },
        view,
        INSPECTOR_W,
      );
    }
  } else if (selection?.kind === 'wire') {
    const w = layout.wires.find((x) => x.edge === edgeIndexOf(graph, selection.from, selection.to));
    if (w) {
      const y = py(w.labelAt[1]);
      anchor = placeInspector(
        { cx: px(w.labelAt[0]), top: y - 10, bottom: y + 10 },
        view,
        INSPECTOR_W,
      );
    }
  }

  return (
    <div
      ref={ref}
      className="tk-canvas"
      data-testid="canvas"
      onKeyDown={(e) => {
        if (e.key === 'Escape') select(null);
      }}
    >
      <button
        type="button"
        className="tk-grain"
        data-testid="grain-toggle"
        aria-pressed={grain === 'neuron'}
        title="Switch between layer view and neuron view"
        onClick={() => setGrain(grain === 'neuron' ? 'layer' : 'neuron')}
      >
        Neurons
      </button>
      {!isNeuron &&
        Math.abs(t.scale - fit.scale) + Math.abs(t.tx - fit.tx) + Math.abs(t.ty - fit.ty) > 1 && (
          <button
            type="button"
            className="tk-grain tk-resetview"
            data-testid="fit-view"
            title="Fit the whole graph in view"
            onClick={() => setUserView('fit')}
          >
            Fit
          </button>
        )}
      {!isNeuron && graph.positions && (
        <button
          type="button"
          className="tk-grain tk-resetview tk-tidy"
          data-testid="tidy-layout"
          title="Put every piece back where the automatic layout puts it"
          onClick={() => {
            const r = apply(clearPositions(graph));
            if (r.ok) setUserView('fit');
          }}
        >
          Tidy
        </button>
      )}
      {isNeuron && userView && (
        <button
          type="button"
          className="tk-grain tk-resetview"
          data-testid="reset-view"
          title="Reset pan and zoom"
          onClick={() => setUserView(null)}
        >
          Reset view
        </button>
      )}
      {!isNeuron &&
        (
          [
            ['left', t.tx < -1],
            ['right', t.tx + layout.size[0] * t.scale > view.w + 1],
            ['top', t.ty < HEADER_H - 1],
            ['bottom', t.ty + layout.size[1] * t.scale > view.h + 1],
          ] as const
        ).map(([edge, on]) =>
          on ? (
            <div key={edge} className="tk-clip" data-edge={edge} data-testid={`clip-${edge}`} />
          ) : null,
        )}
      <svg
        className="tk-svg"
        width={view.w}
        height={view.h}
        data-testid="canvas-svg"
        onPointerDown={onBackgroundDown}
        onPointerDownCapture={onArmedDown}
      >
        <g transform={`translate(${t.tx} ${t.ty}) scale(${t.scale})`}>
          {neuronLayout ? (
            <NeuronView
              graph={graph}
              analysis={analysis}
              layout={neuronLayout}
              fallback={weights}
              errorEdges={errEdges}
              errorNodes={errNodes}
              selection={selection}
              select={select}
              scale={t.scale}
              coarse={coarse}
              toAbstract={toAbstract}
            />
          ) : (
            <LayerView
              graph={graph}
              analysis={analysis}
              layout={layerLayout}
              weights={weights}
              errorEdges={errEdges}
              errorNodes={errNodes}
              errorPins={errPins}
              selection={selection}
              select={select}
              scale={t.scale}
              coarse={coarse}
              drag={drag}
              candidates={candidates}
              mover={mover}
              moving={moving?.id ?? null}
            />
          )}
        </g>
        {drag?.payload.kind === 'pin' &&
          (() => {
            const box = ref.current?.getBoundingClientRect();
            if (!box) return null;
            return (
              <g
                className="tk-pin-preview"
                data-testid="pin-preview"
                data-role={drag.payload.kind === 'pin' ? drag.payload.role : undefined}
                data-target={drag.target ? 'node' : 'none'}
                pointerEvents="none"
              >
                <PinDisc
                  role={drag.payload.kind === 'pin' ? drag.payload.role : 'pred'}
                  cx={drag.x - box.left}
                  cy={drag.y - box.top}
                  r={14}
                />
              </g>
            );
          })()}
        {drag &&
          wireSource(drag.payload) !== undefined &&
          (() => {
            const from = outputPortAt(layout, wireSource(drag.payload) ?? '');
            const box = ref.current?.getBoundingClientRect();
            if (!from || !box) return null;
            const tg = drag.target;
            let end: [number, number] = [drag.x - box.left, drag.y - box.top];
            if (tg?.kind === 'port') {
              const node = graph.nodes.find((n) => n.id === tg.node);
              const k = node
                ? Math.max(
                    0,
                    inputPortsOfType(node.type).findIndex(
                      (p) => tg.port === node.id || tg.port === `${node.id}.${p}`,
                    ),
                  )
                : 0;
              const at =
                node && inputPortAt(layout, node.id, k, inputPortsOfType(node.type).length);
              if (at) end = [px(at[0]), py(at[1])];
            }
            return (
              <path
                className="tk-wire-preview"
                data-testid="wire-preview"
                d={`M${px(from[0])} ${py(from[1])} L${end[0]} ${end[1]}`}
              />
            );
          })()}
      </svg>
      {armed && (
        <div className="tk-armed" data-testid="armed-hint" role="status">
          {armed.kind === 'macro'
            ? `Tap a wire or empty canvas to add ${armed.label}.`
            : `Tap a highlighted wire or piece to place ${armed.label}.`}
        </div>
      )}
      {graph.nodes.length === 0 ? (
        <div className="tk-empty" data-testid="empty-hint">
          {HINTS.empty}
        </div>
      ) : (
        !edited && (
          <div className="tk-hints" data-testid="canvas-hints">
            <p>{HINTS.insert}</p>
            {!isNeuron && <p>{HINTS.connect}</p>}
            {!isNeuron && <p>{HINTS.rewire}</p>}
          </div>
        )
      )}
      <Toast />
      {placed.global.length > 0 && (
        <div className="tk-global-issues" data-testid="issue-global" role="status">
          {placed.global.map((m) => (
            <p key={m}>{m}</p>
          ))}
        </div>
      )}
      {[...placed.byEdge].map(([edge, text]) => {
        const w = layout.wires.find((x) => x.edge === edge);
        if (!w) return null;
        return (
          <Callout
            key={`e${edge}`}
            testid={`issue-wire-${edge}`}
            x={px(w.labelAt[0])}
            y={py(w.labelAt[1]) + 14}
            viewW={view.w}
            text={text}
          />
        );
      })}
      {[...placed.byPin].map(([key, text]) => {
        const [role, id] = key.split(':') as ['pred' | 'loss', string];
        const box = id ? layout.nodes[id] : undefined;
        if (!box) return null;
        const [cx, cy] = pinSpots.get(key) ?? [0, 0];
        return (
          <Callout
            key={`p${key}`}
            testid={`issue-pin-${role}-${id}`}
            x={px(cx)}
            y={py(cy) + (role === 'pred' ? -16 : 16)}
            above={role === 'pred'}
            viewW={view.w}
            text={text}
          />
        );
      })}
      {[...placed.byNode].map(([id, text]) => {
        const box = layout.nodes[id];
        if (!box) return null;
        return (
          <Callout
            key={`n${id}`}
            testid={`issue-node-${id}`}
            x={px(box.x + box.w / 2)}
            y={py(box.y + box.h) + 8}
            viewW={view.w}
            text={text}
          />
        );
      })}
      {selection && anchor && (
        <div className="tk-inspector-pos" style={{ ...anchor, width: INSPECTOR_W }}>
          <Inspector
            selection={selection}
            analysis={analysis}
            weights={weights}
            wireMessages={
              selection.kind === 'wire'
                ? (placed.byEdge.get(edgeIndexOf(graph, selection.from, selection.to)) ?? [])
                : []
            }
          />
        </div>
      )}
    </div>
  );
}
