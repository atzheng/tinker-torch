import { useEffect, useMemo, useRef, useState, useSyncExternalStore } from 'react';
import { inputPortsOf as inputPortsOfType, layoutLayers } from '@tinker/core';
import './canvas.css';
import { useEditShortcuts } from './actions.js';
import {
  dnd,
  editForDrop,
  inputPortAt,
  outputPortAt,
  resolveDrop,
  type DragPayload,
} from './dnd.js';
import { Inspector } from './Inspector.js';
import { LayerView } from './LayerView.js';
import { Toast } from './Toast.js';
import {
  edgeIndexOf,
  errorNodes,
  fitTransform,
  placeInspector,
  placeIssues,
  type InspectorPlacement,
  wireOfNode,
  type Transform,
} from './pure.js';
import { useAnalysis, useSelection, useWeights } from '../state/derived.js';
import { useStore } from '../state/store.js';

const DEFAULT_SIZE = { w: 800, h: 500 };
const INSPECTOR_W = 248;

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

function Callout(props: { x: number; y: number; viewW: number; text: string[]; testid: string }) {
  const half = 100;
  const left = Math.min(Math.max(props.x, half + 4), Math.max(half + 4, props.viewW - half - 4));
  return (
    <div className="tk-callout" data-testid={props.testid} style={{ left, top: props.y }}>
      {props.text.map((t) => (
        <p key={t}>{t}</p>
      ))}
    </div>
  );
}

/**
 * The canvas region: the layer view drawn to fit, issue callouts, and the inspector beside the
 * selected piece. Selection lives in the view slice (`selection`); tapping the background clears it.
 * Neuron view and the grain toggle are tinker-ran.1.
 */
export function Canvas() {
  const ref = useRef<HTMLDivElement>(null);
  const view = useElementSize(ref);
  const graph = useStore((s) => s.graph);
  const select = useStore((s) => s.select);
  const analysis = useAnalysis();
  const weights = useWeights(analysis);
  const selection = useSelection();
  const coarse = useCoarse();
  const apply = useStore((s) => s.apply);
  const selectNode = useStore((s) => s.selectNode);
  const drag = useSyncExternalStore(dnd.subscribe, dnd.get, () => null);
  useEditShortcuts();

  const layout = useMemo(() => layoutLayers(graph, analysis), [graph, analysis]);
  const t: Transform = useMemo(() => fitTransform(layout.size, view), [layout.size, view]);
  const placed = useMemo(
    () =>
      placeIssues(
        analysis.issues,
        new Set(layout.wires.map((w) => w.edge)),
        new Set(graph.nodes.map((n) => n.id)),
        (id) => wireOfNode(graph, analysis, id),
      ),
    [analysis, layout, graph],
  );
  const errNodes = useMemo(() => errorNodes(analysis.issues), [analysis]);
  const errEdges = useMemo(() => new Set(placed.byEdge.keys()), [placed]);

  // Drag and drop: the palette and the port handles ask the canvas what is under the pointer.
  useEffect(() => {
    const toAbstract = (cx: number, cy: number) => {
      const r = ref.current?.getBoundingClientRect();
      return {
        x: (cx - (r?.left ?? 0) - t.tx) / t.scale,
        y: (cy - (r?.top ?? 0) - t.ty) / t.scale,
      };
    };
    return dnd.setHandler({
      resolve: (payload: DragPayload, cx, cy) =>
        resolveDrop(
          payload,
          graph,
          layout,
          analysis.edgeWidth,
          t.scale,
          toAbstract(cx, cy),
          coarse ? 32 : 20,
        ),
      commit: (payload, target) => {
        const edit = editForDrop(graph, payload, target);
        if (!edit) return;
        const result = apply(edit);
        if (result.ok && result.select !== undefined) selectNode(result.select);
      },
    });
  }, [graph, layout, analysis, t, coarse, apply, selectNode]);

  const px = (x: number) => x * t.scale + t.tx;
  const py = (y: number) => y * t.scale + t.ty;

  // The inspector goes below the piece (or wire label), above when there is no room below, so it
  // does not cover the wires beside the piece.
  let anchor: InspectorPlacement | null = null;
  if (selection?.kind === 'node') {
    const box = layout.nodes[selection.id];
    if (box) {
      anchor = placeInspector(
        { cx: px(box.x + box.w / 2), top: py(box.y), bottom: py(box.y + box.h) },
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
      <svg
        className="tk-svg"
        width={view.w}
        height={view.h}
        data-testid="canvas-svg"
        onPointerDown={() => select(null)}
      >
        <g transform={`translate(${t.tx} ${t.ty}) scale(${t.scale})`}>
          <LayerView
            graph={graph}
            analysis={analysis}
            layout={layout}
            weights={weights}
            errorEdges={errEdges}
            errorNodes={errNodes}
            selection={selection}
            select={select}
            scale={t.scale}
            coarse={coarse}
            drag={drag}
          />
        </g>
        {drag?.payload.kind === 'wire' &&
          (() => {
            const from = outputPortAt(layout, drag.payload.from);
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
