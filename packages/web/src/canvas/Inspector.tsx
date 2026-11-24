import { useEffect, useRef, type ComponentType } from 'react';
import {
  addPrediction,
  docViewFor,
  parsePort,
  setLoss,
  type Analysis,
  type Graph,
  type HostParams,
  type Node,
  type PieceType,
} from '@tinker/core';
import { WEB_PIECES } from './drawings/index.js';
import { deleteSelection } from './actions.js';
import { lockedIds } from '../state/locks.js';
import { pinLive } from './pure.js';
import { usePinRoles } from '../state/disclosure.js';
import { biasAt, weightAt } from './neuronPure.js';
import { edgeIndexOf } from './pure.js';
import { InspectorMatrix } from './InspectorMatrix.js';
import { ValueField } from './ValueField.js';
import { TexText } from '../tex/Tex.js';
import { hostShape } from '../state/hostShape.js';
import { useStore } from '../state/store.js';
import { PieceCard, cardOwner, useCardOpen } from '../palette/PieceCard.js';
import type { Selection } from '../state/viewSlice.js';

export const PIECE_LABEL: Record<PieceType, string> = {
  input: 'Input',
  target: 'Target',
  feature: 'Feature',
  linear: 'Linear',
  activation: 'Activation',
  dropout: 'Dropout',
  add: 'Add',
  link: 'Link',
  subtract: 'Subtract',
  square: 'Square',
  abs: 'Absolute value',
  huber: 'Huber',
  log_loss: 'Log loss',
  cross_entropy: 'Cross-entropy',
  mean: 'Mean',
  scale: 'Scale',
  weights: 'Weights',
  norm: 'Norm',
  column: 'Column',
  scalew: 'Weight',
  bias: 'Bias',
  sum: 'Sum',
  onehot: 'One-hot',
  lookup: 'Lookup',
  tree: 'Tree',
};

function Issues({ messages }: { messages: string[] }) {
  if (messages.length === 0) return null;
  return (
    <ul className="tk-issues" data-testid="inspector-issues">
      {messages.map((m) => (
        <li key={m}>{m}</li>
      ))}
    </ul>
  );
}

/**
 * The "?" in a piece inspector's title row, next to the close button (CONTRACTS.md section 15). Opens
 * the live card for the node beside the inspector; it follows width edits (it re-renders with the
 * graph and analysis) and closes when the selection moves to another node or away.
 */
function NodeDoc({ node, analysis }: { node: Node; analysis: Analysis }) {
  const id = 'inspector';
  const open = useCardOpen(id);
  const btn = useRef<HTMLButtonElement>(null);
  const graph = useStore((s) => s.graph);
  const data = useStore((s) => s.dataInfo);
  useEffect(() => () => cardOwner.close(id), [node.id]);
  // The positioned box (clipped to the canvas), not the aside, which may be taller when it scrolls.
  const panel = btn.current?.closest<HTMLElement>('.tk-inspector-pos, .tk-inspector') ?? null;
  const canvas = btn.current?.closest<HTMLElement>('.tk-canvas') ?? null;
  return (
    <>
      <button
        ref={btn}
        type="button"
        className="tk-doc-btn tk-inspector-doc"
        data-testid="piece-doc-inspector"
        aria-label={`About ${PIECE_LABEL[node.type]}`}
        aria-expanded={open}
        aria-controls={open ? 'piece-card-inspector' : undefined}
        onClick={() => (open ? cardOwner.close(id) : cardOwner.open(id))}
      >
        ?
      </button>
      {open && panel && canvas && (
        <PieceCard
          id="piece-card-inspector"
          view={docViewFor(node, analysis, data, graph)}
          opener={btn.current}
          placement={{
            kind: 'inspector',
            panel,
            canvas,
            piece: canvas.querySelector(`[data-testid="node-${node.id}"]`),
          }}
          onClose={() => cardOwner.close(id)}
        />
      )}
    </>
  );
}

function NodeInspector({ node, analysis }: { node: Node; analysis: Analysis }) {
  const Fields = WEB_PIECES[node.type].Fields as ComponentType<{ node: Node }>;
  const inW = analysis.inWidth[node.id];
  const count = analysis.params[node.id];
  const messages = analysis.issues
    .filter((i) => i.at.node === node.id && !i.at.pin)
    .map((i) => i.message);
  const graph = useStore((s) => s.graph);
  const apply = useStore((s) => s.apply);
  const select = useStore((s) => s.select);
  const openWeights = useStore((s) => s.openWeights);
  const roles = usePinRoles();
  const docs = useStore((s) => s.show?.docs) !== false; // `show.docs: false` removes the "?"
  const canPred = roles.includes('pred') && !pinLive(graph, 'pred', node.id);
  const canLoss = roles.includes('loss') && !pinLive(graph, 'loss', node.id);
  const pin = (role: 'pred' | 'loss') => {
    const opts = { locked: lockedIds() };
    const r = role === 'pred' ? addPrediction(graph, node.id, opts) : setLoss(graph, node.id, opts);
    if (apply(r).ok) select({ kind: 'pin', role, node: node.id });
  };
  return (
    <>
      {docs && <NodeDoc node={node} analysis={analysis} />}
      <h2 className="tk-inspector-title" data-docs={docs}>
        {PIECE_LABEL[node.type]} <code>{node.id}</code>
      </h2>
      {node.type === 'linear' && (
        <p data-testid="inspector-params" className="tk-diag">
          <TexText>
            {inW === undefined || count === undefined
              ? 'Parameters: unknown until the input width is known.'
              : `$${inW} \\times ${node.out}${node.bias ? ` + ${node.out}` : ''} = ${count}$ parameters`}
          </TexText>
        </p>
      )}
      {inW !== undefined && node.type !== 'linear' && (
        <p className="tk-diag" data-testid="inspector-width-in">
          Width {inW}
        </p>
      )}
      <Fields node={node} />
      {node.type === 'linear' && <InspectorMatrix node={node} />}
      {(node.type === 'linear' || node.type === 'scalew' || node.type === 'bias') && (
        <button
          type="button"
          className="tk-inspect-weights"
          data-testid="inspector-open-weights"
          onClick={() => openWeights(node.id)}
        >
          Inspect weights
        </button>
      )}
      {(canPred || canLoss) && (
        <div className="tk-pin-actions">
          {canPred && (
            <button type="button" data-testid="inspector-pin-pred" onClick={() => pin('pred')}>
              Make this the prediction {'\u0177'}
            </button>
          )}
          {canLoss && (
            <button type="button" data-testid="inspector-pin-loss" onClick={() => pin('loss')}>
              Make this the loss L
            </button>
          )}
        </div>
      )}
      <Issues messages={messages} />
    </>
  );
}

/**
 * A selected role pin. The L pin is moved (drag it, or "Make this the loss L" on another piece), never removed;
 * a ŷ pin can also be removed here, or with Delete. Shows the pin's own issues.
 */
function PinInspector({
  role,
  node,
  analysis,
  graph,
}: {
  role: 'pred' | 'loss';
  node: string;
  analysis: Analysis;
  graph: Graph;
}) {
  const index = graph.predictions.findIndex((p) => parsePort(p).node === node);
  const messages = analysis.issues
    .filter((i) =>
      role === 'loss'
        ? i.at.pin?.role === 'loss'
        : i.at.pin?.role === 'prediction' && i.at.pin.index === index,
    )
    .map((i) => i.message);
  return (
    <>
      <h2 className="tk-inspector-title">
        {role === 'pred' ? 'Prediction' : 'Loss'}{' '}
        <TexText>{role === 'pred' ? '$\\hat{y}$' : '$L$'}</TexText>
      </h2>
      <p className="tk-muted">
        on <code>{node}</code>
      </p>
      <Issues messages={messages} />
    </>
  );
}

function WireInspector({
  graph,
  analysis,
  from,
  to,
  messages,
}: {
  graph: Graph;
  analysis: Analysis;
  from: string;
  to: string;
  messages: string[];
}) {
  const i = edgeIndexOf(graph, from, to);
  const width = i >= 0 ? analysis.edgeWidth[i] : null;
  return (
    <>
      <h2 className="tk-inspector-title">Wire</h2>
      <p className="tk-muted">
        <code>{from}</code> to <code>{to}</code>
      </p>
      <p data-testid="inspector-width" className="tk-diag">
        {width === null || width === undefined ? 'Width unknown' : `Width ${width}`}
      </p>
      <Issues messages={messages} />
    </>
  );
}

/** The live value of a weight; follows the train slice's `host` while it runs. */
function EdgeInspector(p: {
  graph: Graph;
  analysis: Analysis;
  node: string;
  i: number;
  j: number;
  fallback: HostParams | undefined;
}) {
  const host = useStore((s) => s.host) ?? p.fallback;
  const inW = p.analysis.inWidth[p.node];
  const n = p.graph.nodes.find((x) => x.id === p.node);
  const w =
    n?.type === 'linear' && inW !== undefined
      ? weightAt(
          host,
          p.node,
          p.i,
          p.j,
          inW,
          n.out,
          hostShape(host, p.node, useStore.getState().dataset),
        )
      : undefined;
  return (
    <>
      <h2 className="tk-inspector-title">Weight</h2>
      <p className="tk-muted">
        <code>{p.node}</code>: input {p.i} to unit {p.j}
      </p>
      <p data-testid="inspector-weight" className="tk-diag">
        {w === undefined ? 'No value yet' : `Value ${w.toFixed(4)}`}
      </p>
      {w !== undefined && (
        <ValueField
          testid="inspector-weight-input"
          label="Set value"
          value={w}
          target={{ kind: 'W', node: p.node, i: p.i, j: p.j }}
        />
      )}
    </>
  );
}

function UnitInspector(p: {
  graph: Graph;
  analysis: Analysis;
  node: string;
  index: number;
  fallback: HostParams | undefined;
}) {
  const host = useStore((s) => s.host) ?? p.fallback;
  const n = p.graph.nodes.find((x) => x.id === p.node);
  if (!n) return null;
  const b = n.type === 'linear' && n.bias ? biasAt(host, n.id, p.index) : undefined;
  const inW = p.analysis.inWidth[n.id];
  return (
    <>
      <h2 className="tk-inspector-title">
        Unit {p.index} <code>{n.id}</code>
      </h2>
      <p className="tk-muted">{PIECE_LABEL[n.type]}</p>
      {n.type === 'linear' && inW !== undefined && (
        <p data-testid="inspector-fanin" className="tk-diag">
          {inW} incoming weights
        </p>
      )}
      {b !== undefined && (
        <p data-testid="inspector-bias" className="tk-diag">
          Bias {b.toFixed(4)}
        </p>
      )}
      {b !== undefined && (
        <ValueField
          testid="inspector-bias-input"
          label="Set bias"
          value={b}
          target={{ kind: 'b', node: n.id, j: p.index }}
        />
      )}
    </>
  );
}

/**
 * Settings and facts for the selected piece or wire. Positioned by the canvas beside the piece.
 * Every change goes through core edits and `graphSlice.apply` (the Fields do that), so undo
 * restores it. Hover information (wire width) is also here, for touch (CONTRACTS.md section 10).
 */
export function Inspector({
  selection,
  analysis,
  wireMessages,
  weights,
}: {
  selection: Selection;
  analysis: Analysis;
  /** The sentences the canvas shows on the selected wire (`placeIssues().byEdge`), so both agree. */
  wireMessages: string[];
  /** Weights shown for a selected edge or unit until the train slice has its own. */
  weights?: HostParams | undefined;
}) {
  const graph = useStore((s) => s.graph);
  const select = useStore((s) => s.select);
  const roles = usePinRoles();
  const docs = useStore((s) => s.show?.docs) !== false;
  const node =
    selection.kind === 'node' ? graph.nodes.find((n) => n.id === selection.id) : undefined;
  return (
    <aside
      className="tk-inspector"
      data-testid="inspector"
      aria-label="Inspector"
      data-docs={docs && selection.kind === 'node'}
    >
      <button
        type="button"
        className="tk-inspector-close"
        aria-label="Close inspector"
        data-testid="inspector-close"
        onClick={() => select(null)}
      >
        {'×'}
      </button>
      {(selection.kind === 'node' ||
        selection.kind === 'wire' ||
        (selection.kind === 'pin' && selection.role === 'pred' && roles.includes('pred'))) && (
        <button
          type="button"
          className="tk-inspector-delete"
          data-testid="inspector-delete"
          onClick={deleteSelection}
        >
          Delete
        </button>
      )}
      {selection.kind === 'node' && node && <NodeInspector node={node} analysis={analysis} />}
      {selection.kind === 'pin' && (
        <PinInspector
          role={selection.role}
          node={selection.node}
          analysis={analysis}
          graph={graph}
        />
      )}
      {selection.kind === 'edge' && (
        <EdgeInspector
          graph={graph}
          analysis={analysis}
          node={selection.node}
          i={selection.i}
          j={selection.j}
          fallback={weights}
        />
      )}
      {selection.kind === 'unit' && (
        <UnitInspector
          graph={graph}
          analysis={analysis}
          node={selection.node}
          index={selection.index}
          fallback={weights}
        />
      )}
      {selection.kind === 'wire' && (
        <WireInspector
          graph={graph}
          analysis={analysis}
          from={selection.from}
          to={selection.to}
          messages={wireMessages}
        />
      )}
    </aside>
  );
}
