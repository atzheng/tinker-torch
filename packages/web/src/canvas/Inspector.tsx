import type { ComponentType } from 'react';
import type { Analysis, Graph, HostParams, Node, PieceType } from '@tinker/core';
import { WEB_PIECES } from './drawings/index.js';
import { deleteSelection } from './actions.js';
import { biasAt, weightAt } from './neuronPure.js';
import { edgeIndexOf } from './pure.js';
import { hostShape } from '../state/hostShape.js';
import { useStore } from '../state/store.js';
import type { Selection } from '../state/viewSlice.js';

export const PIECE_LABEL: Record<PieceType, string> = {
  input: 'Input',
  feature: 'Feature',
  linear: 'Linear',
  activation: 'Activation',
  dropout: 'Dropout',
  add: 'Add',
  output: 'Output',
  target: 'Target',
  loss: 'Loss',
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

function NodeInspector({ node, analysis }: { node: Node; analysis: Analysis }) {
  const Fields = WEB_PIECES[node.type].Fields as ComponentType<{ node: Node }>;
  const inW = analysis.inWidth[node.id];
  const count = analysis.params[node.id];
  const messages = analysis.issues.filter((i) => i.at.node === node.id).map((i) => i.message);
  return (
    <>
      <h2 className="tk-inspector-title">
        {PIECE_LABEL[node.type]} <code>{node.id}</code>
      </h2>
      {node.type === 'linear' && (
        <p data-testid="inspector-params" className="tk-muted">
          {inW === undefined || count === undefined
            ? 'Parameters: unknown until the input width is known.'
            : `${inW} x ${node.out}${node.bias ? ` + ${node.out}` : ''} = ${count} parameters`}
        </p>
      )}
      {inW !== undefined && node.type !== 'linear' && <p className="tk-muted">Width {inW}</p>}
      <Fields node={node} />
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
      <p data-testid="inspector-width">
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
          hostShape(host, p.node, useStore.getState().dataInfo),
        )
      : undefined;
  return (
    <>
      <h2 className="tk-inspector-title">Weight</h2>
      <p className="tk-muted">
        <code>{p.node}</code>: input {p.i} to unit {p.j}
      </p>
      <p data-testid="inspector-weight">
        {w === undefined ? 'No value yet' : `Value ${w.toFixed(4)}`}
      </p>
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
        <p data-testid="inspector-fanin">{inW} incoming weights</p>
      )}
      {b !== undefined && <p data-testid="inspector-bias">Bias {b.toFixed(4)}</p>}
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
  const node =
    selection.kind === 'node' ? graph.nodes.find((n) => n.id === selection.id) : undefined;
  return (
    <aside className="tk-inspector" data-testid="inspector" aria-label="Inspector">
      <button
        type="button"
        className="tk-inspector-close"
        aria-label="Close inspector"
        data-testid="inspector-close"
        onClick={() => select(null)}
      >
        {'×'}
      </button>
      {(selection.kind === 'node' || selection.kind === 'wire') && (
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
