import type { ComponentType } from 'react';
import type { Analysis, Graph, Node, PieceType } from '@tinker/core';
import { WEB_PIECES } from './drawings/index.js';
import { edgeIndexOf } from './pure.js';
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
}: {
  graph: Graph;
  analysis: Analysis;
  from: string;
  to: string;
}) {
  const i = edgeIndexOf(graph, from, to);
  const width = i >= 0 ? analysis.edgeWidth[i] : null;
  const messages = analysis.issues.filter((x) => x.at.edge === i).map((x) => x.message);
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

/**
 * Settings and facts for the selected piece or wire. Positioned by the canvas beside the piece.
 * Every change goes through core edits and `graphSlice.apply` (the Fields do that), so undo
 * restores it. Hover information (wire width) is also here, for touch (CONTRACTS.md section 10).
 */
export function Inspector({ selection, analysis }: { selection: Selection; analysis: Analysis }) {
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
      {selection.kind === 'node' && node && <NodeInspector node={node} analysis={analysis} />}
      {selection.kind === 'wire' && (
        <WireInspector graph={graph} analysis={analysis} from={selection.from} to={selection.to} />
      )}
    </aside>
  );
}
