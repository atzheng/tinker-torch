import { parsePort } from '@tinker/core';
import { useAnalysis } from '../../state/derived.js';
import { useStore } from '../../state/store.js';
import type { NodeOf, WebPiece } from './types.js';

/** Layer drawing: a plus circle with two black input dots on its left. */
export const addPiece: WebPiece<'add'> = {
  Layer: ({ w, h }) => {
    const r = Math.min(w, h) / 2 - 8;
    const cx = w / 2;
    const cy = h / 2;
    return (
      <g data-piece="add">
        <circle className="tk-shape" cx={cx} cy={cy} r={r} />
        <path
          className="tk-stroke"
          d={`M${cx - r / 2} ${cy}H${cx + r / 2}M${cx} ${cy - r / 2}V${cy + r / 2}`}
        />
        <circle className="tk-dot" cx={cx - r} cy={cy - r / 2} r={3.5} />
        <circle className="tk-dot" cx={cx - r} cy={cy + r / 2} r={3.5} />
      </g>
    );
  },
  Neuron: () => <g data-placeholder="add-neuron" />,
  Fields: ({ node }) => <AddFields node={node} />,
};

/** No settings: shows the width on each of the two inputs, or that they disagree. */
function AddFields({ node }: { node: NodeOf<'add'> }) {
  const graph = useStore((s) => s.graph);
  const analysis = useAnalysis();
  const width = (port: string): string => {
    const i = graph.edges.findIndex((e) => {
      const t = parsePort(e[1]);
      return t.node === node.id && t.port === port;
    });
    const wd = i >= 0 ? analysis.edgeWidth[i] : null;
    return wd === null || wd === undefined ? 'not known' : String(wd);
  };
  const a = width('a');
  const b = width('b');
  return (
    <>
      <p data-testid="add-widths">
        Input a: {a}. Input b: {b}.
      </p>
    </>
  );
}
