import { parsePort } from '@tinker/core';
import { useAnalysis } from '../../state/derived.js';
import { useStore } from '../../state/store.js';
import { NodeShell } from './NodeShell.js';
import type { NodeOf, WebPiece } from './types.js';

/** Layer drawing: the shared node with two input ports (a, b) and a plus glyph. */
export const addPiece: WebPiece<'add'> = {
  Layer: ({ w, h, scale }) => (
    <NodeShell piece="add" w={w} h={h} scale={scale} name="Add" glyph="+" inPorts={2} outPort />
  ),
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
