import { categoryOf } from '@tinker/core';
import { useStore } from '../../state/store.js';
import { useAnalysis, useWeights } from '../../state/derived.js';
import { fitText } from '../../state/names.js';
import { NodeShell } from './NodeShell.js';
import { NumberField, editSetting } from './fields.js';
import type { NodeOf, WebPiece } from './types.js';

const noNeuron = (name: string) => () => <g data-placeholder={`${name}-neuron`} />;

/** The category names a piece reads (or carries), or undefined until it is wired to a category column. */
export function useCategories(id: string) {
  const info = useStore((s) => s.dataInfo);
  const analysis = useAnalysis();
  return categoryOf(analysis, info, id);
}

/** Inspector line: how many categories and their names. */
export function CategoryList({ id }: { id: string }) {
  const cats = useCategories(id);
  if (!cats) {
    return <p className="tk-muted">Connect a category column (an Input on a text column).</p>;
  }
  return (
    <p className="tk-diag" data-testid="category-names">
      {cats.names.length} categories: {cats.names.join(', ')}
    </p>
  );
}

export const onehotPiece: WebPiece<'onehot'> = {
  Layer: function OneHotLayer({ node, w, h, scale }) {
    const cats = useCategories(node.id);
    return (
      <NodeShell
        piece="onehot"
        w={w}
        h={h}
        scale={scale}
        name="One-hot"
        glyph="e_k"
        secondary={[
          cats ? `${cats.names.length} columns` : '? columns',
          cats ? fitText(cats.names.join(', '), w - 20, 11) : 'no category input',
        ]}
        inPorts={1}
        outPort
      />
    );
  },
  Neuron: noNeuron('onehot'),
  Fields: ({ node }) => <CategoryList id={node.id} />,
};

/** One learned row per category: the values training has found, labelled with the names. */
function LookupValues({ node }: { node: NodeOf<'lookup'> }) {
  const cats = useCategories(node.id);
  const analysis = useAnalysis();
  const weights = useWeights(analysis);
  const W = weights?.[node.id]?.['W'];
  if (!cats || !W || W.length !== cats.names.length * node.width) return null;
  return (
    <table className="tk-diag" data-testid="lookup-values">
      <tbody>
        {cats.names.map((name, k) => (
          <tr key={name}>
            <th scope="row" style={{ textAlign: 'left', fontWeight: 'normal', paddingRight: 12 }}>
              {name}
            </th>
            <td>
              {Array.from({ length: node.width }, (_, j) =>
                (W[k * node.width + j] ?? 0).toFixed(3),
              ).join(', ')}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export const lookupPiece: WebPiece<'lookup'> = {
  Layer: function LookupLayer({ node, w, h, paramCount, scale }) {
    const cats = useCategories(node.id);
    return (
      <NodeShell
        piece="lookup"
        w={w}
        h={h}
        scale={scale}
        name="Lookup"
        glyph="E"
        secondary={[
          // like a Linear's "d to m": one category index to `width` numbers
          cats ? `${cats.names.length} to ${node.width}` : `? to ${node.width}`,
          cats ? `${paramCount} params` : 'no category input',
        ]}
        inPorts={1}
        outPort
      />
    );
  },
  Neuron: noNeuron('lookup'),
  Fields: ({ node }) => (
    <>
      <CategoryList id={node.id} />
      <NumberField
        label="Width"
        testid="field-width"
        value={node.width}
        min={1}
        max={64}
        step={1}
        onCommit={(v) => editSetting(node.id, 'width', v)}
      />
      <LookupValues node={node} />
    </>
  ),
};
