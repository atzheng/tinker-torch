import { useStore } from '../../state/store.js';
import { useAnalysis, useWeights } from '../../state/derived.js';
import { ValueField } from '../ValueField.js';
import { featureName, fitText } from '../../state/names.js';
import { NodeShell } from './NodeShell.js';
import { ScalarValuePill } from '../ScalarValuePill.js';
import { SelectField, editSetting } from './fields.js';
import type { LayerProps, NodeOf, WebPiece } from './types.js';

const NoSettings = () => <p className="tk-muted">No settings.</p>;
const noNeuron = (name: string) => () => <g data-placeholder={`${name}-neuron`} />;

/**
 * The one learned number of a Weight (`W`, a single value) or a Bias (`b`): read-out and a field to
 * type it. The weight is also set by dragging its pill on the canvas; both go through
 * `beginWeightEdit`, one undo step each, and need a paused run.
 */
function ScalarValue(p: { node: NodeOf<'scalew'> | NodeOf<'bias'> }) {
  const analysis = useAnalysis();
  const weights = useWeights(analysis);
  const isWeight = p.node.type === 'scalew';
  const arr = weights?.[p.node.id]?.[isWeight ? 'W' : 'b'];
  const v = arr?.length === 1 ? arr[0] : undefined;
  if (v === undefined) return <p className="tk-muted">No value yet.</p>;
  return (
    <>
      <p className="tk-diag" data-testid="inspector-scalar">
        Value {v.toFixed(4)}
      </p>
      <ValueField
        testid="inspector-scalar-input"
        label="Set value"
        value={v}
        target={{ kind: isWeight ? 'scalew' : 'bias', node: p.node.id }}
      />
    </>
  );
}

/** Which dataset column a Column piece reads. */
function ColumnFields({ node }: { node: NodeOf<'column'> }) {
  const info = useStore((s) => s.dataInfo);
  const columns = info.columns;
  const names = columns.includes(node.name) ? columns : [node.name, ...columns];
  return (
    <SelectField
      label="Column"
      testid="column-name"
      value={node.name}
      options={names.map((c) => [c, featureName(info, c)] as const)}
      onChange={(v) => editSetting(node.id, 'name', v)}
    />
  );
}

export const columnPiece: WebPiece<'column'> = {
  Layer: function ColumnLayer({ node, w, h, scale }) {
    const info = useStore((s) => s.dataInfo);
    const fname = featureName(info, node.name);
    return (
      <NodeShell
        piece="column"
        role="input"
        w={w}
        h={h}
        scale={scale}
        name="Column"
        glyph={`x_{${node.name.replace(/^x/, '')}}`}
        {...(fname !== node.name ? { secondary: [fitText(fname, w - 48)] } : {})}
        inPorts={0}
        outPort
      />
    );
  },
  Neuron: noNeuron('column'),
  Fields: ColumnFields,
};
/** A Weight is a module that multiplies by one learnable number; it shows its value like a Bias (tinker-g3w). */
function WeightLayer(p: LayerProps<'scalew'>) {
  const known = p.weightValue !== undefined;
  return (
    <NodeShell
      piece="scalew"
      w={p.w}
      h={p.h}
      scale={p.scale}
      name="Weight"
      {...(known ? {} : { glyph: 'w' })}
      inPorts={1}
      outPort
    >
      <ScalarValuePill
        node={p.node.id}
        kind="scalew"
        w={p.w}
        h={p.h}
        scale={p.scale}
        value={p.weightValue}
      />
    </NodeShell>
  );
}
export const scalewPiece: WebPiece<'scalew'> = {
  Layer: WeightLayer,
  Neuron: noNeuron('scalew'),
  Fields: ScalarValue,
};
/** A Bias shows its live value on the piece, as a pill that can be dragged (tinker-gw6). */
function BiasLayer(p: LayerProps<'bias'>) {
  const known = p.biasValue !== undefined;
  return (
    <NodeShell
      piece="bias"
      w={p.w}
      h={p.h}
      scale={p.scale}
      name="Bias"
      {...(known ? {} : { glyph: '+b' })}
      inPorts={1}
      outPort
    >
      <ScalarValuePill node={p.node.id} w={p.w} h={p.h} scale={p.scale} value={p.biasValue} />
    </NodeShell>
  );
}
export const biasPiece: WebPiece<'bias'> = {
  Layer: BiasLayer,
  Neuron: noNeuron('bias'),
  Fields: ScalarValue,
};
export const sumPiece: WebPiece<'sum'> = {
  Layer: ({ w, h, scale }) => (
    <NodeShell
      piece="sum"
      w={w}
      h={h}
      scale={scale}
      name="Sum"
      glyph={'\\Sigma'}
      inPorts={1}
      outPort
    />
  ),
  Neuron: noNeuron('sum'),
  Fields: () => <NoSettings />,
};
