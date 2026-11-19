import { LOSS_FNS, type LossFn } from '@tinker/core';
import { NodeShell } from './NodeShell.js';
import { NumberField, SelectField, editPenalty, editSetting } from './fields.js';
import type { WebPiece } from './types.js';

export const LOSS_LABEL: Record<LossFn, string> = {
  squared_error: 'Squared error',
  absolute_error: 'Absolute error',
  huber: 'Huber',
  log_loss: 'Log loss',
  cross_entropy: 'Cross-entropy',
};

/** Layer drawing: the shared node in the loss role, badge L; the loss name (and delta) is its secondary info, a penalty its tag. */
export const lossPiece: WebPiece<'loss'> = {
  Layer: ({ node, w, h, scale }) => {
    const lines: string[] = [];
    const extra = [
      node.fn === 'huber' ? `$\\delta=${node.delta ?? 1}$` : '',
      node.penalty ? `$L_${node.penalty.kind.slice(1)},\\ \\lambda=${node.penalty.lambda}$` : '',
    ].filter(Boolean);
    if (extra.length) lines.push(extra.join(' '));
    return (
      <NodeShell
        piece="loss"
        role="loss"
        badge
        w={w}
        h={h}
        scale={scale}
        name={LOSS_LABEL[node.fn]}
        glyph="L"
        secondary={lines.length ? lines : undefined}
        inPorts={2}
        outPort={false}
      >
        {node.penalty && <g data-testid={`penalty-tag-${node.id}`} />}
      </NodeShell>
    );
  },
  Neuron: () => <g data-placeholder="loss-neuron" />,
  Fields: ({ node }) => (
    <>
      <SelectField
        label="Loss"
        testid="field-loss-fn"
        value={node.fn}
        options={LOSS_FNS.map((f) => [f, LOSS_LABEL[f]] as const)}
        onChange={(v) => editSetting(node.id, 'fn', v)}
      />
      {node.fn === 'huber' && (
        <NumberField
          label="Threshold $\delta$"
          testid="field-delta"
          value={node.delta ?? 1}
          min={0}
          step={0.1}
          onCommit={(v) => editSetting(node.id, 'delta', v)}
        />
      )}
      <SelectField
        label="Weight penalty"
        testid="field-penalty-kind"
        value={node.penalty?.kind ?? 'none'}
        options={[
          ['none', 'None'],
          ['l2', 'L2'],
          ['l1', 'L1'],
        ]}
        onChange={(v) =>
          editPenalty(
            node.id,
            v === 'none' ? null : { kind: v, lambda: node.penalty?.lambda ?? 0.01 },
          )
        }
      />
      {node.penalty && (
        <NumberField
          label="Strength $\lambda$"
          testid="field-penalty-lambda"
          value={node.penalty.lambda}
          min={0}
          max={100}
          step={0.01}
          onCommit={(v) => editPenalty(node.id, { kind: node.penalty?.kind ?? 'l2', lambda: v })}
        />
      )}
    </>
  ),
};
