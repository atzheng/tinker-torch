import { LOSS_FNS, type LossFn } from '@tinker/core';
import { legibleAt } from '../pure.js';
import { NumberField, SelectField, editPenalty, editSetting } from './fields.js';
import type { WebPiece } from './types.js';

export const LOSS_LABEL: Record<LossFn, string> = {
  squared_error: 'Squared error',
  absolute_error: 'Absolute error',
  huber: 'Huber',
  log_loss: 'Log loss',
  cross_entropy: 'Cross-entropy',
};

const LOSS_LINES: Record<LossFn, string[]> = {
  squared_error: ['Squared', 'error'],
  absolute_error: ['Absolute', 'error'],
  huber: ['Huber'],
  log_loss: ['Log', 'loss'],
  cross_entropy: ['Cross-', 'entropy'],
};

/** Layer drawing: a loss circle with the loss name (two short lines when it has two words). */
export const lossPiece: WebPiece<'loss'> = {
  Layer: ({ node, w, h, scale }) => {
    const showDelta = node.fn === 'huber' && legibleAt(scale, 12.5);
    const showTag = node.penalty !== undefined && legibleAt(scale, 9);
    const lines = LOSS_LINES[node.fn];
    return (
      <g data-piece="loss">
        <circle className="tk-shape" cx={w / 2} cy={h / 2} r={Math.min(w, h) / 2 - 2} />
        {showDelta && (
          <text
            className="tk-text tk-small tk-muted"
            x={w / 2}
            y={h / 2 + 13}
            textAnchor="middle"
            dominantBaseline="middle"
          >
            {`δ=${node.delta ?? 1}`}
          </text>
        )}
        {node.penalty && showTag && (
          <g data-testid={`penalty-tag-${node.id}`}>
            <rect className="tk-shape" x={w - 34} y={h / 2 - 17} width={34} height={34} rx={5} />
            <text
              className="tk-text"
              x={w - 17}
              y={h / 2 - 5}
              textAnchor="middle"
              dominantBaseline="middle"
              style={{ fontSize: 11 }}
            >
              {node.penalty.kind.toUpperCase()}
            </text>
            <text
              className="tk-text tk-muted"
              x={w - 17}
              y={h / 2 + 8}
              textAnchor="middle"
              dominantBaseline="middle"
              style={{ fontSize: 9 }}
            >
              {String(node.penalty.lambda)}
            </text>
          </g>
        )}
        {lines.map((t, i) => (
          <text
            key={t}
            className="tk-text tk-small"
            x={w / 2}
            y={h / 2 + (i - (lines.length - 1) / 2) * 13 - (showDelta ? 7 : 0)}
            textAnchor="middle"
            dominantBaseline="middle"
          >
            {t}
          </text>
        ))}
      </g>
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
          label="Threshold delta"
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
          label="Strength lambda"
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
