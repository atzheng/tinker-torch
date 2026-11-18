import type { WebPiece } from './types.js';

/** Layer drawing: a dot labelled y. */
export const targetPiece: WebPiece<'target'> = {
  Layer: ({ w, h }) => (
    <g data-piece="target">
      <circle className="tk-dot" cx={w * 0.3} cy={h / 2} r={8} />
      <text className="tk-text" x={w * 0.3 + 16} y={h / 2} dominantBaseline="middle">
        y
      </text>
    </g>
  ),
  Neuron: () => <g data-placeholder="target-neuron" />,
  Fields: () => <p className="tk-muted">No settings.</p>,
};
