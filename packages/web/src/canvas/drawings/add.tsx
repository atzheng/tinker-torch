import type { WebPiece } from './types.js';

/** Layer drawing: a plus circle. (Minimal: the swapping epic may refine it.) */
export const addPiece: WebPiece<'add'> = {
  Layer: ({ w, h }) => {
    const r = Math.min(w, h) / 2 - 8;
    return (
      <g data-piece="add">
        <circle className="tk-shape" cx={w / 2} cy={h / 2} r={r} />
        <path
          className="tk-stroke"
          d={`M${w / 2 - r / 2} ${h / 2}H${w / 2 + r / 2}M${w / 2} ${h / 2 - r / 2}V${h / 2 + r / 2}`}
        />
      </g>
    );
  },
  Neuron: () => <g data-placeholder="add-neuron" />,
  Fields: () => <p className="tk-muted">No settings.</p>,
};
