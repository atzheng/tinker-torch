import { InputFields } from './InputFields.js';
import type { WebPiece } from './types.js';

const MAX_DOTS = 4;

/** Layer drawing: one labelled dot per selected column (x1 to xd). Fields is tinker-27m.5. */
export const inputPiece: WebPiece<'input'> = {
  Layer: ({ node, w, h }) => {
    const shown = node.columns.slice(0, MAX_DOTS);
    const more = node.columns.length - shown.length;
    const rows = shown.length + (more > 0 ? 1 : 0);
    const pitch = Math.min(18, (h - 8) / Math.max(1, rows));
    const y0 = (h - pitch * (rows - 1)) / 2;
    return (
      <g data-piece="input">
        {shown.map((c, i) => (
          <g key={c}>
            <circle className="tk-dot" cx={w * 0.3} cy={y0 + i * pitch} r={5} />
            <text className="tk-text" x={w * 0.3 + 12} y={y0 + i * pitch} dominantBaseline="middle">
              {c}
            </text>
          </g>
        ))}
        {more > 0 && (
          <text
            className="tk-text tk-muted"
            x={w * 0.3 + 12}
            y={y0 + shown.length * pitch}
            dominantBaseline="middle"
          >
            +{more} more
          </text>
        )}
      </g>
    );
  },
  Neuron: () => <g data-placeholder="input-neuron" />,
  Fields: InputFields,
};
