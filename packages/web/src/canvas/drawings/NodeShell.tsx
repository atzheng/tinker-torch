import type { ReactNode } from 'react';
import type { PieceType } from '@tinker/core';
import { legibleAt } from '../pure.js';

/**
 * The one node component (CONTRACTS.md section 10, "Visual language"). Every piece in layer view,
 * and every box in neuron view, is drawn by this shell; pieces only choose what goes in the slots.
 * No piece draws its own outline, port or state colour.
 */

/** Ordinary pieces share one style; the three roles below are the only special ones. */
export type NodeRole = 'input' | 'pred' | 'loss';

/** The role of a piece type, or undefined for an ordinary piece. Moves to a pin with graph v2. */
export function roleOf(type: PieceType): NodeRole | undefined {
  switch (type) {
    case 'input':
    case 'target':
      return 'input';
    case 'output':
      return 'pred';
    case 'loss':
      return 'loss';
    default:
      return undefined;
  }
}

/** Corner radius of the body; the state ring sits 4 units outside it. */
export const NODE_RADIUS = 8;
export const PORT_R = 4.5;
const PAD = 12;
const NAME_PX = 15;
const SECONDARY_PX = 13;

export type NodeShellProps = {
  w: number;
  h: number;
  /** Piece type, for `data-piece`. */
  piece: string;
  /** The piece's name, top left. Always drawn. */
  name: string;
  /** Short symbol or formula for the operation, right slot. Ignored when `slot` is given. */
  glyph?: string;
  /** A role badge (x, y-hat, L) instead of plain glyph text. Needs `role`. */
  badge?: boolean;
  /** Short qualifier, top right (Feature options, a weight penalty). Drawn when legible. */
  tag?: string;
  /** Custom content for the right slot (the Linear heat map), drawn in place of the glyph. */
  slot?: ReactNode;
  /** Secondary info, bottom left, one entry per line (at most two). Both are always drawn. */
  secondary?: string[];
  /** `data-testid` of the second secondary line, when a test or a caller needs to find it. */
  secondaryTestId?: string;
  role?: NodeRole;
  /** Input ports on the left edge (spread evenly, as `inputPortAt`) and an output port on the right. */
  inPorts: number;
  outPort: boolean;
  scale?: number;
  children?: ReactNode;
};

export function NodeShell(p: NodeShellProps) {
  const { w, h } = p;
  const lines = p.secondary ?? [];
  const small = legibleAt(p.scale, SECONDARY_PX);
  const secondaryY = lines.length > 1 ? [h - 24, h - 10] : [h - 12];
  return (
    <g data-piece={p.piece} data-role={p.role}>
      <rect className="tk-body" x={0} y={0} width={w} height={h} rx={NODE_RADIUS} />
      <text className="tk-name" x={PAD} y={NAME_PX + 4} fontSize={NAME_PX}>
        {p.name}
      </text>
      {p.tag && small && (
        <text className="tk-secondary" x={w - PAD} y={NAME_PX + 4} fontSize={11} textAnchor="end">
          {p.tag}
        </text>
      )}
      {lines.map((t, i) => (
        <text
          key={i}
          className="tk-secondary"
          data-testid={i === 1 ? p.secondaryTestId : undefined}
          x={PAD}
          y={secondaryY[i]}
          fontSize={SECONDARY_PX}
        >
          {t}
        </text>
      ))}
      {p.slot ??
        (p.badge ? (
          <g className="tk-badge" transform={`translate(${w - PAD - 10} ${NAME_PX + 4 - 5})`}>
            <circle r={10} />
            <text textAnchor="middle" dominantBaseline="central" y={1.5}>
              {p.glyph}
            </text>
          </g>
        ) : (
          p.glyph !== undefined && (
            <text
              className="tk-glyph"
              x={w - PAD}
              y={h / 2 + 7}
              textAnchor="end"
              fontSize={p.glyph.length > 4 ? 15 : 19}
            >
              {p.glyph}
            </text>
          )
        ))}
      {p.children}
      {Array.from({ length: p.inPorts }, (_, k) => (
        <circle
          key={k}
          className="tk-port-static"
          cx={0}
          cy={(h * (k + 1)) / (p.inPorts + 1)}
          r={PORT_R}
        />
      ))}
      {p.outPort && <circle className="tk-port-static" cx={w} cy={h / 2} r={PORT_R} />}
    </g>
  );
}
