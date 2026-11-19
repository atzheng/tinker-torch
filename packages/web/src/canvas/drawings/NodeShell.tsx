import type { ReactNode } from 'react';
import type { PieceType } from '@tinker/core';
import { fitName, legibleAt } from '../pure.js';
import { SvgTex } from '../../tex/Tex.js';
import { splitTex } from '../../tex/Tex.js';

/**
 * The one node component (CONTRACTS.md section 10, "Visual language"). Every piece in layer view,
 * and every box in neuron view, is drawn by this shell; pieces only choose what goes in the slots.
 * No piece draws its own outline, port or state colour.
 */

/** Ordinary pieces share one style; the three roles below are the only special ones. */
export type NodeRole = 'input' | 'pred' | 'loss';

/**
 * The role of a node, or undefined for an ordinary one. The single place that maps graph data to
 * roles: graph v2 (role pins on an output port) changes only this function.
 */
export function roleOfNode(node: { type: PieceType }): NodeRole | undefined {
  switch (node.type) {
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

/** Box (node-local units) a custom glyph `slot` must stay inside: the KaTeX bead's foreignObject goes here. */
export function slotBox(w: number, h: number) {
  return { x: w - 52, y: h / 2 - 10, width: 40, height: 32 };
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
  /** TeX (no dollar signs) for the operation or role badge, right slot. Ignored when `slot` is given. */
  glyph?: string;
  /** A role badge (x, y-hat, L) instead of plain glyph text. Needs `role`. */
  badge?: boolean;
  /** TeX qualifier, top right (Feature options, a weight penalty). Drawn when legible. */
  tag?: string;
  /** Custom content for the right slot (the Linear heat map), drawn in place of the glyph. */
  slot?: ReactNode;
  /** Secondary info, bottom left, one entry per line (at most two). Both are always drawn. A line with `$...$` is typeset as maths. */
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
  const name = fitName(p.scale, p.name, w - PAD - 6, NAME_PX);
  const secondaryY = lines.length > 1 ? [h - 24, h - 10] : [h - 12];
  return (
    <g data-piece={p.piece} data-role={p.role}>
      <rect className="tk-body" x={0} y={0} width={w} height={h} rx={NODE_RADIUS} />
      <text className="tk-name" x={PAD} y={NAME_PX + 4} style={{ fontSize: name.size }}>
        {name.text}
      </text>
      {p.tag && small && (
        <SvgTex
          src={p.tag}
          box={{ x: w / 2, y: 3, width: w / 2 - PAD, height: 22 }}
          fontSize={11}
          align="end"
          className="tk-secondary-tex"
        />
      )}
      {lines.map((t, i) =>
        splitTex(t).some((x) => x.tex) ? (
          <SvgTex
            key={i}
            src={t}
            mixed
            box={{ x: PAD, y: (secondaryY[i] ?? 0) - 12, width: w - 2 * PAD, height: 16 }}
            fontSize={SECONDARY_PX}
            align="start"
            className="tk-secondary-tex"
            testid={i === 1 ? p.secondaryTestId : undefined}
          />
        ) : (
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
        ),
      )}
      {p.slot ??
        (p.badge ? (
          <g className="tk-badge" transform={`translate(${w - PAD - 10} ${h / 2 + 4})`}>
            <circle r={10} />
            <SvgTex
              src={p.glyph ?? ''}
              box={{ x: -10, y: -10, width: 20, height: 20 }}
              fontSize={13}
              align="center"
              className="tk-badge-tex"
            />
          </g>
        ) : (
          p.glyph !== undefined && (
            <SvgTex
              src={p.glyph}
              box={slotBox(w, h)}
              fontSize={Math.min(name.size, p.glyph.length > 4 ? 13 : 15)}
              align="end"
            />
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
