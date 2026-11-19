import type { PointerEvent as ReactPointerEvent } from 'react';
import { SvgTex } from '../tex/Tex.js';
import { ROLE_GLYPH } from './drawings/NodeShell.js';
import { PIN, pinCentre, type PinRole } from './pure.js';

type Rect = { x: number; y: number; w: number; h: number };

/** The disc and its ŷ or L glyph, centred on (cx, cy). Shared by the pin and the drag preview. */
export function PinDisc({
  role,
  cx,
  cy,
  r = PIN.r,
}: {
  role: PinRole;
  cx: number;
  cy: number;
  r?: number;
}) {
  return (
    <>
      <circle className="tk-pin-body" cx={cx} cy={cy} r={r} />
      <SvgTex
        src={ROLE_GLYPH[role]}
        box={{ x: cx - r, y: cy - r, width: 2 * r, height: 2 * r }}
        fontSize={r * 1.15}
        align="center"
        className="tk-pin-tex"
      />
    </>
  );
}

export type PinMarkProps = {
  role: PinRole;
  node: string;
  /** The box of the node the pin is attached to, in abstract units. */
  box: Rect;
  /** Where the disc is centred (`placePins`); left out, the default spot. */
  at?: readonly [number, number];
  /** Press rectangle (`pinHitRect`). */
  hit: Rect;
  selected?: boolean;
  error?: boolean;
  /** The pin being dragged: drawn faint where it was. */
  lifted?: boolean;
  /** An add affordance (dashed, not yet a pin) on the selected node. */
  ghost?: boolean;
  title: string;
  onPointerDown: (e: ReactPointerEvent<Element>) => void;
  onActivate: () => void;
};

/**
 * One role pin: a short wire stub from the node's output port ending in a disc labelled ŷ or L.
 * It takes the role colours from the bare `[data-role]` selector (CONTRACTS.md section 10). The
 * ghost form is the dashed "add" version shown on the selected node.
 */
export function PinMark(p: PinMarkProps) {
  const [cx, cy] = p.at ?? pinCentre(p.box, p.role);
  const px = p.box.x + p.box.w;
  const py = p.box.y + p.box.h / 2;
  const len = Math.hypot(cx - px, cy - py);
  const ex = cx - ((cx - px) / len) * PIN.r;
  const ey = cy - ((cy - py) / len) * PIN.r;
  return (
    <g
      className="tk-pin"
      data-testid={p.ghost ? `pin-add-${p.role}-${p.node}` : `pin-${p.role}-${p.node}`}
      data-role={p.role}
      data-ghost={p.ghost === true}
      data-selected={p.selected === true}
      data-error={p.error === true}
      data-lifted={p.lifted === true}
      role="button"
      tabIndex={0}
      aria-label={p.title}
      onPointerDown={p.onPointerDown}
      onKeyDown={(e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          p.onActivate();
        }
      }}
    >
      <rect className="tk-hit" x={p.hit.x} y={p.hit.y} width={p.hit.w} height={p.hit.h} />
      <path className="tk-pin-stub" d={`M${px} ${py}L${ex} ${ey}`} />
      <circle className="tk-pin-ring" cx={cx} cy={cy} r={PIN.r + 4} />
      <PinDisc role={p.role} cx={cx} cy={cy} />
      <title>{p.title}</title>
    </g>
  );
}
