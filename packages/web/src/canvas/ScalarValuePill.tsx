import { useStore } from '../state/store.js';
import { SvgTex } from '../tex/Tex.js';
import { pointerDownWith } from './tap.js';
import { fmtWeight } from './pure.js';
import { weightDragHandlers } from './weightEdit.js';

const NBSP = String.fromCharCode(160);
const FS = 12;
const PH = 20;

/**
 * The live value of a Bias, drawn on its piece as `b = 0.50` (tinker-gw6). The same control as the
 * weight-wire pill: raised, outlined, `ns-resize`; a vertical drag sets the value through
 * `beginWeightEdit` (one undo step, refused while running with the usual toast), a tap selects the
 * piece. It has its own press rectangle (at least 32 px tall, growing downward past the box at small scales so the upper body stays free to move the piece) so the rest of the body still moves
 * the piece. Drawn in the piece's local units, bottom of the box, clear of the output port's
 * 32 px handle. Nothing is drawn until the model has a value.
 */
/** The press rectangle's geometry (node-local units): from the pill's top edge down, so it never reaches up into the body. */
export function pillHit(w: number, h: number, scale = 1) {
  const top = h - 5 - PH;
  return { x: 12, y: top, w: w - 12 - 20, h: Math.max(PH, 32 / scale) };
}

export function ScalarValuePill(p: {
  node: string;
  w: number;
  h: number;
  scale?: number;
  /** Current `b[0]`, from `layerProps`; nothing is drawn without it. */
  value: number | undefined;
}) {
  const v = p.value;
  const selected = useStore((s) => s.selection?.kind === 'node' && s.selection.id === p.node);
  if (v === undefined) return null;
  const x = 12;
  const pw = p.w - 12 - 20; // keeps 20 units clear of the right edge (the output handle)
  const cy = p.h - 5 - PH / 2;
  const hit = pillHit(p.w, p.h, p.scale);
  const handlers = weightDragHandlers(() => ({ kind: 'bias', node: p.node }));
  return (
    <g
      data-testid={`bias-pill-${p.node}`}
      data-value={String(v)}
      data-selected={selected}
      className="tk-wpill"
      onPointerDown={pointerDownWith({
        onTap: () => useStore.getState().select({ kind: 'node', id: p.node }),
        ...handlers,
      })}
    >
      <rect className="tk-hit" x={hit.x} y={hit.y} width={hit.w} height={hit.h} />
      <rect className="tk-wpill-bg" x={x} y={cy - PH / 2} width={pw} height={PH} rx={PH / 2} />
      <SvgTex
        src={`$b$${NBSP}=${NBSP}${fmtWeight(v)}`}
        mixed
        box={{ x, y: cy - PH / 2, width: pw, height: PH }}
        fontSize={FS}
        align="center"
        className="tk-wpill-tex"
      />
      <title>Drag up or down to set this number, tap to type a value</title>
    </g>
  );
}
