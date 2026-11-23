import { useStore } from '../state/store.js';
import { useAnalysis, useWeights } from '../state/derived.js';
import { SvgTex } from '../tex/Tex.js';
import { pointerDownWith } from './tap.js';
import { fmtWeight } from './pure.js';
import { weightDragHandlers } from './weightEdit.js';

const NBSP = String.fromCharCode(160);
const FS = 12;
const PH = 20;

/** The value of a Bias (`b[0]`) when the model has one. Reactive: it follows training and edits. */
export function useBiasValue(node: string): number | undefined {
  const analysis = useAnalysis();
  const arr = useWeights(analysis)?.[node]?.['b'];
  return arr?.length === 1 ? arr[0] : undefined;
}

/**
 * The live value of a Bias, drawn on its piece as `b = 0.50` (tinker-gw6). The same control as the
 * weight-wire pill: raised, outlined, `ns-resize`; a vertical drag sets the value through
 * `beginWeightEdit` (one undo step, refused while running with the usual toast), a tap selects the
 * piece. It has its own press rectangle (at least 32 px tall) so the rest of the body still moves
 * the piece. Drawn in the piece's local units, bottom of the box, clear of the output port's
 * 32 px handle. Nothing is drawn until the model has a value.
 */
export function ScalarValuePill(p: { node: string; w: number; h: number; scale?: number }) {
  const v = useBiasValue(p.node);
  const selected = useStore((s) => s.selection?.kind === 'node' && s.selection.id === p.node);
  if (v === undefined) return null;
  const x = 12;
  const pw = p.w - 12 - 20; // keeps 20 units clear of the right edge (the output handle)
  const cy = p.h - 5 - PH / 2;
  const hh = Math.max(PH, 32 / (p.scale ?? 1));
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
      <rect className="tk-hit" x={x} y={cy - hh / 2} width={pw} height={hh} />
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
