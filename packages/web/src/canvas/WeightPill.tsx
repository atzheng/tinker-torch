import { pointerDownWith } from './tap.js';
import { SvgTex } from '../tex/Tex.js';
import { fmtWeight, pillSize, weightTex } from './pure.js';
import { weightDragHandlers } from './weightEdit.js';

const NBSP = String.fromCharCode(160);

export type WeightPillProps = {
  /** The Weight piece this wire stands for. */
  node: string;
  /** Index of the wire (for tests and drop highlights); never kept across edits. */
  edge: number;
  at: [number, number];
  /** Current value, undefined until the model has one. */
  value: number | undefined;
  selected: boolean;
  error: boolean;
  drop?: boolean;
  valid?: boolean;
  replace?: boolean;
  scale: number;
  /** Press rectangle (`layerHitShapes`): at least 32 px each way. */
  hit: { x: number; y: number; w: number; h: number };
  /** Text that fixes the pill's width (`pillTexts`). */
  sizing: string;
  onSelect: () => void;
  onHover?: (edge: number) => void;
};

/**
 * The label of a weight wire: `w_1 = 0.42`. It is a control (raised, outlined, accent when
 * selected or touched): a tap selects the weight and opens the inspector, a vertical drag sets the
 * value (up raises it, as in neuron view), and the inspector has a field to type it. A sibling of
 * the wires, painted above the pieces so it is always the topmost target for its wire.
 */
export function WeightPill(p: WeightPillProps) {
  const { fs, lw, lh } = pillSize(p.sizing, p.scale);
  const handlers = weightDragHandlers(() => ({ kind: 'scalew', node: p.node }));
  const text = p.value === undefined ? '' : `${NBSP}=${NBSP}${fmtWeight(p.value)}`;
  return (
    <g
      data-testid={`weight-pill-${p.node}`}
      data-edge={p.edge}
      data-value={p.value === undefined ? '' : String(p.value)}
      data-error={p.error}
      data-selected={p.selected}
      data-drop={p.drop === true}
      data-valid={p.valid === true}
      data-replace={p.replace === true}
      className="tk-wire tk-wpill"
      onPointerEnter={(e) => e.pointerType === 'mouse' && p.onHover?.(p.edge)}
      onPointerLeave={(e) => e.pointerType === 'mouse' && p.onHover?.(-1)}
      onPointerDown={pointerDownWith({ onTap: p.onSelect, ...handlers })}
    >
      <rect className="tk-hit" x={p.hit.x} y={p.hit.y} width={p.hit.w} height={p.hit.h} />
      <rect
        className="tk-wpill-bg"
        x={p.at[0] - lw / 2}
        y={p.at[1] - lh / 2}
        width={lw}
        height={lh}
        rx={lh / 2}
      />
      <SvgTex
        src={`$${weightTex(p.node)}$${text}`}
        mixed
        box={{ x: p.at[0] - lw / 2, y: p.at[1] - lh / 2, width: lw, height: lh }}
        fontSize={fs}
        align="center"
        className="tk-wpill-tex"
      />
      <title>Drag up or down to set this weight, tap to type a value</title>
    </g>
  );
}
