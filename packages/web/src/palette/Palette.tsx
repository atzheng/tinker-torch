import { useEffect, useRef, useState, useSyncExternalStore } from 'react';
import {
  PALETTE,
  PIECE_DOCS,
  docView,
  isShown,
  ownsParams,
  paletteTabs,
  type Node,
  type PaletteEntry,
} from '@tinker/core';
import { dnd } from '../canvas/dnd.js';
import { HINTS } from '../hints.js';
import { startDrag } from '../gestures/drag.js';
import { useLayoutMode } from '../layout/useLayoutMode.js';
import { useSplitTabs, useViewTab } from '../state/disclosure.js';
import { useStore } from '../state/store.js';
import { Tex } from '../tex/Tex.js';
import { PieceCard, cardOwner, useCardOpen } from './PieceCard.js';
import './palette.css';

/** The glyph a palette chip shows, the same one the node it creates carries (roles get a badge). */
const GLYPH: Record<string, string> = {
  input: 'x',
  target: 'y',
  feature: 'x^2',
  linear: 'W',
  activation: '\\sigma',
  dropout: 'p',
  add: '+',
  sigmoid: '\\sigma',
  softmax: '\\mathrm{softmax}',
  subtract: 'a-b',
  square: 'x^2',
  abs: '|x|',
  huber: '\\mathrm{H}_\\delta',
  log_loss: '-\\log p',
  cross_entropy: '-\\log p_y',
  mean: '\\bar{x}',
  scale: 'c\\,x',
  weights: 'W',
  l2: '\\sum w^2',
  l1: '\\sum|w|',
  onehot: 'e_k',
  lookup: 'E',
  tree: '\\mathrm{leaf}',
  weight_penalty: '\\lambda',
};
function chipOf(entry: PaletteEntry): {
  role?: 'input' | 'pred' | 'loss' | 'param';
  glyph?: string;
} {
  if (entry.kind === 'macro') {
    return entry.family === 'losses' ? { role: 'loss', glyph: 'L' } : { glyph: GLYPH[entry.key] };
  }
  const role =
    entry.key === 'input' || entry.key === 'target'
      ? 'input'
      : ownsParams({ id: 'chip', ...entry.make() } as Node)
        ? 'param'
        : undefined;
  return { role, glyph: GLYPH[entry.key] };
}

/** Palette entries that make a value (a pill, as `shapeOfNode`): a new Bias is unwired, so a value. */
const VALUE_ENTRIES: ReadonlySet<string> = new Set([
  'input',
  'target',
  'column',
  'weights',
  'bias',
]);

/**
 * The "?" of one palette entry (CONTRACTS.md section 15). A sibling of the entry button, so a press
 * on it never reaches the entry's drag or arm handlers. Opens the symbolic card beside the palette.
 */
function EntryDoc({ entry }: { entry: PaletteEntry }) {
  const id = `palette:${entry.key}`;
  const open = useCardOpen(id);
  const btn = useRef<HTMLButtonElement>(null);
  const row = btn.current?.parentElement ?? null;
  return (
    <>
      <button
        ref={btn}
        type="button"
        className="tk-doc-btn tk-palette-doc"
        data-testid={`piece-doc-${entry.key}`}
        aria-label={`About ${entry.label}`}
        aria-expanded={open}
        onClick={() => (open ? cardOwner.close(id) : cardOwner.open(id))}
      >
        ?
      </button>
      {open && row && (
        <PieceCard
          view={docView(entry.key)}
          opener={btn.current}
          placement={{ kind: 'palette', row }}
          onClose={() => cardOwner.close(id)}
        />
      )}
    </>
  );
}

/**
 * The palette's groups, in order. Most are one family; "Ready-made losses" is every macro (the five
 * classic losses and the weight penalty), which share the Losses and Parameters families with
 * primitives. `open` is the start state: the pieces a first network needs are open, the primitives
 * for building a loss by hand are one click away.
 */
type Group = { id: string; label: string; open: boolean; has: (e: PaletteEntry) => boolean };
const inFamily = (f: PaletteEntry['family']) => (e: PaletteEntry) =>
  e.kind === 'piece' && e.family === f;
export const GROUPS: Group[] = [
  { id: 'inputs', label: 'Inputs', open: true, has: inFamily('inputs') },
  { id: 'categories', label: 'Categories', open: false, has: inFamily('categories') },
  { id: 'features', label: 'Features', open: false, has: inFamily('features') },
  { id: 'linear', label: 'Linear', open: true, has: inFamily('linear') },
  { id: 'activations', label: 'Activations', open: true, has: inFamily('activations') },
  { id: 'structure', label: 'Structure', open: false, has: inFamily('structure') },
  { id: 'links', label: 'Links', open: true, has: inFamily('links') },
  { id: 'math', label: 'Math', open: false, has: inFamily('math') },
  { id: 'errors', label: 'Errors', open: false, has: inFamily('errors') },
  { id: 'reductions', label: 'Reductions', open: false, has: inFamily('reductions') },
  { id: 'losses', label: 'Losses', open: false, has: inFamily('losses') },
  { id: 'scalars', label: 'Scalars', open: true, has: inFamily('scalars') },
  { id: 'parameters', label: 'Parameters', open: false, has: inFamily('parameters') },
  { id: 'ready', label: 'Ready-made losses', open: true, has: (e) => e.kind === 'macro' },
];

/**
 * The pieces and ready-made groups by family. Dragging one (Pointer Events, mouse or touch) onto the
 * canvas inserts it on a wire, swaps it for a piece of its family, or adds a group; the canvas
 * decides what is under the pointer (`canvas/dnd.ts`).
 */
export function Palette() {
  // Portrait: a horizontally scrolling strip above the canvas (CONTRACTS.md section 10).
  const strip = useLayoutMode() === 'portrait';
  const drag = useSyncExternalStore(dnd.subscribe, dnd.get, () => null);
  const dragging = drag?.payload.kind === 'palette' ? drag.payload.entry : null;
  const armed = useSyncExternalStore(dnd.subscribe, dnd.armed, () => null);
  // A preset's palette allow-list (CONTRACTS.md section 12): listed entries, or every entry of a listed group.
  const allowed = useStore((s) => s.show?.palette);
  const docs = useStore((s) => s.show?.docs) !== false; // `show.docs: false` removes every "?"
  // Toggles the person made; a group not toggled is open at its default, or open when the preset narrows the palette.
  const [toggled, setToggled] = useState<Record<string, boolean>>({});
  const isOpen = (g: Group) => toggled[g.id] ?? (allowed !== undefined || g.open);
  // Model pieces on the model tab, loss and penalty pieces on the loss tab (CONTRACTS.md section 9).
  const tab = useViewTab();
  const split = useSplitTabs();
  const visibleIn = (g: Group) =>
    PALETTE.filter(g.has)
      .filter((e) => !split || paletteTabs(e).includes(tab))
      .filter((e) => isShown(allowed, g.id) || isShown(allowed, e.key));
  // Strip: a mouse wheel (vertical delta) scrolls it sideways; a trackpad's horizontal delta already does.
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const el = ref.current;
    if (!strip || !el) return;
    const onWheel = (e: WheelEvent) => {
      if (Math.abs(e.deltaY) <= Math.abs(e.deltaX)) return;
      e.preventDefault();
      el.scrollLeft += e.deltaY;
    };
    el.addEventListener('wheel', onWheel, { passive: false });
    return () => el.removeEventListener('wheel', onWheel);
  }, [strip]);
  useEffect(() => () => dnd.cancel(), []); // unmounted mid-drag: leave no ghost
  // Tap-to-add ends on Escape, or on a press that is neither on the palette nor on the drawing.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && dnd.armed()) dnd.arm(null);
    };
    const onDown = (e: PointerEvent) => {
      if (!dnd.armed()) return;
      const t = e.target instanceof Element ? e.target : null;
      if (
        t?.closest('.tk-palette-entry, .tk-palette-doc, .tk-piece-card, [data-testid="canvas-svg"]')
      )
        return;
      dnd.arm(null);
    };
    window.addEventListener('keydown', onKey);
    window.addEventListener('pointerdown', onDown);
    return () => {
      window.removeEventListener('keydown', onKey);
      window.removeEventListener('pointerdown', onDown);
    };
  }, []);
  return (
    <div className="tk-palette" data-testid="palette" data-strip={strip} ref={ref}>
      <div className="tk-palette-head">
        <h2 className="tk-region-title">Pieces</h2>
        <p className="tk-palette-hint" data-testid="palette-hint">
          {HINTS.palette}
        </p>
      </div>
      {GROUPS.filter((g) => visibleIn(g).length > 0).map((g) => (
        <section key={g.id} className="tk-palette-group" aria-label={g.label} data-open={isOpen(g)}>
          <h3>
            <button
              type="button"
              className="tk-palette-toggle"
              data-testid={`palette-group-${g.id}`}
              aria-expanded={isOpen(g)}
              onClick={() => setToggled((o) => ({ ...o, [g.id]: !isOpen(g) }))}
            >
              <span aria-hidden="true" className="tk-palette-caret" />
              {g.label}
            </button>
          </h3>
          <div hidden={!isOpen(g)}>
            {visibleIn(g).map((entry) => (
              <div key={entry.key} className="tk-palette-row" data-armed={armed?.key === entry.key}>
                <button
                  type="button"
                  className="tk-palette-entry"
                  data-testid={`palette-${entry.key}`}
                  data-dragging={dragging?.key === entry.key}
                  data-role={chipOf(entry).role}
                  data-shape={
                    entry.kind === 'piece' && VALUE_ENTRIES.has(entry.key) ? 'value' : 'op'
                  }
                  data-family={entry.family}
                  data-kind={entry.kind}
                  aria-description={PIECE_DOCS[entry.key]?.summary}
                  aria-pressed={armed?.key === entry.key}
                  onClick={(e) => {
                    if (e.detail === 0) dnd.arm(entry); // keyboard: Enter or Space picks the entry
                  }}
                  onPointerDown={(e) => {
                    if (e.pointerType === 'mouse' && e.button !== 0) return;
                    // Touch: the palette scrolls along its long axis (`touch-action: pan-y`, or `pan-x`
                    // in the strip), so a drag begins only once the movement is mostly across it.
                    const touch = e.pointerType !== 'mouse';
                    let begun = false;
                    startDrag(e, {
                      onMove: (p) => {
                        if (!begun) {
                          if (
                            touch &&
                            (strip
                              ? Math.abs(p.dy) < Math.abs(p.dx)
                              : Math.abs(p.dx) < Math.abs(p.dy))
                          )
                            return;
                          begun = true;
                          dnd.begin({ kind: 'palette', entry }, p.x, p.y);
                        }
                        dnd.move(p.x, p.y);
                      },
                      onTap: () => dnd.arm(entry),
                      onEnd: (p) => (begun ? dnd.end(p.x, p.y) : undefined),
                      onCancel: () => dnd.cancel(),
                    });
                  }}
                >
                  <span className="tk-chip-label">{entry.label}</span>
                  {chipOf(entry).glyph && (
                    <span className="tk-chip-glyph" aria-hidden="true">
                      <Tex src={chipOf(entry).glyph ?? ''} />
                    </span>
                  )}
                </button>
                {docs && <EntryDoc entry={entry} />}
              </div>
            ))}
          </div>
        </section>
      ))}
      {dragging && drag && (
        <div
          className="tk-palette-ghost"
          data-testid="palette-ghost"
          data-target={drag.target ? drag.target.kind : 'none'}
          style={{ left: drag.x, top: drag.y }}
        >
          {dragging.label}
        </div>
      )}
    </div>
  );
}
