import { useEffect, useSyncExternalStore } from 'react';
import { PALETTE, type PaletteEntry } from '@tinker/core';
import { dnd } from '../canvas/dnd.js';
import { HINTS } from '../hints.js';
import { startDrag } from '../gestures/drag.js';
import './palette.css';

/** The glyph a palette chip shows, the same one the node it creates carries (roles get a badge). */
const GLYPH: Record<string, string> = {
  input: 'x',
  target: 'y',
  output: 'ŷ',
  feature: 'x²',
  linear: 'W',
  activation: 'σ',
  dropout: 'p',
  add: '+',
};
function chipOf(entry: PaletteEntry): { role?: 'input' | 'pred' | 'loss'; glyph?: string } {
  if (entry.family === 'losses') return { role: 'loss', glyph: 'L' };
  if (entry.family === 'penalties') return { glyph: 'λ' };
  const role =
    entry.key === 'input' || entry.key === 'target'
      ? 'input'
      : entry.key === 'output'
        ? 'pred'
        : undefined;
  return { role, glyph: GLYPH[entry.key] };
}

/** One plain phrase per entry: what the piece is and what it connects to (the chip's tooltip). */
export const ENTRY_TITLES: Record<string, string> = {
  input: 'Input: the data columns. Connects out to the next piece.',
  feature: 'Feature: new columns made from the inputs. Goes between two pieces.',
  linear: 'Linear: weighted sum of its inputs. Goes between two pieces.',
  activation: 'Activation: bends the numbers. Goes between two pieces.',
  dropout: 'Dropout: randomly zeroes values in training. Goes between two pieces.',
  add: 'Add: sums two inputs. Goes between two pieces.',
  output: 'Output: the prediction. Connects to the loss.',
  target: 'Target: the true answers. Connects to the loss.',
  squared_error: 'Squared error loss. Swap it onto the loss piece.',
  absolute_error: 'Absolute error loss. Swap it onto the loss piece.',
  huber: 'Huber loss. Swap it onto the loss piece.',
  log_loss: 'Log loss. Swap it onto the loss piece.',
  cross_entropy: 'Cross-entropy loss. Swap it onto the loss piece.',
  weight_penalty: 'Weight penalty: discourages large weights. Drop it onto the loss piece.',
};

const GROUPS: { family: PaletteEntry['family']; label: string }[] = [
  { family: 'inputs', label: 'Inputs' },
  { family: 'features', label: 'Features' },
  { family: 'linear', label: 'Linear' },
  { family: 'activations', label: 'Activations' },
  { family: 'structure', label: 'Structure' },
  { family: 'outputs', label: 'Outputs' },
  { family: 'losses', label: 'Losses' },
  { family: 'penalties', label: 'Penalties' },
];

/**
 * The 14 pieces by family. Dragging one (Pointer Events, mouse or touch) onto the canvas inserts,
 * swaps or sets a penalty; the canvas decides what is under the pointer (`canvas/dnd.ts`). Pieces
 * whose engine support is not built yet are still draggable: the graph is just data.
 */
export function Palette() {
  const drag = useSyncExternalStore(dnd.subscribe, dnd.get, () => null);
  const dragging = drag?.payload.kind === 'palette' ? drag.payload.entry : null;
  const armed = useSyncExternalStore(dnd.subscribe, dnd.armed, () => null);
  useEffect(() => () => dnd.cancel(), []); // unmounted mid-drag: leave no ghost
  // Tap-to-add ends on Escape, or on a press that is neither on the palette nor on the drawing.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && dnd.armed()) dnd.arm(null);
    };
    const onDown = (e: PointerEvent) => {
      if (!dnd.armed()) return;
      const t = e.target instanceof Element ? e.target : null;
      if (t?.closest('.tk-palette-entry, [data-testid="canvas-svg"]')) return;
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
    <div className="tk-palette" data-testid="palette">
      <div className="tk-palette-head">
        <h2 className="tk-region-title">Pieces</h2>
        <p className="tk-palette-hint" data-testid="palette-hint">
          {HINTS.palette}
        </p>
      </div>
      {GROUPS.map((g) => (
        <section key={g.family} className="tk-palette-group" aria-label={g.label}>
          <h3>{g.label}</h3>
          {PALETTE.filter((e) => e.family === g.family).map((entry) => (
            <button
              key={entry.key}
              type="button"
              className="tk-palette-entry"
              data-testid={`palette-${entry.key}`}
              data-dragging={dragging?.key === entry.key}
              data-role={chipOf(entry).role}
              data-family={entry.family}
              title={ENTRY_TITLES[entry.key] ?? entry.label}
              aria-pressed={armed?.key === entry.key}
              onClick={(e) => {
                if (e.detail === 0) dnd.arm(entry); // keyboard: Enter or Space picks the entry
              }}
              onPointerDown={(e) => {
                if (e.pointerType === 'mouse' && e.button !== 0) return;
                // Touch: the palette scrolls vertically (`touch-action: pan-y`), so a drag begins
                // only once the movement is mostly horizontal; a vertical swipe is left to scroll.
                const touch = e.pointerType !== 'mouse';
                let begun = false;
                startDrag(e, {
                  onMove: (p) => {
                    if (!begun) {
                      if (touch && Math.abs(p.dx) < Math.abs(p.dy)) return;
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
                  {chipOf(entry).glyph}
                </span>
              )}
            </button>
          ))}
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
