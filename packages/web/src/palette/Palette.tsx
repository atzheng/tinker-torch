import { useEffect, useSyncExternalStore } from 'react';
import { PALETTE, type PaletteEntry } from '@tinker/core';
import { dnd } from '../canvas/dnd.js';
import { HINTS } from '../hints.js';
import { startDrag } from '../gestures/drag.js';
import { Tex } from '../tex/Tex.js';
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
  weight_penalty: '\\lambda',
};
function chipOf(entry: PaletteEntry): { role?: 'input' | 'pred' | 'loss'; glyph?: string } {
  if (entry.kind === 'macro') {
    return entry.family === 'losses' ? { role: 'loss', glyph: 'L' } : { glyph: GLYPH[entry.key] };
  }
  const role = entry.key === 'input' || entry.key === 'target' ? 'input' : undefined;
  return { role, glyph: GLYPH[entry.key] };
}

const GROUP = 'Adds the pieces that make it. Drop it on a wire or on empty canvas.';

/** One plain phrase per entry: what the piece is and what it connects to (the chip's tooltip). */
export const ENTRY_TITLES: Record<string, string> = {
  input: 'Input: the data columns. Connects out to the next piece.',
  target: 'Target: the true answers. Connects to an error or a loss.',
  feature: 'Feature: new columns made from the inputs. Goes between two pieces.',
  linear: 'Linear: weighted sum of its inputs. Goes between two pieces.',
  activation: 'Activation: bends the numbers. Goes between two pieces.',
  dropout: 'Dropout: randomly zeroes values in training. Goes between two pieces.',
  add: 'Add: sums two inputs. Goes between two pieces.',
  sigmoid: 'Sigmoid: turns a number into a probability. Goes between two pieces.',
  softmax: 'Softmax: turns numbers into probabilities that sum to 1. Goes between two pieces.',
  subtract: 'Subtract: a minus b. Goes between two pieces.',
  square: 'Square: each number times itself. Goes between two pieces.',
  abs: 'Absolute value: drops the sign. Goes between two pieces.',
  huber: 'Huber: squared for small values, linear for large ones. Goes between two pieces.',
  log_loss: 'Log loss: per example, from a probability and the target.',
  cross_entropy: 'Cross-entropy: per example, from probabilities and the target class.',
  mean: 'Mean: averages over the examples into one number. Goes between two pieces.',
  scale: 'Scale: multiplies by a constant. Goes between two pieces.',
  weights: 'Weights: reads the weights of a Linear piece.',
  l2: 'L2 norm: sum of the squared weights.',
  l1: 'L1 norm: sum of the absolute weights.',
  loss_squared_error: `Squared error loss. ${GROUP}`,
  loss_absolute_error: `Absolute error loss. ${GROUP}`,
  loss_huber: `Huber loss. ${GROUP}`,
  loss_log_loss: `Log loss, averaged. ${GROUP}`,
  loss_cross_entropy: `Cross-entropy loss, averaged. ${GROUP}`,
  weight_penalty: `Weight penalty: discourages large weights. ${GROUP}`,
};

const GROUPS: { family: PaletteEntry['family']; label: string }[] = [
  { family: 'inputs', label: 'Inputs' },
  { family: 'features', label: 'Features' },
  { family: 'linear', label: 'Linear' },
  { family: 'activations', label: 'Activations' },
  { family: 'structure', label: 'Structure' },
  { family: 'links', label: 'Links' },
  { family: 'math', label: 'Math' },
  { family: 'errors', label: 'Errors' },
  { family: 'reductions', label: 'Reductions' },
  { family: 'losses', label: 'Losses' },
  { family: 'parameters', label: 'Parameters' },
];

/**
 * The pieces and ready-made groups by family. Dragging one (Pointer Events, mouse or touch) onto the
 * canvas inserts it on a wire, swaps it for a piece of its family, or adds a group; the canvas
 * decides what is under the pointer (`canvas/dnd.ts`).
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
              data-kind={entry.kind}
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
                  <Tex src={chipOf(entry).glyph ?? ''} />
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
