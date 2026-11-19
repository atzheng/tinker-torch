import { useEffect, useSyncExternalStore } from 'react';
import { PALETTE, type PaletteEntry } from '@tinker/core';
import { dnd } from '../canvas/dnd.js';
import { startDrag } from '../gestures/drag.js';
import { Tex } from '../tex/Tex.js';
import './palette.css';

/** The glyph a palette chip shows, the same one the node it creates carries (roles get a badge). */
const GLYPH: Record<string, string> = {
  input: 'x',
  target: 'y',
  output: '\\hat{y}',
  feature: 'x^2',
  linear: 'W',
  activation: '\\sigma',
  dropout: 'p',
  add: '+',
};
function chipOf(entry: PaletteEntry): { role?: 'input' | 'pred' | 'loss'; glyph?: string } {
  if (entry.family === 'losses') return { role: 'loss', glyph: 'L' };
  if (entry.family === 'penalties') return { glyph: '\\lambda' };
  const role =
    entry.key === 'input' || entry.key === 'target'
      ? 'input'
      : entry.key === 'output'
        ? 'pred'
        : undefined;
  return { role, glyph: GLYPH[entry.key] };
}

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
  useEffect(() => () => dnd.cancel(), []); // unmounted mid-drag: leave no ghost
  return (
    <div className="tk-palette" data-testid="palette">
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
