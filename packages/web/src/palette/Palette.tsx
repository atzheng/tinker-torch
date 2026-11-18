import { useSyncExternalStore } from 'react';
import { PALETTE, type PaletteEntry } from '@tinker/core';
import { dnd } from '../canvas/dnd.js';
import { startDrag } from '../gestures/drag.js';
import './palette.css';

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
              onPointerDown={(e) => {
                if (e.pointerType === 'mouse' && e.button !== 0) return;
                startDrag(e, {
                  onStart: (p) => dnd.begin({ kind: 'palette', entry }, p.x, p.y),
                  onMove: (p) => dnd.move(p.x, p.y),
                  onEnd: (p) => dnd.end(p.x, p.y),
                  onCancel: () => dnd.cancel(),
                });
              }}
            >
              {entry.label}
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
