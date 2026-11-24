import {
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
  useSyncExternalStore,
  type ReactNode,
} from 'react';
import { createPortal } from 'react-dom';
import type { DocView } from '@tinker/core';
import { Tex } from '../tex/Tex.js';
import './pieceCard.css';

/**
 * The piece reference card (CONTRACTS.md section 15): a non-modal popover. One component for both
 * entry points (the palette's "?" and the inspector's "?"). Only one card is open at a time: the
 * open card's owner id lives in a tiny store, so opening another closes this one.
 */
let openOwner: string | null = null;
const subs = new Set<() => void>();
const emit = () => subs.forEach((f) => f());
const subscribe = (f: () => void) => {
  subs.add(f);
  return () => void subs.delete(f);
};
export const cardOwner = {
  get: () => openOwner,
  open(id: string) {
    openOwner = id;
    emit();
  },
  /** Closes the card, or only `id`'s card when given. */
  close(id?: string) {
    if (id !== undefined && openOwner !== id) return;
    openOwner = null;
    emit();
  },
};
/** True while the card `id` is the open one. */
export function useCardOpen(id: string): boolean {
  return useSyncExternalStore(subscribe, cardOwner.get, () => null) === id;
}

export const CARD_W = 320;
const GAP = 8;
const EDGE = 4;

type Rect = { left: number; top: number; right: number; bottom: number };
export type CardPos = { left: number; top: number; width: number; maxHeight: number };
/** The narrowest the card gets before it overlaps what it sits beside. */
const CARD_MIN_W = 200;

/**
 * Where the card (natural `height`) goes inside `box`. Beside `side` (the palette or the inspector):
 * on the right, else the left, whichever has room (the larger room when neither fits whole), its
 * top lined up with `align`. With `under`, below `side` instead (the portrait palette strip), at
 * `align` horizontally. The card scrolls inside `maxHeight` when `box` is short.
 */
export function placeCard(
  box: Rect,
  side: { left: number; right: number; bottom: number },
  align: number,
  height: number,
  under = false,
): CardPos {
  let w = Math.min(CARD_W, box.right - box.left - 2 * EDGE);
  let left: number;
  let wanted: number;
  if (under) {
    left = Math.min(Math.max(box.left + EDGE, align), box.right - EDGE - w);
    wanted = side.bottom + GAP;
  } else {
    const roomR = box.right - EDGE - (side.right + GAP);
    const roomL = side.left - GAP - (box.left + EDGE);
    const onRight = roomR >= Math.min(w, Math.max(roomL, CARD_MIN_W));
    const room = onRight ? roomR : roomL;
    if (room >= CARD_MIN_W) w = Math.min(w, room); // beside it, narrower when the canvas is tight
    left = onRight
      ? Math.min(side.right + GAP, box.right - EDGE - w)
      : Math.max(box.left + EDGE, side.left - GAP - w);
    wanted = align;
  }
  const maxHeight = Math.max(box.bottom - box.top - 2 * EDGE, 0);
  const h = Math.min(height, maxHeight);
  const top = Math.min(Math.max(wanted, box.top + EDGE), box.bottom - EDGE - h);
  return {
    left,
    top,
    width: w,
    maxHeight: Math.max(Math.min(maxHeight, box.bottom - EDGE - top), 0),
  };
}

export type CardPlacement =
  /** Beside the palette (under its strip in portrait), lined up with the entry; fixed to the viewport. */
  | { kind: 'palette'; row: HTMLElement }
  /** Beside the inspector panel, inside the canvas. */
  | { kind: 'inspector'; panel: HTMLElement; canvas: HTMLElement };

function Section({ label, children }: { label: string; children: ReactNode }) {
  return (
    <section className="tk-card-section">
      <h4>{label}</h4>
      {children}
    </section>
  );
}

function CardBody({ view }: { view: DocView }) {
  return (
    <>
      <p className="tk-card-summary" data-testid="piece-card-summary">
        {view.summary}
      </p>
      <p className="tk-card-equation" data-testid="piece-card-equation">
        <Tex src={view.equation} />
      </p>
      {view.shapes && (
        <Section label="Shapes">
          <dl className="tk-card-dl" data-testid="piece-card-shapes">
            {Object.entries(view.shapes.in).map(([name, tex]) => (
              <div key={name}>
                <dt>{name}</dt>
                <dd>
                  <Tex src={tex} />
                </dd>
              </div>
            ))}
            <div>
              <dt>out</dt>
              <dd>
                <Tex src={view.shapes.out} />
              </dd>
            </div>
          </dl>
        </Section>
      )}
      {view.params && (
        <Section label="Parameters">
          <p data-testid="piece-card-params">
            <Tex src={view.params} />
          </p>
        </Section>
      )}
      {view.settings.length > 0 && (
        <Section label="Settings">
          <dl className="tk-card-dl">
            {view.settings.map((s) => (
              <div key={s.field}>
                <dt>
                  <code>{s.field}</code>
                </dt>
                <dd>{s.text}</dd>
              </div>
            ))}
          </dl>
        </Section>
      )}
      {view.torch && (
        <Section label="Torch">
          <code className="tk-card-torch" data-testid="piece-card-torch">
            {view.torch}
          </code>
        </Section>
      )}
      {view.notes.length > 0 && (
        <ul className="tk-card-notes">
          {view.notes.map((n) => (
            <li key={n}>{n}</li>
          ))}
        </ul>
      )}
      {view.seeAlso.length > 0 && (
        <p className="tk-card-see" data-testid="piece-card-see-also">
          See also: {view.seeAlso.map((s) => s.label).join(', ')}
        </p>
      )}
    </>
  );
}

/**
 * The card. Closes on Escape (focus goes back to `opener`, the "?" button), on a pointerdown outside
 * it and outside the opener, and on its ×. `onClose` is called for each.
 */
export function PieceCard({
  view,
  opener,
  placement,
  onClose,
}: {
  view: DocView;
  opener: HTMLElement | null;
  placement: CardPlacement;
  onClose: () => void;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const [pos, setPos] = useState<CardPos | null>(null);
  const close = useRef(onClose);
  close.current = onClose;

  // Placed after every render (the view changes with live edits, the inspector moves with the
  // canvas); state only changes when the numbers do.
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    const h = el.scrollHeight;
    let next: CardPos;
    if (placement.kind === 'palette') {
      const bar = placement.row.closest('.tk-palette');
      const b = (bar ?? placement.row).getBoundingClientRect();
      const row = placement.row.getBoundingClientRect();
      const vp = { left: 0, top: 0, right: window.innerWidth, bottom: window.innerHeight };
      const strip = bar?.getAttribute('data-strip') === 'true';
      next = placeCard(vp, b, strip ? row.left : row.top, h, strip);
    } else {
      const c = placement.canvas.getBoundingClientRect();
      const p = placement.panel.getBoundingClientRect();
      const side = { left: p.left - c.left, right: p.right - c.left, bottom: p.bottom - c.top };
      next = placeCard(
        { left: 0, top: 0, right: c.width, bottom: c.height },
        side,
        p.top - c.top,
        h,
      );
    }
    setPos((o) =>
      o &&
      o.left === next.left &&
      o.top === next.top &&
      o.width === next.width &&
      o.maxHeight === next.maxHeight
        ? o
        : next,
    );
  });

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== 'Escape') return;
      // The canvas and the palette treat Escape as deselect or disarm; here it only closes the card.
      e.stopPropagation();
      close.current();
      opener?.focus();
    };
    const onDown = (e: PointerEvent) => {
      const t = e.target instanceof Node ? e.target : null;
      if (t && (ref.current?.contains(t) || opener?.contains(t))) return;
      close.current();
    };
    window.addEventListener('keydown', onKey, true);
    window.addEventListener('pointerdown', onDown, true);
    return () => {
      window.removeEventListener('keydown', onKey, true);
      window.removeEventListener('pointerdown', onDown, true);
    };
  }, [opener]);

  const fixed = placement.kind === 'palette';
  const card = (
    <div
      ref={ref}
      className="tk-piece-card"
      data-testid="piece-card"
      data-key={view.key}
      data-placement={placement.kind}
      role="dialog"
      aria-modal="false"
      aria-label={`${view.label} reference`}
      style={{
        position: fixed ? 'fixed' : 'absolute',
        left: pos?.left ?? 0,
        top: pos?.top ?? 0,
        width: pos?.width,
        maxHeight: pos?.maxHeight,
        visibility: pos ? 'visible' : 'hidden',
      }}
    >
      <div className="tk-card-head">
        <h3>{view.label}</h3>
        <button
          type="button"
          className="tk-card-close"
          aria-label="Close reference"
          data-testid="piece-card-close"
          onClick={() => {
            close.current();
            opener?.focus();
          }}
        >
          {'×'}
        </button>
      </div>
      <CardBody view={view} />
    </div>
  );
  return createPortal(card, fixed ? document.body : placement.canvas);
}
