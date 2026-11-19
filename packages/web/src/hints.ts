/**
 * The one-line gesture hints (CONTRACTS.md section 10 lists them as the exception to "no lesson
 * text"). Keep them few, plain and one sentence each.
 */
export const HINTS = {
  palette:
    'Drag a piece onto a wire, or tap it and then tap a wire. Any piece or ready-made group can also go on empty canvas; it then needs wiring in before it is used.',
  insert: 'Drag a piece onto a wire to add it.',
  connect: "Drag from a piece's right dot to connect it.",
  rewire: "Drag a wire's end to another piece's left dot to rewire.",
  empty: 'Nothing here yet. Open a preset from the top bar.',
} as const;
